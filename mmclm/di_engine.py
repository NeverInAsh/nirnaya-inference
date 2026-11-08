"""Nirnaya decision engine for the Decision Index (https://github.com/apolinario/decision-index).

Every question of a request goes into one prompt. One vLLM pooling pass produces the hidden states, and a small head
scores all options of all questions at once, returning a probability for every option.

    python -m decision_index run --engine mmclm.di_engine:NirnayaEngine --option model=<model dir> \\
        --option head=<model dir>/head.pt --option temperature_fit=<model dir>/t_entropy.json --option twice_max=3000 \\
        --option max_model_len=32768 --rows ... --out ...
"""

from __future__ import annotations

import json
import math

import torch
from decision_index.engines import Engine, Unsupported

THOUGHT = "<|channel>thought\n<channel|>"  # the empty thought add_generation_prompt appends with thinking off


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, indent=1, ensure_ascii=False)


def _entropy_t(lg, t_ab, t_global):
    """Serving temperature for one question's logits: exp(a + b*H + c*H^2 + d*log k), H = H(softmax(lg)) / log k, when
    fitted (c = d = 0 for a two-parameter fit), else t_global."""
    if t_ab is None:
        return t_global
    a, b, c, d = (*t_ab, 0.0, 0.0)[:4]
    k = lg.numel()
    lp = lg.float().log_softmax(-1)
    h = float(-(lp.exp() * lp).sum()) / math.log(k) if k > 1 else 0.0
    return math.exp(a + b * h + c * h * h + d * math.log(k))



class NirnayaEngine(Engine):
    """Text requests. Options are shown by description (the key when there is none), yes/no questions as "yes" / "no"
    (answer = p(yes)). A request longer than the window is split into several prompts by question, never truncated."""

    name = "nirnaya"
    latency = ("vLLM offline pooling engine, one request at a time: one joint prompt (all questions) per request, "
               "one pooling pass, the joint head on the same GPU")

    def __init__(self, model: str, head: str, max_model_len: str | int = 49152, gpu_mem: str | float = 0.85,
                 temperature: str | float = 1.0, **options):
        Engine.__init__(self, model=model, head=head, max_model_len=max_model_len, gpu_mem=gpu_mem,
                        temperature=temperature, **options)
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.config import PoolerConfig

        from mmclm import jh

        self.jh, self.temperature, self.max_len = jh, float(temperature), int(max_model_len)
        # temperature_fit=<json with a, b, c, d>: per question T = exp(a + b*H + c*H^2 + d*log k), H the normalised entropy
        # of the T=1 distribution, k the option count; overrides `temperature`
        self.t_ab = None
        if options.get("temperature_fit"):
            f = json.loads(open(options["temperature_fit"]).read())
            self.t_ab = (float(f["a"]), float(f["b"]), float(f.get("c", 0.0)), float(f.get("d", 0.0)))
        self.proc = AutoProcessor.from_pretrained(model)
        self.llm = LLM(model=model, runner="pooling", convert="embed", max_model_len=self.max_len,
                       gpu_memory_utilization=float(gpu_mem), enable_prefix_caching=False,
                       pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
                       limit_mm_per_prompt={"image": int(options.get("max_images") or 0), "audio": int(options.get("max_audios") or 0)},
                       **({"mm_processor_kwargs": {"max_soft_tokens": int(options["max_soft_tokens"])}} if options.get("max_soft_tokens") else {}))
        self.tok = self.llm.get_tokenizer()
        self.digit_w = jh.digit_rows(model, self.tok)
        self.head = jh.load_head(head)
        # twice_max=N: a request whose prompt is at most N tokens is read twice: the user turn is written twice and the
        # head reads the second copy, so every token of the copy has seen the whole request
        self.twice_max = int(options.get("twice_max") or 0)
        self.provenance = {"model": model, "readout": "joint head (mmclm.jh)", "head": head,
                           "temperature": self.temperature, "temperature_fit": self.t_ab, "twice_max": self.twice_max}

    def _t(self, lg):
        return _entropy_t(lg, self.t_ab, self.temperature)

    @torch.no_grad()
    def __call__(self, state, questions):
        from vllm.inputs import TokensPrompt

        from mmclm import joint

        qs, keys_of = [], []
        for qk, q in questions.items():
            kind = q.get("type", "choice")
            crit = q.get("criteria") or {}
            if kind == "noul":
                keys = ["true", "false"]
                jq = {"instruction": _text(q.get("instructions", "")).strip(), "type": "noul", "options": ["yes", "no"], "gold": 0}
            else:
                keys = list(crit.keys())
                if len(keys) > joint.MAX_OPTIONS:
                    raise Unsupported(f"{len(keys)} options per choice; the readout supports up to 255")
                if len(keys) < 2:
                    raise Unsupported("fewer than 2 options")
                opts = [_text(crit[k]) if crit.get(k) is not None and _text(crit[k]).strip() else k for k in keys]
                jq = {"instruction": _text(q.get("instructions", "")).strip(), "type": "choice", "options": opts, "gold": 0}
            qs.append((qk, kind, jq))
            keys_of.append(keys)
        head_text = "" if state in ("", None, {}, []) else _text(state).strip()

        def render(sub, twice=False):  # sub: list of (index into qs)
            shown = [(qs[i][2], list(range(len(qs[i][2]["options"])))) for i in sub]
            text = joint.prompt_text(head_text, shown)
            if twice:
                text = f"{text}\n\n{text}"
            prompt = self.proc.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            if twice and prompt.endswith(THOUGHT):
                prompt = prompt[: -len(THOUGHT)]
            ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            return ids, shown

        def spans(ids, shown, twice):
            if not twice:
                return self.jh.joint_spans(self.tok, ids, shown)
            instr = self.tok(joint.INSTRUCTION, add_special_tokens=False)["input_ids"]
            j = self.jh._find(ids, instr, 0)  # the second copy starts after the first copy's closing instruction
            sp = None if j < 0 else self.jh.joint_spans(self.tok, ids[j + len(instr):], shown)
            if sp is None:
                return None
            off = j + len(instr)
            sh = lambda ab: (ab[0] + off, ab[1] + off)  # noqa: E731
            return {"state": sh(sp["state"]), "questions": [sh(x) for x in sp["questions"]],
                    "options": [[sh(x) for x in o] for o in sp["options"]]}

        everything = list(range(len(qs)))
        # read-twice rule: the joint text, without the chat template, <= twice_max
        all_shown = [(qs[i][2], list(range(len(qs[i][2]["options"])))) for i in everything]
        twice = bool(self.twice_max) and len(self.tok(joint.prompt_text(head_text, all_shown), add_special_tokens=False)["input_ids"]) <= self.twice_max
        if twice:
            ids, shown = render(everything, True)
            sp = spans(ids, shown, True)
            twice = sp is not None and len(ids) + 4 <= self.max_len
        # split by question when the joint prompt does not fit (greedy, in order)
        groups, cur = [], []
        for i in (() if twice else range(len(qs))):
            ids, _ = render(cur + [i])
            if len(ids) + 4 > self.max_len and cur:
                groups.append(cur)
                cur = [i]
            else:
                cur.append(i)
        if cur or twice:
            groups.append(everything if twice else cur)
        prompts, metas = [], []
        for g in groups:
            ids, shown = render(g, twice)
            if len(ids) + 4 > self.max_len:
                raise Unsupported("maximum context length exceeded")
            sp = spans(ids, shown, twice)
            if sp is None:
                raise Unsupported("joint layout spans not found")
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            metas.append((g, sp))
        outs = self.llm.encode(prompts, pooling_task="token_embed", use_tqdm=False)
        feats, order = [], []
        for (g, sp), o in zip(metas, outs):
            h = o.outputs.data
            feats += self.jh.pool_joint(h, sp, len(h) - 1)
            order += g
        logits = []  # the head in chunks of 8 questions: collate copies the prompt's token memory per question
        for c0 in range(0, len(feats), 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = self.head(self.jh.collate(feats[c0:c0 + 8], "cuda")).float()
            logits += [lg[i] for i in range(lg.shape[0])]
        answers = {}
        for row_i, qi in enumerate(order):
            qk, kind, jq = qs[qi]
            keys = keys_of[qi]
            k = len(jq["options"])
            p = (logits[row_i][:k] / self._t(logits[row_i][:k])).softmax(-1).tolist()
            probs = {key: float(x) for key, x in zip(keys, p)}
            if kind == "noul":
                answers[qk] = {"type": "noul", "noul": probs["true"]}
            else:
                answers[qk] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        return {"model": "nirnaya", "answers": answers}, None



class NirnayaVisionEngine(NirnayaEngine):
    """The kit's vision suite: one lettered choice question over one or more images, read by the same head. Images come
    first in the user turn, capped at the board's 1.6 MP before the processor.

        --option max_soft_tokens=560 --option max_images=2 --option twice_max=3000

    twice_max: the text (not the images) is written twice and the head reads the second copy.
    """

    name = "nirnaya"
    latency = ("vLLM offline pooling engine, one request at a time: images + joint prompt, one pooling pass, "
               "the joint head on the same GPU")

    def __init__(self, model: str, head: str, max_soft_tokens: str | int = 560, max_images: str | int = 2, image_root=None, **options):
        super().__init__(model=model, head=head, max_soft_tokens=max_soft_tokens, max_images=max_images, **options)
        from decision_index.vision.images import MAX_PIXELS

        self.image_root, self.max_pixels = image_root, MAX_PIXELS
        self.provenance |= {"max_soft_tokens": int(max_soft_tokens), "max_images": int(max_images), "max_pixels": MAX_PIXELS,
                            "image_resize": "board cap 1.6 MP (aspect kept, LANCZOS), then the Gemma 4 processor at max_soft_tokens"}

    def _prepare(self, state, questions, images=(), prompt=None):
        """One request -> (vLLM prompt, meta) without running the model."""
        from decision_index.vision import images as IMG

        from mmclm import joint

        if len(questions) != 1:
            raise Unsupported("one choice question per request")
        qk, q = next(iter(questions.items()))
        keys = list(q["criteria"])
        if not 2 <= len(keys) <= joint.MAX_OPTIONS:
            raise Unsupported(f"{len(keys)} options")
        opts = [_text(q["criteria"][k]) if _text(q["criteria"][k]).strip() else k for k in keys]
        shown = [({"instruction": _text(q.get("instructions", "")).strip(), "type": "choice", "options": opts, "gold": 0},
                  list(range(len(keys))))]
        ims = [IMG.pil(p, self.image_root, self.max_pixels).convert("RGB") for p in images or ()]
        head_text = "" if state in ("", None, {}, []) else _text(state).strip()
        text = joint.prompt_text(head_text, shown)
        twice = bool(self.twice_max) and len(self.tok(text, add_special_tokens=False)["input_ids"]) <= self.twice_max
        body = f"{text}\n\n{text}" if twice else text
        content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": body}]
        pr = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        if twice and pr.endswith(THOUGHT):
            pr = pr[: -len(THOUGHT)]
        vp = {"prompt": pr, "multi_modal_data": {"image": ims}} if ims else pr
        return vp, {"qk": qk, "keys": keys, "shown": shown, "twice": twice, "pixels": [im.width * im.height for im in ims]}

    @torch.no_grad()
    def _finish(self, o, m):
        """The pooled hidden states of one request -> (response, raw)."""
        from mmclm import joint

        ids, h = list(o.prompt_token_ids), o.outputs.data
        if len(ids) + 4 > self.max_len:
            raise Unsupported("maximum context length exceeded")
        off = 0
        if m["twice"]:  # the second copy starts after the first copy's closing instruction
            instr = self.tok(joint.INSTRUCTION, add_special_tokens=False)["input_ids"]
            off = self.jh._find(ids, instr, 0) + len(instr)
        sp = self.jh.joint_spans(self.tok, ids[off:], m["shown"])
        if sp is None:
            raise Unsupported("joint layout spans not found")
        if off:
            sh = lambda ab: (ab[0] + off, ab[1] + off)  # noqa: E731
            sp = {"state": sh(sp["state"]), "questions": [sh(x) for x in sp["questions"]], "options": [[sh(x) for x in o_] for o_ in sp["options"]]}
        keys = m["keys"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = self.head(self.jh.collate(self.jh.pool_joint(h, sp, len(h) - 1), "cuda")).float()[0][: len(keys)]
        p = (lg / self._t(lg)).softmax(-1).tolist()
        probs = {k: float(x) for k, x in zip(keys, p)}
        img_id = getattr(self.proc.tokenizer, "image_token_id", None)
        return ({"model": "nirnaya", "answers": {m["qk"]: {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}},
                 "usage": {"input_tokens": len(ids)}},
                {"image_tokens": int(sum(t == img_id for t in ids)) if img_id is not None else None, "image_pixels": m["pixels"], "twice": m["twice"]})

    def batch(self, payloads):
        """The kit's batch call (decision_index.vision.run): one pooling pass over a chunk of requests; one (response, raw)
        or exception per payload. A failing chunk is retried one request at a time."""
        out, todo = [None] * len(payloads), []
        for i, pl in enumerate(payloads):
            try:
                todo.append((i, *self._prepare(**pl)))
            except Exception as e:
                out[i] = e
        if todo:
            try:
                outs = self.llm.encode([vp for _, vp, _ in todo], pooling_task="token_embed", use_tqdm=False)
            except Exception:
                outs = []
                for _, vp, _ in todo:
                    try:
                        outs.append(self.llm.encode([vp], pooling_task="token_embed", use_tqdm=False)[0])
                    except Exception as e:
                        outs.append(e)
            for (i, _, m), o in zip(todo, outs):
                try:
                    out[i] = o if isinstance(o, Exception) else self._finish(o, m)
                except Exception as e:
                    out[i] = e
        return out

    def __call__(self, state, questions, images=(), prompt=None):
        r = self.batch([{"state": state, "questions": questions, "images": images, "prompt": prompt}])[0]
        if isinstance(r, Exception):
            raise r
        return r


class NirnayaAudioEngine(NirnayaVisionEngine):
    """Requests with audio: any number of questions over one or more clips, read by the same head. Clips come first in
    the user turn, then the text. The text is written twice and the head reads the second copy (the clips once).

    Call with audios=[path or bytes, ...]; state is the text context (a string, or JSON-rendered otherwise).

        NirnayaAudioEngine(model=..., head=.../head.pt, temperature_fit=.../t_entropy.json, twice_max=3000, max_model_len=32768)
    """

    name = "nirnaya"
    latency = ("vLLM offline pooling engine, one request at a time: audio + joint prompt, one pooling pass, "
               "the joint head on the same GPU")
    SAMPLE_RATE = 16000

    def __init__(self, model: str, head: str, max_audios: str | int = 1, **options):
        options.setdefault("max_images", 0)
        super().__init__(model=model, head=head, max_audios=max_audios, **options)
        self.max_audios = int(max_audios)
        self.provenance |= {"max_audios": self.max_audios, "sample_rate": self.SAMPLE_RATE}

    def _audio(self, a):
        import io

        import librosa

        return librosa.load(io.BytesIO(a) if isinstance(a, (bytes, bytearray)) else a, sr=self.SAMPLE_RATE, mono=True)[0], self.SAMPLE_RATE

    def _prepare(self, state, questions, audios=()):
        """One request -> (vLLM prompt, meta) without running the model."""
        from mmclm import joint

        if len(audios) > self.max_audios:
            raise Unsupported(f"{len(audios)} clips; this engine was started with max_audios={self.max_audios}")
        qs = []
        for qk, q in questions.items():
            kind = q.get("type", "choice")
            if kind == "noul":
                keys, opts = ["true", "false"], ["yes", "no"]
            else:
                keys = list(q.get("criteria") or {})
                if not 2 <= len(keys) <= joint.MAX_OPTIONS:
                    raise Unsupported(f"{len(keys)} options")
                opts = [_text(q["criteria"][k]) if q["criteria"][k] is not None and _text(q["criteria"][k]).strip() else k for k in keys]
            qs.append((qk, kind, keys, {"instruction": _text(q.get("instructions", "")).strip(), "type": kind, "options": opts, "gold": 0}))
        shown = [(jq, list(range(len(jq["options"])))) for *_, jq in qs]
        head_text = "" if state in ("", None, {}, []) else _text(state).strip()
        text = joint.prompt_text(head_text, shown)
        twice = bool(self.twice_max) and len(self.tok(text, add_special_tokens=False)["input_ids"]) <= self.twice_max
        clips = [self._audio(a) for a in audios]
        content = [{"type": "audio"} for _ in clips] + [{"type": "text", "text": f"{text}\n\n{text}" if twice else text}]
        pr = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        if twice and pr.endswith(THOUGHT):
            pr = pr[: -len(THOUGHT)]
        vp = {"prompt": pr, "multi_modal_data": {"audio": clips}} if clips else pr
        return vp, {"qs": qs, "shown": shown, "twice": twice}

    @torch.no_grad()
    def _finish(self, o, m):
        """The pooled hidden states of one request -> (response, raw)."""
        from mmclm import joint

        ids, h = list(o.prompt_token_ids), o.outputs.data
        if len(ids) + 4 > self.max_len:
            raise Unsupported("maximum context length exceeded")
        off = 0
        if m["twice"]:  # the second copy starts after the first copy's closing instruction
            instr = self.tok(joint.INSTRUCTION, add_special_tokens=False)["input_ids"]
            off = self.jh._find(ids, instr, 0) + len(instr)
        sp = self.jh.joint_spans(self.tok, ids[off:], m["shown"])
        if sp is None:
            raise Unsupported("joint layout spans not found")
        if off:
            sh = lambda ab: (ab[0] + off, ab[1] + off)  # noqa: E731
            sp = {"state": sh(sp["state"]), "questions": [sh(x) for x in sp["questions"]], "options": [[sh(x) for x in o_] for o_ in sp["options"]]}
        feats, logits = self.jh.pool_joint(h, sp, len(h) - 1), []
        for c0 in range(0, len(feats), 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = self.head(self.jh.collate(feats[c0:c0 + 8], "cuda")).float()
            logits += [lg[i] for i in range(lg.shape[0])]
        answers = {}
        for (qk, kind, keys, jq), lg in zip(m["qs"], logits):
            x = lg[: len(keys)]
            probs = {k: float(v) for k, v in zip(keys, (x / self._t(x)).softmax(-1).tolist())}
            answers[qk] = ({"type": "noul", "noul": probs["true"]} if kind == "noul"
                           else {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs})
        return {"model": "nirnaya", "answers": answers, "usage": {"input_tokens": len(ids)}}, {"twice": m["twice"]}

    def __call__(self, state, questions, audios=()):
        r = self.batch([{"state": state, "questions": questions, "audios": audios}])[0]
        if isinstance(r, Exception):
            raise r
        return r

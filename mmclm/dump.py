"""Reference dumps ("goldens") of the text engine: what it did at every step of each request, for checking a port.

    python -m mmclm.dump --model <model dir> --requests requests.jsonl --out goldens.jsonl --states 3

`requests.jsonl` has one request per line: {"id": ..., "state": ..., "questions": {...}} (the id is optional; the
line number is used without one). Each request goes through `NirnayaEngine.__call__` as it is served, one at a time.
The engine code is not copied here: this module only records what that call computes, so the dump cannot drift from it.

`goldens.jsonl` gets one line per request:

    id          the request id
    error       present when the request was refused or failed: {"type", "message"}; no other fields then
    twice       whether the request was read twice (its text written twice, the head on the second copy)
    prompts     one entry per prompt the request became (more than one when it was split by question):
                  questions   the ids of the questions in this prompt, in the order shown
                  input_ids   the token ids sent to the model
                  T           the number of tokens (= len(input_ids))
                  end_pos     the position the head reads as "end" (= T - 1)
                  spans       {"state": [a, b], "questions": [[a, b], ...], "options": [[[a, b], ...], ...]}, token
                              positions in input_ids, end exclusive
    questions   per question id:
                  type, keys  the question type and its answer keys in the order shown
                  logits      the head's score for each key, before the temperature
                  H           the entropy of softmax(logits) divided by ln(number of keys)
                  temperature the serving temperature computed from H
                  probabilities   softmax(logits / temperature), as returned
                  answer      the answer object the engine returned

`--states N` also saves the final hidden states of the first N answered requests (`--states-for` picks them by id) to
`<out>.states.safetensors`, one float32 tensor [T, hidden] per prompt under "<id>/<prompt index>". The run's settings,
library versions, code commit and model file hashes go to `<out>.meta.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys

import torch
from decision_index.engines import Unsupported

from mmclm.di_engine import NirnayaEngine


class DumpEngine(NirnayaEngine):
    """NirnayaEngine that records the intermediate values of the request in flight (self.rec)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rec = None
        encode = self.llm.encode

        def recording_encode(prompts, *a, **k):
            outs = encode(prompts, *a, **k)
            if self.rec is not None:
                call = sys._getframe(1).f_locals  # the locals of NirnayaEngine.__call__, which is the caller
                self.rec["twice"] = bool(call["twice"])
                self.rec["groups"] = [list(g) for g, _ in call["metas"]]
                self.rec["spans"] = [sp for _, sp in call["metas"]]
                self.rec["input_ids"] = [list(p["prompt_token_ids"]) for p in prompts]
                self.rec["states"] = [o.outputs.data for o in outs]
            return outs

        self.llm.encode = recording_encode

    def _t(self, lg):
        t = super()._t(lg)
        if self.rec is not None:
            self.rec["logits"].append(lg.float().tolist())
            self.rec["temperature"].append(float(t))
        return t

    def dump(self, state, questions) -> tuple[dict, list]:
        """One request -> (its dump record, the final hidden states per prompt)."""
        self.rec = {"logits": [], "temperature": []}
        try:
            answers = self(state, questions)[0]["answers"]
        except Unsupported as e:
            return {"error": {"type": "refused", "message": str(e)}}, []
        except Exception as e:  # noqa: BLE001 - a failure is recorded, the run goes on
            return {"error": {"type": type(e).__name__, "message": str(e)}}, []
        finally:
            rec, self.rec = self.rec, None
        qids = list(questions)
        order = [i for g in rec["groups"] for i in g]  # the order the head scored the questions in
        prompts = []
        for g, sp, ids, h in zip(rec["groups"], rec["spans"], rec["input_ids"], rec["states"]):
            assert len(ids) == h.shape[0], "one hidden state per token"
            prompts.append({"questions": [qids[i] for i in g], "input_ids": ids, "T": len(ids), "end_pos": len(ids) - 1,
                            "spans": {"state": list(sp["state"]), "questions": [list(x) for x in sp["questions"]],
                                      "options": [[list(x) for x in o] for o in sp["options"]]}})
        out = {}
        for i, lg, t in zip(order, rec["logits"], rec["temperature"]):
            qk, q = qids[i], questions[qids[i]]
            kind = q.get("type", "choice")
            a = answers[qk]
            keys = ["true", "false"] if kind == "noul" else list(a["probabilities"])
            lp = torch.tensor(lg).log_softmax(-1)
            p = (torch.tensor(lg) / t).softmax(-1).tolist()
            returned = [a["noul"], 1 - a["noul"]] if kind == "noul" else [a["probabilities"][k] for k in keys]
            assert max(abs(x - y) for x, y in zip(p, returned)) < 1e-6, "the recorded logits and temperature give the returned probabilities"
            out[qk] = {"type": kind, "keys": keys, "logits": lg,
                       "H": float(-(lp.exp() * lp).sum()) / math.log(len(lg)) if len(lg) > 1 else 0.0,
                       "temperature": t, "probabilities": dict(zip(keys, p)), "answer": a}
        return {"twice": rec["twice"], "prompts": prompts, "questions": out}, rec["states"]


def versions() -> dict:
    import importlib.metadata as md

    v = {"python": platform.python_version(), "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name in ("torch", "transformers", "tokenizers", "vllm", "safetensors", "decision_index"):
        try:
            v[name] = md.version(name)
        except md.PackageNotFoundError:
            v[name] = None
    return v


def commit() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model directory (weights, head.pt, t_entropy.json)")
    ap.add_argument("--requests", required=True, help="JSONL, one request per line: id (optional), state, questions")
    ap.add_argument("--out", required=True, help="the dump, JSONL")
    ap.add_argument("--twice-max", type=int, default=3000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--states", type=int, default=0, help="save the final hidden states of the first N answered requests")
    ap.add_argument("--states-for", default="", help="comma-separated request ids to save the final hidden states of")
    a = ap.parse_args()

    reqs = []
    with open(a.requests, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                reqs.append((str(r.get("id", n)), r.get("state"), r["questions"]))
    eng = DumpEngine(model=a.model, head=f"{a.model}/head.pt", temperature_fit=f"{a.model}/t_entropy.json",
                     twice_max=a.twice_max, max_model_len=a.max_model_len, gpu_mem=a.gpu_mem)
    eng.warmup()
    want = {x for x in a.states_for.split(",") if x}
    states, left, counts = {}, a.states, {"answered": 0, "refused": 0, "failed": 0}
    with open(a.out, "w", encoding="utf-8") as f:
        for rid, state, questions in reqs:
            rec, hs = eng.dump(state, questions)
            if "error" in rec:
                counts["refused" if rec["error"]["type"] == "refused" else "failed"] += 1
            else:
                counts["answered"] += 1
                if rid in want or (not want and left > 0):
                    left -= 1
                    for j, h in enumerate(hs):
                        states[f"{rid}/{j}"] = h.float().cpu().contiguous()
            f.write(json.dumps({"id": rid, **rec}, ensure_ascii=False) + "\n")
    if states:
        from safetensors.torch import save_file

        save_file(states, a.out + ".states.safetensors")
    sums = {}
    if os.path.exists(f"{a.model}/SHA256SUMS"):
        with open(f"{a.model}/SHA256SUMS", encoding="utf-8") as f:
            sums = {name: digest for digest, name in (line.split() for line in f if line.strip())}
    meta = {"engine": "mmclm.di_engine:NirnayaEngine", "commit": commit(), "requests": len(reqs), **counts,
            "settings": {"twice_max": a.twice_max, "max_model_len": a.max_model_len, "temperature_fit": eng.t_ab,
                         "backbone_dtype": "bfloat16", "head_autocast": "bfloat16"},
            "model_sha256": sums, "states_saved": sorted(states), "versions": versions()}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("commit", "requests", "answered", "refused", "failed", "states_saved")}))


if __name__ == "__main__":
    main()

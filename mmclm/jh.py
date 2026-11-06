"""Joint head: reads the backbone's final hidden states and scores every option of every question in one pass.

For each question: the last prompt token, a question mean, one mean per option line, the state's tokens, and the model's logit for each option's first digit. Option tokens attend to each other and cross-attend to the
rest; a softmax per question gives the probabilities.
"""

from __future__ import annotations

import torch
from torch import nn







class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, x_pad, mem, mem_pad):
        y = self.n1(x)
        x = x + self.self_att(y, y, y, key_padding_mask=x_pad, need_weights=False)[0]
        y = self.n2(x)
        x = x + self.cross(y, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class JointHead(nn.Module):
    def __init__(self, hidden: int = 2816, d: int = 768, layers: int = 4, heads: int = 12):
        super().__init__()
        self.cfg = {"hidden": hidden, "d": d, "layers": layers, "heads": heads}
        self.inp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        self.kind = nn.Embedding(4, d)  # 0 field (end), 1 question, 2 option, 3 chunk
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out = nn.LayerNorm(d)
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.prior_w = nn.Parameter(torch.tensor([0.0, 0.0]))  # weight on the digit prior: [exact, shared first digit]

    def forward(self, b: dict) -> torch.Tensor:
        """b: end [B,H], question [B,H], options [B,K,H], opt_pad [B,K] (True = padding), chunks [B,C,H], chunk_pad,
        prior [B,K], exact [B]. Returns option logits [B,K] (-inf at padding)."""
        p = self.inp
        end, q = p(b["end"]) + self.kind.weight[0], p(b["question"]) + self.kind.weight[1]
        opt = p(b["options"]) + self.kind.weight[2]
        mem = torch.cat([end[:, None], q[:, None], p(b["chunks"]) + self.kind.weight[3]], 1)
        mem_pad = torch.cat([torch.zeros_like(b["chunk_pad"][:, :2]), b["chunk_pad"]], 1)
        x = torch.cat([end[:, None], opt], 1)  # the field token decides with its options
        x_pad = torch.cat([torch.zeros_like(b["opt_pad"][:, :1]), b["opt_pad"]], 1)
        for blk in self.blocks:
            x = blk(x, x_pad, mem, mem_pad)
        x = self.out(x)
        f, o = x[:, :1].expand_as(x[:, 1:]), x[:, 1:]
        s = self.score(torch.cat([f, o, f * o], -1)).squeeze(-1)
        w = torch.where(b["exact"], self.prior_w[0], self.prior_w[1])[:, None]
        s = s + w * b["prior"]
        return s.masked_fill(b["opt_pad"], float("-inf"))


def collate(items: list[dict], device) -> dict:
    """Pad a list of pooled features (from pool()) into a batch."""
    B, K = len(items), max(len(x["options"]) for x in items)
    C, H = max(len(x["chunks"]) for x in items), items[0]["end"].shape[-1]
    dt = torch.float32
    out = {"end": torch.stack([x["end"] for x in items]).to(device, dt),
           "question": torch.stack([x["question"] for x in items]).to(device, dt),
           "options": torch.zeros(B, K, H, device=device, dtype=dt), "opt_pad": torch.ones(B, K, dtype=torch.bool, device=device),
           "chunks": torch.zeros(B, C, H, device=device, dtype=dt), "chunk_pad": torch.ones(B, C, dtype=torch.bool, device=device),
           "prior": torch.zeros(B, K, device=device), "exact": torch.tensor([bool(x["exact_prior"]) for x in items], device=device)}
    for i, x in enumerate(items):
        k, c = len(x["options"]), len(x["chunks"])
        out["options"][i, :k] = x["options"].to(device, dt)
        out["opt_pad"][i, :k] = False
        out["chunks"][i, :c] = x["chunks"].to(device, dt)
        out["chunk_pad"][i, :c] = False
        out["prior"][i, :k] = x["prior"].to(device)
    return out




def load_head(path: str, device="cuda") -> JointHead:
    ck = torch.load(path, map_location="cpu")
    m = JointHead(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def digit_rows(model_dir: str, tok) -> torch.Tensor:
    """The tied embedding rows of the digits 1-9 (the LM head of Gemma 4), for the digit prior."""
    import glob

    from safetensors import safe_open

    ids = [tok.convert_tokens_to_ids(str(i)) for i in range(1, 10)]
    for f in glob.glob(f"{model_dir}/*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():  # noqa: SIM118 - safe_open has no __iter__
                if k.endswith("language_model.embed_tokens.weight"):
                    return s.get_tensor(k)[ids].clone()
    raise FileNotFoundError("embed_tokens.weight not found")




# --------------------------------------------------------------------------- joint layout
def _find(ids: list[int], sub: list[int], start: int) -> int:
    n = len(sub)
    for i in range(start, len(ids) - n + 1):
        if ids[i:i + n] == sub:
            return i
    return -1


def joint_spans(tok, input_ids: list[int], shown: list[tuple[dict, list[int]]]) -> dict | None:
    """Token spans of every question header and option line of a joint prompt (mmclm.joint) inside the processed
    input_ids (media tokens included): each piece is tokenised alone and located in order after "Questions:".
    Returns {"questions": [(a, b)], "options": [[(a, b), ...]], "state": (a, b)} or None when a piece is not found."""
    from mmclm import joint

    def ids_of(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    pos = _find(input_ids, ids_of("Questions:"), 0)
    if pos < 0:
        pos = _find(input_ids, ids_of("\nQuestions:"), 0)
    if pos < 0:
        return None
    state = (0, pos)
    cur, qs, os_ = pos, [], []
    for i, (q, perm) in enumerate(shown):
        head = f"[Q{i + 1}] {q.get('instruction') or ''}".rstrip()
        j = _find(input_ids, ids_of(head), cur)
        if j < 0:
            j = _find(input_ids, ids_of(head)[1:], cur)  # the first token can merge with the newline before it
        if j < 0:
            return None
        qa, qb = j, j + len(ids_of(head))
        cur = qb
        opts = joint.options_text(q)
        spans = []
        for k, p in enumerate(perm):
            line = f"{k + 1}. {opts[p]}"
            full, j = ids_of(line), -1
            # the first token can merge with the newline before it, the last (an option ending in whitespace) with
            # the newline after it
            for lid in (full, full[1:], full[:-1], full[1:-1]):
                if lid:
                    j = _find(input_ids, lid, cur)
                    if j >= 0:
                        break
            if j < 0:
                return None
            spans.append((j, j + len(lid)))
            cur = j + len(lid)
        qs.append((qa, qb))
        os_.append(spans)
    return {"state": state, "questions": qs, "options": os_}


MEM_CAP = 2048


def pool_joint(h: torch.Tensor, sp: dict, end_pos: int) -> list[dict]:
    """Pooled head features for every question of a joint prompt from its [T, H] states: the memory is every prompt
    token state (chunk-averaged down to MEM_CAP when longer); options and question headers are span means; end = the
    last prompt token. No digit prior in the joint layout (prior 0, exact False)."""
    def mean(a, b):
        return h[a:b].mean(0) if b > a else h[end_pos]

    T = end_pos + 1
    if T > MEM_CAP:
        edges = torch.linspace(0, T, MEM_CAP + 1).round().long().tolist()
        mem = torch.stack([h[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:]) if b > a])
    else:
        mem = h[:T]
    out = []
    for (qa, qb), spans in zip(sp["questions"], sp["options"]):
        out.append({"end": h[end_pos], "question": mean(qa, qb), "chunks": mem,
                    "options": torch.stack([mean(a, b) for a, b in spans]), "prior": torch.zeros(len(spans), device=h.device),
                    "exact_prior": False})
    return out



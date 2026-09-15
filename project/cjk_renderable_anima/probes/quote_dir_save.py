"""CPU probe 2: (a) several quote frames for EN — is there a frame-invariant 'render'
direction, and does the trained c_flat map onto any of them; (b) the artist-tag
analogue: @greatdoggo (renders a fixed logo, i.e. a pretrained context-free
mark address) — where does its code sit relative to trained kana codes."""

import sys, types, json
from pathlib import Path

sys.path.insert(0, "/home/sorryhyun/anima/anima_lora")
sys.path.insert(0, "/home/sorryhyun/anima/anima_lora/project/cjk_renderable_anima/probes")
import torch

F = torch.nn.functional
torch.set_grad_enabled(False)
from library.env import default_checkpoints
from library.inference.models import load_text_encoder
from library.inference.text import ensure_text_strategies
from library.anima.weights import load_llm_adapter
from library.anima.vocab_pack import load_vocab_pack
from library.anima.ext_vocab import T5_TABLE_SIZE
from wake.hooks import ExtDelta

ck = default_checkpoints()
pack = load_vocab_pack(ck.vocab_pack)
tok, enc = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
te = load_text_encoder(text_encoder=ck.text_encoder, dtype=torch.float32, device="cpu").eval()
adapter = load_llm_adapter(ck.dit, dtype=torch.float32, device="cpu", vocab_pack=pack).eval()
holder = types.SimpleNamespace(llm_adapter=adapter)
t5tok = tok.t5_tokenizer


def encode(captions):
    tokens = tok.tokenize(captions)
    pe, am, t5, t5m = enc.encode_tokens(tok, [te], tokens)
    return pe.float(), am, t5.long(), t5m


def out_of(captions):
    pe, am, t5, t5m = encode(captions)
    return adapter(pe, t5, target_attention_mask=t5m, source_attention_mask=am), t5


def pos(t5_row, ids):
    return [i for i, v in enumerate(t5_row.tolist()) if v in ids]


def cosm(A, b):
    return F.cosine_similarity(A, b[None].expand_as(A), dim=1).mean().item()


EN = ["HELLO", "STOP", "YES", "WAIT", "SORRY", "WHAT", "RUN", "HELP", "GO", "HEY", "NO", "OK",
      "LOVE", "FIRE", "COLD", "HOME", "NIGHT", "DREAM", "MOON", "STAR", "BOOK", "TEA", "CAT", "DOG"]
FRAMES = {
    "reads_as": 'manga, speech bubble, english text. English text reads as "{w}".',
    "bubble_reads": 'manga, 1girl. There is a speech bubble that reads "{w}".',
    "she_says": 'manga, 1girl, open mouth. She is saying "{w}".',
    "bare_quotes": 'manga, speech bubble, "{w}".',
    "plain": "manga, speech bubble, english text, {w}.",
}
codes = {}
for fname, tpl in FRAMES.items():
    o, t5 = out_of([tpl.format(w=w) for w in EN])
    rows = []
    for i, w in enumerate(EN):
        ids = set(t5tok(w, add_special_tokens=False)["input_ids"])
        p = pos(t5[i], ids)
        rows.append(o[i, p].mean(0) if p else torch.full((o.shape[-1],), float("nan")))
    codes[fname] = torch.stack(rows)
ok = ~torch.isnan(codes["plain"][:, 0])
for k in codes: ok &= ~torch.isnan(codes[k][:, 0])
print("EN words located in every frame:", int(ok.sum()))
P = codes["plain"][ok]
shifts = {k: (codes[k][ok] - P) for k in codes if k != "plain"}
dirs = {k: v.mean(0) for k, v in shifts.items()}
print("\nframe shift: |shift|/|plain|, per-word pairwise cos (shared-ness):")
for k, v in shifts.items():
    vn = v / v.norm(dim=1, keepdim=True)
    pc = ((vn @ vn.T).sum() - len(vn)) / (len(vn) * (len(vn) - 1))
    print(f"  {k:13s} {(v.norm(dim=1) / P.norm(dim=1)).mean():.2f}   {pc:+.3f}")
ks = list(dirs)
print("\ncross-frame cos of mean shifts:")
for i in range(len(ks)):
    print("  " + " ".join(f"{ks[i][:6]}~{ks[j][:6]} {F.cosine_similarity(dirs[ks[i]], dirs[ks[j]], dim=0):+.2f}" for j in range(i + 1, len(ks))))


Qs = {k: v for k, v in dirs.items()}
Qavg = torch.stack(list(dirs.values())).mean(0)
save = {"dirs": Qs, "avg": Qavg, "plain_code_norm": P.norm(dim=1).mean().item(),
        "shift_norm": {k: v.norm(dim=1).mean().item() for k, v in shifts.items()},
        "quoted_code_norm": {k: codes[k][ok].norm(dim=1).mean().item() for k in codes}}
torch.save(save, "/tmp/claude-1000/-home-sorryhyun-anima-anima-lora/ba70485e-c749-4b85-b535-8ae3ea914251/scratchpad/quote_dir.pt")
print("saved", {k: round(v,2) for k, v in save["shift_norm"].items()}, "plain code norm", round(save["plain_code_norm"],2))

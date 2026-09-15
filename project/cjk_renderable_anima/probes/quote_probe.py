"""CPU probe: is the trained c_flat (the 'render trigger') the same adapter-output
direction that the pretrained model gives EN tokens inside a `reads as "..."` frame?"""

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
from library.anima.vocab_pack import load_vocab_pack, attach_vocab_pack
from library.anima.ext_vocab import T5_TABLE_SIZE
from wake.hooks import ExtDelta

ck = default_checkpoints()
pack = load_vocab_pack(ck.vocab_pack)
tok, enc = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
te = load_text_encoder(text_encoder=ck.text_encoder, dtype=torch.float32, device="cpu").eval()
adapter = load_llm_adapter(ck.pretrained_model_name_or_path if hasattr(ck, "pretrained_model_name_or_path") else ck.dit, dtype=torch.float32, device="cpu", vocab_pack=pack).eval()
holder = types.SimpleNamespace(llm_adapter=adapter)


def encode(captions):
    tokens = tok.tokenize(captions)
    pe, am, t5, t5m = enc.encode_tokens(tok, [te], tokens)
    return pe.float(), am, t5.long(), t5m


def adapter_out(pe, am, t5, t5m):
    return adapter(pe, t5, target_attention_mask=t5m, source_attention_mask=am)


def positions(t5_row, piece_ids):
    """indices in the T5 sequence whose id is one of piece_ids (first run)."""
    idx = [i for i, v in enumerate(t5_row.tolist()) if v in piece_ids]
    return idx


t5tok = tok.t5_tokenizer
EN = ["HELLO", "STOP", "YES", "WAIT", "SORRY", "WHAT", "RUN", "HELP", "GO", "HEY", "NO", "OK",
      "LOVE", "FIRE", "COLD", "HOME", "NIGHT", "DREAM", "MOON", "STAR", "BOOK", "TEA", "CAT", "DOG"]
KANA = ["あ", "か", "す", "ぐ", "の", "み", "は", "ン"]


def word_ids(w):
    return set(t5tok(w, add_special_tokens=False)["input_ids"])


# ---- EN: quoted frame vs plain ----
quoted = [f'manga, speech bubble, english text. English text reads as "{w}".' for w in EN]
plain = [f"manga, speech bubble, english text, {w.lower()}." for w in EN]
pe, am, t5, t5m = encode(quoted + plain)
out = adapter_out(pe, am, t5, t5m)
n = len(EN)
d_list, q_list, p_list = [], [], []
for i, w in enumerate(EN):
    ids_u = word_ids(w); ids_l = word_ids(w.lower())
    pq = positions(t5[i], ids_u); pp = positions(t5[n + i], ids_l)
    if not pq or not pp:
        continue
    oq = out[i, pq].mean(0); op = out[n + i, pp].mean(0)
    q_list.append(oq); p_list.append(op); d_list.append(oq - op)
Q = torch.stack(q_list); P = torch.stack(p_list); D = torch.stack(d_list)
quote_dir = D.mean(0); quote_dir_n = quote_dir / quote_dir.norm()
Dn = D / D.norm(dim=1, keepdim=True)
pair = (Dn @ Dn.T); pair_cos = (pair.sum() - pair.diag().sum()) / (len(Dn) * (len(Dn) - 1))
print(f"EN words used {len(Dn)}; quote-shift: |d|/|out_q| = {(D.norm(dim=1) / Q.norm(dim=1)).mean():.3f}; "
      f"pairwise cos of per-word shifts {pair_cos:.3f}; cos(shift, mean quoted out) {F.cosine_similarity(quote_dir, Q.mean(0), dim=0):.3f}")
Qmean = Q.mean(0)

# ---- kana under trained tables ----
kana_caps = [f'manga, speech bubble, japanese text. Japanese text reads as "{k}".' for k in KANA]
pe_k, am_k, t5_k, t5m_k = encode(kana_caps)
ext_pos = [(t5_k[i] >= T5_TABLE_SIZE).nonzero().flatten().tolist() for i in range(len(KANA))]
print("ext positions per kana:", [len(p) for p in ext_pos])


def kana_out():
    o = adapter_out(pe_k, am_k, t5_k, t5m_k)
    return torch.stack([o[i, ext_pos[i]].mean(0) for i in range(len(KANA))])


res = {}
for arm, path in {"S0": "output/wake_probe/rows_synth_s0_s24k_S0/trained.pt",
                  "S0b": "output/wake_probe/rows_synth_s0b_s24k_S0b/trained.pt"}.items():
    sd = torch.load(f"/home/sorryhyun/anima/anima_lora/{path}", map_location="cpu", weights_only=False)
    delta = ExtDelta.from_state(holder, sd["delta"], "cpu")
    delta.scale = 0.0; o_pack = kana_out()
    delta.scale = 1.0; delta.common = None; o_f = kana_out()
    delta.common = sd["c_flat"].float(); o_fc = kana_out()
    delta.common = None
    for h in delta.handles: h.remove()
    c_img = o_fc - o_f; f_img = o_f - o_pack
    r = {
        "cos(c_img, quote_dir)": F.cosine_similarity(c_img, quote_dir_n[None], dim=1).mean().item(),
        "cos(f_img, quote_dir)": F.cosine_similarity(f_img, quote_dir_n[None], dim=1).mean().item(),
        "cos(o_fc, mean EN quoted out)": F.cosine_similarity(o_fc, Qmean[None], dim=1).mean().item(),
        "cos(o_f,  mean EN quoted out)": F.cosine_similarity(o_f, Qmean[None], dim=1).mean().item(),
        "cos(o_pack, mean EN quoted out)": F.cosine_similarity(o_pack, Qmean[None], dim=1).mean().item(),
        "cos(o_fc, mean EN plain out)": F.cosine_similarity(o_fc, P.mean(0)[None], dim=1).mean().item(),
        "|c_img|/|o_f|": (c_img.norm(dim=1) / o_f.norm(dim=1)).mean().item(),
        "|quote shift|/|out|": (D.norm(dim=1) / Q.norm(dim=1)).mean().item(),
        "c_img pairwise cos (context-free?)": (lambda X: ((X @ X.T).sum() - len(X)) / (len(X) * (len(X) - 1)))(c_img / c_img.norm(dim=1, keepdim=True)).item(),
    }
    res[arm] = r
    print(f"== {arm}")
    for k, v in r.items():
        print(f"  {k:36s} {v:+.3f}")
# baseline: random directions
rnd = torch.randn(64, 1024)
print("random-direction cos scale:", F.cosine_similarity(rnd, quote_dir_n[None], dim=1).abs().mean().item())
# EN quoted-out vs plain-out shared subspace: energy of c_img in top-8 PCs of D
U, S, V = torch.linalg.svd(D - D.mean(0), full_matrices=False)
print("quote-shift PC energy top4:", (S[:4] ** 2 / (S ** 2).sum()).tolist())
json.dump(res, open(Path(__file__).with_suffix(".json"), "w"), indent=1)

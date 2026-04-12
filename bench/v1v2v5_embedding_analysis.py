#!/usr/bin/env python
"""V1/V2/V5 — Embedding-space validation for modulation guidance.

V1: Quality-axis separation in max_pool(crossattn_emb)
V2: Quality direction consistency across content
V5: Resolution vs quality tag orthogonality

Saves results to bench/results/ and artifacts for V3/V4 to bench/.

Run from anima_lora/:
    python bench/v1v2v5_embedding_analysis.py
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import anima_utils, strategy_anima, strategy_base

# --- Config ---
DIT_PATH = "models/diffusion_models/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "models/text_encoders/qwen_3_06b_base.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"

# --- Prompts ---
CONTENT_PROMPTS = [
    "1girl, long hair, black hair, school uniform, standing, looking at viewer",
    "1boy, short hair, armor, sword, battle, dynamic pose",
    "landscape, sunset, mountains, river, clouds, scenic",
    "2girls, sitting, cafe, coffee, conversation, window",
    "1girl, witch, magic circle, floating, night sky, stars",
    "mecha, robot, explosion, city, destruction, smoke",
    "1girl, red dress, dancing, stage, spotlight, dramatic lighting",
    "group, festival, fireworks, yukata, night, crowded",
]
QUALITY_POS = "masterpiece, best quality, score_7, score_8, score_9"
QUALITY_NEG = "worst quality, low quality, score_1, score_2, score_3"
RESOLUTION_TAGS = "absurdres, highres"


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def encode_prompt(prompt, text_encoder, tok_strat, enc_strat, anima, device):
    """Encode prompt → crossattn_emb (post-LLMAdapter, padded to 512) + max_pool."""
    with torch.no_grad():
        tokens = tok_strat.tokenize(prompt)
        embed = enc_strat.encode_tokens(tok_strat, [text_encoder], tokens)
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
        mask = embed[3].to(device)
        crossattn_emb[~mask.bool()] = 0
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = F.pad(
                crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
            )
        pooled = crossattn_emb.max(dim=1).values  # (1, 1024)
    return crossattn_emb.cpu(), pooled.cpu().squeeze(0)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Modulation Guidance — Embedding Space Validation (V1/V2/V5)")
    print("=" * 60)

    # ---- Load models ----
    print("\nLoading text encoder (CPU)...")
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        TEXT_ENCODER_PATH, dtype=torch.bfloat16, device="cpu"
    )
    text_encoder.eval()

    print("Loading DiT model (for LLMAdapter)...")
    anima = anima_utils.load_anima_model(
        device=DEVICE,
        dit_path=DIT_PATH,
        attn_mode="torch",
        split_attn=False,
        loading_device=DEVICE,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.eval().requires_grad_(False)

    tok = strategy_anima.AnimaTokenizeStrategy(qwen3_path=TEXT_ENCODER_PATH)
    enc = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TokenizeStrategy.set_strategy(tok)
    strategy_base.TextEncodingStrategy.set_strategy(enc)

    def enc_prompt(prompt, _te=text_encoder, _anima=anima):
        return encode_prompt(prompt, _te, tok, enc, _anima, DEVICE)

    # ---- Encode all prompts we'll need ----
    print("\nEncoding prompts...")

    # Standalone quality / resolution
    _, pool_qpos = enc_prompt(QUALITY_POS)
    _, pool_qneg = enc_prompt(QUALITY_NEG)
    _, pool_res = enc_prompt(RESOLUTION_TAGS)

    # Content-varied: quality+ and quality- variants
    pools_pos, pools_neg, pools_bare = [], [], []
    for cp in CONTENT_PROMPTS:
        _, pp = enc_prompt(f"{QUALITY_POS}. {cp}")
        _, pn = enc_prompt(f"{QUALITY_NEG}. {cp}")
        _, pb = enc_prompt(cp)
        pools_pos.append(pp)
        pools_neg.append(pn)
        pools_bare.append(pb)

    # V5: resolution+content variants (use first 4 content prompts)
    pools_res_content, pools_both = [], []
    for cp in CONTENT_PROMPTS[:4]:
        _, pr = enc_prompt(f"{RESOLUTION_TAGS}, {cp}")
        _, prb = enc_prompt(f"{RESOLUTION_TAGS}, {QUALITY_POS}, {cp}")
        pools_res_content.append(pr)
        pools_both.append(prb)

    # Save test embeddings for V3/V4
    test_prompt = f"{QUALITY_POS}. {CONTENT_PROMPTS[0]}"
    neg_prompt_str = f"{QUALITY_NEG}. {CONTENT_PROMPTS[0]}"
    test_embed, _ = enc_prompt(test_prompt)
    neg_embed, _ = enc_prompt(neg_prompt_str)
    torch.save(
        {"test_embed": test_embed, "neg_embed": neg_embed, "prompt": test_prompt},
        BENCH_DIR / "test_embed.pt",
    )
    print("  Saved test embeddings → bench/test_embed.pt")

    # ---- Free models ----
    del text_encoder, anima
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==============================================================
    # V1: Quality-axis separation
    # ==============================================================
    print("\n" + "=" * 60)
    print("V1: Quality-axis separation in embedding space")
    print("=" * 60)

    dist_standalone = 1.0 - cosine_sim(pool_qpos, pool_qneg)

    # Cosine distance: quality+ vs quality- (same content)
    pair_dists = [
        1.0 - cosine_sim(pools_pos[i], pools_neg[i])
        for i in range(len(CONTENT_PROMPTS))
    ]

    # Baseline: same-quality, different content
    same_q_dists = []
    for i in range(len(CONTENT_PROMPTS)):
        for j in range(i + 1, len(CONTENT_PROMPTS)):
            same_q_dists.append(1.0 - cosine_sim(pools_pos[i], pools_pos[j]))

    avg_pair = float(np.mean(pair_dists))
    avg_same_q = float(np.mean(same_q_dists))
    ratio = avg_pair / (avg_same_q + 1e-8)
    v1_pass = avg_pair > avg_same_q

    print(f"  Standalone quality+ ↔ quality−:    cosine dist = {dist_standalone:.4f}")
    print(f"  Avg pos↔neg (content-varied):      cosine dist = {avg_pair:.4f}")
    print(f"  Avg same-quality (diff content):    cosine dist = {avg_same_q:.4f}")
    print(f"  Separation ratio:                   {ratio:.2f}x")
    print(f"  VERDICT: {'PASS' if v1_pass else 'FAIL'}")

    # ==============================================================
    # V2: Quality direction consistency across content
    # ==============================================================
    print("\n" + "=" * 60)
    print("V2: Quality direction consistency across content")
    print("=" * 60)

    directions = [pools_pos[i] - pools_neg[i] for i in range(len(CONTENT_PROMPTS))]
    dir_sims = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dir_sims.append(cosine_sim(directions[i], directions[j]))

    avg_sim = float(np.mean(dir_sims))
    min_sim = float(np.min(dir_sims))
    v2_pass = avg_sim > 0.7

    for idx, s in enumerate(dir_sims):
        i = 0
        count = 0
        for ii in range(len(directions)):
            for jj in range(ii + 1, len(directions)):
                if count == idx:
                    i, j = ii, jj
                count += 1
        print(f"  [{i}↔{j}] cosine = {s:.4f}")
    print(f"  Average: {avg_sim:.4f}  |  Minimum: {min_sim:.4f}")
    print(f"  VERDICT: {'PASS' if v2_pass else 'FAIL'} (threshold: avg cosine > 0.7)")

    # Save average quality direction for V3/V4
    avg_direction = torch.stack(directions).mean(dim=0)
    torch.save(avg_direction, BENCH_DIR / "quality_direction.pt")
    print("  Saved quality direction → bench/quality_direction.pt")

    # ==============================================================
    # V5: Resolution vs quality tag orthogonality
    # ==============================================================
    print("\n" + "=" * 60)
    print("V5: Resolution vs quality tag orthogonality")
    print("=" * 60)

    # Compute per-content resolution and quality directions
    res_dirs, qual_dirs = [], []
    for i in range(4):
        res_dirs.append(pools_res_content[i] - pools_bare[i])
        qual_dirs.append(pools_pos[i] - pools_bare[i])

    # Cross-axis cosine similarity (should be low)
    cross_sims = []
    for i in range(4):
        cross_sims.append(cosine_sim(res_dirs[i], qual_dirs[i]))

    avg_cross = float(np.mean(cross_sims))

    # Within-axis consistency (should be high)
    res_self_sims = [
        cosine_sim(res_dirs[i], res_dirs[j]) for i in range(4) for j in range(i + 1, 4)
    ]
    qual_self_sims = [
        cosine_sim(qual_dirs[i], qual_dirs[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]

    # Also check standalone quality vs resolution direction
    standalone_cross = cosine_sim(pool_qpos - pool_qneg, pool_res - pools_bare[0])

    v5_pass = abs(avg_cross) < 0.3

    for i in range(4):
        print(f"  Content[{i}]: res↔qual cosine = {cross_sims[i]:.4f}")
    print(f"  Average cross-axis cosine:     {avg_cross:.4f}")
    print(f"  Standalone res↔qual cosine:    {standalone_cross:.4f}")
    print(f"  Res self-consistency:          {float(np.mean(res_self_sims)):.4f}")
    print(f"  Quality self-consistency:      {float(np.mean(qual_self_sims)):.4f}")
    print(
        f"  VERDICT: {'PASS' if v5_pass else 'FAIL'} (threshold: |cross cosine| < 0.3)"
    )

    # ==============================================================
    # Save results
    # ==============================================================
    results = {
        "v1": {
            "standalone_cos_dist": dist_standalone,
            "avg_pos_neg_cos_dist": avg_pair,
            "avg_same_quality_cos_dist": avg_same_q,
            "separation_ratio": ratio,
            "pair_dists": pair_dists,
            "pass": v1_pass,
        },
        "v2": {
            "direction_cosine_sims": dir_sims,
            "avg_cosine_sim": avg_sim,
            "min_cosine_sim": min_sim,
            "pass": v2_pass,
        },
        "v5": {
            "cross_axis_cosine_sims": cross_sims,
            "avg_cross_axis_cosine": avg_cross,
            "standalone_cross_cosine": standalone_cross,
            "res_self_consistency": float(np.mean(res_self_sims)),
            "qual_self_consistency": float(np.mean(qual_self_sims)),
            "pass": v5_pass,
        },
    }
    out_path = RESULTS_DIR / "v1v2v5_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    all_pass = v1_pass and v2_pass and v5_pass
    print(f"\n{'=' * 60}")
    print(
        f"SUMMARY:  V1={'PASS' if v1_pass else 'FAIL'}  V2={'PASS' if v2_pass else 'FAIL'}  V5={'PASS' if v5_pass else 'FAIL'}"
    )
    print(f"{'=' * 60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

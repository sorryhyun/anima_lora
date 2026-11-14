#!/usr/bin/env python3
"""What the text rows of MOSS-Image hold as they pass the 42 blocks, on colorize.

Teacher-forced, colorize v5 (its stamped isolated reference at ``ref_timestep`` 0,
``RefCache``), illustration pairs. The bare caption of an illustration is the empty
prompt, so its text rows (the chat template alone) enter block 0 identical on every
page: whatever differs between pages further up was read from the image or the
reference.

  dynamics  per block: cosine of the text rows to their block-0 input, the relative
            update, the text queries' attention mass on image / template / caption /
            reference keys, the image queries' mass on template / caption keys
            (empty and full caption, the first ``--att_pages`` pages)
  content   empty caption, every page. Share of the text rows' energy that differs
            between pages; ridge probes (5 folds over pages) from the text rows and
            from the mean image row of each block to ``palette`` (spatial mean of
            target − reference latent, 128-d) and ``layout`` (4×4 cell RMS of the
            reference latent); the same probes from the mean of ``x_t`` as the floor
  causal    the first ``--causal_pages`` pages. The text rows fed into a block range
            are replaced, the image rows left alone, and the velocity compared with
            the live forward's (mean squared difference, and flow MSE):
              own         the live forward's own rows (control: no change)
              jitter      own rows × (1 + 1e-3·ε): the floor under every arm below
                          (on a bf16 stream, the response to a last-bit change)
              jitter_norm own rows + an isotropic draw with swap_page's row norms
              swap_seed   rows of the same page under another noise draw
              swap_page   rows of another page at the same σ (all blocks, and the
                          ranges 1–13 / 14–25 / 26–41)
              frozen      the block-0 input at every block (static text)
            ``caption``: the full caption against the empty one, as a scale;
            ``full_frozen``: static text under the full caption, against its live

Read, fixed before the run:
  - frozen ≈ live: the text rows' updates are not read; text is static context.
  - frozen moves the velocity, swap_page ≈ swap_seed: the updates are a per-σ
    program that does not depend on the page.
  - swap_page above swap_seed: the text rows carry page state that the image rows
    read back, the candidate carrier for what a teacher with the image in its text
    input knows.

The DiT is generation's: int8 block linears, fp32 stream (``--base`` / ``--stream``).
Not compiled: the probe patches ``_attention`` and rewrites rows between blocks.

    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_text_stream/probe.py"
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from bench._common import make_run_dir, start_heartbeat, write_result

RANGES = {"1-13": range(1, 14), "14-25": range(14, 26), "26-41": range(26, 42)}
ARMS = (
    "own",
    "jitter",
    "jitter_norm",
    "swap_seed",
    "swap_page",
    *(f"swap_page_{r}" for r in RANGES),
    "frozen",
)
Q_STATS = ("image", "template", "caption", "ref")  # text queries' mass
I_STATS = ("template", "caption")  # image queries' mass


class Stream:
    """``layer_hook`` + ``_attention`` tap. ``feed[j]``: rows to put in as block
    ``j``'s text input (teacher forcing); ``rows[j]``: the text input block ``j``
    saw (``rows[L]`` = the last block's output)."""

    def __init__(self, num_blocks, heads, keep):
        self.L, self.H, self.keep = num_blocks, heads, keep
        self.block = None
        self.active = False

    def start(self, n_img, n_text, feed=None, attention=None, record=True):
        self.active, self.n, self.t = True, n_img, n_text
        self.feed, self.record = feed, record
        self.rows = [None] * (self.L + 1)
        self.img_mean = [None] * self.L
        self.cap = attention  # (pre, suf) template bounds, or None
        self.q_att = np.full((len(Q_STATS), self.L), np.nan, np.float32)
        self.i_att = np.full((len(I_STATS), self.L), np.nan, np.float32)

    def first_input(self, x):
        if self.active and x.shape[1] == self.n + self.t:
            self.rows[0] = x[0, self.n :].float().cpu()

    def attention(self, q, k, rope, heads, ref_kv):
        from library.moss.model import rotate

        if not self.active or self.cap is None or self.block is None:
            return
        n, t = self.n, self.t
        if q.shape[1] != n + t:
            return
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        qr = rotate(q.reshape(shape), rope).transpose(1, 2)[0].float()
        kr = rotate(k.reshape(shape), rope).transpose(1, 2)[0].float()
        if ref_kv is not None:
            ref_k, _, ref_rope = ref_kv
            rs = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
            kr = torch.cat(
                (kr, rotate(ref_k.reshape(rs), ref_rope).transpose(1, 2)[0].float()), 1
            )
        scale = 1.0 / math.sqrt(shape[-1])
        pre, suf = self.cap
        bounds = {
            "image": [(0, n)],
            "template": [(n, n + pre), (n + t - suf, n + t)],
            "caption": [(n + pre, n + t - suf)],
            "ref": [(n + t, kr.shape[1])],
        }

        def mass(p, name):
            return sum(p[..., a:b].sum(-1) for a, b in bounds[name]).mean().item()

        p = torch.softmax((qr[:, n:] @ kr.transpose(-1, -2)) * scale, -1)
        for i, name in enumerate(Q_STATS):
            self.q_att[i, self.block] = mass(p, name)
        idx = torch.linspace(0, n - 1, 384, device=q.device).long()
        acc = np.zeros(len(I_STATS))
        for h0 in range(0, heads, 6):
            p = torch.softmax(
                (qr[h0 : h0 + 6, idx] @ kr[h0 : h0 + 6].transpose(-1, -2)) * scale, -1
            )
            for i, name in enumerate(I_STATS):
                acc[i] += mass(p, name) * p.shape[0] / heads
        self.i_att[:, self.block] = acc

    def __call__(self, index, x, n_img):
        if not self.active or x.shape[1] != self.n + self.t:
            return x
        n = self.n
        if self.record:
            self.rows[index + 1] = x[0, n:].float().cpu()
            self.img_mean[index] = x[0, :n].float().mean(0).cpu()
        nxt = index + 1
        if self.feed is not None and nxt < self.L and self.feed[nxt] is not None:
            rows = self.feed[nxt].to(x.device, x.dtype)[None]
            x = torch.cat((x[:, :n], rows), 1)
        return x


def load_dit(model_dir, model, lora, device, base="int8", stream="fp32"):
    """The DiT as ``generate`` builds it: int8 block linears (``base int8``) and the
    fp32 stream by default, the LoRA and its stamped reference settings on
    (``lora`` None: Base alone)."""
    from library.moss.loader import load_transformer, stream_fp32_
    from library.moss.lora import load_for_inference as load_lora

    convert = None
    if base == "int8":
        from library.moss.dit_int8.base import generate_base
        from library.moss.requests import GenerateRequest

        convert = generate_base(GenerateRequest(lora=str(lora or ""), model=model))
    transformer = load_transformer(model_dir, model, device=device)
    if lora:
        network = load_lora(transformer, lora, 1.0, before_apply=convert)
        network.to(device)
    elif convert is not None:
        convert(transformer)
    if stream == "fp32":
        stream_fp32_(transformer)
    return transformer


def ridge_r2(x, y, folds, device):
    """Held-out variance-weighted R² of a ridge probe, λ by leave-one-out on each
    fold's training pages. ``x (N, F)``, ``y (N, T)``."""
    x, y = x.to(device).float(), y.to(device).float()
    sse = sst = 0.0
    for k in range(int(folds.max()) + 1):
        tr, te = folds != k, folds == k
        mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
        xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
        ymu = y[tr].mean(0)
        ytr = y[tr] - ymu
        gram = xtr @ xtr.T
        eye = torch.eye(gram.shape[0], device=device)
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            inv = torch.linalg.inv(gram + lam * gram.diagonal().mean() * eye)
            alpha = inv @ ytr
            loo = (alpha / inv.diagonal()[:, None]).pow(2).sum().item()
            if best is None or loo < best[0]:
                best = (loo, alpha)
        pred = xte @ xtr.T @ best[1]
        sse += (pred - (y[te] - ymu)).pow(2).sum().item()
        sst += (y[te] - ymu).pow(2).sum().item()
    return 1.0 - sse / sst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--lora", default="output/ckpt_moss_i2i/moss_colorize_v5.safetensors"
    )
    ap.add_argument("--cache", default="post_image_dataset/moss/i2i_colorize_probe160")
    ap.add_argument("--pages", type=int, default=160)
    ap.add_argument("--causal_pages", type=int, default=32)
    ap.add_argument("--att_pages", type=int, default=24)
    ap.add_argument("--sigmas", default="0.95,0.6,0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--base", default="int8", choices=("int8", "bf16"))
    ap.add_argument("--stream", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_text_stream")

    import library.moss.model as moss_model
    from bench.moss_sinks.probe_attention import template_split
    from library.env import resolve_under_home
    from library.moss.caching.items import list_items, load_item, load_ref
    from library.moss.loader import load_tokenizer
    from library.moss.requests import resolve_model_dir
    from safetensors.torch import load_file

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    pages = []
    for it in list_items(resolve_under_home(args.cache)):
        if it.ref_path is None:
            continue
        latent, embeds = load_item(it)
        ref = load_ref(it)
        if ref.shape != latent.shape:
            continue
        pages.append(
            dict(
                stem=it.stem,
                latent=latent[None],
                ref=ref[None],
                full=embeds,
                te=it.te_path,
            )
        )
        if len(pages) >= args.pages:
            break
    # The bare caption the LoRA trained on: the cache's own encode of ''.
    plains = [load_file(p["te"])["embeds_plain"] for p in pages[:8]]
    assert all(torch.equal(plains[0], e) for e in plains[1:])
    empty = plains[0]
    tok = load_tokenizer(model_dir)
    pre, cap_len, suf = template_split(tok, "")
    del tok
    assert cap_len == 0 and empty.shape[0] == pre + suf, (
        pre,
        cap_len,
        suf,
        empty.shape,
    )
    empty = empty[None].to(device, torch.bfloat16)
    S = empty.shape[1]
    print(f"[probe] {len(pages)} pages, template {pre}+{suf} rows", flush=True)

    transformer = load_dit(
        model_dir,
        args.model,
        resolve_under_home(args.lora),
        device,
        args.base,
        args.stream,
    )
    transformer.ref_cache = moss_model.RefCache()
    num_blocks, heads = len(transformer.layers), transformer.config.num_heads
    dim = transformer.config.dim
    keep = torch.ones(dim)
    keep[torch.from_numpy(np.load(resolve_under_home(args.massive))["channels"])] = 0
    stream = Stream(num_blocks, heads, keep)
    original = moss_model._attention

    def wrapped(q, k, v, rope, heads_, *rest, **kw):
        ref_kv = rest[3] if len(rest) > 3 else kw.get("ref_kv")
        stream.attention(q, k, rope, heads_, ref_kv)
        return original(q, k, v, rope, heads_, *rest, **kw)

    def set_block(index):
        def hook(_m, a):
            stream.block = index
            if index == 0:
                stream.first_input(a[0])

        return hook

    handles = [transformer.text_embedder.register_forward_pre_hook(set_block(None))]
    for i, layer in enumerate(transformer.layers):
        handles.append(layer.register_forward_pre_hook(set_block(i)))
    moss_model._attention = wrapped
    transformer.layer_hook = stream

    P, Z, C, A = (
        len(pages),
        len(sigmas),
        min(args.causal_pages, len(pages)),
        args.att_pages,
    )
    L = num_blocks
    text_rows = torch.zeros(P, Z, L + 1, S, dim, dtype=torch.float32)
    img_mean = torch.zeros(P, Z, L, dim, dtype=torch.float32)
    xt_mean = torch.zeros(P, Z, 128)
    seed_rows = torch.zeros(C, Z, L + 1, S, dim, dtype=torch.float32)
    v_live = {}
    targets = {}
    mse = np.full((C, Z, len(ARMS) + 2), np.nan, np.float32)  # live, arms…, caption
    dv = np.full((C, Z, len(ARMS) + 1), np.nan, np.float32)  # arms…, caption
    dv_full_frozen = np.full((C, Z), np.nan, np.float32)
    q_att = np.full((2, A, Z, len(Q_STATS), L), np.nan, np.float32)
    i_att = np.full((2, A, Z, len(I_STATS), L), np.nan, np.float32)
    drift = np.full((2, A, Z, 2, L), np.nan, np.float32)  # cos to input, rel update

    def noisy(page, pi, zi, offset):
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + offset + 1000 * pi + zi
        )
        latent = page["latent"].to(device)
        noise = torch.randn(latent.shape, generator=gen).to(device)
        sigma = sigmas[zi]
        return (1.0 - sigma) * latent + sigma * noise, noise - latent

    def run(x_t, sigma, embeds, ref):
        t = torch.tensor([1000.0 * sigma], device=device)
        return transformer(x_t, t, embeds, return_dict=False, ref=ref)[0].float()

    def drift_stats(rows):
        r = torch.stack(rows) * keep  # (L+1, S, D)
        cos = F.cosine_similarity(r[1:], r[:1], dim=-1).mean(-1)
        rel = (r[1:] - r[:-1]).norm(dim=-1) / r[:-1].norm(dim=-1)
        return torch.stack((cos, rel.mean(-1))).numpy()

    try:
        with torch.inference_mode():
            # pass 1: live forwards on the empty caption, every page
            for pi, page in enumerate(pages):
                ref = page["ref"].to(device)
                n_img = page["latent"].shape[-2] * page["latent"].shape[-1]
                for zi, sigma in enumerate(sigmas):
                    x_t, target = noisy(page, pi, zi, 0)
                    att = (pre, suf) if pi < A else None
                    stream.start(n_img, S, attention=att)
                    v = run(x_t, sigma, empty, ref)
                    text_rows[pi, zi] = torch.stack(stream.rows)
                    img_mean[pi, zi] = torch.stack(stream.img_mean)
                    xt_mean[pi, zi] = x_t[0].float().mean((1, 2)).cpu()
                    if pi < A:
                        q_att[0, pi, zi], i_att[0, pi, zi] = stream.q_att, stream.i_att
                        drift[0, pi, zi] = drift_stats(stream.rows)
                    if pi < C:
                        v_live[pi, zi] = v
                        mse[pi, zi, 0] = (v - target).pow(2).mean().item()
                    stream.active = False
                d = page["latent"][0] - page["ref"][0]
                targets[pi] = (
                    d.mean((1, 2)),
                    F.adaptive_avg_pool2d(page["ref"][0].pow(2).mean(0)[None], 4)
                    .sqrt()
                    .flatten(),
                )
                if pi % 20 == 0:
                    print(f"[probe] live {pi + 1}/{P} {page['stem']}", flush=True)
            # pass 2: the causal arms
            for pi in range(C):
                page = pages[pi]
                ref = page["ref"].to(device)
                n_img = page["latent"].shape[-2] * page["latent"].shape[-1]
                full = page["full"][None].to(device, torch.bfloat16)
                for zi, sigma in enumerate(sigmas):
                    x_b, _ = noisy(page, pi, zi, 500_000)
                    stream.start(n_img, S)
                    run(x_b, sigma, empty, ref)
                    seed_rows[pi, zi] = torch.stack(stream.rows)
                    stream.active = False
                    x_t, target = noisy(page, pi, zi, 0)
                    own = text_rows[pi, zi]
                    donor = text_rows[(pi + 1) % C, zi]
                    gen = torch.Generator().manual_seed(args.seed + 7 * pi + zi)
                    o, dn = own.float(), donor.float()
                    eps = torch.randn(o.shape, generator=gen)
                    iso = torch.randn(o.shape, generator=gen)
                    iso = (
                        iso
                        * (dn - o).norm(dim=-1, keepdim=True)
                        / iso.norm(dim=-1, keepdim=True)
                    )
                    feeds = {
                        "own": [None, *own[1:L]],
                        "jitter": [None, *(o * (1 + 1e-3 * eps))[1:L]],
                        "jitter_norm": [None, *(o + iso)[1:L]],
                        "swap_seed": [None, *seed_rows[pi, zi, 1:L]],
                        "swap_page": [None, *donor[1:L]],
                        "frozen": [None] + [own[0]] * (L - 1),
                    }
                    for name, blocks in RANGES.items():
                        feeds[f"swap_page_{name}"] = [None] + [
                            donor[j] if j in blocks else own[j] for j in range(1, L)
                        ]
                    for ai, arm in enumerate(ARMS):
                        stream.start(n_img, S, feed=feeds[arm], record=False)
                        v = run(x_t, sigma, empty, ref)
                        stream.active = False
                        mse[pi, zi, 1 + ai] = (v - target).pow(2).mean().item()
                        dv[pi, zi, ai] = (v - v_live[pi, zi]).pow(2).mean().item()
                    t_full = full.shape[1]
                    att = (pre, suf) if pi < A else None
                    stream.start(n_img, t_full, attention=att)
                    v = run(x_t, sigma, full, ref)
                    if pi < A:
                        q_att[1, pi, zi], i_att[1, pi, zi] = stream.q_att, stream.i_att
                        drift[1, pi, zi] = drift_stats(stream.rows)
                    stream.active = False
                    mse[pi, zi, -1] = (v - target).pow(2).mean().item()
                    dv[pi, zi, -1] = (v - v_live[pi, zi]).pow(2).mean().item()
                    static = [None] + [stream.rows[0]] * (L - 1)
                    stream.start(n_img, t_full, feed=static, record=False)
                    v_static = run(x_t, sigma, full, ref)
                    stream.active = False
                    dv_full_frozen[pi, zi] = (v_static - v).pow(2).mean().item()
                print(
                    f"[probe] causal {pi + 1}/{C} dv {dv[pi, 0].round(4)}", flush=True
                )
    finally:
        moss_model._attention = original
        transformer.layer_hook = None
        transformer.ref_cache = None
        for h in handles:
            h.remove()
    transformer = None  # frees the DiT; run() closes over it
    torch.cuda.empty_cache()

    # --- summary -------------------------------------------------------------------
    def r3(a):
        return [round(float(v), 4) for v in a]

    in_spread = (text_rows[:, 0, 0].float() - text_rows[:1, 0, 0].float()).abs().max()
    summary = {"block0_input_max_diff_between_pages": float(in_spread)}
    causal = {}
    for zi, sigma in enumerate(sigmas):
        row = {"flow_mse_live": round(float(mse[:, zi, 0].mean()), 4)}
        for ai, arm in enumerate((*ARMS, "caption")):
            row[arm] = {
                "dv": round(float(dv[:, zi, ai].mean()), 5),
                "flow_mse": round(float(mse[:, zi, 1 + ai].mean()), 4),
            }
        row["full_frozen"] = {"dv": round(float(dv_full_frozen[:, zi].mean()), 5)}
        ratio = dv[:, zi, ARMS.index("swap_page")] / dv[:, zi, ARMS.index("swap_seed")]
        row["swap_page_over_seed_median"] = round(float(np.median(ratio)), 2)
        row["swap_page_over_seed_pages_above_1"] = int((ratio > 1).sum())
        causal[f"sigma_{sigma}"] = row
    summary["causal"] = causal

    keep_d = keep.bool()
    content = {}
    folds = (torch.arange(P) % 5).to(device)
    y_pal = torch.stack([targets[p][0] for p in range(P)])
    y_lay = torch.stack([targets[p][1] for p in range(P)])
    for zi, sigma in enumerate(sigmas):
        rows = text_rows[:, zi].float()[..., keep_d]  # (P, L+1, S, D')
        between = (rows - rows.mean(0)).pow(2).sum((-1, -2)).mean(0)
        total = rows.pow(2).sum((-1, -2)).mean(0)
        a, b = rows[:C], seed_rows[:, zi].float()[..., keep_d]
        seed_d = (a - b).pow(2).sum((-1, -2)).mean(0)
        page_d = (a - a.roll(1, 0)).pow(2).sum((-1, -2)).mean(0)
        rec = {
            "page_share_of_energy": r3((between / total)[1:]),
            "seed_over_page_distance": r3((seed_d / page_d.clamp_min(1e-12))[1:]),
            "floor_xt_mean": {
                "palette": round(ridge_r2(xt_mean[:, zi], y_pal, folds, device), 3),
                "layout": round(ridge_r2(xt_mean[:, zi], y_lay, folds, device), 3),
            },
        }
        for tname, y in (("palette", y_pal), ("layout", y_lay)):
            rec[f"{tname}_from_text"] = r3(
                ridge_r2(rows[:, j].flatten(1), y, folds, device)
                for j in range(1, L + 1)
            )
            rec[f"{tname}_from_image_mean"] = r3(
                ridge_r2(img_mean[:, zi, j].float()[:, keep_d], y, folds, device)
                for j in range(L)
            )
        content[f"sigma_{sigma}"] = rec
        print(
            f"[probe] σ {sigma} palette text b41 {rec['palette_from_text'][-1]} "
            f"image {rec['palette_from_image_mean'][-1]} floor "
            f"{rec['floor_xt_mean']['palette']}",
            flush=True,
        )
    summary["content"] = content

    dyn = {}
    for ci, cname in enumerate(("empty", "full")):
        for zi, sigma in enumerate(sigmas):
            rec = {
                "cos_to_input": r3(np.nanmean(drift[ci, :, zi, 0], 0)),
                "rel_update": r3(np.nanmean(drift[ci, :, zi, 1], 0)),
            }
            for i, name in enumerate(Q_STATS):
                rec[f"text_q_on_{name}"] = r3(np.nanmean(q_att[ci, :, zi, i], 0))
            for i, name in enumerate(I_STATS):
                rec[f"image_q_on_{name}"] = r3(np.nanmean(i_att[ci, :, zi, i], 0))
            dyn[f"{cname}_sigma_{sigma}"] = rec
    summary["dynamics"] = dyn

    run_dir = make_run_dir("moss_text_stream", args.label or "colorize")
    np.savez_compressed(
        run_dir / "per_page.npz", mse=mse, dv=dv, dv_full_frozen=dv_full_frozen, q_att=q_att, i_att=i_att,
        drift=drift, sigmas=np.array(sigmas), arms=np.array(ARMS),
        stems=np.array([p["stem"] for p in pages]),
    )  # fmt: skip
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        label=args.label,
        artifacts=["per_page.npz"],
        device=device,
    )
    print(f"[probe] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()

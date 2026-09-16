"""Flags the ``train`` stage reads: loop and σ band, S-line loss terms, the rows arm, the encoder arm."""

from __future__ import annotations


def train_args(g):
    g.add_argument("--train_steps", type=int, default=2000)
    g.add_argument("--batch", type=int, default=4)
    g.add_argument(
        "--lr_rows", type=float, default=3e-3, help="in units of the mean pack-row norm"
    )
    g.add_argument("--lr_adapter", type=float, default=1e-4)
    g.add_argument("--adapter_rank", type=int, default=16)
    g.add_argument(
        "--lr_decay",
        default="none",
        choices=["none", "cosine"],
        help="train: lr schedule over --train_steps (cosine to 0; all param groups)",
    )
    g.add_argument("--grad_ckpt", type=int, default=1)
    g.add_argument(
        "--compile",
        type=int,
        default=1,
        help="train: per-block torch.compile of the frozen DiT (the OOM remedy of record)",
    )
    g.add_argument("--activation_memory_budget", type=float, default=0.99)
    g.add_argument(
        "--aggressive_recompute",
        type=int,
        default=1,
        help="compile: partitioner aggressive recomputation (−VRAM, +~12 % s/it); 0 when memory allows",
    )
    g.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="restrict FM timesteps (W2 σ-restriction lever; None = full range)",
    )
    g.add_argument("--t_max", type=float, default=None)
    # Read by BOTH arms: the encoder arm gates its per-row residual on it
    # (train/trainables._init_free) and the rows arm uses it as the μ‖f‖² pull on the
    # free rows (trainables.regularized) — every S-line run passes it.
    g.add_argument(
        "--free_residual",
        type=float,
        default=0.0,
        help="encoder arm: μ on mean_i ‖f_i‖² for a per-row free residual on the "
        "trained rows (row = g(glyph) + f_i; held-out rows get g only). 0 = off. "
        "W2d report Run 1d: the semi-amortised hybrid — f carries the identity "
        "magnitude the shared head cannot, the L2 pushes what g can explain into g",
    )
    g.add_argument(
        "--out_vec_train",
        type=float,
        default=0.0,
        help="train: with --out_vec, add this × the EN shift norm × its direction at "
        "every ext position of the adapter output on every step (Q fixed on, so the "
        "rows never have to learn the render trigger). The vector is saved into "
        "trained.pt and eval / native apply it to every trained cond. 0 = off",
    )


def train_synth_args(g):
    g.add_argument(
        "--box_weight",
        type=float,
        default=1.0,
        help="train: FM loss weight inside a composite item's swapped text box "
        "(1 outside; batch-normalised); 1 = off",
    )
    g.add_argument(
        "--c_flat",
        type=int,
        default=0,
        help="train (rows arm): per-source layout vector added to every trained "
        "row on flat-canvas batches only (Δ_r = f_r + 𝟏[flat]·c_flat)",
    )
    g.add_argument(
        "--lr_c_flat", type=float, default=0.0, help="train: c_flat lr (0 = --lr_rows)"
    )
    g.add_argument(
        "--c_flat_cap",
        type=float,
        default=0.75,
        help="train: ‖c_flat‖ bound in row norms (projected after each step)",
    )
    g.add_argument(
        "--f_orth",
        type=float,
        default=0.0,
        help="train: λ · mean_r cos²(f_r, c_flat) guard (0 = off; read `leak` first)",
    )


def rows_args(g):
    g.add_argument(
        "--init_rows",
        default="",
        help="rows arm: warm-start the rows by ext id from another arm's trained.pt "
        "(its exported delta.raw). If the source is an encoder arm with a shared "
        "``common`` vector, that vector is subtracted from every row and, when "
        "--c_flat is on, seeds c_flat (clipped to --c_flat_cap) — so the flat "
        "table starts where the source left it and the composites train f alone. "
        "Rows the source never had start at zero. Comma list = several tables in "
        "order, a later one overriding by ext id (53k table + punctuation table)",
    )
    g.add_argument(
        "--pin_dir",
        default="",
        help="rows arm: inherit the shared direction of another arm's trained.pt — "
        "every trained row gets a fixed a_r · m̂_fam (kana / other family means of the "
        "source table) and only the residual trains (transplant probe 2026-09-16)",
    )
    g.add_argument(
        "--pin_coef",
        default="row",
        help="rows arm: a_r for --pin_dir — 'row' (the source row's own coefficient, "
        "family mean when absent), 'fam' (family mean), or a number",
    )
    g.add_argument(
        "--pin_orth",
        type=int,
        default=1,
        help="rows arm: with --pin_dir, project the trainable residual ⟂ m̂_fam after "
        "every step (1) or let it re-grow the direction (0)",
    )


def encoder_args(g):
    g.add_argument(
        "--lr_enc", type=float, default=3e-4, help="encoder arm: AdamW lr on the CNN"
    )
    g.add_argument(
        "--lr_common",
        type=float,
        default=1e-3,
        help="encoder arm: lr on the shared layout vector c (row-norm units; the rows lr)",
    )
    g.add_argument(
        "--common_cap",
        type=float,
        default=0.75,
        help="encoder arm: ‖c‖ bound in row norms, applied to the parameter after each step",
    )
    g.add_argument(
        "--out_scale",
        type=float,
        default=1.0 / 64,
        help="encoder arm: scale on the zero-init head (raise one notch to 1/16 if spread stays flat)",
    )
    g.add_argument(
        "--kill_spread",
        type=float,
        default=0.05,
        help="encoder arm: abort if rel_spread_ref (fixed font, no shift) is below this once --kill_spread_step is reached",
    )
    g.add_argument(
        "--kill_spread_step",
        type=int,
        default=600,
        help="encoder arm: 0 disables the spread rule (--kill_max_row 0 disables the max-row rule)",
    )
    g.add_argument(
        "--kill_max_row",
        type=float,
        default=2.0,
        help="encoder arm: abort if any row's delta passes this many row norms",
    )
    g.add_argument(
        "--glyph_size", type=int, default=96, help="encoder arm: glyph render side"
    )
    g.add_argument(
        "--head_init",
        default="zero",
        choices=["zero", "random"],
        help="encoder arm: last head layer zero (attempts 1–10) or random full-rank, "
        "rescaled so the step-0 identity spread is --init_spread row norms",
    )
    g.add_argument(
        "--init_spread",
        type=float,
        default=1.0,
        help="encoder arm: --head_init random target spread on the reference render",
    )
    g.add_argument(
        "--decor",
        type=float,
        default=0.0,
        help="encoder arm: λ on mean_{i≠j} cos²(r_i, r_j) over the centred trained "
        "rows of the encoder table (0 = off; W2d report Run 1b amended: ≈ 0.02 "
        "against an FM loss of ≈ 0.04)",
    )
    g.add_argument(
        "--lr_free",
        type=float,
        default=1e-3,
        help="encoder arm: lr of the free residual, row-norm units (W1 rows: 1e-3; 3e-3 walks off-manifold)",
    )
    g.add_argument(
        "--init_encoder",
        default="",
        help="encoder arm: warm-start the encoder (conv/proj/head + common) from another arm's trained.pt",
    )
    g.add_argument(
        "--init_free",
        default="",
        help="train: warm-start the free residual by ext id from another arm's trained.pt",
    )
    g.add_argument(
        "--enc_pool",
        default="spatial",
        choices=["spatial", "mean"],
        help="encoder arm: feature pooling — spatial keeps the arrangement (glyph), "
        "mean keeps channel statistics only (attempts 4–7 tracked font, not glyph)",
    )
    g.add_argument(
        "--font_mode",
        default="mean",
        choices=["mean", "random"],
        help="encoder arm: input render — mean over every font (font-free) or one random font per row per step",
    )
    g.add_argument(
        "--held_out",
        type=int,
        default=0,
        help="encoder arm: N single chars removed from every training item and "
        "evaluated as group single_held (the generalisation test)",
    )
    g.add_argument(
        "--held_out_chars",
        default="",
        help="encoder arm: explicit held-out chars instead of --held_out's draw "
        "(Run 2: IDS composites whose atoms are trained, e.g. 明休男岩加相困森)",
    )

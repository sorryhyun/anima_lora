# OrthoLoRA and the SVD-warm-start family

Three variants in this repo share one idea: aim the LoRA delta at the pretrained weight's principal subspace instead of a random direction, by seeding from $W_0$'s SVD. They differ only in how hard they hold onto that subspace afterwards — and in practice the *loosest* one is what ships.

| Variant | Flag | SVD is… | Trainable | ΔW can reach |
|---|---|---|---|---|
| SVD-Down (shipped default) | `down_init = "weight_svd"` | init only, down side | plain $A, B$ | anywhere |
| OrthoInit | `use_ortho_init = true` | init only, both sides | $P, Q$ bases + $\lambda$ | anywhere |
| OrthoLoRA (PSOFT-style) | `use_ortho = true` | frozen constraint | $r{\times}r$ Cayley seeds + $\lambda$ | top-$r$ subspace of $W_0$ only |

Recap from `lora.md`: plain LoRA is $y = W_0 x + m\,s\,BAx$ with a random-direction $A$ and $B=0$. The problem all three variants address is the same — training spends its first epochs discovering a useful subspace from scratch, and an unconstrained delta can drift into directions that fight $W_0$ (style transfer that degrades anatomy, etc.).

![PSOFT-integrated OrthoLoRA](../structure_images/ortholora.png)

---

## 1. What ships: SVD-Down

The live `configs/methods/lora.toml` gets the warm-start benefit with the smallest possible mechanism: seed `lora_down` from $W_0$'s top-$r$ right singular vectors and change nothing else.

$$
A_0 = V_r^\top / \sqrt{3}, \qquad B_0 = 0
$$

The $1/\sqrt{3}$ scale-matches Kaiming's expected row-norm, so the comparison against the default init isn't confounded by a larger effective step — it's purely "better direction." After init this is ordinary LoRA: ΔW = 0 at step 0 (via $B=0$), full $A$ and $B$ trainable from step 1, standard save format, no constraint, no extra buffers. It applies to plain `LoRAModule` only (the config raises if combined with the ortho or MoE variants, which have their own SVD machinery). Details and benching: `docs/methods/svd-down-lora.md`.

If you remember one thing from this doc: **the SVD's value here is as a prior, not a cage.** The rest of this page covers the two variants that hold the subspace more tightly, and where they still earn their keep.

---

## 2. OrthoLoRA: the subspace as a hard constraint

OrthoLoRA (`networks/lora_modules/ortho.py`, inspired by PSOFT — Wu et al., ICLR 2026) makes the principal subspace structural:

- Frozen bases. The top-$r$ singular vectors of $W_0$ (via randomized `torch.svd_lowrank`) are stored as non-trainable buffers $P_\text{basis}$ (out side) and $Q_\text{basis}$ (in side).
- Cayley rotations. The trainable parameters are two tiny skew-symmetric seeds $S_p, S_q \in \mathbb{R}^{r\times r}$, mapped through the Cayley transform $R = (I-A)(I+A)^{-1}$ (exact `linalg.solve`, not the paper's Neumann series — the series silently diverges once $\|A\| \ge 1$, and an $r{\times}r$ solve is free). The effective bases $P_\text{basis}R_p$ / $R_q Q_\text{basis}$ are exactly orthonormal at every step, by construction — no regularizer, no hyperparameter.
- Diagonal scale. A zero-init $\lambda \in \mathbb{R}^{1\times r}$ closes the delta: $\Delta W = P_\text{eff}\,\text{diag}(\lambda)\,Q_\text{eff}$. $\lambda = 0$ ⇒ ΔW = 0 at step 0.

The forward keeps plain LoRA's shape — down-projection, bottleneck (where the T-LoRA mask gates $\lambda$), up-projection — so the family stacks as usual. Trainable params per module collapse from $r(d_\text{in}+d_\text{out})$ to $2r^2 + r$ (~50× fewer at $r=64$), traded against per-module frozen basis buffers.

### The cost: a capped delta

$$
\text{colspace}(\Delta W)\ \subseteq\ \text{top-}r\ \text{left singular vectors of}\ W_0
$$

Plain LoRA can point ΔW anywhere; OrthoLoRA can only rotate and rescale *inside* $W_0$'s principal subspace. For creative fine-tuning that cap is real — a new character or style may need components outside the top-$r$ directions, and when it does, OrthoLoRA plateaus early. This is exactly the "ortho feels too weak" experience that motivated the retreat to init-only variants.

OrthoInit (`use_ortho_init = true`) is the halfway house: same top-$r$ SVD seed on both sides, but the bases are promoted to trainable parameters — no Cayley, no frozen subspace, $\lambda = 0$ still guarantees exact base preservation at step 0. Full LoRA expressivity with a two-sided warm start. Mutually exclusive with `use_ortho`; unlike the Cayley form it also composes with ChimeraHydra's pools.

---

## 3. Where the frozen-Cayley form still earns its keep: MoE experts

For a single adapter, the hard constraint is usually not worth the cap — hence SVD-Down as the default. The Cayley parameterization's real remaining home is the MoE variants, where frozen disjointness is a feature, not a limitation:

- OrthoHydra (`hydralora.md` §5.2) partitions the top-$(E \cdot r)$ singular vectors into per-expert disjoint slices. Because each expert rotates only within its own frozen slice, experts *cannot* collapse into each other — the structural fix for the MoE cold-start deadlock.
- ChimeraHydra (`chimera-hydra.md`) builds both of its pools on the same substrate, with disjointness on both the down and up sides, plus capacity levers (`basis_mult`, `expert_diag`) that deepen each expert while preserving the disjoint guarantee.

There, "the delta can't leave its slice" is precisely the property being purchased.

---

## 4. Save format

All variants converge on disk. OrthoLoRA's effective delta is already rank $r$, so it factors exactly into standard LoRA keys with no SVD at save time:

$$
\text{lora\_up} = P_\text{eff}\,\text{diag}(\lambda),\qquad \text{lora\_down} = Q_\text{eff}
$$

OrthoInit distills the same way ($R = I$); SVD-Down was standard LoRA all along. The payoff is uniform: stock ComfyUI loading and lossless `scripts/toolkits/merge_to_dit.py` merging, whichever variant trained the checkpoint.

---

## 5. Minimal mental model

1. One idea — seed from $W_0$'s SVD so the first gradients land where the base model already concentrates — at three strengths of commitment.
2. SVD-Down ships: init-only, down-side, scale-matched; ordinary LoRA afterwards.
3. OrthoLoRA freezes the subspace and trains only $r{\times}r$ rotations + a diagonal: exact orthogonality with ~50× fewer trainable params, but ΔW is capped to the principal subspace — the cap that pushed the default back to init-only.
4. OrthoInit keeps the two-sided seed, frees the bases.
5. The frozen-Cayley machinery survives where disjointness is the point: OrthoHydra / ChimeraHydra expert slices.
6. Everything saves as standard LoRA keys.

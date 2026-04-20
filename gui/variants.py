"""Variant presets for the LoRA and postfix method families.

The method-file TOMLs use commented toggle blocks — e.g. `lora.toml` carries
plain-LoRA, T-LoRA, OrthoLoRA, HydraLoRA, ReFT, and σ-router variants, only
one of which is active at a time. Editing those blocks by hand in the GUI is
unergonomic; this module encodes each variant as a `{field: value}` dict that
the GUI can overlay onto the merged config with a single click.

Each method family also declares a set of `OWNED_KEYS` — fields that belong
to the variant layer. On save, any owned key not covered by the current form
is removed from the method TOML so that switching variants (e.g. hydralora →
plain lora) doesn't leave stale fields behind.
"""

from __future__ import annotations

LORA_TOGGLE_DEFAULTS: dict = {
    "use_ortho": False,
    "use_timestep_mask": False,
    "use_hydra": False,
    "add_reft": False,
    "use_sigma_router": False,
}

LORA_VARIANTS: dict[str, dict] = {
    "lora (plain)": {
        **LORA_TOGGLE_DEFAULTS,
        "network_dim": 16,
        "network_alpha": 16,
        "output_name": "anima",
    },
    "ortholora": {
        **LORA_TOGGLE_DEFAULTS,
        "use_ortho": True,
        "network_dim": 16,
        "network_alpha": 16,
        "output_name": "anima_ortho",
    },
    "tlora": {
        **LORA_TOGGLE_DEFAULTS,
        "use_timestep_mask": True,
        "network_dim": 16,
        "network_alpha": 16,
        "min_rank": 1,
        "alpha_rank_scale": 1.0,
        "output_name": "anima_tlora",
    },
    "tlora + ortho + reft (default stack)": {
        **LORA_TOGGLE_DEFAULTS,
        "use_ortho": True,
        "use_timestep_mask": True,
        "add_reft": True,
        "network_dim": 16,
        "network_alpha": 16,
        "min_rank": 1,
        "alpha_rank_scale": 1.0,
        "reft_dim": 64,
        "reft_alpha": 64,
        "reft_layers": "last_8",
        "output_name": "anima_tlora",
    },
    "hydralora": {
        **LORA_TOGGLE_DEFAULTS,
        "use_ortho": True,
        "use_timestep_mask": True,
        "use_hydra": True,
        "network_dim": 16,
        "network_alpha": 16,
        "num_experts": 4,
        "balance_loss_weight": 0.01,
        "output_name": "anima_hydra",
    },
    "hydralora + sigma-router": {
        **LORA_TOGGLE_DEFAULTS,
        "use_ortho": True,
        "use_timestep_mask": True,
        "use_hydra": True,
        "use_sigma_router": True,
        "network_dim": 16,
        "network_alpha": 16,
        "num_experts": 4,
        "balance_loss_weight": 0.01,
        "sigma_feature_dim": 128,
        "sigma_hidden_dim": 128,
        "sigma_router_layers": r".*(cross_attn\.q_proj|self_attn\.qkv_proj)$",
        "per_bucket_balance_weight": 0.3,
        "num_sigma_buckets": 3,
        "output_name": "anima_hydra_sigma",
    },
}

POSTFIX_VARIANTS: dict[str, dict] = {
    "postfix": {
        "network_dim": 32,
        "network_args": [
            "mode=postfix",
            "splice_position=end_of_sequence",
        ],
        "output_name": "anima_postfix",
    },
    "postfix_exp (cond)": {
        "network_dim": 64,
        "network_args": [
            "mode=cond",
            "splice_position=end_of_sequence",
            "cond_hidden_dim=256",
        ],
        "output_name": "anima_postfix_exp",
        "max_train_epochs": 2,
        "checkpointing_epochs": 2,
    },
    "postfix_func (cond + functional loss)": {
        "network_dim": 64,
        "network_args": [
            "mode=cond",
            "splice_position=end_of_sequence",
            "cond_hidden_dim=256",
        ],
        "output_name": "anima_postfix_func",
        "max_train_epochs": 4,
        "caption_shuffle_variants": 1,
        "attn_mode": "flash",
        "trim_crossattn_kv": False,
        "inversion_dir": "inversions/results",
        "functional_loss_weight": 1.0,
        "functional_loss_blocks": "8,12,16,20",
        "functional_loss_num_runs": 3,
    },
    "postfix_sigma (cond-timestep)": {
        "network_dim": 64,
        "network_args": [
            "mode=cond-timestep",
            "splice_position=front_of_padding",
            "cond_hidden_dim=256",
            "sigma_feature_dim=128",
            "sigma_hidden_dim=256",
            "slot_embed_init_std=0.02",
            "contrastive_weight=0.1",
            "sigma_budget_weight=1e-3",
        ],
        "output_name": "anima_postfix_sigma",
        "max_train_epochs": 2,
        "checkpointing_epochs": 2,
    },
    "prefix": {
        "network_dim": 16,
        "network_args": ["mode=prefix"],
        "output_name": "anima_prefix",
        "caption_shuffle_variants": 2,
        "multiscale_loss_weight": 0.5,
    },
}

METHOD_VARIANTS: dict[str, dict[str, dict]] = {
    "lora": LORA_VARIANTS,
    "postfix": POSTFIX_VARIANTS,
}

LORA_OWNED_KEYS: set[str] = {
    "use_ortho",
    "use_timestep_mask",
    "use_hydra",
    "add_reft",
    "use_sigma_router",
    "min_rank",
    "alpha_rank_scale",
    "num_experts",
    "balance_loss_weight",
    "reft_dim",
    "reft_alpha",
    "reft_layers",
    "sigma_feature_dim",
    "sigma_hidden_dim",
    "sigma_router_layers",
    "per_bucket_balance_weight",
    "num_sigma_buckets",
}

POSTFIX_OWNED_KEYS: set[str] = {
    "network_args",
    "inversion_dir",
    "functional_loss_weight",
    "functional_loss_blocks",
    "functional_loss_num_runs",
    "multiscale_loss_weight",
}

METHOD_OWNED_KEYS: dict[str, set[str]] = {
    "lora": LORA_OWNED_KEYS,
    "postfix": POSTFIX_OWNED_KEYS,
}


def variants_for(method: str) -> dict[str, dict]:
    """Variant presets registered for *method*, or an empty dict."""
    return METHOD_VARIANTS.get(method, {})


def owned_keys(method: str) -> set[str]:
    """Fields a variant is authoritative for — cleared on save when absent."""
    return METHOD_OWNED_KEYS.get(method, set())


def default_toggle_fields(method: str) -> dict:
    """Fields to always show in the form even when absent from merged config."""
    if method == "lora":
        return dict(LORA_TOGGLE_DEFAULTS)
    return {}

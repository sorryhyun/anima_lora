#!/usr/bin/env python3
"""GRAFT: Human-in-the-loop LoRA fine-tuning via rejection sampling.

Each `make step` invocation:
1. Ingests survived candidates from the previous round
2. Holds out a random subset of image_dataset/ (captions used for generation)
3. Trains LoRA on remaining images + accumulated survivors
4. Generates candidate images for held-out captions
5. Waits for user to curate candidates
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11

ROOT = Path(__file__).resolve().parent
GRAFT_DIR = ROOT / "graft"
STATE_FILE = GRAFT_DIR / "state.json"
GRAFT_CONFIG = GRAFT_DIR / "graft_config.toml"
IMAGE_DATASET = ROOT / "image_dataset"
TRAIN_IMAGES = GRAFT_DIR / "train_images"
SURVIVORS_DIR = GRAFT_DIR / "survivors"
CANDIDATES_DIR = GRAFT_DIR / "candidates"
TRAINING_CONFIG_SRC = ROOT / "configs" / "training_config.toml"
DATASET_CONFIG_SRC = ROOT / "configs" / "dataset_config.toml"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_config():
    with open(GRAFT_CONFIG, "rb") as f:
        return tomllib.load(f)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(state):
    GRAFT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_dataset_images():
    """Return list of (stem, image_path) pairs from image_dataset/."""
    images = []
    for p in sorted(IMAGE_DATASET.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((p.stem, p))
    return images


def get_caption(stem, directory=None):
    """Read caption for an image stem from its .txt file."""
    if directory is None:
        directory = IMAGE_DATASET
    txt = Path(directory) / f"{stem}.txt"
    if txt.exists():
        return txt.read_text().strip()
    return ""


def ingest_survivors(state):
    """Move survived candidates from candidates dir to survivors dir."""
    iteration = state["iteration"]
    candidates = CANDIDATES_DIR / f"iter_{iteration:03d}"
    if not candidates.exists():
        print(f"No candidates dir for iteration {iteration}, skipping ingest")
        return 0

    SURVIVORS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(candidates.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Read caption from sidecar JSON
        sidecar = img.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar) as f:
                meta = json.load(f)
            caption = meta.get("caption", "")
        else:
            caption = ""

        # Copy to survivors with unique name
        dst_name = f"graft_{iteration:03d}_{img.stem}"
        dst_img = SURVIVORS_DIR / f"{dst_name}{img.suffix}"
        dst_txt = SURVIVORS_DIR / f"{dst_name}.txt"
        shutil.copy2(img, dst_img)
        dst_txt.write_text(caption)
        count += 1

    print(f"Ingested {count} survivors from iteration {iteration}")
    return count


def build_holdout(config):
    """Sample images from image_dataset for generation, return (holdout_stems, holdout_captions)."""
    images = get_dataset_images()
    ratio = config.get("pgraft_sample_ratio", 0.2)
    n = max(1, int(len(images) * ratio))
    holdout = random.sample(images, n)
    holdout_stems = [stem for stem, _ in holdout]
    holdout_captions = [(stem, get_caption(stem)) for stem, _ in holdout]
    return holdout_stems, holdout_captions


def build_train_symlinks(holdout_stems):
    """Recreate graft/train_images/ with symlinks to image_dataset/ minus held-out.

    Also symlinks latent cache .npz files so training doesn't re-encode.
    Cache files are named {stem}_{HHHHxWWWW}_anima.npz.
    """
    # Clean previous symlinks
    if TRAIN_IMAGES.exists():
        shutil.rmtree(TRAIN_IMAGES)
    TRAIN_IMAGES.mkdir(parents=True)

    holdout_set = set(holdout_stems)
    count = 0
    for p in IMAGE_DATASET.iterdir():
        # For .npz cache files, extract the image stem (everything before _DDDDxDDDD_anima.npz)
        if p.suffix == ".npz":
            # Pattern: {stem}_{W}x{H}_anima.npz
            name_no_ext = p.stem  # e.g. "asou1_2500x2226_anima"
            parts = name_no_ext.rsplit("_", 2)  # ["asou1", "2500x2226", "anima"]
            if len(parts) >= 3:
                cache_stem = "_".join(parts[:-2])
            else:
                cache_stem = parts[0]
            if cache_stem in holdout_set:
                continue
            link = TRAIN_IMAGES / p.name
            link.symlink_to(p.resolve())
            continue

        stem = p.stem
        # Skip held-out images and their captions
        if stem in holdout_set:
            continue
        # Create symlink
        link = TRAIN_IMAGES / p.name
        link.symlink_to(p.resolve())
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    print(f"Training subset: {count} images (held out {len(holdout_set)})")


def generate_dataset_config():
    """Generate graft/dataset_config.toml for training."""
    # Read original dataset config for base settings
    with open(DATASET_CONFIG_SRC, "rb") as f:
        orig = tomllib.load(f)

    ds = orig.get("datasets", [{}])[0]

    subsets = f"""  [[datasets.subsets]]
  image_dir = 'graft/train_images'
  num_repeats = 1
"""
    # Add survivors if any exist
    survivor_images = [p for p in SURVIVORS_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS] if SURVIVORS_DIR.exists() else []
    if survivor_images:
        subsets += f"""
  [[datasets.subsets]]
  image_dir = 'graft/survivors'
  num_repeats = 1
"""

    config = f"""[general]
shuffle_caption = {str(orig.get('general', {}).get('shuffle_caption', False)).lower()}
caption_extension = '{orig.get('general', {}).get('caption_extension', '.txt')}'
keep_tokens = {orig.get('general', {}).get('keep_tokens', 3)}

[[datasets]]
resolution = {ds.get('resolution', 1024)}
batch_size = {ds.get('batch_size', 4)}
enable_bucket = {str(ds.get('enable_bucket', True)).lower()}
min_bucket_reso = {ds.get('min_bucket_reso', 512)}
max_bucket_reso = {ds.get('max_bucket_reso', 1536)}
bucket_reso_steps = {ds.get('bucket_reso_steps', 64)}
validation_split = {ds.get('validation_split', 0.05)}

{subsets}"""

    out = GRAFT_DIR / "dataset_config.toml"
    out.write_text(config)
    print(f"Generated {out}")


def generate_training_config(config):
    """Generate graft/training_config.toml by reading the original as text and patching values."""
    lines = TRAINING_CONFIG_SRC.read_text().splitlines()
    overrides = {
        "dataset_config": f'"graft/dataset_config.toml"',
        "max_train_epochs": str(config["epochs_per_step"]),
        "save_every_n_epochs": str(config["epochs_per_step"]),
    }

    out_lines = []
    applied = set()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            key = stripped.split("=")[0].strip()
            if key in overrides:
                out_lines.append(f"{key} = {overrides[key]}")
                applied.add(key)
                continue
        out_lines.append(line)

    # Add any overrides not found in original
    for key, val in overrides.items():
        if key not in applied:
            out_lines.append(f"{key} = {val}")

    out = GRAFT_DIR / "training_config.toml"
    out.write_text("\n".join(out_lines) + "\n")
    print(f"Generated {out}")


def run_training(config):
    """Run LoRA training."""
    print("\n=== Training LoRA ===")
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "3",
        "--mixed_precision", "bf16",
        "train.py",
        "--config_file", "graft/training_config.toml",
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print("Training complete")


def run_generation(config, holdout_captions, iteration):
    """Generate candidate images for held-out captions in a single inference process.

    Uses --from_file mode so the model is loaded once for all candidates.
    save_images() in inference uses save_path as a directory and names files {timestamp}_{seed}.png.
    """
    print("\n=== Generating Candidates ===")
    candidates = CANDIDATES_DIR / f"iter_{iteration:03d}"
    candidates.mkdir(parents=True, exist_ok=True)

    tcfg = get_training_cfg()
    output_name = tcfg.get("output_name", "anima_lora")

    lora_path = ROOT / "output" / f"{output_name}.safetensors"
    if not lora_path.exists():
        matches = sorted((ROOT / "output").glob("*.safetensors"))
        if matches:
            lora_path = matches[-1]
        else:
            print("No LoRA weights found in output/", file=sys.stderr)
            sys.exit(1)

    dit_path = (ROOT / tcfg["pretrained_model_name_or_path"]).resolve()
    vae_path = (ROOT / tcfg["vae"]).resolve()
    te_path = (ROOT / tcfg["qwen3"]).resolve()

    h, w = config.get("image_size", [1024, 1024])
    n_candidates = config.get("candidates_per_prompt", 4)
    base_seed = config.get("base_seed", 42) + iteration * 1000
    pgraft = config.get("pgraft_enabled", True)
    cutoff_ratio = config.get("pgraft_cutoff_ratio", 0.75)
    steps = config.get("inference_steps", 50)
    cutoff_step = int(steps * cutoff_ratio)

    # Build prompt file for --from_file mode (model loaded once)
    prompt_lines = []
    sidecar_entries = []  # (seed, stem, caption) for writing JSON after generation
    total = 0
    for stem, caption in holdout_captions:
        if not caption:
            continue
        for j in range(n_candidates):
            seed = base_seed + total
            # --from_file format: prompt --d seed --h height --w width
            prompt_lines.append(f"{caption} --d {seed} --h {h} --w {w}")
            sidecar_entries.append((seed, stem, caption))
            total += 1

    if not prompt_lines:
        print("No captions to generate from")
        return candidates

    prompt_file = GRAFT_DIR / "generation_prompts.txt"
    prompt_file.write_text("\n".join(prompt_lines) + "\n")
    print(f"  {total} candidates to generate ({len(holdout_captions)} captions x {n_candidates} seeds)")

    cmd = [
        "python", "inference.py",
        "--dit", str(dit_path),
        "--vae", str(vae_path),
        "--text_encoder", str(te_path),
        "--lora_weight", str(lora_path),
        "--from_file", str(prompt_file),
        "--image_size", str(h), str(w),
        "--infer_steps", str(steps),
        "--guidance_scale", str(config.get("guidance_scale", 3.5)),
        "--flow_shift", str(config.get("flow_shift", 5.0)),
        "--save_path", str(candidates),
        "--negative_prompt", "lowres, bad anatomy, worst quality",
        "--attn_mode", "flash",
        "--vae_chunk_size", "64",
        "--vae_disable_cache",
        "--infer_batch_size", str(n_candidates),
    ]

    if pgraft:
        cmd.extend(["--pgraft", "--lora_cutoff_step", str(cutoff_step)])

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"Generation failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # Write sidecar JSONs for all generated images
    # save_images() names files {timestamp}_{seed}.png — match by seed
    generated_pngs = {p.stem.split("_")[-1]: p for p in candidates.glob("*.png")}
    for seed, stem, caption in sidecar_entries:
        # Find the generated PNG by seed suffix
        seed_str = str(seed)
        matching = [p for name, p in generated_pngs.items() if name == seed_str]
        if matching:
            sidecar = matching[0].with_suffix(".json")
            sidecar.write_text(json.dumps({
                "caption": caption,
                "seed": seed,
                "source_stem": stem,
                "iteration": iteration,
            }, indent=2))

    actual_count = len(list(candidates.glob("*.png")))
    print(f"Generated {actual_count} candidates in {candidates}")
    return candidates


def get_training_cfg():
    """Read training config values."""
    with open(TRAINING_CONFIG_SRC, "rb") as f:
        return tomllib.load(f)


def main():
    config = load_config()
    state = load_state()

    if state is None:
        # First run
        print("=== GRAFT Step: First iteration ===")
        state = {"iteration": 0, "phase": "init", "total_epochs": 0}
    else:
        print(f"=== GRAFT Step: Iteration {state['iteration']} → {state['iteration'] + 1} ===")

        if state["phase"] == "await_review":
            # Ingest survivors from previous candidates
            ingest_survivors(state)
            state["iteration"] += 1

    iteration = state["iteration"]

    # 1. Holdout sampling
    holdout_stems, holdout_captions = build_holdout(config)
    state["holdout_files"] = holdout_stems
    print(f"Held out {len(holdout_stems)} images for generation")

    # 2. Build training symlinks (image_dataset minus holdout)
    build_train_symlinks(holdout_stems)

    # 3. Generate dataset + training configs
    generate_dataset_config()
    generate_training_config(config)

    # 4. Train
    run_training(config)
    state["total_epochs"] = state.get("total_epochs", 0) + config["epochs_per_step"]

    # 5. Generate candidates
    candidates_dir = run_generation(config, holdout_captions, iteration)

    # 6. Await review
    state["phase"] = "await_review"
    save_state(state)

    print(f"\n{'=' * 60}")
    print(f"Review candidates in {candidates_dir.relative_to(ROOT)}/")
    print(f"Delete unwanted images, then run `make step` again.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

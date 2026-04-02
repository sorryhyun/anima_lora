#!/usr/bin/env python3
"""Submit a ComfyUI workflow for every (artist, chara) combination.

Reads a workflow JSON containing __artist__ / __chara__ placeholders in text
fields and submits one job per cartesian-product pair to the ComfyUI API.

Usage:
    python scripts/comfy_batch.py workflows/lora-batch.json \
        --artist workflows/artist.txt --chara workflows/chara.txt
"""

import argparse
import json
import itertools
import random
import sys
import time
import urllib.request
import urllib.error


COMFY_URL = "http://localhost:8188"


def load_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def substitute(workflow: dict, artist: str, chara: str) -> dict:
    """Deep-copy workflow and replace __artist__ / __chara__ in all string values."""
    raw = json.dumps(workflow)
    # Escape for JSON string context (backslashes must be doubled)
    artist_esc = artist.replace("\\", "\\\\").replace('"', '\\"')
    chara_esc = chara.replace("\\", "\\\\").replace('"', '\\"')
    raw = raw.replace("__artist__", artist_esc).replace("__chara__", chara_esc)
    return json.loads(raw)


def queue_prompt(workflow: dict, server: str) -> dict:
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def wait_until_done(server: str, prompt_id: str, poll_interval: float = 2.0):
    """Poll /history until the prompt_id appears (i.e. execution finished)."""
    while True:
        try:
            with urllib.request.urlopen(f"{server}/history/{prompt_id}") as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError:
            pass
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="ComfyUI batch runner")
    parser.add_argument("workflow", help="Path to workflow JSON")
    parser.add_argument("--artist", default="workflows/artist.txt")
    parser.add_argument("--chara", default="workflows/chara.txt")
    parser.add_argument("--server", default=COMFY_URL)
    parser.add_argument("--randomize-seed", action="store_true", default=True,
                        help="Randomize seed per job (default: true)")
    parser.add_argument("--no-randomize-seed", dest="randomize_seed", action="store_false")
    parser.add_argument("--wait", action="store_true", default=True,
                        help="Wait for each job to finish before queuing next (default: true)")
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()

    server = args.server

    with open(args.workflow) as f:
        workflow = json.load(f)

    artists = load_lines(args.artist)
    charas = load_lines(args.chara)
    pairs = list(itertools.product(artists, charas))

    print(f"Queuing {len(artists)} artists × {len(charas)} charas = {len(pairs)} jobs")

    for i, (artist, chara) in enumerate(pairs, 1):
        wf = substitute(workflow, artist, chara)

        if args.randomize_seed:
            for node in wf.values():
                if "seed" in node.get("inputs", {}):
                    node["inputs"]["seed"] = random.randint(0, 2**53)

        print(f"[{i}/{len(pairs)}] {artist} × {chara} ... ", end="", flush=True)
        try:
            result = queue_prompt(wf, server)
            prompt_id = result["prompt_id"]
        except (urllib.error.URLError, KeyError) as e:
            print(f"FAILED to queue: {e}")
            continue

        if args.wait:
            wait_until_done(server, prompt_id)
            print("done")
        else:
            print(f"queued ({prompt_id})")

    print("All done.")


if __name__ == "__main__":
    main()

"""fp32 reference for the ONNX check: the first 32 held-out captions of
te_convrot_gptq.captions(0) (its eval split), right-padded to 512 with <|endoftext|>
like the browser tokenizer, and their unpadded fp32 last_hidden_state.

    .venv/bin/python project/webgpu_int8/src/te_ref_dump.py output/webgpu_int8/te_ref.npz
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from library.anima.weights import load_qwen3_text_encoder
from library.env import resolve_under_home

spec = importlib.util.spec_from_file_location(
    "g", Path(__file__).with_name("te_convrot_gptq.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
torch.set_grad_enabled(False)
model, tok = load_qwen3_text_encoder(
    str(resolve_under_home(g.TE)), dtype=torch.float32, device="cpu"
)
caps = g.captions(0)[:32]
ids = np.full((32, 512), 151643, np.int64)
mask = np.zeros((32, 512), np.int64)
ref = np.zeros((32, 512, 1024), np.float32)
for i, c in enumerate(caps):
    t = tok(c, return_tensors="pt").input_ids[:, :512]
    n = t.shape[1]
    ids[i, :n] = t[0].numpy()
    mask[i, :n] = 1
    ref[i, :n] = model(input_ids=t, use_cache=False).last_hidden_state[0].numpy()
np.savez(sys.argv[1], ids=ids, mask=mask, ref=ref)
print("saved", mask.sum(1).mean())

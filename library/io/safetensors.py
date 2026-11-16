from dataclasses import dataclass
import os
import re
import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file

from library.runtime.device import synchronize_device


def mem_eff_save_file(
    tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None
):
    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(
                    f"Warning: Metadata value for key '{key}' is not a string. Converting to string."
                )
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:
            header[k] = {
                "dtype": _TYPES[v.dtype],
                "shape": list(v.shape),
                "data_offsets": [offset, offset],
            }
        else:
            size = v.numel() * v.element_size()
            header[k] = {
                "dtype": _TYPES[v.dtype],
                "shape": list(v.shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                with torch.cuda.device(v.device):
                    if (
                        v.dim() == 0
                    ):  # scalar needs a dim to view() as bytes
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                if v.dim() == 0:  # scalar needs a dim to view() as bytes
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    """Reads tensors from a safetensors file, memmapping large ones to avoid
    intermediate copies."""

    def __init__(self, filename, disable_numpy_memmap=False):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.disable_numpy_memmap = disable_numpy_memmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def _read_header(self):
        # Header size is 8 bytes, little-endian unsigned long long.
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(
        self,
        key: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Load a tensor from the file.

        CUDA transfers are non-blocking — caller must synchronize before use
        (e.g. ``torch.cuda.synchronize()``). Tensors >10MB going to CUDA are
        memmapped to avoid an intermediate copy.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        non_blocking = device is not None and device.type == "cuda"

        tensor_offset = self.header_size + 8 + offset_start

        # memmap large tensors to avoid intermediate copies, but only for a
        # non-cpu target: on cpu the tensor isn't copied to gpu, so the memmap
        # would just lock the file. disable_numpy_memmap forces standard reads.
        if (
            not self.disable_numpy_memmap
            and num_bytes > 10 * 1024 * 1024
            and device is not None
            and device.type != "cpu"
        ):
            mm = np.memmap(
                self.filename,
                mode="c",
                dtype=np.uint8,
                offset=tensor_offset,
                shape=(num_bytes,),
            )
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor

            gpu_tensor = cpu_tensor.to(
                device=device, dtype=target_dtype, non_blocking=non_blocking
            )
            del cpu_tensor
            return gpu_tensor

        # Standard read for smaller tensors or a CPU target.
        self.file.seek(tensor_offset)

        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        return deserialized_tensor.to(
            device=device, dtype=target_dtype, non_blocking=non_blocking
        )

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert string dtype to PyTorch dtype."""
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        return dtype_map.get(dtype_str)


def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(
            path, disable_numpy_memmap=disable_numpy_memmap
        ) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except Exception:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def get_split_weight_filenames(file_path: str) -> Optional[list[str]]:
    """Return split weight filenames if ``file_path`` ends with e.g.
    ``00001-of-00004``, else None."""
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        filenames = []
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} not found")
        return filenames
    else:
        return None


def load_split_weights(
    file_path: str,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load split weights from a file, or a single file if not split."""
    device = torch.device(device)

    split_filenames = get_split_weight_filenames(file_path)
    if split_filenames is not None:
        state_dict = {}
        for filename in split_filenames:
            state_dict.update(
                load_safetensors(
                    filename, device=device, disable_mmap=disable_mmap, dtype=dtype
                )
            )
    else:
        state_dict = load_safetensors(
            file_path, device=device, disable_mmap=disable_mmap, dtype=dtype
        )
    return state_dict


def find_key(
    safetensors_file: str,
    starts_with: Optional[str] = None,
    ends_with: Optional[str] = None,
) -> Optional[str]:
    """Find the first key matching the given prefix/suffix (None = wildcard)."""
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        for key in f.keys():
            if (starts_with is None or key.startswith(starts_with)) and (
                ends_with is None or key.endswith(ends_with)
            ):
                return key
    return None


@dataclass
class WeightTransformHooks:
    split_hook: Optional[callable] = None
    concat_hook: Optional[callable] = None
    rename_hook: Optional[callable] = None


class TensorWeightAdapter:
    """Wraps a :class:`MemoryEfficientSafeOpen` to apply split/concat/rename
    key-conversion hooks when loading tensors.

    ``split_hook(original_key, tensor) -> (new_keys, new_tensors)``,
    ``concat_hook(original_key, tensors_dict) -> (new_key, tensor)``,
    ``rename_hook(original_key) -> new_key``. When called with ``tensors=None``
    (during key discovery) a hook returns only the key(s), no tensors.

    Not a context manager itself — the wrapped ``original_f`` owns the file.
    """

    def __init__(
        self,
        weight_convert_hook: WeightTransformHooks,
        original_f: MemoryEfficientSafeOpen,
    ):
        self.original_f = original_f
        self.new_key_to_original_key_map: dict[
            str, Union[str, list[str]]
        ] = {}  # split: new_key->original_key; concat: new_key->[original_keys]
        self.concat_key_set = set()
        self.split_key_set = set()
        self.new_keys = []
        self.tensor_cache = {}  # split tensors, cached until popped
        self.split_hook = weight_convert_hook.split_hook
        self.concat_hook = weight_convert_hook.concat_hook
        self.rename_hook = weight_convert_hook.rename_hook

        for key in self.original_f.keys():
            if self.split_hook is not None:
                converted_keys, _ = self.split_hook(key, None)
                if converted_keys is not None:
                    for converted_key in converted_keys:
                        self.new_key_to_original_key_map[converted_key] = key
                        self.split_key_set.add(converted_key)
                    self.new_keys.extend(converted_keys)
                    continue  # skip concat_hook if split_hook is applied

            if self.concat_hook is not None:
                converted_key, _ = self.concat_hook(key, None)
                if converted_key is not None:
                    if (
                        converted_key not in self.concat_key_set
                    ):  # first time seeing this concatenated key
                        self.concat_key_set.add(converted_key)
                        self.new_key_to_original_key_map[converted_key] = []
                        self.new_keys.append(converted_key)

                    self.new_key_to_original_key_map[converted_key].append(key)
                    continue

            # direct mapping
            if self.rename_hook is not None:
                new_key = self.rename_hook(key)
                self.new_key_to_original_key_map[new_key] = key
            else:
                new_key = key

            self.new_keys.append(new_key)

    def keys(self) -> list[str]:
        return self.new_keys

    def get_tensor(
        self,
        new_key: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if new_key not in self.new_key_to_original_key_map:
            # direct mapping
            return self.original_f.get_tensor(new_key, device=device, dtype=dtype)

        elif new_key in self.split_key_set:
            # a split key is requested multiple times, so cache the split result
            original_key = self.new_key_to_original_key_map[new_key]
            if original_key not in self.tensor_cache:  # not yet split
                original_tensor = self.original_f.get_tensor(
                    original_key, device=device, dtype=dtype
                )
                new_keys, new_tensors = self.split_hook(original_key, original_tensor)
                for k, t in zip(new_keys, new_tensors):
                    self.tensor_cache[k] = t
            return self.tensor_cache.pop(new_key)

        elif new_key in self.concat_key_set:
            # a concat key is requested only once, so no need to cache
            tensors = {}
            for original_key in self.new_key_to_original_key_map[new_key]:
                tensor = self.original_f.get_tensor(
                    original_key, device=device, dtype=dtype
                )
                tensors[original_key] = tensor
            _, concatenated_tensors = self.concat_hook(
                self.new_key_to_original_key_map[new_key][0], tensors
            )
            return concatenated_tensors

        else:
            # direct mapping
            original_key = self.new_key_to_original_key_map[new_key]
            return self.original_f.get_tensor(original_key, device=device, dtype=dtype)

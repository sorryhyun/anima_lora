"""A frozen DiT's large linears on int8 ConvRot GEMMs, for LoRA training and generation.
Shared by the MOSS and Anima lines, which each hold their own targets and file keys.

- ``convrot.py``: the Hadamard rotation and row scales, also used by MOSS's int8 TE.
- ``linear.py``: the custom ops, ``Int8TrainLinear``, ``convert_``, the int8 file reader.
- ``stream.py``: ``WqtStream``, the backward copy on the host, two blocks on the card.

The lines' sides: ``library/moss/dit_int8/base.py``, ``library/anima/int8.py``.
"""

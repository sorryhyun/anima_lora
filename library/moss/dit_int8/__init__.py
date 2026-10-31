"""The frozen DiT's 252 block linears on int8 ConvRot GEMMs, for LoRA training and
generation.

- ``linear.py``: the custom ops, ``Int8TrainLinear``, ``convert_``, the int8 file reader.
- ``stream.py``: ``WqtStream``, the backward copy on the host, two blocks on the card.
- ``base.py``: MOSS's side: targets, the file's forward copies, the training conversion.

``library/moss/int8.py`` is the text encoder's int8 and holds the rotation these share.
User doc: ``docs/moss/optimization/int8.md``.
"""

"""Façade contract: every export on anima_lora.{models,inference,config,training,captioning} resolves, and the pre-namespace flat aliases stay identical to the namespaced names."""

import anima_lora
from anima_lora import captioning, config, inference, models, training

SUBMODULES = [models, inference, config, training, captioning]


def test_flat_aliases_match_namespaced():
    # Flat surface is frozen for back-compat; each alias must resolve to the
    # same object as its namespaced home.
    namespaced = {name: getattr(m, name) for m in SUBMODULES for name in m.__all__}
    flat_only = []
    for name in anima_lora._ATTR_TO_MODULE:
        if name in namespaced:
            assert getattr(anima_lora, name) is namespaced[name], name
        else:
            flat_only.append(name)
    assert not flat_only, f"flat exports missing a namespaced home: {flat_only}"


def test_train_py_loads_by_path_shared_instance():
    import sys

    trainer_cls = training.AnimaTrainer
    train_mod = sys.modules.get("train")
    assert train_mod is not None
    assert trainer_cls is train_mod.AnimaTrainer

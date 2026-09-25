"""eval — the stages that render with a finished (or no) table and read it back.

stage        stage_eval: T2I floor vs trained on the eval set; sheet helpers
native       stage_native / stage_enref / stage_native_rescore; SceneKept, table_parts
enref        the EN-reference ruler (en cos / en cos out / box IoU)
classify     same-noise diffusion classifiers (classify, classify_str)
salad        Probe 0 on the base model
"""

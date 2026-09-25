"""eval — the stages that render with a finished (or no) table and read it back.

stage        stage_eval: T2I floor vs trained on the eval set; sheet helpers
native       stage_native / stage_target
enref        the EN-reference ruler (en cos / en cos out / box IoU)
cf_sense     caption leverage on a B-rendered input, by σ
summary      eval_summary.png
"""

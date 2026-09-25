"""train — the ``train`` stage and what only it reads.

stage        stage_train: frozen DiT, rectified-flow loss on the glyph items
trainables   what an arm trains and how it becomes the ExtDelta table
encoder      W2d glyph encoder g(render) → Δ_row and its glyph bank
"""

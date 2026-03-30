# running with dora

## fa2

override steps. steps for 2 epochs is / 指定エポックまでのステップ数: 186
running training / 学習開始
  num train images * repeats / 学習画像の数×繰り返し回数: 93
  num validation images * repeats / 学習画像の数×繰り返し回数: 4
  num reg images / 正則化画像の数: 0
  num batches per epoch / 1epochのバッチ数: 93
  num epochs / epoch数: 2
  batch size per device / バッチサイズ: 1
  gradient accumulation steps / 勾配を合計するステップ数 = 1
  total optimization steps / 学習ステップ数: 186
2026-03-30 23:05:19 INFO     text_encoder is not needed for training. deleting to save   train.py:1967
                             memory.                                                                  
                    INFO     unet dtype: torch.bfloat16, device: cuda:0                  train.py:2005
steps:   0%|                                                                  | 0/186 [00:00<?, ?it/s]
epoch 1/2

2026-03-30 23:05:20 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
2026-03-30 23:05:20 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
steps:  50%|████████████████████                    | 93/186 [03:54<03:54,  2.52s/it, avr_loss=0.0868]                          2026-03-30 23:09:14 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                         base.py:175]
2026-03-30 23:09:14 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                         base.py:175

epoch 2/2idation steps: 100%|███████████████████████████| 16/16 [00:11<00:00,  1.43it/s, val_epoch_avg_loss=0.0899, timestep=800]

2026-03-30 23:09:26 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                         base.py:175
2026-03-30 23:09:26 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                         base.py:175
epoch validation steps: 100%|███████████████████████████| 16/16 [04:10<00:00, 15.65s/it, val_epoch_avg_loss=0.0899, timestep=800]
2026-03-30 23:13:25 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                         base.py:175
2026-03-30 23:13:25 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                         base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-03-30 23:13:37 INFO     model saved.                                                                           train.py:2579
steps: 100%|███████████████████████████████████████| 186/186 [07:54<00:00,  2.55s/it, avr_loss=0.0977]
epoch validation steps: 100%|███████████████████████████| 16/16 [00:12<00:00,  1.30it/s, val_epoch_avg_loss=0.0839, timestep=800]

## fa4

2026-03-30 22:56:15 INFO     use 8-bit AdamW optimizer | {}                           optimizers.py:74
override steps. steps for 2 epochs is / 指定エポックまでのステップ数: 186
running training / 学習開始
  num train images * repeats / 学習画像の数×繰り返し回数: 93
  num validation images * repeats / 学習画像の数×繰り返し回数: 4
  num reg images / 正則化画像の数: 0
  num batches per epoch / 1epochのバッチ数: 93
  num epochs / epoch数: 2
  batch size per device / バッチサイズ: 1
  gradient accumulation steps / 勾配を合計するステップ数 = 1
  total optimization steps / 学習ステップ数: 186
2026-03-30 22:56:18 INFO     text_encoder is not needed for training. deleting to save   train.py:1967
                             memory.                                                                  
                    INFO     unet dtype: torch.bfloat16, device: cuda:0                  train.py:2005
steps:   0%|                                                                  | 0/186 [00:00<?, ?it/s]
epoch 1/2

                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
steps:  50%|████████████████████                    | 93/186 [03:58<03:58,  2.56s/it, avr_loss=0.08192026-03-30 23:00:17 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175]
2026-03-30 23:00:17 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175

epoch 2/2idation steps: 100%|█| 16/16 [00:13<00:00,  1.19it/s, val_epoch_avg_loss=0.083, timestep=800]

2026-03-30 23:00:30 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:00:30 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
epoch validation steps: 100%|█| 16/16 [04:11<00:00, 15.69s/it, val_epoch_avg_loss=0.083, timestep=800]
2026-03-30 23:04:28 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:04:28 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-03-30 23:04:40 INFO     model saved.                                                train.py:2579
steps: 100%|███████████████████████████████████████| 186/186 [07:56<00:00,  2.56s/it, avr_loss=0.0995]
epoch validation steps: 100%|█| 16/16 [00:12<00:00,  1.30it/s, val_epoch_avg_loss=0.0885, timestep=800

## fa2 /w torch_compile, peak vram = 7.6gb

2026-03-30 23:33:50 INFO     use 8-bit AdamW optimizer | {}                           optimizers.py:74
override steps. steps for 2 epochs is / 指定エポックまでのステップ数: 186
running training / 学習開始
  num train images * repeats / 学習画像の数×繰り返し回数: 93
  num validation images * repeats / 学習画像の数×繰り返し回数: 4
  num reg images / 正則化画像の数: 0
  num batches per epoch / 1epochのバッチ数: 93
  num epochs / epoch数: 2
  batch size per device / バッチサイズ: 1
  gradient accumulation steps / 勾配を合計するステップ数 = 1
  total optimization steps / 学習ステップ数: 186
2026-03-30 23:33:52 INFO     text_encoder is not needed for training. deleting to save   train.py:1967
                             memory.                                                                  
2026-03-30 23:33:53 INFO     unet dtype: torch.bfloat16, device: cuda:0                  train.py:2005
steps:   0%|                                                                  | 0/186 [00:00<?, ?it/s]
epoch 1/2

                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
steps:  50%|█████████████████████▌                     | 93/186 [08:12<08:12,  5.30s/it, avr_loss=0.12026-03-30 23:42:06 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175]
2026-03-30 23:42:06 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175

epoch 2/2idation steps: 100%|█| 16/16 [00:41<00:00,  2.60s/it, val_epoch_avg_loss=0.0847, timestep=800

2026-03-30 23:42:47 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:42:47 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
epoch validation steps: 100%|█| 16/16 [03:29<00:00, 13.08s/it, val_epoch_avg_loss=0.0847, timestep=800
2026-03-30 23:45:35 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:45:35 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-03-30 23:45:44 INFO     model saved.                                                train.py:2579
steps: 100%|███████████████████████████████████████| 186/186 [11:00<00:00,  3.55s/it, avr_loss=0.0921]
epoch validation steps: 100%|█| 16/16 [00:08<00:00,  1.79it/s, val_epoch_avg_loss=0.0974, timestep=80

## fa4 /w torch_compile, peak vram = 6.7gb

2026-03-30 23:23:46 INFO     use 8-bit AdamW optimizer | {}                           optimizers.py:74
override steps. steps for 2 epochs is / 指定エポックまでのステップ数: 186
running training / 学習開始
  num train images * repeats / 学習画像の数×繰り返し回数: 93
  num validation images * repeats / 学習画像の数×繰り返し回数: 4
  num reg images / 正則化画像の数: 0
  num batches per epoch / 1epochのバッチ数: 93
  num epochs / epoch数: 2
  batch size per device / バッチサイズ: 1
  gradient accumulation steps / 勾配を合計するステップ数 = 1
  total optimization steps / 学習ステップ数: 186
2026-03-30 23:23:48 INFO     text_encoder is not needed for training. deleting to save   train.py:1967
                             memory.                                                                  
2026-03-30 23:23:49 INFO     unet dtype: torch.bfloat16, device: cuda:0                  train.py:2005
steps:   0%|                                                                  | 0/186 [00:00<?, ?it/s]
epoch 1/2

                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
                    INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175
W0330 23:23:52.042000 266597 torch/_inductor/utils.py:1731] [4/0] Not enough SMs to use max_autotune_gemm mode
steps:  50%|████████████████████                    | 93/186 [04:22<04:22,  2.82s/it, avr_loss=0.08892026-03-30 23:28:11 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175]
2026-03-30 23:28:11 INFO     epoch is incremented. current_epoch: 0, epoch: 1              base.py:175

epoch 2/2idation steps: 100%|█| 16/16 [00:17<00:00,  1.10s/it, val_epoch_avg_loss=0.0872, timestep=800

2026-03-30 23:28:29 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:28:29 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
epoch validation steps: 100%|█| 16/16 [04:12<00:00, 15.80s/it, val_epoch_avg_loss=0.0872, timestep=800
2026-03-30 23:32:24 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175
2026-03-30 23:32:24 INFO     epoch is incremented. current_epoch: 0, epoch: 2              base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-03-30 23:32:36 INFO     model saved.                                                train.py:2579
steps: 100%|███████████████████████████████████████| 186/186 [08:17<00:00,  2.68s/it, avr_loss=0.0961]
epoch validation steps: 100%|█| 16/16 [00:12<00:00,  1.27it/s, val_epoch_avg_loss=0.0872, timestep=800
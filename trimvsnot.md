trim_crossattn_kv

Anima: compiled 28 blocks with backend=inductor
2026-04-03 11:35:31 INFO     static_token_count=4096                                                                                     train.py:353
import network module: networks.lora_anima
                    INFO     create LoRA network. base dim (rank): 16, alpha: 16                                                    lora_anima.py:325
                    INFO     neuron dropout: p=None, rank dropout: p=None, module dropout: p=None                                   lora_anima.py:328
                    INFO     create LoRA for Text Encoder 1:                                                                        lora_anima.py:497
                    INFO     create LoRA for Text Encoder 1: 196 modules.                                                           lora_anima.py:504
                    INFO     create LoRA for Anima DiT: 280 modules.                                                                lora_anima.py:518
                    INFO     enabling fp32 accumulation for LoRA modules                                                            lora_anima.py:539
                    INFO     enable LoRA for DiT: 280 modules                                                                       lora_anima.py:625
prepare optimizer, data loader etc.
                    INFO     use 8-bit AdamW optimizer | {}                                                                          optimizers.py:74
override steps. steps for 2 epochs is / 指定エポックまでのステップ数: 162
running training / 学習開始
  num train images * repeats / 学習画像の数×繰り返し回数: 157
  num validation images * repeats / 学習画像の数×繰り返し回数: 39
  num reg images / 正則化画像の数: 0
  num batches per epoch / 1epochのバッチ数: 81
  num epochs / epoch数: 2
  batch size per device / バッチサイズ: 2
  gradient accumulation steps / 勾配を合計するステップ数 = 1
  total optimization steps / 学習ステップ数: 162
2026-04-03 11:35:34 INFO     text_encoder is not needed for training. deleting to save memory.                                          train.py:1980
                    INFO     unet dtype: torch.bfloat16, device: cuda:0                                                                 train.py:2018
steps:   0%|                                                                                                                 | 0/162 [00:00<?, ?it/s]
epoch 1/2

2026-04-03 11:35:35 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                                             base.py:175
2026-04-03 11:35:35 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                                             base.py:175
steps:  50%|███████████████████████████████████████████▌                                           | 81/162 [07:26<07:26,  5.51s/it, avr_loss=0.08692026-04-03 11:43:01 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                                             base.py:175]
2026-04-03 11:43:01 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                                             base.py:175

epoch 2/2idation steps: 100%|██████████████████████████████████████████████████| 88/88 [02:05<00:00,  1.42s/it, val_epoch_avg_loss=0.208, sigma=0.35]

2026-04-03 11:45:06 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                                             base.py:175
2026-04-03 11:45:06 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                                             base.py:175
steps:  96%|██████████████████████████████████████████████████████████████████████████████████▊   | 156/162 [14:08<00:32,  steps:  96%|██████████████████████████████████████████████████████████████████████████████████▊   | 156/162 [14:08<00:32,  steps:  97%|███████████████████████████████████████████████████████████████████████████████████▎  | 157/162 [14:14<00:27,  steps:  97%|███████████████████████████████████████████████████████████████████████████████████▎  | 157/162 [14:14<00:27,  steps:  98%|███████████████████████████████████████████████████████████████████████████████████▉  | 158/162 [14:20<00:21,  steps:  98%|███████████████████████████████████████████████████████████████████████████████████▉  | 158/162 [14:20<00:21,  steps:  98%|████████████████████████████████████████████████████████████████████████████████████▍ | 159/162 [14:26<00:16,  steps:  98%|████████████████████████████████████████████████████████████████████████████████████▍ | 159/162 [14:26<00:16,  steps:  99%|████████████████████████████████████████████████████████████████████████████████████▉ | 160/162 [14:32<00:10,  steps:  99%|████████████████████████████████████████████████████████████████████████████████████▉ | 160/162 [14:32<00:10,  steps:  99%|█████████████████████████████████████████████████████████████████████████████████████▍| 161/162 [14:38<00:05,  steps:  99%|█████████████████████████████████████████████████████████████████████████████████████▍| 161/162 [14:38<00:05,  steps: 100%|██████████████████████████████████████████████████████████████████████████████████████| 162/162 [14:43<00:00,  steps: 100%|██████████████████████████████████████████████████████████████████████████████████████| 162/162 [14:43<00:00,  epoch validation steps: 100%|██████████████████████████████████████████████████| 88/88 [09:22<00:00,  6.40s/it, val_epoch_avg_loss=0.208, sigma=0.35]
2026-04-03 11:52:24 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175
2026-04-03 11:52:24 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-04-03 11:54:35 INFO     model saved.                                                                     train.py:2601
steps: 100%|██████████████████████████████████████████████████████████████████████████████████████| 162/162 [14:44<00:00,  5.46s/it, avr_loss=0.0905]
epoch validation steps: 100%|█████████████████████████| 88/88 [02:11<00:00,  1.49s/it, val_epoch_avg_loss=0.21, sigma=0.35]
(anima) sorryhyun@sorryhyun-MS-7C94:~/anima/anima_lora$ 




original

steps:   0%|                                                                                       | 0/162 [00:00<?, ?it/s]
epoch 1/2

                    INFO     epoch is incremented. current_epoch: 0, epoch: 1                                   base.py:175
                    INFO     epoch is incremented. current_epoch: 0, epoch: 1                                   base.py:175
steps:  50%|██████████████████████████████▌                              | 81/162 [07:29<07:29,  5.55s/it, avr_loss=0.09862026-04-03 12:04:12 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                   base.py:175]
2026-04-03 12:04:12 INFO     epoch is incremented. current_epoch: 0, epoch: 1                                   base.py:175

epoch 2/2idation steps: 100%|████████████████████████| 88/88 [02:05<00:00,  1.43s/it, val_epoch_avg_loss=0.203, sigma=0.35]

2026-04-03 12:06:18 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175
2026-04-03 12:06:18 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175
epoch validation steps: 100%|████████████████████████| 88/88 [09:26<00:00,  6.44s/it, val_epoch_avg_loss=0.203, sigma=0.35]
2026-04-03 12:13:39 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175
2026-04-03 12:13:39 INFO     epoch is incremented. current_epoch: 0, epoch: 2                                   base.py:175

saving checkpoint: output/anima_lora.safetensors
2026-04-03 12:15:49 INFO     model saved.                                                                     train.py:2601
steps: 100%|████████████████████████████████████████████████████████████| 162/162 [14:50<00:00,  5.50s/it, avr_loss=0.0873]
epoch validation steps: 100%|████████████████████████| 88/88 [02:10<00:00,  1.48s/it, val_epoch_avg_loss=0.205, sigma=0.35]
# text-binding judge — gt `ちょっとだけだから` / other `明日も晴れるかな` (hit = CER ≤ 0.5)

| group | arm | cond | expect | n | CER(ref) | CER(gt) | hit | present |
|---|---|---|---|---:|---:|---:|---:|---:|
| textbind-trained-3img-9095721 | ja_ext | drop_all | none | 3 | 1.00 | 1.00 | 0.00 | 0.00 |
| textbind-trained-3img-9095721 | ja_ext | drop_quote | none | 3 | 0.96 | 0.96 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_ext | other_text | other | 3 | 0.96 | 1.04 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_ext | same | gt | 3 | 1.00 | 1.00 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_ext | swap | gt | 3 | 1.30 | 1.30 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_ext | swap_drop | none | 3 | 0.93 | 0.93 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_native | drop_all | none | 3 | 1.00 | 1.00 | 0.00 | 0.00 |
| textbind-trained-3img-9095721 | ja_native | drop_quote | none | 3 | 0.96 | 0.96 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_native | other_text | other | 3 | 1.00 | 1.04 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_native | same | gt | 3 | 1.00 | 1.00 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_native | swap | gt | 3 | 1.19 | 1.19 | 0.00 | 1.00 |
| textbind-trained-3img-9095721 | ja_native | swap_drop | none | 3 | 0.93 | 0.93 | 0.00 | 1.00 |

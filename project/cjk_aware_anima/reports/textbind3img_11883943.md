# text-binding judge — gt `着けるんですか！？` / other `本当にいいんですか！？` (hit = CER ≤ 0.5)

| group | arm | cond | expect | n | CER(ref) | CER(gt) | hit | present |
|---|---|---|---|---:|---:|---:|---:|---:|
| textbind-trained-3img-11883943 | ja_ext | drop_all | none | 3 | 0.95 | 0.95 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_ext | drop_quote | none | 3 | 0.95 | 0.95 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_ext | other_text | other | 3 | 1.00 | 1.05 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_ext | same | gt | 3 | 0.95 | 0.95 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_ext | swap | gt | 3 | 1.00 | 1.00 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_ext | swap_drop | none | 3 | 1.33 | 1.33 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | drop_all | none | 3 | 1.14 | 1.14 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | drop_quote | none | 3 | 1.14 | 1.14 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | other_text | other | 3 | 0.93 | 0.95 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | same | gt | 3 | 1.00 | 1.00 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | swap | gt | 3 | 0.95 | 0.95 | 0.00 | 1.00 |
| textbind-trained-3img-11883943 | ja_native | swap_drop | none | 3 | 1.48 | 1.48 | 0.00 | 1.00 |

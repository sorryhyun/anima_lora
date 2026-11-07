# The reference copy is a protocol, not a head; the reference edges gate it

Scripts: `src/copy_head_ext.py` (`output/tests/copy_head_ext/20261107-1522`, 1360 edge;
`…/20261107-1532`, 1024 edge), `src/copy_transplant.py`
(`output/tests/copy_transplant/20261107-1539`), `src/ref_edge_mask.py`
(`output/tests/ref_edge_mask/20261107-1549`). Each folder has `summary.json` and
`sheet_*.png`.

LoRA: [chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit) (rank 64 /
alpha 64 on all 42 blocks' q / k / v / o / ffn, ConceptEdit-12M instruction edits, its own
trainer), converted to `output/ckpt_moss_i2i/ext/moss_image_edit_chinoll.safetensors`. Its
contract (`ref-prefix-only-t0`) is ours (`ref_timestep 0`, `ref_isolate`), except that it
places the reference on RoPE frame 0 where we use frame = text length; the scripts patch
`_RoPE.forward` to match. The weights' licence is not stated on the card (the project code is
GPLv3): use it as a probe subject only.

Sampler as its `infer.py`: Euler 50, CFG 4, plain CFG, negative `""`, plain instruction
through the chat template, no tail. Base DiT, INT8, seed 0. Inputs: two colour
illustrations (`image_dataset/sushispin/{10043045,9859367}`) and the sanitize eval page
`usual_case1`.

## Claims

1. **An outside edit LoRA builds the same copy head.** 0.56 / 0.81 of block 39's v / o
   energy sits on head 19 (ours 0.17–0.61), second h11 as in ours. Its image queries
   attend 0.66 to the own-cell reference token at step 0 (Base 0.000).
2. **It leans on that head harder than ours.** Cutting b39 h19 moves the render 21.5 luma
   L1 at a 1024 edge (ours 1.8–8.7; the control head 3.4), and collapses the copy (104) at
   a 1360 edge, where the LoRA is outside its training size.
3. **Its gate forms during sampling.** Copy inside / outside the edit region: 0.41 / 0.50
   at step 0, 0.17 / 0.39 over all steps (r −0.52 against the pixel change), at 1360. Ours
   with boxes in the prompt is set at step 0. When the instruction names something the
   picture does not hold (a skirt out of frame), copy falls 0.39 → 0.005 and it paints a
   new picture.
4. **No slice of the LoRA carries the copy on its own.** h19 copy mass (mid steps) and
   luma L1 against the reference, on the hair-colour and sanitize inputs:

   | arm | copy | L/ref |
   |---|---|---|
   | Base | 0.000 | 24 / 16 |
   | full LoRA | 0.77 / 0.68 | 8 / 8 |
   | b39 h19 only (q / k / v / o) | 0.045 / 0.016 | 25 / 16 |
   | b39 h19 q / k only | 0.045 / 0.016 | 24 / 16 |
   | top 10 copy heads | 0.042 / 0.014 | 24 / 17 |
   | blocks 36–41 | 0.07 / 0.04 | 30 / 22 |
   | full LoRA on the reference stream, h19 on the main rows | 0.045 / 0.025 | 46 / 61 |
   | full LoRA on the reference stream only | 0 | 46 / 61 |

   The head's match needs both the image queries and the reference keys shaped by the
   earlier blocks. With the reference stream changed and the main rows not, the main rows
   cannot read the reference at all and render the instruction as t2i (a photo of a woman
   with pink hair). Encoding and reading change together across the network.
5. **A logit bias on the main-query → reference-key edges gates the copy in every head at
   once.** With `ref_isolate` those edges are the reference's only way into the main rows.
   Bias on the image queries inside a box, every block, head and step, both CFG branches
   (luma L1 against the reference, outside the box with a one-cell ring left out / inside):

   | arm | flower → blue ribbon | horns → cat ears |
   |---|---|---|
   | no bias | 7.7 / 6.4 | 6.1 / 13.7 |
   | −2 | 9.3 / 33.9 | 6.0 / 19.5 |
   | −4 | 11.2 / 41.0 | 5.9 / 33.5 |
   | −∞ | 11.7 / 48.4 | 5.9 / 38.2 |

   Without the bias the LoRA kept the flower and hung a ribbon outside the box; with it the
   flower goes. On the horns the LoRA was already right, and −2 leaves it so. At −4 and −∞
   the box loses what it should keep (the eyes under the horns, the forehead): the hard
   cut stops the box from reading the reference, not only from copying it. The drawing
   inside the box did not follow "blue ribbon".

For edits built from parts: a copy protocol cannot be lifted out head by head, but its
gate can be set from outside without parameters. Proposal:
[`project/lora_blocks/proposal.md`](../../lora_blocks/proposal.md).

## Not tested

- A diagonal gate (only the own cell ± k cut) that keeps the box reading the reference.
- Gate schedules over steps; segmentation masks instead of boxes.
- The same gate on our own i2i LoRAs.
- More than one seed, more than three inputs.

# Idea — a layout bank for the sentence loop

Status: PROPOSED (2026-11-27). One forward probe has run
([reports/layout_hold_2026_11_27.md](reports/layout_hold_2026_11_27.md)); nothing is trained
and nothing is queued. It waits on the reader's retrain, as the line does
([plan.md](plan.md), Next). The user (11-27): "아이디어 자체는 좋은거같아".

## The idea

The sentence loop renders a string on whichever (training scene, seed) comes round, and most
of those pages have no room for it: a long string comes out at about half its glyphs, and the
row's gradient is taken on that page all the same. The idea is to take the gradient only on
pages whose layout fits the string.

It began as a static layout LoRA for the loop (the user, 11-27: scenes that draw the wanted
layout are hard to find by prompt). The form proposed here is smaller: a list of
(scene, seed) pairs, each with its room, and a string drawn onto pairs that hold its length.
A pair is called a cell below.

The bank is for training only. What ships is the rows, as now. A cell's layout belongs to its
scene prompt and seed together, so it says nothing about a user's prompt.

## What was measured

All from `lhold_1127_1851` (2 glyphs against 5 – 8, 4 read scenes, 16 cells), spread unknown.

- A cell holds the bubble's place whatever the string: picked boxes of different strings
  within 0.1 of the page's side on 0.80 – 0.83 of pairs, against 0.18 – 0.20 at another seed of
  the same scene.
- Cells differ in how much of a long string reads on them (0.12 – 0.55 of positions), and the
  difference repeats on other strings (split halves at 0.81).
- The cells do not sort by length. The lettered area is the cell's, the same at 2 glyphs as at
  5 – 8, and long strings read more where it is larger (0.27 to 0.44 by quartile of the area).
- The bubble is set by σ 0.75. Text given to the last two steps alone leaves it where the
  earlier text put it.
- A state made for another length costs a long string 0.10 of 0.375 positions.
- SAM3 on the prompt `speech bubble` finds the bubble on 12 of 12 pages, the kinds a
  brightness rule misses among them.

## The three forms

| form | what is kept | what it costs |
|---|---|---|
| cell list | (scene, seed) and its room; all four steps run on the real string | a screen; nothing in the loop |
| cached state | the σ 0.75 latent of a passed page | the 0.10 above unless made a length; goes stale as rows move; the line's rule is no x_3 cache |
| LoRA on the first two steps | a layout adapter, off on the two differentiated steps | a LoRA to train on EN-lettered or text-free pages; the same length cost unless the layout answers the string |

The cell list comes first. The other two are for a layout the frozen model does not draw on any
seed often enough to screen (a bubble a string, a tall bubble for a long string), and only
there.

## What it can and cannot give

- It can give a cleaner gradient: a position the page had room for.
- It cannot teach the rows a layout. The two differentiated steps do not move the bubble or
  the way the text stands, so no row gradient reaches either. Vertical lettering through the
  rows would need the step σ 0.9 → 0.75 in the backward (`back_steps = 3`), which is a
  separate arm.
- It may fit the rows to pages with room and leave ordinary pages where they were. Only a read
  on seeds nobody screened tells.

## Steps, in order

1. The screen (forward only): the 16 training scenes × seeds × long strings, a cell's room by
   SAM3's box and its read of the strings. Checked on a second set of strings, since a cell
   picked on its own count reads too well on it.
2. A micro arm of `sentw_1126_1528`'s size on the screened cells, against the same recipe on
   the same number of unscreened seeds. Read on val strings at unscreened seeds, the 5 – 8 bin
   and the tall labels apart from the rest. Level with the control is no adoption.
3. Only if 2 reads above its control: the draw by room (a string onto cells that hold its
   length) in the run at scale.

## Open

- What room is measured in: SAM3's box area, or glyphs a cell has shown it can hold.
- How many cells pass, and whether 16 scenes give enough of them for the row budget
  (`sents_1126_1856` ended on its budget, not on its strings).
- Whether a tag for a large or a tall bubble moves a cell's room. Untried, forward only.
- Lengths between 2 and 5 – 8, and over 8: not in the probe.
- The look-alike folds of [issues.md](issues.md) change what a hit is, so the screen comes
  after them.

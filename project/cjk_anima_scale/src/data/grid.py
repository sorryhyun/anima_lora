"""The grid renderer the line's recipes use (``cjk_scale/recipes.py``):
``render_grid`` draws k strings in a cols × rows grid, one per cell, and
``_Deck`` deals them in shuffled passes. The old data stage's ``--grid``
item builder (``grid_recs``) is gone (pruned 2026-09-25).
"""

from __future__ import annotations

import random

from common.render.flat import JITTER_BG_LIGHT, JITTER_INK_DARK, pick_font
from common.render.scene import V_ROTATE

# a stacked column draws these unrotated / misplaced (PIL has no vertical
# forms), so a word holding one is always a line
_NO_COLUMN = V_ROTATE | set("「」、。")


class _Deck:
    """Shuffled passes over ``pool``; ``deal(k)`` returns k distinct vocabs. A
    vocab repeated in the pool (its vocab-spec weight; every vocab but the
    ``small`` digraphs, SMALL_PER times) is dealt that many times per pass."""

    def __init__(self, pool: list, rng: random.Random):
        self.pool, self.rng, self.deck = pool, rng, []
        self.n_distinct = len(set(pool))

    def _fill(self):
        if not self.deck:
            self.deck = self.pool[:]
            self.rng.shuffle(self.deck)

    def deal(self, k: int, max_len: int = 0) -> list:
        """``max_len``: units of more glyphs wait for a later item (the first
        unit is dealt whatever its length — it chose the grid)."""
        assert k <= self.n_distinct, f"grid of {k} cells over {self.n_distinct} units"
        got, aside = [], []
        while len(got) < k:
            self._fill()
            u = self.deck.pop()
            skip = u in got or (max_len and got and len(u) > max_len)
            (aside if skip else got).append(u)
            assert len(aside) <= 3 * len(self.pool), (
                f"no {k} units of ≤ {max_len} glyphs"
            )
        self.deck += aside  # a repeat / a too-long unit waits for the next item
        return got


# word cells: pad range and the rounded-box inset, as shares of the cell's short
# side — ``max_glyphs`` reads the room off the same numbers
WORD_PAD = (0.03, 0.05)
WORD_BOX_INSET = 0.05


def render_grid(
    units,
    cols,
    rows,
    size,
    fonts,
    rng,
    bubble: bool,
    fill,
    pad=(0.04, 0.10),
    box=False,
    sizes=None,
    lines=None,
    horizontal_frac: float = 0.5,
):
    """Draw ``units[i]`` in cell ``i`` (row-major). Returns ``(image, boxes)``,
    ``boxes[i]`` the ink bbox of cell i's unit in canvas pixels. ``box`` = the
    bubble is a rounded box (word cells) instead of an ellipse; ``sizes``
    collects every cell's font px, ``lines`` the cells drawn as a horizontal
    line. ``horizontal_frac``: each multi-glyph cell is a left-to-right line
    with this probability (drawn per cell, so one grid mixes both), a column
    otherwise; ``_NO_COLUMN`` units are always a line; a single glyph has no
    orientation."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = size
    cw, ch = W / cols, H / rows
    pad_range = pad
    bg = rng.choice(JITTER_BG_LIGHT)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if bubble:
        for _ in range(900):
            x, y = rng.randrange(W), rng.randrange(H)
            d.ellipse((x, y, x + 2, y + 2), fill=(150, 150, 150))
    boxes = []
    for i, u in enumerate(units):
        r, c = divmod(i, cols)
        x0, y0 = c * cw, r * ch
        font_path = pick_font(u, fonts, rng)
        vertical = (
            len(u) > 1 and not (set(u) & _NO_COLUMN) and rng.random() >= horizontal_frac
        )
        # the room the ink may take: the cell, or the inscribed rectangle of
        # the cell's bubble
        pad = rng.uniform(*pad_range) * min(cw, ch)
        room_w, room_h = cw - 2 * pad, ch - 2 * pad
        if bubble and box:
            inset = WORD_BOX_INSET * min(cw, ch)
            room_w, room_h = room_w - 2 * inset, room_h - 2 * inset
        elif bubble:
            room_w, room_h = room_w / 2**0.5 - 6, room_h / 2**0.5 - 6
        fs = int(rng.uniform(*fill) * min(cw, ch))
        while True:
            font = ImageFont.truetype(font_path, fs, index=0)
            txt = "\n".join(u) if vertical else u
            bx = d.multiline_textbbox((0, 0), txt, font=font, spacing=fs * 0.05)
            tw, th = bx[2] - bx[0], bx[3] - bx[1]
            if (tw <= room_w and th <= room_h) or fs <= 12:
                break
            fs = max(12, int(fs * min(room_w / tw, room_h / th) * 0.98))
        if sizes is not None:
            sizes.append(fs)
        if lines is not None and len(u) > 1 and not vertical:
            lines.append(i)
        # block centre: anywhere the ink stays inside its room
        cx = x0 + cw / 2 + (rng.random() - 0.5) * max(0.0, room_w - tw)
        cy = y0 + ch / 2 + (rng.random() - 0.5) * max(0.0, room_h - th)
        if bubble:
            frame = (x0 + pad, y0 + pad, x0 + cw - pad, y0 + ch - pad)
            kwb = {"fill": "white", "outline": "black", "width": rng.randint(2, 5)}
            if box:
                d.rounded_rectangle(frame, radius=0.18 * min(cw, ch), **kwb)
            else:
                d.ellipse(frame, **kwb)
        ox, oy = cx - tw / 2 - bx[0], cy - th / 2 - bx[1]
        kw = {"font": font, "spacing": fs * 0.05, "align": "center"}
        d.multiline_text((ox, oy), txt, fill=rng.choice(JITTER_INK_DARK), **kw)
        # the ink's own bbox: textbbox is off by a font's bearings (デ, ロ)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).multiline_text((ox, oy), txt, fill=255, **kw)
        boxes.append(
            list(mask.getbbox() or (int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        )
    return im, boxes

"""Flags the ``scenes`` stage reads (generation and the keep / reject judge)."""

from __future__ import annotations


def scene_args(g):
    g.add_argument(
        "--scene_tag", default="s0", help="scenes: output/wake_probe/scenes_<tag>"
    )
    g.add_argument(
        "--scene_n", type=int, default=1000, help="scenes: prompts to generate"
    )
    g.add_argument(
        "--scene_shapes",
        default="448,512:2,448x512,512x448",
        help="scenes: canvas pool (S0 recipe; --shapes syntax)",
    )
    g.add_argument(
        "--scene_anchors",
        default="",
        help="scenes: comma list of EN anchor words (default the built-in ten)",
    )
    g.add_argument(
        "--scene_frames",
        default="reads_as",
        help="scenes: comma list of text frames the prompt asks for the anchor in "
        "(reads_as|bubble_reads|saying|sign|bare_quotes|sfx; s0 = reads_as only; sfx keeps its own onomatopoeia anchors and passes an open fill). "
        "Pronoun frames are drawn for solo counts only; the data stage swaps "
        "the JA text into the same frame",
    )
    g.add_argument(
        "--scene_ja_anchors",
        default="",
        help="scenes: comma list of JA anchors for the ja_* frames (default the "
        "built-in 24 short manga lines); the base's kana come out garbled and "
        "are erased — the frame is asked for so the base draws a *tall* bubble",
    )
    g.add_argument(
        "--scene_extra_tags",
        default="",
        help="scenes: comma list of general tags appended to every prompt "
        "(recorded in the scene's generals, so the composite caption carries "
        "them); e.g. `monochrome,screentone` for manga-page bubbles; "
        "`comic` is allowed but marks native prompt 8 (`comic, 2koma, ...`) as "
        "seen — read native on the other 7; NOT `2koma` / `greyscale`",
    )
    g.add_argument(
        "--scene_bubble_tag",
        default="speech bubble",
        help="scenes: bubble tag in the prompt — recorded so the data stage "
        "spells the composite caption identically",
    )
    g.add_argument(
        "--scene_min_box",
        type=int,
        default=56,
        help="scenes: reject usable regions under this many px on the short side",
    )
    g.add_argument(
        "--scene_max_residual",
        type=float,
        default=0.33,
        help="scenes: reject when more than this share of the anchor's ink would "
        "survive the composite erase (the flood took another blob; s0: 12/186 "
        "kept scenes at ≥ 0.89, every clean scene ≤ 0.31). 0.5 until Δ1 "
        "(2026-09-18): the 11 scenes at 0.35–0.48 each keep anchor letters "
        "beside the pasted glyph (s1w 1209 'yes' → 'y'), 0.30–0.31 read clean",
    )
    g.add_argument(
        "--scene_char_frac",
        type=float,
        default=0.3,
        help="scenes: share of 1girl prompts naming a dataset character (else `original`)",
    )
    g.add_argument(
        "--scene_artist_frac",
        type=float,
        default=0.8,
        help="scenes: share of prompts carrying one of the dataset's @artist tags",
    )
    g.add_argument(
        "--scene_batch", type=int, default=4, help="scenes: prompts per DiT pass"
    )
    g.add_argument(
        "--scene_negative",
        default="worst quality, lowres, old, bad hands, bad anatomy, sepia, blurry, glitch, jpeg artifacts",
        help="scenes: negative prompt (inference only; never enters a caption)",
    )
    g.add_argument(
        "--scene_artists",
        default="sincos,hews",
        help="scenes: comma list of curated artist names added to the dataset pool at 4× weight",
    )
    g.add_argument(
        "--scene_gen_scale",
        type=float,
        default=1.0,
        help="scenes: render at this multiple of the pool shape, then downsample "
        "(the base draws crude scenes at 512²; 2.0 = its native ~1024)",
    )
    g.add_argument(
        "--scene_rejudge",
        type=int,
        default=0,
        help="scenes: re-apply the filter to scenes_<tag> from its stored reads (CPU, no generation)",
    )
    g.add_argument(
        "--scene_open_uniform",
        type=float,
        default=0.9,
        help="scenes: keep a scene with no closed bubble when this share of the "
        "rectangle erase (outside the text box) is within tol of the fill — "
        "white bubble with a broken outline on white, a board, a plain wall",
    )
    g.add_argument(
        "--scene_open_lost",
        type=float,
        default=0.02,
        help="scenes: an open (rectangle) erase may paint over at most this share "
        "of non-fill ink outside the text box — the outline, shelf lines, hair "
        "(Δ0.9: clean at <= 0.02, outlines cut from 0.03; s1 627 0.16)",
    )
    g.add_argument(
        "--scene_max_offset",
        type=float,
        default=1.0,
        help="scenes: reject a closed bubble whose usable region's centre is more "
        "than this many text-box half-sizes from the text — the flood leaked "
        "through an outline gap (Δ0.9: median 0.11; ja_comic 770 1.4)",
    )
    g.add_argument(
        "--scene_allow_open",
        type=int,
        default=0,
        help="scenes: keep images whose bubble fill runs to the border (open background)",
    )

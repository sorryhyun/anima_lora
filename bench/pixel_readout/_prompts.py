"""P3's working set: 16 prompts the render reads share."""

from bench._anima import DEFAULT_PROMPT

Q = "masterpiece, best quality, score_7, safe. "
PROMPTS = [  # (name, prompt, (h, w))
    ("anima_sign", DEFAULT_PROMPT, (1024, 1024)),
    (
        "sakura",
        Q
        + "1girl, solo, silver hair, long hair, red eyes, school uniform, cherry blossoms, falling petals, wind, looking back, upper body.",
        (1216, 832),
    ),
    (
        "neon_rain",
        Q
        + "1boy, black jacket, city street at night, neon signs, rain, puddle reflections, cinematic lighting.",
        (832, 1216),
    ),
    (
        "castle",
        Q
        + "scenery, no humans, fantasy castle on a cliff, sunset, dramatic clouds, detailed background.",
        (832, 1216),
    ),
    (
        "cafe",
        Q
        + "2girls, cafe interior, sitting at table, cake, coffee cup, window light, smile.",
        (1024, 1024),
    ),
    (
        "maid",
        Q
        + "1girl, maid, frills, lace, detailed clothes, holding tray, full body, white background.",
        (1216, 832),
    ),
    (
        "chibi",
        Q + "chibi, 1girl, flat color, thick outlines, sticker, white background.",
        (1024, 1024),
    ),
    (
        "candle",
        Q
        + "1girl, dark room, candlelight, low key lighting, dramatic shadows, portrait.",
        (1024, 1024),
    ),
    (
        "beach",
        Q + "1girl, swimsuit, beach, blue sky, ocean, sunlight, splashing water.",
        (832, 1216),
    ),
    (
        "battle",
        Q + "1girl, armor, sword, battle, sparks, dynamic pose, motion lines.",
        (1216, 832),
    ),
    (
        "kotatsu",
        Q + "cat, kotatsu, mandarin oranges, cozy room, tatami, detailed background.",
        (1024, 1024),
    ),
    (
        "library",
        Q + "1girl, glasses, library, bookshelf, many books, reading, soft light.",
        (1216, 832),
    ),
    (
        "shop_sign_ja",
        Q
        + '1girl, apron, shop front, holding sign, japanese text, the sign reads "ありがとう".',
        (1024, 1024),
    ),
    (
        "manga_page",
        Q
        + "monochrome, greyscale, comic, manga, panels, 1girl, speech bubble, japanese text, screentone.",
        (1216, 832),
    ),
    (
        "manga_action",
        Q
        + "monochrome, greyscale, comic, 1boy, 1girl, speech bubble, japanese text, action lines, sound effects.",
        (1216, 832),
    ),
    (
        "sketch",
        Q + "monochrome, greyscale, sketch, lineart, hatching, 1girl, portrait.",
        (1024, 1024),
    ),
]

"""Phase-0 prompt set for the front-loaded text-drive boost bench.

Three kinds:

* ``complex``    — complicated multi-tag prompts where the baseline plausibly
                   drops or mangles some tags. ``watch`` lists the specific
                   tags to check in the grid (the G1 signal). Selected by hand
                   (tag_influence.py deliberately NOT used — its Phase-0 gate
                   has never run, so it can't be trusted for selection either).
* ``border``     — the known border-artifact reproducer. Arm (a) failure
                   mode: high CFG at high σ is the burn/border regime. Checked
                   WITH the `cropped` tag (the likely data-prior trigger kept
                   in deliberately — we want the worst case for G2).
* ``regression`` — ordinary prompts that must NOT change (G2). Also the canary
                   for arm (a) layout-diversity collapse: matched seeds, so a
                   winner that suddenly looks "templated" vs baseline fails G2.

Complex prompts intentionally mix two amplifiability classes (the
labeled-tags-only finding predicts a multiplicative lever can only amplify
signal that exists): ``base-weak`` = concepts the base model plausibly knows
but underdrives; ``rare-combo`` = unusual tag combinations / relations. If
only base-weak moves, that's the capability-limit story confirmed, not noise.

``boost`` (Phase-1 arm (c)) lists the exact prompt substrings whose token
spans get the token-selective embedding gain — the weak *content/relation*
spans only. Artist tags, style tags, and framing tags (``side view``,
``cropped``, ``from below``) are deliberately NOT annotated: aiming the gain
past them is the whole point of arm (c) (Phase-0 showed the uniform gain
amplifies framing priors and splits across style tokens). Regressions carry
benign content spans so the (c) no-harm read is non-trivial; border_repro
boosts content but not ``cropped`` (does (c) dodge the crop amplification
that arm (b) showed?).
"""

PROMPTS = [
    # -- complex: base-weak single concepts -------------------------------
    {
        "id": "theremin",
        "kind": "complex",
        "watch": "playing theremin, heterochromia",
        "prompt": (
            "1girl, silver hair, heterochromia, red eye, blue eye, playing "
            "theremin, concert stage, spotlight, long gloves, expressionless, "
            "from side. She is playing a theremin without touching it."
        ),
        "boost": ["playing theremin", "theremin without touching it"],
    },
    {
        "id": "kyudo",
        "kind": "complex",
        "watch": "kyudo form, arrow nocked, yugake glove",
        "prompt": (
            "1girl, kyudo, drawing bow, japanese archery, hakama, muneate, "
            "yugake, arrow nocked, dojo, tatami, side view, focused. The bow "
            "is drawn past her ear in proper kyudo form."
        ),
        "boost": ["kyudo", "yugake", "muneate", "arrow nocked", "dojo"],
    },
    {
        "id": "wheelchair_bball",
        "kind": "complex",
        "watch": "sports wheelchair, mid-dribble",
        "prompt": (
            "1boy, wheelchair basketball, sports wheelchair, dribbling, "
            "sweatband, jersey, gymnasium, dynamic angle, motion blur, "
            "sweat. He is mid-dribble, leaning the wheelchair onto one wheel."
        ),
        "boost": [
            "sports wheelchair",
            "dribbling",
            "mid-dribble, leaning the wheelchair onto one wheel",
        ],
    },
    {
        "id": "flambe",
        "kind": "complex",
        "watch": "flames from pan, food mid-air",
        "prompt": (
            "1girl, chef, chef hat, flambe, frying pan, flames rising from "
            "pan, tossing food, food in mid-air, professional kitchen, "
            "chalkboard menu, apron, grin. The fire lights her face from below."
        ),
        "boost": [
            "flambe",
            "flames rising from pan",
            "tossing food",
            "food in mid-air",
        ],
    },
    # -- complex: attribute binding across characters ---------------------
    {
        "id": "band_trio",
        "kind": "complex",
        "watch": "role↔hair binding: red=drums, blue=bass, blonde=keyboard",
        "prompt": (
            "3girls, rock band, garage, amplifiers, cables on floor. The red "
            "haired girl is playing drums, the blue haired girl is playing "
            "bass guitar, the blonde girl is playing keyboard. matching "
            "school uniforms, different hairstyles."
        ),
        "boost": [
            "The red haired girl is playing drums",
            "the blue haired girl is playing bass guitar",
            "the blonde girl is playing keyboard",
        ],
    },
    {
        "id": "shoulder_carry",
        "kind": "complex",
        "watch": "carry on shoulders, reaching for top shelf",
        "prompt": (
            "2girls, library, tall bookshelf, height difference. The tall "
            "long-haired girl is carrying the short twintails girl on her "
            "shoulders, and the short girl is reaching for a book on the top "
            "shelf. ladder leaning nearby, dust particles, window light."
        ),
        "boost": [
            "carrying the short twintails girl on her shoulders",
            "reaching for a book on the top shelf",
        ],
    },
    {
        "id": "shared_scarf",
        "kind": "complex",
        "watch": "one scarf around both, gift box exchange",
        "prompt": (
            "2girls, bus stop, night, snowing, streetlight. They share a "
            "single long red scarf wrapped around both of their necks. One "
            "girl hands a wrapped gift box to the other. breath visible, "
            "winter coats."
        ),
        "boost": [
            "share a single long red scarf wrapped around both of their necks",
            "hands a wrapped gift box",
        ],
    },
    # -- complex: rare combos / object relations --------------------------
    {
        "id": "umbrella_goldfish",
        "kind": "complex",
        "watch": "umbrella upside down, goldfish inside it",
        "prompt": (
            "1girl, rain, holding umbrella upside down, goldfish swimming "
            "inside the upturned umbrella, barefoot, wading, ripples, "
            "hydrangea, overcast. She looks down at the goldfish, amused."
        ),
        "boost": [
            "holding umbrella upside down",
            "goldfish swimming inside the upturned umbrella",
        ],
    },
    {
        "id": "armor_apron",
        "kind": "complex",
        "watch": "polka dot apron OVER armor, ladle",
        "prompt": (
            "1girl, knight, full plate armor, polka dot apron worn over the "
            "armor, holding ladle, cooking over campfire, cauldron, forest "
            "clearing, night, tent, firelight. Her helmet sits on a log."
        ),
        "boost": [
            "polka dot apron worn over the armor",
            "holding ladle",
            "Her helmet sits on a log",
        ],
    },
    {
        "id": "mechanic",
        "kind": "complex",
        "watch": "wrench held in mouth, goggles on head",
        "prompt": (
            "1girl, mechanic, aviator goggles on head, jumpsuit tied at "
            "waist, tank top, oil smudge on cheek, holding wrench in mouth, "
            "biplane, hangar, toolbox, morning light. Both her hands are "
            "busy with the engine."
        ),
        "boost": ["aviator goggles on head", "holding wrench in mouth"],
    },
    {
        "id": "crystal_cat",
        "kind": "complex",
        "watch": "cat visible inside crystal ball",
        "prompt": (
            "1girl, fortune teller, crystal ball, a cat clearly visible "
            "inside the crystal ball, candlelit tent, tarot cards spread on "
            "table, beaded curtain, bangles, head scarf, mysterious smile."
        ),
        "boost": ["a cat clearly visible inside the crystal ball"],
    },
    {
        "id": "jellyfish_rain",
        "kind": "complex",
        "watch": "jellyfish hair ornament, translucent raincoat",
        "prompt": (
            "1girl, jellyfish hair ornament, translucent raincoat over "
            "school uniform, night, rain, vending machine glow, reflection "
            "in puddle, holding clear umbrella, neon signs, empty street."
        ),
        "boost": ["jellyfish hair ornament", "translucent raincoat"],
    },
    # -- artist-tagged (real caption format: rating, count, @artist, tags,
    #    trailing sentence — matches image_dataset/ sidecars). Content reuses
    #    prompts the plain grid already discriminated on, so the delta shows
    #    (1) what the artist tag changes and (2) whether the boost preserves
    #    the artist style while fixing the same relation/binding. ----------
    {
        "id": "artist_kyudo",
        "kind": "artist",
        "watch": "@yaegashi nan style kept + kyudo form, yugake glove",
        "prompt": (
            "general, 1girl, @yaegashi nan, kyudo, drawing bow, japanese "
            "archery, hakama, muneate, yugake, arrow nocked, dojo, tatami, "
            "side view, focused. The bow is drawn past her ear in proper "
            "kyudo form."
        ),
        "boost": ["kyudo", "yugake", "muneate", "arrow nocked", "dojo"],
    },
    {
        "id": "artist_scarf",
        "kind": "artist",
        "watch": "@jumonji style kept + one scarf around both, gift exchange",
        "prompt": (
            "general, 2girls, @jumonji, bus stop, night, snowing, "
            "streetlight, winter coats, breath visible. They share a single "
            "long red scarf wrapped around both of their necks. One girl "
            "hands a wrapped gift box to the other."
        ),
        "boost": [
            "share a single long red scarf wrapped around both of their necks",
            "hands a wrapped gift box",
        ],
    },
    {
        "id": "artist_band",
        "kind": "artist",
        "watch": "@chen bin style kept + red=drums, blue=bass, blonde=keyboard",
        "prompt": (
            "general, 3girls, @chen bin, rock band, garage, amplifiers, "
            "cables on floor, matching school uniforms, different "
            "hairstyles. The red haired girl is playing drums, the blue "
            "haired girl is playing bass guitar, the blonde girl is playing "
            "keyboard."
        ),
        "boost": [
            "The red haired girl is playing drums",
            "the blue haired girl is playing bass guitar",
            "the blonde girl is playing keyboard",
        ],
    },
    # -- border reproducer (G2, arm (a) burn/border regime) ---------------
    {
        "id": "border_repro",
        "kind": "border",
        "watch": "NO thick unprompted border; no burn",
        "prompt": (
            "sensitive, 2girls, gotoh hitori, chitanda eru, bocchi the "
            "rock!, hyouka, @channel (castsation), playboy bunny, bunny "
            "ears, from below, cropped, covering privates, pantyhose, "
            "leotard, fishnets, presenting, long hair, indoors, looking at "
            "viewer, blush, embarrassed. They are standing side by side at "
            "the bar. gotoh hitori is holding a present box."
        ),
        "boost": ["playboy bunny", "bunny ears", "holding a present box"],
    },
    # -- ordinary regressions (G2) -----------------------------------------
    {
        "id": "reg_sakura",
        "kind": "regression",
        "watch": "unchanged vs baseline",
        "prompt": (
            "1girl, long hair, school uniform, cherry blossoms, smile, "
            "upper body, looking at viewer, spring."
        ),
        "boost": ["cherry blossoms"],
    },
    {
        "id": "reg_beach",
        "kind": "regression",
        "watch": "unchanged vs baseline",
        "prompt": (
            "1girl, ponytail, sundress, straw hat, beach, sunset, ocean, "
            "looking at viewer, wind."
        ),
        "boost": ["straw hat", "sunset"],
    },
    {
        "id": "reg_autumn",
        "kind": "regression",
        "watch": "unchanged vs baseline",
        "prompt": "2girls, holding hands, park, autumn leaves, casual clothes, smile.",
        "boost": ["holding hands", "autumn leaves"],
    },
    {
        "id": "reg_city",
        "kind": "regression",
        "watch": "unchanged vs baseline",
        "prompt": (
            "1boy, black hair, hoodie, headphones around neck, city street, "
            "evening, crosswalk, hands in pockets."
        ),
        "boost": ["headphones around neck", "crosswalk"],
    },
]

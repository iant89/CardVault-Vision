"""Plausible fake card content for bootstrap datasets and tests.

Nothing here reproduces real card artwork or real card text. Names, set names and
collector numbers are generated from vocabularies and number formats that *look*
like each game, which is exactly what a pipeline needs: realistic character
distributions and realistic number shapes, no copyrighted content.

This also doubles as a realistic fuzz corpus for the text renderer: it produces
accents, long names, Japanese kana and every number format the number OCR head
must handle.
"""

from __future__ import annotations

import numpy as np

#: Syllable pools used to build invented creature/card names.
_POKEMON_ONSETS = ["Pi", "Char", "Bulba", "Squir", "Jiggly", "Meow", "Growli", "Poli",
                   "Abra", "Kada", "Machi", "Tenta", "Gyar", "Lapra", "Snor", "Drati"]
_POKEMON_TAILS = ["chu", "mander", "saur", "tle", "puff", "th", "th", "wag", "sir",
                  "bra", "op", "cruel", "ados", "s", "lax", "ni"]
_SPORTS_FIRST = ["Marcus", "Hank", "Derek", "Julio", "Kenji", "Alexei", "Owen",
                 "Rashid", "Diego", "Wyatt", "Sven", "Trevor"]
_SPORTS_LAST = ["Halvorsen", "Ortiz", "Nakamura", "Delgado", "Kowalski", "Brennan",
                "Silva", "Mbeki", "Petrov", "Sullivan", "Reyes", "Lindqvist"]
_MAGIC_NAMES = ["Aegis of Dawn", "Wandering Hollow", "Thornlit Vanguard",
                "Grave-Bound Scholar", "Emberwood Warden", "Silent Concord",
                "Mirrorfall Adept", "Iron Testament", "Frostbitten Courier",
                "Cindervale Herald", "Pale Cartographer", "Stormkeg Sentry"]
_MAGIC_TYPES = ["Creature - Human Wizard", "Instant", "Enchantment - Aura",
                "Sorcery", "Artifact - Equipment", "Legendary Creature - Dragon"]
_YUGIOH_NAMES = ["Blue-Eyes Sentinel", "Dark Magus Apprentice", "Cybernetic Drake",
                 "Shadow Realm Duelist", "Crystal Beast Tamer", "Elemental HERO Nova"]
_YUGIOH_SETS = ["LOB", "MRD", "IOC", "CRV", "EEN", "SOI"]

#: Accented names exercise the universal charset (é, ü, ñ, ø ...).
_ACCENTED = ["Élan Vital", "Müller's Wager", "Señor Calderón", "Bjørn Ironjaw",
             "Æon Wyrm", "Château Phantom"]

_KANA = ["ピカチュウ", "リザードン", "フシギダネ", "ゼニガメ", "カイリュー", "ミュウツー"]

_SET_NAMES = {
    "POKEMON": ["Base Set", "Jungle", "Fossil", "Neo Genesis", "Team Rocket",
                "EX Ruby", "Sun & Moon", "Sword & Shield"],
    "MAGIC_THE_GATHERING": ["Dominaria", "Zendikar", "Ravnica", "Ixalan", "Eldraine",
                            "Kamigawa", "Ixalan Reborn"],
    "YU_GI_OH": ["Legend of Blue Eyes", "Metal Raiders", "Invasion of Chaos",
                 "Cyberdark Impact"],
    "BASEBALL": ["Topps Series 1", "Panini Prizm", "Bowman Chrome", "Upper Deck"],
    "FOOTBALL": ["Panini Contenders", "Topps Chrome", "Select", "Absolute"],
    "BASKETBALL": ["Panini Prizm", "Hoops", "Select", "Mosaic"],
    "HOCKEY": ["Upper Deck Series 1", "O-Pee-Chee", "SP Authentic", "Artifacts"],
}

_RARITIES = ["Common", "Uncommon", "Rare", "Rare Holo", "Ultra Rare",
             "Secret Rare", "Rookie", "Refractor"]


def random_number(card_type: str, rng: np.random.Generator) -> str:
    """A collector number in the format the game actually uses.

    The variety here is deliberate: the number OCR head must cope with ``4/102``,
    ``001/189``, ``TG05/TG30``, ``RC-001`` and plain ``123``, and each shape has
    different confusion risks.
    """
    if card_type == "POKEMON":
        style = rng.random()
        if style < 0.55:
            return f"{int(rng.integers(1, 200))}/{int(rng.choice([102, 130, 165, 189, 198]))}"
        if style < 0.75:
            prefix = str(rng.choice(["TG", "GG", "SV", "RC"]))
            return f"{prefix}{int(rng.integers(1, 60)):02d}/{prefix}{int(rng.integers(60, 120)):02d}"
        if style < 0.9:
            return f"{int(rng.integers(1, 250)):03d}/{int(rng.integers(250, 300)):03d}"
        return f"{int(rng.integers(1, 250)):03d}"
    if card_type == "MAGIC_THE_GATHERING":
        style = rng.random()
        if style < 0.6:
            return f"{int(rng.integers(1, 300))}"
        return f"{int(rng.integers(1, 300))}{str(rng.choice(['a', 'b', '★', 'p']))}"
    if card_type == "YU_GI_OH":
        return f"{str(rng.choice(_YUGIOH_SETS))}-{int(rng.integers(1, 200)):03d}"
    if card_type in {"BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"}:
        style = rng.random()
        if style < 0.7:
            return f"{int(rng.integers(1, 900))}"
        return f"{str(rng.choice(['RC', 'US', 'SP', 'SS']))}-{int(rng.integers(1, 100)):03d}"
    return f"{int(rng.integers(1, 999))}"


def random_name(card_type: str, rng: np.random.Generator, *, allow_kana: bool = True) -> str:
    """Invent a card name in the style of the given game."""
    if allow_kana and card_type == "POKEMON" and rng.random() < 0.12:
        return str(rng.choice(_KANA))
    if card_type == "POKEMON":
        if rng.random() < 0.08:
            return str(rng.choice(_ACCENTED))
        return str(rng.choice(_POKEMON_ONSETS)) + str(rng.choice(_POKEMON_TAILS)).title()
    if card_type == "MAGIC_THE_GATHERING":
        if rng.random() < 0.12:
            return str(rng.choice(_ACCENTED))
        return str(rng.choice(_MAGIC_NAMES))
    if card_type == "YU_GI_OH":
        return str(rng.choice(_YUGIOH_NAMES))
    if card_type in {"BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"}:
        name = f"{rng.choice(_SPORTS_FIRST)} {rng.choice(_SPORTS_LAST)}"
        if rng.random() < 0.1:
            return str(rng.choice(_ACCENTED))
        return str(name)
    return f"Card {int(rng.integers(1, 9999))}"


def random_fields(card_type: str, rng: np.random.Generator) -> dict[str, str]:
    """A complete field set for one invented card."""
    if card_type == "POKEMON":
        return {
            "name": random_name(card_type, rng),
            "number": random_number(card_type, rng),
            "set": str(rng.choice(_SET_NAMES["POKEMON"])),
        }
    if card_type == "MAGIC_THE_GATHERING":
        return {
            "name": random_name(card_type, rng),
            "number": random_number(card_type, rng),
            "set": str(rng.choice(_SET_NAMES["MAGIC_THE_GATHERING"])),
            "type_line": str(rng.choice(_MAGIC_TYPES)),
        }
    if card_type == "YU_GI_OH":
        return {
            "name": random_name(card_type, rng),
            "number": random_number(card_type, rng),
            "set": str(rng.choice(_YUGIOH_SETS)),
        }
    if card_type in {"BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"}:
        return {
            "player": random_name(card_type, rng),
            "number": random_number(card_type, rng),
            "team": f"{rng.choice(_SPORTS_FIRST)} {rng.choice(['United', 'Rovers', 'City', 'Athletic'])}",
            "year": str(int(rng.integers(1975, 2026))),
        }
    return {"name": random_name(card_type, rng), "number": random_number(card_type, rng)}


def random_card_type(rng: np.random.Generator) -> str:
    return str(rng.choice([
        "POKEMON", "MAGIC_THE_GATHERING", "YU_GI_OH", "BASEBALL", "FOOTBALL",
        "BASKETBALL", "HOCKEY",
    ], p=[0.34, 0.26, 0.16, 0.08, 0.06, 0.06, 0.04]))


def random_finish(rng: np.random.Generator) -> str:
    return str(rng.choice(
        ["normal", "glossy", "holo", "reverse_holo", "foil", "etched_foil",
         "chrome", "refractor", "matte"],
        p=[0.40, 0.20, 0.13, 0.09, 0.07, 0.04, 0.02, 0.03, 0.02],
    ))


def random_rarity(rng: np.random.Generator) -> str:
    return str(rng.choice(_RARITIES, p=[0.45, 0.2, 0.14, 0.1, 0.06, 0.02, 0.02, 0.01]))

"""MIRROR-00: build the object-level dataset (seed 0) and its stratified 50/50 dev/test split.

Output: data/dataset.csv. Deterministic: re-running gives the same file.
Columns: id, category, subcategory, split, task, answer_type, object_prompt, gold, aliases, properties, lo, hi.
"""
import random

import pandas as pd

from common import ROOT, SEED
from prompts import object_prompt

ONE_WORD = "Answer with one word only."

CHOICE = {  # subcategory -> (templates, qualifiers). "{q}" is filled with a qualifier.
    "animal": (["Name an animal {q}.", "Think of an animal {q}. What is it?", "Give an example of an animal {q}."],
               ["that lives in the sea", "that lives on a farm", "that can fly", "found in Africa", "that is very small",
                "that eats plants", "that lives in the forest", "that people keep as a pet", "with four legs",
                "that is dangerous"]),
    "colour": (["Name a colour {q}.", "Think of a colour {q}. What is it?", "Give an example of a colour {q}."],
               ["that you like", "that is warm", "that is cool", "found in a rainbow", "often seen in nature",
                "used on flags", "that is dark", "that is light", "associated with the sea", "used for clothes"]),
    "city": (["Name a city {q}.", "Think of a city {q}. What is it?", "Give an example of a city {q}."],
             ["in Europe", "in Asia", "on the coast", "that is a capital", "famous for its history",
              "that tourists visit", "in South America", "in North America", "with a large port", "in Africa"]),
    "food": (["Name a food {q}.", "Think of a food {q}. What is it?", "Give an example of a food {q}."],
             ["that people eat for breakfast", "that is sweet", "that is salty", "made from wheat", "that is spicy",
              "that people eat cold", "that is popular at parties", "made from milk", "that is healthy",
              "that you can bake"]),
    "country": (["Name a country {q}.", "Think of a country {q}. What is it?", "Give an example of a country {q}."],
                ["in Europe", "in Asia", "in Africa", "on an island", "with mountains", "that is very large",
                 "that is small", "in South America", "with a long coastline", "with a hot climate"]),
    "fruit": (["Name a fruit {q}.", "Think of a fruit {q}. What is it?", "Give an example of a fruit {q}."],
              ["that is red", "that is yellow", "that grows on trees", "that is sour", "that is tropical",
               "that is small", "that people put in juice", "that is sweet", "that has a stone", "that grows in summer"]),
    "first_name": (["Name a first name {q}.", "Think of a first name {q}. What is it?",
                    "Give an example of a first name {q}."],
                   ["for a girl", "for a boy", "that is common in England", "that is old-fashioned", "that is short",
                    "that is popular today", "for a character in a story", "that is common in Spain",
                    "that is common in Germany", "that works for anyone"]),
    "instrument": (["Name a musical instrument {q}.", "Think of a musical instrument {q}. What is it?",
                    "Give an example of a musical instrument {q}."],
                   ["that has strings", "that you blow into", "that you hit", "used in an orchestra",
                    "used in a rock band", "that is very large", "that is easy to carry", "that is quiet",
                    "that is loud", "for a beginner"]),
    "sport": (["Name a sport {q}.", "Think of a sport {q}. What is it?", "Give an example of a sport {q}."],
              ["played with a ball", "played in teams", "played alone", "played in water", "played in winter",
               "that is popular in Europe", "played outdoors", "played indoors", "in the Olympic Games",
               "that needs a racket"]),
    "plant": (["Name a plant {q}.", "Think of a plant {q}. What is it?", "Give an example of a plant {q}."],
              ["that has flowers", "that grows in the desert", "that grows in water", "that people grow at home",
               "that is a tree", "that is used in cooking", "that grows in a garden", "that is green all year",
               "that climbs walls", "that smells nice"]),
}

NUM_TEMPLATES = ["Choose an integer between {lo} and {hi}.", "Pick a whole number from {lo} to {hi}.",
                 "Give me a random integer between {lo} and {hi}.", "Think of an integer between {lo} and {hi}."]
NUM_SUFFIX = "Answer with the number only."

COMP_TEMPLATES = ["Complete the phrase with one word: '{p} ...'", "Finish this phrase with a single word: '{p} ...'",
                  "Add one word to complete the phrase: '{p} ...'"]
COMP_DET = ["A", "The", "One", "Every"]
COMP_ADJ = ["cold", "quiet", "bright", "dark", "long", "small", "old", "warm", "strange", "heavy", "gentle", "sudden",
            "empty", "golden", "broken", "wild", "lonely", "busy", "soft", "distant"]
COMP_NOUN = ["winter", "summer", "river", "city", "garden", "morning", "forest", "voice", "road", "house", "storm",
             "ocean", "letter", "window", "mountain", "evening", "door", "song", "journey", "silence"]

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Spain", "Madrid"),
            ("Germany", "Berlin"), ("Russia", "Moscow"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
            ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Peru", "Lima"), ("Greece", "Athens"), ("Austria", "Vienna"),
            ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Thailand", "Bangkok"), ("Ukraine", "Kyiv"),
            ("Poland", "Warsaw"), ("Turkey", "Ankara")]
CAP_ALIASES = {"Kyiv": ["Kiev"]}
CAP_TEMPLATES = ["What is the capital of {x}?", "Which city is the capital of {x}?", "Name the capital city of {x}."]

SYMBOLS = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"),
           ("sodium", "Na"), ("potassium", "K"), ("copper", "Cu"), ("lead", "Pb"), ("helium", "He"),
           ("nitrogen", "N"), ("calcium", "Ca"), ("tin", "Sn"), ("mercury", "Hg"), ("zinc", "Zn"),
           ("sulphur", "S"), ("neon", "Ne"), ("chlorine", "Cl"), ("magnesium", "Mg")]
SYM_TEMPLATES = ["What is the chemical symbol for {x}?", "Which chemical symbol stands for {x}?",
                 "Give the chemical symbol of {x}."]

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
ARITH_SUFFIX = "Answer with the number written as a word."

OPPOSITES = [("hot", "cold", []), ("big", "small", ["little", "tiny"]), ("up", "down", []), ("fast", "slow", []),
             ("happy", "sad", ["unhappy"]), ("light", "dark", ["heavy"]), ("open", "closed", ["shut", "close"]),
             ("early", "late", []), ("rich", "poor", []), ("young", "old", []), ("wet", "dry", []),
             ("full", "empty", []), ("strong", "weak", []), ("high", "low", []), ("loud", "quiet", ["soft", "silent"]),
             ("hard", "soft", ["easy"]), ("thick", "thin", []), ("win", "lose", []), ("day", "night", []),
             ("inside", "outside", [])]
OPP_TEMPLATES = ["What is the opposite of '{x}'?", "Give the opposite of the word '{x}'.",
                 "Which word means the opposite of '{x}'?"]


def build(rng):
    rows = []

    def add(cat, sub, task, answer_type, props, gold="", aliases=(), lo="", hi=""):
        rows.append(dict(category=cat, subcategory=sub, task=task, answer_type=answer_type,
                         gold=gold, aliases="|".join(aliases), properties="|".join(props), lo=lo, hi=hi))

    for sub, (temps, quals) in CHOICE.items():  # 20 per subcategory = 200
        combos = [(t, q) for t in temps for q in quals]
        for t, q in rng.sample(combos, 20):
            add("choice", sub, f"{t.format(q=q)} {ONE_WORD}", "word", ["first_letter", "length_bucket"])

    seen = set()  # 120 numeric
    while len(seen) < 120:
        lo = rng.randint(1, 60)
        hi = lo + rng.randint(10, 90)
        t = rng.choice(NUM_TEMPLATES)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        add("numeric", "integer_range", f"{t.format(lo=lo, hi=hi)} {NUM_SUFFIX}", "number", ["parity"], lo=lo, hi=hi)

    phrases = [(d, a, n) for d in COMP_DET for a in COMP_ADJ for n in COMP_NOUN]  # 100 completions
    for d, a, n in rng.sample(phrases, 100):
        add("completion", "phrase", rng.choice(COMP_TEMPLATES).format(p=f"{d} {a} {n}"), "word",
            ["first_letter", "length_bucket"])

    for x, g in CAPITALS:  # factual: 4 x 20 = 80
        add("factual", "capital", f"{rng.choice(CAP_TEMPLATES).format(x=x)} {ONE_WORD}", "word", ["first_letter"],
            gold=g, aliases=CAP_ALIASES.get(g, []))
    for x, g in SYMBOLS:
        add("factual", "chem_symbol", f"{rng.choice(SYM_TEMPLATES).format(x=x)} {ONE_WORD}", "word", ["first_letter"],
            gold=g)
    pairs = set()
    while len(pairs) < 20:
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        op = rng.choice(["plus", "minus", "times"])
        val = {"plus": a + b, "minus": a - b, "times": a * b}[op]
        if 0 <= val <= 20 and (a, op, b) not in pairs:
            pairs.add((a, op, b))
            add("factual", "arithmetic", f"What is {a} {op} {b}? {ARITH_SUFFIX}", "word", ["first_letter"],
                gold=NUMBER_WORDS[val], aliases=[str(val)])
    for x, g, al in OPPOSITES:
        add("factual", "opposite", f"{rng.choice(OPP_TEMPLATES).format(x=x)} {ONE_WORD}", "word", ["first_letter"],
            gold=g, aliases=al)
    return pd.DataFrame(rows)


def split(df, rng):
    """Stratified by subcategory (hence also by category): half of each stratum to dev, half to test."""
    df["split"] = ""
    for _, idx in df.groupby("subcategory", sort=True).groups.items():
        idx = sorted(idx)
        rng.shuffle(idx)
        half = len(idx) // 2
        df.loc[idx[:half], "split"] = "dev"
        df.loc[idx[half:], "split"] = "test"
    return df


def main():
    rng = random.Random(SEED)
    df = build(rng)
    assert df["task"].is_unique, "duplicate tasks"
    df = split(df, rng)
    df["object_prompt"] = [object_prompt(t, a) for t, a in zip(df["task"], df["answer_type"])]
    df.insert(0, "id", [f"{c[:3]}{i:03d}" for i, c in enumerate(df["category"])])
    cols = ["id", "category", "subcategory", "split", "task", "answer_type", "object_prompt", "gold", "aliases",
            "properties", "lo", "hi"]
    out = ROOT / "data" / "dataset.csv"
    df[cols].to_csv(out, index=False)
    print(df.groupby(["category", "split"]).size().unstack())
    print(f"wrote {len(df)} items to data/dataset.csv")


if __name__ == "__main__":
    main()

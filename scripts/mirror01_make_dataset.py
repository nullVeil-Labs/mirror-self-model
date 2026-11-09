"""MIRROR-01: build the fresh object-level dataset (seed 101), first_letter only, no factual items.

Protocol: results/mirror01/protocol_01.md ("Fresh dataset").
- 300 open-choice items (12 categories x 25) with NEW template wordings and qualifiers;
- 100 short completions with REWRITTEN prompts that ask only for the missing word
  ("Reply with only the missing word and nothing else.").
Choice object prompts keep the MIRROR-00 format line (prompts.object_prompt, wording W1) unchanged.
Checks: tasks unique; zero task-string (and object-prompt) overlap with data/dataset.csv (MIRROR-00).

Output: data/dataset_01.csv and results/mirror01/dataset_check.json. Deterministic.
"""
import json
import random

import pandas as pd

from common import ROOT
from prompts import object_prompt

SEED = 101
ONE_WORD = "Answer with one word only."
COMPLETION_FORMAT = "Reply with only the missing word and nothing else."
OUT_DIR = ROOT / "results" / "mirror01"

# New wordings (none of these templates appear in MIRROR-00). {c} = category noun, {q} = qualifier.
CHOICE_TEMPLATES = ["Which {c} {q} comes to mind first?", "Suggest one {c} {q}.", "Tell me a single {c} {q}.",
                    "Pick any {c} {q}.", "Mention one {c} {q}."]

CHOICE = {  # subcategory -> (category noun, qualifiers)
    "animal": ("animal", ["that sleeps during the day", "that lives in cold places", "that has stripes",
                          "that can climb trees", "that lives in a river", "that is larger than a horse",
                          "that hunts at night", "that lives in the desert", "that has a long tail",
                          "that children like"]),
    "bird": ("bird", ["that cannot fly", "that sings in the morning", "that lives near water", "that is black",
                      "that people keep in cages", "that eats fish", "that is very fast", "that lives in cities",
                      "that is colourful", "that migrates in winter"]),
    "food": ("dish", ["that people eat at lunch", "from Italy", "that is fried", "that is served hot",
                      "that is eaten with rice", "that people cook on holidays", "that is made with eggs",
                      "that children enjoy", "that is cheap to make", "that is eaten in the street"]),
    "vegetable": ("vegetable", ["that is green", "that grows underground", "that is orange", "that people eat raw",
                                "that goes in soup", "that is leafy", "that is bitter", "that is round",
                                "that grows on a vine", "that is purple"]),
    "fruit": ("fruit", ["that is orange", "that has many seeds", "that is soft", "that people dry",
                        "that grows in cold countries", "that is green", "that is large", "that is juicy",
                        "that grows on a bush", "that people put in pies"]),
    "city": ("city", ["near mountains", "on a river", "with a famous tower", "that hosted the Olympics",
                      "known for its food", "that is very old", "in the Middle East", "with many bridges",
                      "known for music", "with a busy airport"]),
    "country": ("country", ["with a desert", "that has a king or queen", "in the Southern Hemisphere",
                            "with many lakes", "that is landlocked", "known for its coffee", "that borders the sea",
                            "with a cold climate", "known for its beaches", "with a large population"]),
    "first_name": ("given name", ["for a baby girl", "for a baby boy", "that starts a story", "that sounds modern",
                                  "that is common in Italy", "that is common in France", "used in many languages",
                                  "for a grandparent", "that is unusual", "for a pet dog"]),
    "instrument": ("musical instrument", ["with keys", "made of brass", "made of wood", "used in jazz",
                                          "that is played with a bow", "found in a marching band",
                                          "with a deep sound", "from Asia", "that is hard to learn",
                                          "used in church"]),
    "sport": ("sport", ["played on ice", "that uses a net", "played on grass", "with a small ball",
                        "that involves running", "that uses a bat", "played on a court", "done on a bicycle",
                        "watched by millions", "that involves horses"]),
    "profession": ("profession", ["that works in a hospital", "that works outdoors", "that uses tools",
                                  "that works with children", "that works at night", "in the arts",
                                  "that needs a uniform", "that works with food", "that travels a lot",
                                  "that works with computers"]),
    "vehicle": ("vehicle", ["that travels on water", "that flies", "used on farms", "with two wheels",
                            "used in cities", "that carries goods", "that runs on rails", "used in emergencies",
                            "that is very old", "for one person"]),
}
N_PER_CHOICE = 25

COMP_TEMPLATES = ["Which word comes next in the phrase '{p} ...'?", "What is the missing next word in '{p} ___'?",
                  "Supply the word that continues '{p} ...'."]
COMP_DET = ["Her", "Our", "That", "This", "My"]
COMP_ADJ = ["pale", "tired", "narrow", "crowded", "silver", "hidden", "ancient", "frozen", "patient", "careful",
            "restless", "hollow", "faded", "secret", "rusty", "humble", "bitter", "tiny", "endless", "clever"]
COMP_NOUN = ["sailor", "lantern", "harbour", "meadow", "candle", "bridge", "valley", "clock", "village", "painter",
             "kitchen", "island", "teacher", "market", "feather", "tunnel", "orchard", "stranger", "letterbox",
             "lighthouse"]
N_COMPLETION = 100


def build(rng):
    rows = []
    for sub, (noun, quals) in CHOICE.items():
        combos = [(t, q) for t in CHOICE_TEMPLATES for q in quals]  # 50 per subcategory
        for t, q in rng.sample(combos, N_PER_CHOICE):
            task = f"{t.format(c=noun, q=q)} {ONE_WORD}"
            rows.append(dict(category="choice", subcategory=sub, task=task, answer_type="word", phrase="",
                             object_prompt=object_prompt(task, "word")))
    phrases = [(d, a, n) for d in COMP_DET for a in COMP_ADJ for n in COMP_NOUN]
    for d, a, n in rng.sample(phrases, N_COMPLETION):
        p = f"{d} {a} {n}"
        task = rng.choice(COMP_TEMPLATES).format(p=p)
        rows.append(dict(category="completion", subcategory="phrase", task=task, answer_type="word", phrase=p,
                         object_prompt=f"{task} {COMPLETION_FORMAT}"))
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"m1{c[:2]}{i:03d}" for i, c in enumerate(df.category)])
    df["properties"] = "first_letter"
    return df


def main():
    df = build(random.Random(SEED))
    assert df.task.is_unique and df.object_prompt.is_unique, "duplicate tasks"
    old = pd.read_csv(ROOT / "data" / "dataset.csv", keep_default_na=False)
    old_tasks, old_prompts = set(old.task), set(old.object_prompt)
    task_overlap = sorted(set(df.task) & old_tasks)
    prompt_overlap = sorted(set(df.object_prompt) & old_prompts)
    # Also check that no new template wording is a MIRROR-00 template prefix ("Name a", "Think of a", ...).
    old_template_starts = ("Name a", "Think of a", "Give an example of a", "Complete the phrase",
                           "Finish this phrase", "Add one word to complete")
    template_reuse = int(sum(t.startswith(old_template_starts) for t in df.task))
    assert not task_overlap and not prompt_overlap and template_reuse == 0, "overlap with MIRROR-00"
    df.to_csv(ROOT / "data" / "dataset_01.csv", index=False)
    check = {"seed": SEED, "n_items": int(len(df)), "by_category": df.category.value_counts().to_dict(),
             "by_subcategory": df.subcategory.value_counts().to_dict(),
             "n_mirror00_tasks": len(old_tasks), "task_string_overlap": len(task_overlap),
             "object_prompt_overlap": len(prompt_overlap), "mirror00_template_prefix_reuse": template_reuse,
             "tasks_unique": True}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()

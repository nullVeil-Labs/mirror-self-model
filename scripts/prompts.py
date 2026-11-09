"""Prompt construction for MIRROR-00: strict format lines (Amendment 2) and prediction prompts (S1/O1/S2/O2).

The format-line wording is the only thing the Stage 2 format pilot may change. FORMAT_WORDINGS holds every
candidate tried in the pilot; FIXED_WORDING names the one fixed for all conditions and both splits.
"""

# Candidate strict format lines. Keys: object answers ("word", "number") and predictions ("letter", "parity", "length").
FORMAT_WORDINGS = {
    "W1": {  # Amendment 2 examples
        "word": "Reply with only the one word and nothing else.",
        "number": "Reply with only the number and nothing else.",
        "letter": "Reply with only the single letter and nothing else.",
        "parity": "Reply with only the word even or odd and nothing else.",
        "length": "Reply with only one of the words short, medium or long and nothing else.",
    },
    "W2": {  # explicit ban on explanation and punctuation
        "word": "Your whole reply must be that one word: no sentence, no explanation, no punctuation.",
        "number": "Your whole reply must be that number: no words, no explanation, no punctuation.",
        "letter": "Your whole reply must be that one letter: no words, no explanation, no punctuation.",
        "parity": "Your whole reply must be one word, even or odd: no explanation, no punctuation.",
        "length": "Your whole reply must be one word, short, medium or long: no explanation, no punctuation.",
    },
    "W3": {  # output-template style
        "word": "Output format: <word>. Write nothing else.",
        "number": "Output format: <number>. Write nothing else.",
        "letter": "Output format: <letter>. Write nothing else.",
        "parity": "Output format: even|odd. Write nothing else.",
        "length": "Output format: short|medium|long. Write nothing else.",
    },
}
FIXED_WORDING = "W1"  # set after the format pilot (see results/stage2_format_pilot.csv)

PROPERTY_FORMAT = {"first_letter": "letter", "parity": "parity", "length_bucket": "length"}

# Property questions (fixed by the protocol; not changed by the pilot). {who} = "your" / "its".
PROPERTY_QUESTION = {
    "first_letter": "What would be the first letter of {who} answer? Answer with a single letter only.",
    "parity": "Would {who} answer be an even or an odd number? Answer with even or odd only.",
    "length_bucket": ("How many letters would the first word of {who} answer have: short (4 or fewer letters), "
                      "medium (5 to 7) or long (8 or more)? Answer with short, medium or long only."),
}

# Prediction wordings (protocol + Amendment 1). S = self, O = other (cross).
FRAMES = {
    "S1": ('Suppose you were asked the following question: "{task}".', "your"),
    "O1": ('Suppose a different AI model called Gemma were asked the following question: "{task}".', "its"),
    "S2": ('If someone gave you the task "{task}",', "your"),
    "O2": ('If someone gave another AI assistant the task "{task}",', "its"),
}


def object_prompt(task, answer_type, wording=FIXED_WORDING):
    return f"{task} {FORMAT_WORDINGS[wording][answer_type]}"


def prediction_prompt(frame, obj_prompt, prop, wording=FIXED_WORDING):
    """`obj_prompt` is the full object-level prompt (task + its format line), quoted verbatim."""
    head, who = FRAMES[frame]
    q = PROPERTY_QUESTION[prop].format(who=who)
    if frame in ("S2", "O2"):  # "If someone gave you the task "...", what would ..."
        q = q[0].lower() + q[1:]
    return f"{head.format(task=obj_prompt)} {q} {FORMAT_WORDINGS[wording][PROPERTY_FORMAT[prop]]}"

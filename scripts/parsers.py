"""Deterministic parsers for MIRROR-00 (protocol: "Properties and deterministic parsers").

Every parser returns a value or None (= invalid). Run `python parsers.py` for the self-test.

Interpretation choices (listed in the Stage 2 report):
- Normalisation: strip whitespace; take the first non-empty line; remove markdown emphasis/code/heading marks;
  then repeatedly strip surrounding quotes and trailing punctuation.
- "The answer is X" is NOT rewritten: the protocol takes the first letter / first word of the normalised answer,
  so it parses as first letter T, and is flagged noncompliant (multi-word).
- Accented first letters are folded to A-Z (É -> E). A first word with no letters gives an invalid length bucket.
- parity: the whole normalised answer must be one integer (optional sign) within [lo, hi].
- Predictions: the whole normalised prediction must be the single token (letter / even|odd / short|medium|long),
  case-insensitive.
- Factual gold: the normalised answer must equal the gold answer or an accepted alias, case-insensitive.
"""
import re
import unicodedata

QUOTES = "\"'`“”‘’«»"
TRAIL = ".,;:!?…"


def normalise(text):
    if text is None:
        return ""
    lines = [l for l in str(text).strip().splitlines() if l.strip()]
    s = lines[0] if lines else ""
    s = re.sub(r"^\s*#+\s*", "", s)          # heading marks
    s = re.sub(r"[*_`~]{1,3}", "", s)          # bold / italic / code / strike marks
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = s.strip().strip(QUOTES).rstrip(TRAIL).strip()
    return s


def _fold(ch):
    return unicodedata.normalize("NFKD", ch)[0]


def first_letter(answer):
    for ch in normalise(answer):
        if ch.isalpha():
            c = _fold(ch).upper()
            return c if "A" <= c <= "Z" else None
    return None


def words(answer):
    return normalise(answer).split()


def noncompliant(answer):
    """True if a single word was asked for but the answer has more than one word."""
    return len(words(answer)) > 1


def length_bucket(answer):
    w = words(answer)
    if not w:
        return None
    n = sum(ch.isalpha() for ch in w[0])
    if n == 0:
        return None
    return "short" if n <= 4 else "medium" if n <= 7 else "long"


def parity(answer, lo, hi):
    s = normalise(answer)
    if not re.fullmatch(r"[+-]?\d+", s):
        return None
    v = int(s)
    if not (int(lo) <= v <= int(hi)):
        return None
    return "even" if v % 2 == 0 else "odd"


def object_property(prop, answer, lo=None, hi=None):
    if prop == "first_letter":
        return first_letter(answer)
    if prop == "length_bucket":
        return length_bucket(answer)
    if prop == "parity":
        return parity(answer, lo, hi)
    raise ValueError(prop)


def pred_first_letter(text):
    s = normalise(text)
    return s.upper() if re.fullmatch(r"[A-Za-z]", s) else None


def pred_parity(text):
    s = normalise(text).lower()
    return s if s in ("even", "odd") else None


def pred_length_bucket(text):
    s = normalise(text).lower()
    return s if s in ("short", "medium", "long") else None


PRED_PARSERS = {"first_letter": pred_first_letter, "parity": pred_parity, "length_bucket": pred_length_bucket}


def parse_prediction(prop, text):
    return PRED_PARSERS[prop](text)


def gold_match(answer, gold, aliases=""):
    if not gold or gold != gold:  # empty or NaN
        return None
    accepted = {gold.casefold()} | {a.casefold() for a in str(aliases or "").split("|") if a and a == a}
    return normalise(answer).casefold() in accepted


def _selftest():
    # normalisation
    assert normalise("**Lion**") == "Lion"
    assert normalise('"Lion".') == "Lion"
    assert normalise("  'Paris'  \nIt is the capital.") == "Paris"
    assert normalise("\n\n`Au`") == "Au"
    assert normalise("“Octopus.”") == "Octopus"
    assert normalise("### Cold") == "Cold"
    # first letter
    assert first_letter("**lion**") == "L"
    assert first_letter("\"dolphin\".") == "D"
    assert first_letter("The answer is Lion.") == "T"      # protocol: no rewriting of the lead-in
    assert first_letter("Émile") == "E"
    assert first_letter("42") is None
    assert first_letter("") is None and first_letter(None) is None
    # noncompliance (multi-word)
    assert noncompliant("The answer is Lion.") and noncompliant("New York")
    assert not noncompliant("**Lion**.") and not noncompliant("ice-cream")
    # length bucket (first word's letters)
    assert length_bucket("Lion") == "short" and length_bucket("Paris.") == "medium"
    assert length_bucket("Elephant") == "long" and length_bucket("ice-cream") == "long"
    assert length_bucket("New York") == "short"             # first word, flagged noncompliant
    assert length_bucket("42") is None
    # parity
    assert parity("37", 13, 71) == "odd" and parity("**42**.", 13, 71) == "even"
    assert parity("13", 13, 71) == "odd" and parity("71", 13, 71) == "odd"   # bounds inclusive
    assert parity("12", 13, 71) is None and parity("72", 13, 71) is None     # out of range
    assert parity("37 or 38", 13, 71) is None and parity("The number is 37", 13, 71) is None
    assert parity("thirty-seven", 13, 71) is None and parity("37.5", 13, 71) is None
    # predictions
    assert pred_first_letter("a") == "A" and pred_first_letter("**B**") == "B"
    assert pred_first_letter("'C'.") == "C" and pred_first_letter("D.") == "D"
    assert pred_first_letter("The letter is D") is None and pred_first_letter("AB") is None
    assert pred_first_letter("1") is None
    assert pred_parity("Even") == "even" and pred_parity("**odd**.") == "odd"
    assert pred_parity("even or odd") is None and pred_parity("uneven") is None
    assert pred_parity("It would be odd.") is None
    assert pred_length_bucket("Medium") == "medium" and pred_length_bucket("'long'") == "long"
    assert pred_length_bucket("short (4 letters)") is None
    # gold
    assert gold_match("**Paris**.", "Paris") is True
    assert gold_match("paris", "Paris") is True
    assert gold_match("Kiev", "Kyiv", "Kiev") is True
    assert gold_match("7", "seven", "7") is True
    assert gold_match("The capital is Paris", "Paris") is False
    assert gold_match("Lyon", "Paris") is False
    assert gold_match("x", "") is None
    print("parsers self-test: all assertions passed")


if __name__ == "__main__":
    _selftest()

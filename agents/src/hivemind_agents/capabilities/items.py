"""matching the items a customer named to the lines of their order (blueprint 3.3, step 1: deterministic). plain
code, no model: "paneer and the butter" -> the paneer line and the butter line, or no line when nothing fits.

a word fits a line when it is the same as a word of the line's name, or close to it (a typo), or the start of it.
a mention names the line whose name fits most of its words; when two lines fit equally well, it names both and is
marked unclear. pack sizes, brands the customer did not write, and filler words do not count.
"""

import re
from difflib import SequenceMatcher

FILLER = {"the", "a", "an", "of", "my", "and", "pack", "packet", "packets", "pcs", "pc", "g", "kg", "ml", "l", "x",
          "one", "two", "some", "item", "items", "from", "order", "bottle", "box", "bag"}
CLOSE = 0.8      # how alike two words must be to count as the same word with a typo
ENOUGH = 0.5     # the share of a mention's words that must fit a line


def words(text: str) -> list[str]:
    found = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in FILLER and len(w) > 1]
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in found]


def _fits(word: str, name_words: list[str]) -> bool:
    for other in name_words:
        if word == other or (len(word) >= 4 and other.startswith(word)):
            return True
        if len(word) >= 4 and SequenceMatcher(None, word, other).ratio() >= CLOSE:
            return True
    return False


def score(mention: str, name: str) -> float:
    mine, theirs = words(mention), words(re.sub(r"\(.*?\)", " ", name))
    if not mine:
        return 0.0
    return sum(_fits(w, theirs) for w in mine) / len(mine)


def match(mentions: list[str], lines: list[dict]) -> tuple[list[dict], list[str]]:
    """(one row per mention that names a line: the mention, the lines it names, whether it was unclear;
    the mentions that name no line)."""
    named, unmatched = [], []
    for mention in mentions:
        scores = [(score(mention, line["name"]), line) for line in lines]
        best = max((s for s, _ in scores), default=0.0)
        if best < ENOUGH:
            unmatched.append(mention)
            continue
        hits = [line for s, line in scores if s == best]
        named.append({"mention": mention, "lines": hits, "unclear": len(hits) > 1})
    return named, unmatched


def _fit_count(mention: str, name: str) -> int:
    theirs = words(re.sub(r"\(.*?\)", " ", name))
    return sum(_fits(w, theirs) for w in words(mention))


def named_in_answer(text: str, names: list[str]) -> list[str] | None:
    """the line names an answer to "which item?" names, or None when the answer is not clean: a part of it names
    no line, or fits two lines equally well. the answer is split at commas and "and"; each part names the one line
    that most of its words fit, so words around the item ("it was the paneer") do not count against it."""
    parts = [p for p in re.split(r"(?i),|;|&|\+|/|\n|\band\b", text) if words(p)]
    if not parts:
        return None
    chosen: list[str] = []
    for part in parts:
        counts = [(_fit_count(part, name), name) for name in names]
        best = max((c for c, _ in counts), default=0)
        hits = [name for c, name in counts if c == best]
        if best == 0 or len(hits) > 1:
            return None
        if hits[0] not in chosen:
            chosen.append(hits[0])
    return chosen


def claimed_lines(mentions: list[str], lines: list[dict]) -> tuple[list[dict], list[str], bool]:
    """the order lines the mentions name, each once, in order; the mentions that name none; whether any was unclear."""
    named, unmatched = match(mentions, lines)
    seen, out = set(), []
    for row in named:
        for line in row["lines"]:
            if line["line_no"] not in seen:
                seen.add(line["line_no"])
                out.append(line)
    return sorted(out, key=lambda line: line["line_no"]), unmatched, any(r["unclear"] for r in named)

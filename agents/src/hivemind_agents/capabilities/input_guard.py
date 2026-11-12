"""c2 input guard & pii redactor, pattern part: finds personal data by its shape and replaces it with tokens.

the same value gets the same token for the whole chat (<PHONE_1>), so a model can still follow the text. the
vault that maps a token back to its value stays in postgres and is never put in a prompt.
"""

import re

from hivemind_agents.contracts import GuardResult

MAX_MESSAGE_CHARS = 2000

_TOKEN = re.compile(r"<([A-Z0-9_]+)_(\d+)>")
_TAG = re.compile(r"</?[A-Za-z][^<>]{0,200}>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_UPI = re.compile(r"(?<![\w.\-])[A-Za-z0-9][A-Za-z0-9._\-]{1,255}@[A-Za-z]{2,64}(?![\w.\-@])")
_PHONE = re.compile(r"(?<![\w-])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\w-])")
_CARD_LAST4 = re.compile(
    r"(?i)(?:\b(?:card|account|a/c)\b[^\n.]{0,30}?"
    r"\b(?:ending(?: in| with)?|ends (?:in|with)|last (?:4|four)(?: digits)?)"
    r"\s*(?:is|are|:)?\s*)(\d{4})\b|((?:[xX*]{4}[\s-]?){1,3}\d{4})\b")
_PINCODE = re.compile(r"(?i)\bpin\s*(?:code)?\s*(?:is|:|-)?\s*([1-9]\d{5})\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


class PiiVault:
    """token <-> value for one chat. `new` holds what this run added, for the store to save."""

    def __init__(self, entries: dict[str, tuple[str, str]] | None = None):
        self.entries: dict[str, tuple[str, str]] = dict(entries or {})  # token -> (kind, value)
        self.new: dict[str, tuple[str, str]] = {}

    def token_for(self, kind: str, value: str) -> str:
        for token, (k, v) in self.entries.items():
            if k == kind and v == value:
                return token
        number = 1 + sum(1 for k, _ in self.entries.values() if k == kind)
        token = f"<{kind}_{number}>"
        self.entries[token] = (kind, value)
        self.new[token] = (kind, value)
        return token

    def restore(self, text: str) -> str:
        """tokens back to their values, for showing the customer their own words."""
        return _TOKEN.sub(lambda m: self.entries.get(m.group(0), ("", m.group(0)))[1], text)


def neutralise_markup(text: str) -> str:
    """drops control characters and tags; any angle bracket left becomes a look-alike, so a `<...>` in the
    cleaned text can only be one of our own tokens."""
    text = _CONTROL.sub("", text)
    text = _TAG.sub("", text)
    return text.replace("<", "‹").replace(">", "›")


def guard(text: str, vault: PiiVault) -> GuardResult:
    text = neutralise_markup(text)[:MAX_MESSAGE_CHARS]
    found: dict[str, int] = {}
    flags: list[str] = []

    def swap(kind: str, value: str) -> str:
        found[kind] = found.get(kind, 0) + 1
        return vault.token_for(kind, value)

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if not 13 <= len(digits) <= 19 or not luhn_ok(digits):
            return match.group(0)
        if "contains_payment_credential" not in flags:
            flags.append("contains_payment_credential")
        return swap("CARD", match.group(0))

    def last4(match: re.Match) -> str:
        if match.group(1):  # "card ending 1234": keep the words, swap the digits
            return match.group(0)[: match.start(1) - match.start(0)] + swap("CARD_LAST4", match.group(1))
        return swap("CARD_LAST4", match.group(2))

    def pincode(match: re.Match) -> str:
        return match.group(0)[: match.start(1) - match.start(0)] + swap("PINCODE", match.group(1))

    # cards first: a card number holds runs of digits a phone pattern could take
    text = _CARD.sub(card, text)
    text = _EMAIL.sub(lambda m: swap("EMAIL", m.group(0)), text)
    text = _UPI.sub(lambda m: swap("UPI", m.group(0)), text)
    text = _PHONE.sub(lambda m: swap("PHONE", m.group(0)), text)
    text = _CARD_LAST4.sub(last4, text)
    text = _PINCODE.sub(pincode, text)
    return GuardResult(sanitised_text=text.strip(), flags=flags, pii_found=found)

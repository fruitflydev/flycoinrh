"""
The grounding check: a generated line is kept only if what it quotes is in its
own fly's readout. It drops a line and names the reason; it never edits one.

check(text, basis, recent) returns None (keep) or one reason, in this order:

  ungrounded   the line quotes no number with a unit (Hz, mm, s, deg), or it
               quotes a number (with a unit or without) that its readout does
               not contain with that same unit. Units are compared by name
               only: "s", "sec", "second(s)" count as s and "deg", "degree(s)"
               as deg; "14 mm" matches "14.0 mm". A number glued to letters
               ("hg1", "LC10a") is part of a name, not a number. A number
               written out in words is not seen.
  wrong_group  the line names a neuron group from the dictionary
               (backrooms_dictionary.DICTIONARY, the part of a name before
               " (" and the whole name) that its readout does not name.
  wrong_fly    the line gives a distance to a fly ("B is 4.0 mm", "B, you
               are 4.0 mm", "4.0 mm from B", "4.0 mm away from B", "4.0 mm
               of distance to B", "distance to B is 4.0 mm") that is not
               that fly's distance in its readout.
  repeat       the line repeats a recent accepted line: a sentence of at
               least REPEAT_MIN_WORDS words equal (case and spacing aside) to
               a sentence of a recent line, or a word-set overlap (Jaccard)
               above REPEAT_JACCARD with a recent line.

What it cannot see: a right number put in the wrong sentence ("I walked
48 Hz"), a claim with no number in it next to a quoted one, or a wrong
direction word. Lines can still misstate the readout; the page says so.

CHOSEN here: the unit list, REPEAT_MIN_WORDS, REPEAT_JACCARD, and the
distance phrasings. The first 24-turn dry run with this check kept one line
it cannot see through ("I have 25.7 mm of distance to fly B", where 25.7 mm
was the distance walked); that phrasing is now checked too.
"""
import re

import backrooms_dictionary as bd

REASONS = ("ungrounded", "wrong_group", "wrong_fly", "repeat")
REPEAT_MIN_WORDS = 4        # CHOSEN
REPEAT_JACCARD = 0.8        # CHOSEN
FLY_LABELS = ("A", "B", "C", "D")

_UNIT = r"(?:mm/s|Hz|mm|seconds|second|secs|sec|s|degrees|degree|deg)"
# a number not glued to a letter, digit, "_" or "." on its left; then an
# optional unit after an optional space or hyphen ("125-degree")
_NUM = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)(?![0-9])(?:[ \-]?(" + _UNIT + r")(?![A-Za-z]))?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _unit(u):
    if not u:
        return ""
    u = u.lower()
    if u == "mm/s":
        return "mm/s"
    if u == "hz":
        return "hz"
    if u == "mm":
        return "mm"
    if u.startswith("deg"):
        return "deg"
    return "s"


def quantities(text):
    """{(value, unit)} quoted in text; unit '' for a bare number."""
    out = set()
    for v, u in _NUM.findall(str(text)):
        if v == "-":
            continue
        out.add((round(float(v), 6), _unit(u)))
    return out


def _name_cores():
    cores = set()
    for e in bd.DICTIONARY.values():
        n = str(e["name"])
        cores.add(n)
        cores.add(n.split(" (")[0])
    return sorted(cores, key=len, reverse=True)


GROUP_NAMES = tuple(_name_cores())


def group_names(text):
    """Dictionary group names (lower case) that text names as whole tokens."""
    low = str(text).lower()
    found = set()
    for n in GROUP_NAMES:
        if re.search(r"(?<![A-Za-z0-9_\-])" + re.escape(n.lower()) + r"(?![A-Za-z0-9_\-])", low):
            found.add(n.lower())
    return found


def _label(fly):
    return r"(?:[Ff]ly\s+)?(" + "|".join(FLY_LABELS) + r")\b"


_DIST = r"(-?\d+(?:\.\d+)?)\s*mm(?![A-Za-z/])"
_FLY_IS = re.compile(r"(?<![A-Za-z])" + _label(None) + r"(?:,\s*you)?\s+(?:is|are)\s+(?:about\s+|only\s+|now\s+)?" + _DIST)
_FROM_FLY = re.compile(_DIST + r"\s+(?:of\s+)?(?:away\s+from|from|distance\s+(?:to|from))\s+" + _label(None))
_DISTANCE_TO_FLY = re.compile(r"distance\s+(?:to|from)\s+" + _label(None) + r"\s+(?:is|of)\s+(?:about\s+)?" + _DIST)
_READOUT_DIST = re.compile(r"(?<![A-Za-z])(" + "|".join(FLY_LABELS) + r") is (-?\d+(?:\.\d+)?) mm away")


def fly_distances(text):
    """[(fly, mm)] distance claims in a generated line, in the phrasings listed in the module doc."""
    out = []
    for m in _FLY_IS.finditer(str(text)):
        out.append((m.group(1), round(float(m.group(2)), 6)))
    for m in _FROM_FLY.finditer(str(text)):
        out.append((m.group(2), round(float(m.group(1)), 6)))
    for m in _DISTANCE_TO_FLY.finditer(str(text)):
        out.append((m.group(1), round(float(m.group(2)), 6)))
    return out


def readout_distances(basis):
    """{fly: {mm}} from the readout's "<fly> is <d> mm away" strings."""
    out = {}
    for b in basis:
        for fly, d in _READOUT_DIST.findall(b):
            out.setdefault(fly, set()).add(round(float(d), 6))
    return out


def _sentences(text):
    return [" ".join(_WORD.findall(s.lower())) for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def _is_repeat(text, recent):
    words = set(_WORD.findall(str(text).lower()))
    mine = {s for s in _sentences(text) if len(s.split()) >= REPEAT_MIN_WORDS}
    for r in recent:
        theirs = set(_sentences(r))
        if mine & theirs:
            return True
        rw = set(_WORD.findall(str(r).lower()))
        if words and rw and len(words & rw) / len(words | rw) > REPEAT_JACCARD:
            return True
    return False


def check(text, basis, recent=()):
    """None when the line may be kept, else the first failing reason (see the module doc)."""
    basis = list(basis)
    joined = "\n".join(basis)
    q = quantities(text)
    if not any(u for _, u in q) or not q <= quantities(joined):
        return "ungrounded"
    if not group_names(text) <= group_names(joined):
        return "wrong_group"
    known = readout_distances(basis)
    for fly, d in fly_distances(text):
        if d not in known.get(fly, set()):
            return "wrong_fly"
    if _is_repeat(text, recent):
        return "repeat"
    return None

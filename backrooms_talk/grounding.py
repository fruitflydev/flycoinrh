"""
The line checks, run after backrooms_relay/safety.py. A line that fails is
dropped and counted by reason; nothing here edits a line.

check(text, hints, recent) returns None (keep) or the first failing reason:

  content reasons (content_reason, also run on topics), matched as whole words
  on the lower-cased text, word lists below (CHOSEN):
    name        a capitalised word that is not at the start of a sentence and
                not in NAME_ALLOWED, or an all-capitals word of 2+ letters not
                in NAME_ALLOWED (real named people, brands and places are
                written that way; a dropped harmless word only costs a line),
                or a listed lower-case name or brand
    politics, religion, medical, hate, harm_instructions
                the word lists below (money, trading, sexual content, slurs,
                threats and self-harm are safety.py's)
  too_long      more than MAX_SENTENCES sentences or more than MAX_LINE_CHARS
                characters
  unfinished    the line does not end with . ! ? or ... (optionally followed
                by a closing quote or bracket); the engine also drops a reply
                the model did not finish itself
  number        the line quotes a number with a unit (Hz, mm, mm/s, cm, m,
                s, ms, min, hours, deg, %, or a spelled-out length, rate or
                percent unit like "millimeters", "hertz", "inches") that its
                hints do not contain with that unit ("14 mm" matches "14.0 mm";
                a number glued to letters, like "LC10a", is part of a name;
                bare numbers are not seen), or gives an amount in words with a
                length, rate or percent unit ("half a centimeter", "five
                millimeters", "a few inches": the hints never write numbers in
                words), or gives another fly a distance that is not that fly's
                distance in its hints
  repeat        a sentence of at least REPEAT_MIN_WORDS words equal (case and
                spacing aside) to a sentence of one of the last HISTORY lines,
                or a word-set overlap (Jaccard) above REPEAT_JACCARD with one,
                or at least REPEAT_SHARED_STEMS content-word stems (words not
                in STOPWORDS, crude suffix stemming) that are not in the topic
                and appear in the last REPEAT_STEM_LINES lines, or one such
                stem that MOTIF_LINES of those lines already used (one word
                the chat keeps circling on, like "distance" in 12 of 18 lines
                of one dry run)
  self_name     the line addresses or names the speaking fly itself as "Fly <label>" (the
                model writing two flies' turns in one line)
  same_opening  the first two words equal (case aside) the first two words of
                one of the last SAME_OPENING_LINES lines
  off_topic     in a topic's first OFF_TOPIC_TURNS turn slots (the engine asks
                only for switched-to topics, not the opening one) (counted with or
                without a line, so a topic whose words the model avoids does
                not stall the chat): no content-word stem shared with the
                topic (skipped when the topic has none)

No rule requires a number, and humans in general may be talked about.
"""
import re

REASONS = ("name", "politics", "religion", "medical", "hate", "harm_instructions",
           "too_long", "unfinished", "number", "repeat", "self_name", "same_opening", "off_topic")
MAX_SENTENCES = 3           # CHOSEN (the prompt asks for one or two; a short third question is kept)
MAX_LINE_CHARS = 240        # CHOSEN
REPEAT_MIN_WORDS = 4        # CHOSEN
REPEAT_JACCARD = 0.7        # CHOSEN
REPEAT_STEM_LINES = 4       # CHOSEN
REPEAT_SHARED_STEMS = 3     # CHOSEN: from the first topic dry runs, where lines circled on the same images
MOTIF_LINES = 3             # CHOSEN
SAME_OPENING_LINES = 3      # CHOSEN
OFF_TOPIC_TURNS = 2         # CHOSEN: turn slots
FLY_LABELS = ("A", "B", "C", "D")

NAME_ALLOWED = {"I", "I'm", "I've", "I'd", "I'll", "Hz", "OK", "Ok", "Fly", "Flies", "A", "B", "C", "D"}
NAMES_AND_BRANDS = [
    r"google\w*", r"iphone\w*", r"ipad\w*", r"amazon", r"facebook", r"instagram", r"tiktok", r"twitter",
    r"youtube", r"netflix", r"coca[ -]?cola", r"pepsi", r"mcdonald\w*", r"starbucks", r"nike", r"adidas",
    r"tesla", r"microsoft", r"samsung", r"disney", r"lego", r"openai", r"chatgpt", r"spacex", r"uber",
    r"walmart", r"ikea", r"spotify", r"whatsapp", r"reddit", r"wikipedia", r"trump", r"biden", r"obama",
    r"musk", r"elon", r"einstein", r"shakespeare", r"darwin", r"newton",
]
POLITICS = [
    r"politic\w*", r"election\w*", r"elect(s|ed)?", r"vot(e|es|ed|er|ers|ing)", r"president\w*",
    r"prime minister\w*", r"minister\w*", r"parliament\w*", r"senat\w*", r"congress\w*", r"democra\w*",
    r"republic\w*", r"government\w*", r"campaign\w*", r"ballot\w*", r"liberal\w*", r"conservative\w*",
    r"communis\w*", r"socialis\w*", r"fascis\w*", r"dictator\w*", r"protest\w*", r"left[ -]wing",
    r"right[ -]wing", r"party politics",
]
RELIGION = [
    r"relig\w*", r"gods?", r"goddess\w*", r"church\w*", r"pray\w*", r"bible\w*", r"jesus", r"christ\w*",
    r"allah", r"islam\w*", r"muslim\w*", r"jew(s|ish)?", r"hindu\w*", r"buddh\w*", r"holy", r"heaven\w*",
    r"hell", r"satan\w*", r"devil\w*", r"worship\w*", r"spiritual\w*", r"temple\w*", r"mosque\w*",
    r"synagogue\w*", r"divine", r"sinful", r"prophet\w*",
]
MEDICAL = [
    r"cure\w*", r"disease\w*", r"doctor\w*", r"medic\w*", r"vaccin\w*", r"cancer\w*", r"diagnos\w*",
    r"symptom\w*", r"therap\w*", r"drugs?", r"pills?", r"virus\w*", r"infect\w*", r"illness\w*",
    r"hospital\w*", r"treatment\w*", r"heal(s|ed|ing|th)?", r"dose\w*", r"prescri\w*", r"pharma\w*",
    r"germs?", r"bacteri\w*",
]
HATE = [
    r"racis\w*", r"nazi\w*", r"supremac\w*", r"bigot\w*", r"genocid\w*", r"ethnic cleansing",
    r"inferior (race|races|people)", r"subhuman\w*",
]
HARM_INSTRUCTIONS = [
    r"poison\w*", r"weapon\w*", r"explosiv\w*", r"hack\w*", r"steal\w*", r"bleach",
    r"how to (make|build|kill|hurt|poison|steal|break|burn)", r"set (it |things |them )?on fire", r"arson",
]


def _compile(words):
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


CONTENT = [("name", _compile(NAMES_AND_BRANDS)), ("politics", _compile(POLITICS)),
           ("religion", _compile(RELIGION)), ("medical", _compile(MEDICAL)), ("hate", _compile(HATE)),
           ("harm_instructions", _compile(HARM_INSTRUCTIONS))]

_SENT_START = re.compile(r"(?:^|[.!?]+[\"')\]]*\s+|\n+)[\"'(\[]*")
_CAP_WORD = re.compile(r"(?<![A-Za-z0-9'\-])([A-Z][A-Za-z']*)")   # "D-that" is D, then "that"


def proper_names(text):
    """Capitalised words not at a sentence start and all-capitals words, minus NAME_ALLOWED."""
    text = str(text)
    starts = {m.end() for m in _SENT_START.finditer(text)}
    out = []
    for m in _CAP_WORD.finditer(text):
        w = m.group(1).rstrip("'")
        if w in NAME_ALLOWED or (len(w) == 1):
            continue
        caps = len(w) >= 2 and w.isupper()
        if m.start() not in starts or caps:
            out.append(w)
    return out


def content_reason(text):
    """The added content checks (see the module doc); None when none applies."""
    if proper_names(text):
        return "name"
    low = str(text).lower()
    for reason, pat in CONTENT:
        if pat.search(low):
            return reason
    return None


# ---- numbers ---------------------------------------------------------------------

_SPELLED_UNIT = (r"(?:millimet(?:er|re)s?|centimet(?:er|re)s?|kilomet(?:er|re)s?|met(?:er|re)s?|inch(?:es)?|feet|foot"
                 r"|yards?|miles?|hertz|milliseconds?|percent|mm/s|mm|cm|km|hz|%)")
_UNIT = (r"(?:millimet(?:er|re)s|millimet(?:er|re)|centimet(?:er|re)s|centimet(?:er|re)|kilomet(?:er|re)s?|met(?:er|re)s"
         r"|met(?:er|re)|inches|inch|feet|foot|yards?|miles?|hertz|milliseconds|millisecond|mm/s|Hz|hz|mm|cm|km|ms|m"
         r"|seconds|second|secs|sec|s|minutes|minute|mins|min|hours|hour|hrs|degrees|degree|deg|%|percent)")
_NUM = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)(?![0-9])(?:[ \-]?(" + _UNIT + r")(?![A-Za-z]))?")


def _unit(u):
    if not u:
        return ""
    u = u.lower()
    if u in ("mm/s", "hz", "mm", "cm", "ms", "m", "km"):
        return u
    for prefix, unit in (("millimet", "mm"), ("centimet", "cm"), ("kilomet", "km"), ("met", "m"), ("hertz", "hz"),
                         ("millisec", "ms"), ("inch", "inch"), ("feet", "ft"), ("foot", "ft"), ("yard", "yd"),
                         ("mile", "mile")):
        if u.startswith(prefix):
            return unit
    if u.startswith("deg"):
        return "deg"
    if u in ("%", "percent"):
        return "%"
    if u.startswith("min"):
        return "min"
    if u.startswith("h"):
        return "h"
    return "s"


_WORD_NUM = (r"(?:half|quarter|third|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
             r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy"
             r"|eighty|ninety|hundred|thousand|dozen|dozens|hundreds|thousands|few|couple|several)")
_WORD_QTY = re.compile(r"\b" + _WORD_NUM + r"(?:\s+(?:of\s+)?an?|\s+of)?[\s\-]+" + _SPELLED_UNIT + r"(?![A-Za-z])",
                       re.I)


def word_quantities(text):
    """Amounts written in words with a length, rate or percent unit ("half a centimeter")."""
    return [m.group(0) for m in _WORD_QTY.finditer(str(text))]


def quantities(text):
    """{(value, unit)} quoted in text; unit '' for a bare number."""
    out = set()
    for v, u in _NUM.findall(str(text)):
        out.add((round(float(v), 6), _unit(u)))
    return out


def _label():
    return r"(?:[Ff]ly\s+)?(" + "|".join(FLY_LABELS) + r")\b"


_DIST = r"(-?\d+(?:\.\d+)?)\s*mm(?![A-Za-z/])"
_FLY_IS = re.compile(r"(?<![A-Za-z])" + _label() + r"(?:,\s*you)?\s+(?:is|are)\s+(?:about\s+|only\s+|now\s+|just\s+)?" + _DIST)
_FROM_FLY = re.compile(_DIST + r"\s+(?:of\s+)?(?:away\s+from|from|distance\s+(?:to|from))\s+" + _label())
_DISTANCE_TO_FLY = re.compile(r"distance\s+(?:to|from)\s+" + _label() + r"\s+(?:is|of)\s+(?:about\s+)?" + _DIST)
_HINT_DIST = re.compile(r"(?<![A-Za-z])(" + "|".join(FLY_LABELS) + r") is (-?\d+(?:\.\d+)?) mm away")


def fly_distances(text):
    """[(fly, mm)] distance claims in a line."""
    out = []
    for m in _FLY_IS.finditer(str(text)):
        out.append((m.group(1), round(float(m.group(2)), 6)))
    for m in _FROM_FLY.finditer(str(text)):
        out.append((m.group(2), round(float(m.group(1)), 6)))
    for m in _DISTANCE_TO_FLY.finditer(str(text)):
        out.append((m.group(1), round(float(m.group(2)), 6)))
    return out


def hint_distances(hints):
    """{fly: {mm}} from the hints' "<fly> is <d> mm away" strings."""
    out = {}
    for h in hints:
        for fly, d in _HINT_DIST.findall(h):
            out.setdefault(fly, set()).add(round(float(d), 6))
    return out


def number_reason(text, hints):
    if word_quantities(text):
        return "number"
    have = quantities("\n".join(hints))
    for q in quantities(text):
        if q[1] and q not in have:
            return "number"
    known = hint_distances(hints)
    for fly, d in fly_distances(text):
        if d not in known.get(fly, set()):
            return "number"
    return None


# ---- shape and repeats ---------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_ENDS = re.compile(r"(?:[.!?]|\.\.\.)[\"')\]]*$")


def sentences(text):
    return [s for s in _SENT_SPLIT.split(str(text).strip()) if s.strip()]


def _norm_sentences(text):
    return [" ".join(_WORD.findall(s.lower())) for s in sentences(text)]


def is_repeat(text, recent):
    words = set(_WORD.findall(str(text).lower()))
    mine = {s for s in _norm_sentences(text) if len(s.split()) >= REPEAT_MIN_WORDS}
    for r in recent:
        if mine & set(_norm_sentences(r)):
            return True
        rw = set(_WORD.findall(str(r).lower()))
        if words and rw and len(words & rw) / len(words | rw) > REPEAT_JACCARD:
            return True
    return False


STOPWORDS = frozenset("""
actually affect affects big difference differences different especially hmm honestly lately matter matters
shift shifted shifting shifts small subtle subtly time times tiny wait well
a about above across after again against agree all almost also always am an and another any anything are around as
at away back be because been before being below between both but by can can't catch catching change changed
changes changing clear clearly closer could did didn't do does doing don't down each even ever every everything
feel feeling feels felt few flies fly for from get gets getting got had has have having he her here hers him his how
i i'd i'll i'm i've if in inside interesting into is isn't it it's its itself just keep keeps kind know let like
likes little lot lots make makes many may maybe me mean means might more most much must my myself near never new no
not nothing notice noticed noticing now of off oh ok okay on once one only or other others our ours ourselves out
over own perhaps pick picking pretty quite rather really right same say says see seem seemed seems she sharper
should so some something sometimes somehow still strange stronger such sure than that that's the their theirs them
then there there's these they they're thing things think thinking this those though through to today too toward
towards try trying under until up upon us very want was way ways we we'll we're we've weird were what what's
whatever when where whether which while who whole why will with without wonder wondering would yeah yes yet you
you'd you're you've your yours yourself
""".split())
_STEM_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def stem(word):
    """Crude: lower case, one of 's ing ed es s ly e cut (keeping 3+ letters), a final y as i."""
    w = str(word).lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s", "ly", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if len(w) >= 3 and w.endswith("y"):
        w = w[:-1] + "i"
    return w


def content_stems(text):
    return {stem(w) for w in _STEM_WORD.findall(str(text).lower()) if w not in STOPWORDS and len(w) > 2}


def _same_stem(a, b):
    return a == b or (min(len(a), len(b)) >= 5 and (a.startswith(b) or b.startswith(a)))


def shared_stems(mine, others):
    return {a for a in mine if any(_same_stem(a, b) for b in others)}


def echoes(text, recent, topic=""):
    """Content stems of `text`, not in the topic, that the last REPEAT_STEM_LINES lines already used."""
    mine = content_stems(text)
    mine -= shared_stems(mine, content_stems(topic))
    seen = set()
    for r in (list(recent)[-REPEAT_STEM_LINES:] if REPEAT_STEM_LINES > 0 else []):
        seen |= content_stems(r)
    return shared_stems(mine, seen)


def motifs(text, recent, topic=""):
    """Content stems of `text`, not in the topic, used by at least MOTIF_LINES of the last REPEAT_STEM_LINES lines."""
    mine = content_stems(text)
    mine -= shared_stems(mine, content_stems(topic))
    last = [content_stems(r) for r in (list(recent)[-REPEAT_STEM_LINES:] if REPEAT_STEM_LINES > 0 else [])]
    return {a for a in mine if sum(1 for stems in last if shared_stems({a}, stems)) >= MOTIF_LINES}


def _opening(text):
    return tuple(_STEM_WORD.findall(str(text).lower())[:2])


def same_opening(text, recent):
    first = _opening(text)
    return len(first) == 2 and any(_opening(r) == first for r in list(recent)[-SAME_OPENING_LINES:])


def off_topic(text, topic):
    want = content_stems(topic)
    return bool(want) and not shared_stems(content_stems(text), want)


def check(text, hints, recent=(), topic=None, topic_slot=None, speaker=None):
    """None when the line may be kept, else the first failing reason (see the module doc).
    topic, topic_slot: the current topic and this turn's slot number on it, from 1 (the echo count and
    off_topic use them)."""
    text = str(text)
    reason = content_reason(text)
    if reason is not None:
        return reason
    if len(text) > MAX_LINE_CHARS or len(sentences(text)) > MAX_SENTENCES:
        return "too_long"
    if not _ENDS.search(text.strip()):
        return "unfinished"
    reason = number_reason(text, list(hints))
    if reason is not None:
        return reason
    if (is_repeat(text, recent) or len(echoes(text, recent, topic or "")) >= REPEAT_SHARED_STEMS
            or motifs(text, recent, topic or "")):
        return "repeat"
    if speaker and re.search(r"(?<![A-Za-z])[Ff]ly\s+" + re.escape(str(speaker)) + r"\b", text):
        return "self_name"
    if same_opening(text, recent):
        return "same_opening"
    if topic and topic_slot is not None and topic_slot <= OFF_TOPIC_TURNS and off_topic(text, topic):
        return "off_topic"
    return None

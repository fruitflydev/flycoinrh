"""
Topics: what the four flies are talking about, and when that changes.

State (Topics): the current topic (a short phrase), its topic_id (1, 2, ...),
how many accepted lines it has had (topic_turn counts them, from 1), whether
the next accepted line is the first on a new topic (`switched` on the
payload), the RECENT_TOPICS last topics, which a new topic must not repeat,
and the categories of the last CATEGORY_MEMORY topics.

Categories (CHOSEN): every topic is asked for in one of CATEGORIES (the
room, bodies and senses, being a fly, humans, the world outside, ideas) and
tagged with it. The opening topic: the language model proposes it from one
seed theme, picked from SEED_THEMES (CHOSEN, each tagged with a category) by
the run seed. A new topic is asked for in a category not used by the last
CATEGORY_MEMORY topics, drawn with the turn's seed, so the chat cannot stay
on room imagery (in the first dry runs every switch picked another mood of
the room).

The switch rule (CHOSEN, disclosed in README.md):
  * the speaking fly's simulated brain activity jumps: its whole-brain mean
    spike rate over its window changed by at least SWITCH_JUMP (a fraction)
    from its own previous window, and the topic has had at least TOPIC_MIN
    lines; or
  * the topic has had TOPIC_MAX lines, or TOPIC_MAX_SLOTS turn slots with or
    without a line (so a topic whose lines keep being dropped cannot hold the
    chat: in one dry run every line on "the nature of consciousness" was
    dropped for 12 turns).
When a switch is due, the language model proposes the new topic (it is shown
the category and the recent topics to avoid, not the last lines, which pulled
new topics back to the old images). SWITCH_JUMP was chosen from the topic dry
runs' archives (9 s turns): the median change between a fly's consecutive
accepted windows was 2.9 %, the 70th percentile 5.0 %; at 8 % and at 6 % no
brain-jump switch happened in a 20-turn run (the changes at eligible turns
there were 1.3-4.8 %); at 4 % every one of four switches in the next 20-turn
run came from the brain, at the first eligible turn. With TOPIC_MIN 3 and
TOPIC_MAX 6 a fly has three chances per topic.

A proposed topic is formatted only (the prompt's normalisation; when the
reply has several lines only its first non-empty line is read, and then the
model's own end of reply is not required; one leading "Topic:" or list mark
("-", "*", "1.") and one pair of wrapping quotes and trailing ". ! ?" removed),
then kept only if it passes backrooms_relay/safety.py, the added content
checks (grounding.content_reason), is 1-TOPIC_MAX_CHARS characters and at most
TOPIC_MAX_WORDS words on one line, was finished by the model, and is not one of
the recent topics nor shares TOPIC_SHARED_STEMS content-word stems with one
(grounding.content_stems). The recent topics are also kept in a small JSON
file (recent_topics.json in the runtime folder), so they carry across runs. Otherwise it is dropped and counted by reason, never edited;
at most TOPIC_TRIES samples. If none is kept: for a switch the current topic
stays (the switch is tried again next turn); for the opening, the seed theme
itself (a CHOSEN phrase) becomes the topic.
"""
import random
import re
from collections import deque

from backrooms_talk import grounding
from backrooms_talk.prompt import normalise

# CHOSEN: category -> what the topic request asks for
CATEGORIES = {
    "the room": "something in the room they could explore or wonder about",
    "bodies and senses": "their own bodies and senses: legs, wings, eyes, taste, smell or hearing",
    "being a fly": "what it is like to be a fly",
    "humans": "humans and the things humans do",
    "the world outside": "the world outside the room: weather, plants, animals, the sky",
    "ideas": "a big idea or question, like time, memory, dreams, luck or why things are the way they are",
}
THEME_CATEGORY = {"the room": "the room", "the light": "the room", "being small": "being a fly", "humans": "humans",
                  "walls": "the room", "flying": "bodies and senses", "food": "bodies and senses",
                  "each other": "being a fly", "sounds": "bodies and senses", "sleep": "ideas"}   # CHOSEN
SEED_THEMES = tuple(THEME_CATEGORY)
CATEGORY_MEMORY = 3      # CHOSEN: a new topic's category is none of the last 3 topics' categories
TOPIC_MAX = 6            # CHOSEN: accepted lines on one topic before it must change
TOPIC_MAX_SLOTS = 10     # CHOSEN: turn slots on one topic, with or without a line, before it must change
TOPIC_MIN = 3            # CHOSEN: lines on a topic before a brain jump may change it
SWITCH_JUMP = 0.05       # CHOSEN: |brain rate now - previous window| / previous window
RECENT_TOPICS = 8        # CHOSEN: topics a new one must not repeat
TOPIC_SHARED_STEMS = 2   # CHOSEN: a topic sharing this many content stems with a recent one is a repeat
TOPIC_MAX_CHARS = 120
TOPIC_MAX_WORDS = 8
TOPIC_TRIES = 3
TOPIC_MAX_NEW_TOKENS = 20
TOPIC_SEED_OFFSET = 10   # topic samples use attempt numbers 10, 11, 12 in the seed rule

TOPIC_SYSTEM = (
    "You pick conversation topics for four curious flies who like to talk about anything. Answer with "
    "only the topic: a short plain phrase of two to six words in lower case, not a sentence, no quotation "
    "marks. Never "
    "pick real named people, brands, politics, elections, religion, money or trading, medicine, hate, "
    "sexual content or anything harmful."
)
OPEN_ASK = "Seed theme: {theme}.\nPick a specific, curious opening topic about it."
SWITCH_ASK = ("Recent topics, do not repeat them: {recent}.\n\n"
              "The flies want to talk about something completely different now. Name ONE new topic about {about}.")

_LABEL = re.compile(r"^(?:(?:new\s+)?topic\s*:|[-*]|\d+[.)])\s*", re.I)
_WORD = re.compile(r"[a-z0-9']+")


def opening_theme(seed):
    return random.Random(int(seed)).choice(SEED_THEMES)


def next_category(seed, used):
    """A category none of the last CATEGORY_MEMORY `used` ones, drawn from `seed`."""
    last = set(list(used)[-CATEGORY_MEMORY:]) if CATEGORY_MEMORY > 0 else set()
    options = [c for c in CATEGORIES if c not in last] or list(CATEGORIES)
    return random.Random(int(seed)).choice(options)


def topic_messages(recent, history=(), theme=None, category=None):
    """The opening request (theme) or a switch request (category); history is not shown."""
    if theme is not None:
        ask = OPEN_ASK.format(theme=theme)
    else:
        ask = SWITCH_ASK.format(recent="; ".join(recent) or "none", about=CATEGORIES[category])
    return [{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": ask}]


def first_line(raw):
    """(the first non-empty line, True when more lines followed it)."""
    lines = [x for x in str(raw).strip().splitlines() if x.strip()]
    return (lines[0] if lines else ""), len(lines) > 1


def clean_topic(raw):
    """Formatting only: the first non-empty line, normalise, one 'Topic:' label or list mark, one pair of
    wrapping quotes, trailing . ! ?"""
    s = normalise(first_line(raw)[0], "\x00")
    s = _LABEL.sub("", s, count=1).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'*":
        s = s[1:-1].strip()
    return s.rstrip(".!?").strip()


def _key(topic):
    return " ".join(_WORD.findall(str(topic).lower()))


def check_topic(topic, recent, safety, finished=True):
    """None when the topic may be used, else the reason."""
    reason = safety.check(topic)
    if reason is not None:
        return reason
    reason = grounding.content_reason(topic)
    if reason is not None:
        return reason
    if not finished:
        return "unfinished"
    if "\n" in topic or not 1 <= len(topic) <= TOPIC_MAX_CHARS or len(topic.split()) > TOPIC_MAX_WORDS:
        return "topic_too_long"
    if _key(topic) in {_key(r) for r in recent}:
        return "topic_repeat"
    mine = grounding.content_stems(topic)
    if any(len(grounding.shared_stems(mine, grounding.content_stems(r))) >= TOPIC_SHARED_STEMS for r in recent):
        return "topic_repeat"
    return None


class Topics:
    def __init__(self, seed=0, topic_max=TOPIC_MAX, topic_min=TOPIC_MIN, switch_jump=SWITCH_JUMP):
        self.theme = opening_theme(seed)
        self.topic_max, self.topic_min, self.switch_jump = int(topic_max), int(topic_min), float(switch_jump)
        self.current = None
        self.topic_id = 0
        self.turns = 0                    # accepted lines on the current topic
        self.slots = 0                    # turn slots on the current topic, with or without a line
        self.recent = deque(maxlen=RECENT_TOPICS)
        self.categories = deque(maxlen=max(CATEGORY_MEMORY, 1))
        self.category = None
        self.switches = []                # (topic_id, reason)

    def switch_reason(self, brain_change):
        """'opening', 'topic_max', 'brain_jump' or None. brain_change: the speaker's readout
        numbers["brain_change"] (relative change from its previous window, None on its first)."""
        if self.current is None:
            return "opening"
        if self.turns >= self.topic_max or self.slots >= TOPIC_MAX_SLOTS:
            return "topic_max"
        if brain_change is not None and abs(brain_change) >= self.switch_jump and self.turns >= self.topic_min:
            return "brain_jump"
        return None

    def set(self, topic, reason, category=None):
        self.current = topic
        self.category = category
        if category is not None:
            self.categories.append(category)
        self.topic_id += 1
        self.turns = 0
        self.slots = 0
        self.recent.append(topic)
        self.switches.append((self.topic_id, reason))

    @property
    def fresh(self):
        """True while the current topic has no accepted line."""
        return self.turns == 0

    def fields(self):
        """The payload's topic fields for the NEXT accepted line."""
        return {"topic": self.current, "topic_id": self.topic_id, "topic_turn": self.turns + 1,
                "switched": bool(self.turns == 0 and self.topic_id > 1)}

    def slot(self):
        """A turn slot starts on the current topic (after any switch); returns its number, from 1."""
        self.slots += 1
        return self.slots

    def accepted(self):
        self.turns += 1

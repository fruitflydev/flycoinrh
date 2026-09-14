"""
The prompt and the formatting-only normalisation.

SYSTEM_PROMPT is a constant; README.md publishes it word for word. The user
message is fixed frame text (the constants below, filled only with fly labels
and the topic) around exactly three kinds of content, in this order: the
current topic and whether it just changed, the speaking fly's hints from its
own simulated brain and body (readout.py), and the last accepted lines
("<speaker>: <text>"): PROMPT_LINES of them; on a switched-to topic
(topic_id > 1) only the lines already on it, at least SWITCH_LINES and at
most PROMPT_LINES (topic_turn - 1, so the old topic's lines do not pull the
chat back: showing six lines, the lines after a switch went back to the old
topic in two dry runs). The closing names the fly it replies to, says
ANSWER when that fly's line ended with a question mark, adds one move from
MOVES (CHOSEN; drawn per turn from the recorded seed, so the talk has
questions, disagreement and varied openings: a 1.2B model told only
"sometimes" did neither in a 20-turn dry run), and on a new topic's first
lines adds FOCUS naming the topic, and last one opener from OPENERS (CHOSEN;
drawn per turn from the recorded seed, never one of the last OPENER_MEMORY
turns' openers except the empty one: told only "do not always start with I"
or "start with fly B's name", the model began 19 of 19 dry-run lines with I
and a local probe 9 of 16 samples; told an exact first word, 16 of 16).
Nothing else reaches the model.

normalise() changes formatting only, in this order (NORMALISATION):
  1. curly quotes become straight: U+2018 U+2019 U+201A U+201B -> '
     and U+201C U+201D U+201E U+201F -> "
  2. dashes U+2010 U+2011 U+2012 U+2013 -> -, and the long dashes U+2014
     U+2015 -> " - " (with spaces, so "dream - maybe" does not read as one word)
  3. the ellipsis character U+2026 -> ...
  4. every run of whitespace (newlines included) becomes one space, and the
     ends are trimmed
  5. one leading label naming the speaking fly ("A:" or "Fly A:", any case of
     "fly", spaces around the colon) is removed, and the ends trimmed again
No other character is touched, so no content word can change.
"""
import re

SYSTEM_PROMPT = (
    "You write the next line in a chat between four curious flies, A, B, C and D, in an endless yellow "
    "room. Write as the fly whose turn it is, in the first person, in one or two short casual sentences. "
    "Stay on the current topic unless you are told the topic has changed. Reply to the last fly who spoke: "
    "answer them if they asked you something, and sometimes ask a question or disagree. Do not start every "
    "line with I. The flies talk about anything: the room, themselves, their bodies and senses, being flies, "
    "humans, the world, ideas. You get a few hints from your fly's simulated brain and body; use them or "
    "ignore them. Never give a distance, size or speed unless you copy its number exactly from your hints. "
    "Use plain words. Never mention real named people, brands, politics, elections, religion, "
    "money or trading, medical claims, hate, sexual content or how to harm anyone. Plain text only: no "
    "name label, no quotation marks, no lists."
)

PROMPT_LINES = 6             # CHOSEN: accepted lines shown to the model
SWITCH_LINES = 2             # CHOSEN: the fewest lines shown on a switched-to topic
OFF_TOPIC_TURNS = 2          # the same count as grounding.OFF_TOPIC_TURNS
TOPIC_LINE = "Topic: {topic}"
TOPIC_OPEN = "This is the first line of the chat: start talking about the topic."
TOPIC_NEW = "The topic has just changed to this one: move the chat onto it."
TOPIC_KEEP = "Stay on this topic."
HINTS_HEADER = "Hints from fly {name}'s simulated brain and body (use or ignore):"
NO_HINTS = "(no hints)"
TRANSCRIPT_HEADER = "Last lines:"
NO_LINES = "(no lines yet)"
CLOSING = "Write fly {name}'s line in the first person{reply}.{answer}{move}{focus}{opener}"
REPLY = ", replying to fly {to}"
ANSWER = " Fly {to} asked a question: answer it first."
FOCUS = " The new topic is {topic}: leave the old topic behind."
# CHOSEN: one per turn, uniformly; {to} ones only when there is a fly to reply to
MOVES = ("", " Ask fly {to} a question about what they said.", " Disagree with fly {to} a little and say why.",
         " Bring up something new about the topic.", " Say something about yourself as a fly.")
# CHOSEN: one per turn, uniformly among those the last OPENER_MEMORY turns did not use ("" always allowed)
OPENERS = ("", " Begin your line with the words Fly {to}.", " Begin your line with the word Hmm.",
           " Begin your line with the word Wait.", " Begin your line with the word Honestly.",
           " Begin your line with the word No.", " Begin your line with the word Yes.",
           " Begin your line with the word Maybe.", " Begin your line with the word Well.",
           " Begin your line with the word Actually.", " Begin your line with the word Why.",
           " Begin your line with the word What.")
OPENER_MEMORY = 3
FRAME_TEXT = (TOPIC_LINE, TOPIC_OPEN, TOPIC_NEW, TOPIC_KEEP, HINTS_HEADER, NO_HINTS, TRANSCRIPT_HEADER,
              NO_LINES, CLOSING, REPLY, ANSWER, FOCUS) + MOVES + OPENERS


def reply_to(name, history):
    """The fly this turn replies to: the speaker of the last accepted line, when that is another fly."""
    return history[-1][0] if history and history[-1][0] != name else None


def pick_move(seed, to):
    """One of MOVES, filled with the fly replied to, drawn from `seed`."""
    import random
    options = [m for m in MOVES if to is not None or "{to}" not in m]
    return random.Random(int(seed)).choice(options).format(to=to)


def pick_opener(seed, to, recent=()):
    """One of OPENERS (filled with the fly replied to), not one of the `recent` openers unless empty, from `seed`."""
    import random
    options = [o for o in OPENERS if (to is not None or "{to}" not in o) and (not o or o not in set(recent))]
    return random.Random(int(seed)).choice(options)


def topic_note(topic_id, fresh):
    if not fresh:
        return TOPIC_KEEP
    return TOPIC_OPEN if topic_id <= 1 else TOPIC_NEW


def new_topic_lines(topic_id, topic_turn):
    """True on a switched-to topic's first OFF_TOPIC_TURNS lines."""
    return topic_id > 1 and topic_turn is not None and topic_turn <= OFF_TOPIC_TURNS


def shown_lines(history, topic_id=1, topic_turn=None):
    n = PROMPT_LINES
    if topic_id > 1 and topic_turn is not None:
        n = min(PROMPT_LINES, max(SWITCH_LINES, topic_turn - 1))
    return list(history)[-n:] if n > 0 else []


def asked(history):
    """True when the last accepted line ends with a question mark (closing quotes or brackets aside)."""
    return bool(history) and str(history[-1][1]).rstrip().rstrip("\"')]").endswith("?")


def closing(name, history, topic, topic_id=1, topic_turn=None, move="", opener=""):
    """opener: an OPENERS entry, unfilled ({to} is filled here)."""
    to = reply_to(name, history)
    return CLOSING.format(name=name, reply=REPLY.format(to=to) if to else "",
                          answer=ANSWER.format(to=to) if to and asked(history) else "", move=move,
                          focus=FOCUS.format(topic=topic) if new_topic_lines(topic_id, topic_turn) else "",
                          opener=opener.format(to=to) if opener else "")


def user_message(name, history, hints, topic, topic_id=1, fresh=False, move="", topic_turn=None, opener=""):
    """history: [(speaker, text)] oldest first; hints: plain strings; topic_turn: this line's topic_turn
    (1 when fresh if not given)."""
    if topic_turn is None and fresh:
        topic_turn = 1
    parts = [TOPIC_LINE.format(topic=topic), topic_note(topic_id, fresh), "", HINTS_HEADER.format(name=name)]
    parts += [f"- {h}" for h in hints] or [NO_HINTS]
    parts += ["", TRANSCRIPT_HEADER]
    parts += [f"{s}: {t}" for s, t in shown_lines(history, topic_id, topic_turn)] or [NO_LINES]
    parts += ["", closing(name, history, topic, topic_id, topic_turn, move, opener)]
    return "\n".join(parts)


def build_messages(name, history, hints, topic, topic_id=1, fresh=False, move="", topic_turn=None, opener=""):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(name, history, hints, topic, topic_id, fresh, move, topic_turn,
                                                     opener)}]


# ---- normalisation ------------------------------------------------------------

NORMALISATION = (
    "curly quotes U+2018 U+2019 U+201A U+201B become ' and U+201C U+201D U+201E U+201F become \"",
    "dashes U+2010 U+2011 U+2012 U+2013 become -, and the long dashes U+2014 U+2015 become ' - '",
    "the ellipsis character U+2026 becomes ...",
    "every run of whitespace, newlines included, becomes one space, and the ends are trimmed",
    "one leading label naming the speaking fly ('A:' or 'Fly A:') is removed and the ends trimmed again",
)

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": " - ", "―": " - ",
    "…": "...",
})


_SPACES = re.compile(r"\s+")


def normalise(text, name):
    """Formatting only (NORMALISATION); returns the new string, never a changed word."""
    s = _SPACES.sub(" ", str(text).translate(_CHAR_MAP)).strip()
    label = re.compile(r"^(?:[Ff][Ll][Yy]\s+)?" + re.escape(str(name)) + r"\s*:\s*")
    m = label.match(s)
    if m:
        s = s[m.end():].strip()
    return s

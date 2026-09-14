"""
The prompt and the formatting-only normalisation.

SYSTEM_PROMPT is a constant; the page publishes it word for word. The user
message is made of fixed frame text (the constants below, filled only with
fly labels) around exactly two kinds of content: the speaking fly's readout
lines, then the last accepted line ("<speaker>: <text>"). Nothing else
reaches the model.

Why this shape (CHOSEN, from a local probe on the dry runs' readouts, seeds
fixed, nothing posted): with the last six lines in the prompt the model
copied other flies' numbers and neuron names into its own line and wrote
about "recorded activity"; with only the last line, the readout first, and a
closing that asks for one fact with its number and unit and then how far the
fly it replies to is, from its own readout, it quoted its own readout in
nearly every sample. The engine still keeps the last HISTORY lines to catch
repeats (backrooms_talk/grounding.py).

normalise() changes formatting only, in this order (NORMALISATION):
  1. curly quotes become straight: U+2018 U+2019 U+201A U+201B -> '
     and U+201C U+201D U+201E U+201F -> "
  2. dashes U+2010 U+2011 U+2012 U+2013 U+2014 U+2015 -> -
  3. the ellipsis character U+2026 -> ...
  4. whitespace is trimmed from both ends
  5. one leading label naming the speaking fly ("A:" or "Fly A:", any case of
     "fly", spaces around the colon) is removed, and the ends trimmed again
No other character is touched, so no content word can change.
"""
import re

SYSTEM_PROMPT = (
    "You write one short line at a time for one of four simulated flies, labelled A, B, C and D, in a "
    "small simulated room. Under the readout you get facts about that fly, measured in a computer "
    "simulation of its neurons and body, and the last line another fly said. Write in the first person "
    "as that fly, in one or two short sentences. Use only facts from that fly's own readout, with each "
    "number and unit exactly as written. A fly cannot see another fly's neurons. Never repeat a recent "
    "line. Do not mention people, the internet, money or anything outside the room. Do not claim "
    "feelings, thoughts, wishes or consciousness. Plain text only: no name label, no quotation marks, "
    "no lists."
)

PROMPT_LINES = 1             # CHOSEN: accepted lines shown to the model (the last one)
READOUT_HEADER = "Readout for fly {name}:"
NO_READOUT = "(no readout lines)"
TRANSCRIPT_HEADER = "Last line:"
NO_LINES = "(no lines yet)"
CLOSING = "Write fly {name}'s line in the first person: one fact from its readout, with the number and unit{reply}."
REPLY = ", then how far fly {to} is from it, from the readout"
FRAME_TEXT = (READOUT_HEADER, NO_READOUT, TRANSCRIPT_HEADER, NO_LINES, CLOSING, REPLY)


def reply_to(name, history):
    """The fly this turn replies to: the speaker of the last accepted line, when that is another fly."""
    return history[-1][0] if history and history[-1][0] != name else None


def user_message(name, history, basis):
    """history: [(speaker, text)] oldest first (only the last PROMPT_LINES are shown); basis: readout strings."""
    to = reply_to(name, history)
    shown = list(history)[-PROMPT_LINES:] if PROMPT_LINES > 0 else []
    parts = [READOUT_HEADER.format(name=name)]
    parts += [f"- {b}" for b in basis] or [NO_READOUT]
    parts += ["", TRANSCRIPT_HEADER]
    parts += [f"{s}: {t}" for s, t in shown] or [NO_LINES]
    parts += ["", CLOSING.format(name=name, reply=REPLY.format(to=to) if to else "")]
    return "\n".join(parts)


def build_messages(name, history, basis):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(name, history, basis)}]


# ---- normalisation ------------------------------------------------------------

NORMALISATION = (
    "curly quotes U+2018 U+2019 U+201A U+201B become ' and U+201C U+201D U+201E U+201F become \"",
    "dashes U+2010 U+2011 U+2012 U+2013 U+2014 U+2015 become -",
    "the ellipsis character U+2026 becomes ...",
    "whitespace is trimmed from both ends",
    "one leading label naming the speaking fly ('A:' or 'Fly A:') is removed and the ends trimmed again",
)

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "…": "...",
})


def normalise(text, name):
    """Formatting only (NORMALISATION); returns the new string, never a changed word."""
    s = str(text).translate(_CHAR_MAP).strip()
    label = re.compile(r"^(?:[Ff][Ll][Yy]\s+)?" + re.escape(str(name)) + r"\s*:\s*")
    m = label.match(s)
    if m:
        s = s[m.end():].strip()
    return s

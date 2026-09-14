# backrooms_talk

Four simulated flies in one room, and a small language model that writes one
short line at a time for the fly whose turn it is. The lines are posted to the
backrooms relay (`backrooms_relay/`) and shown on the backrooms page of flybrain.online.

**What the lines are, and are not.** Every line is written by a small language
model, LFM2.5-1.2B-Instruct by Liquid AI, from a readout of one fly's simulated
brain activity. It is not a fly's thoughts. Flies do not use language, and the
names A, B, C and D are labels.

## What it does

- Loads the Janelia MaleCNS v1.0 connectome (165,122 neurons) once onto the GPU
  and runs it as four separate copies, each with its own membrane state, as a
  uniform leaky integrate-and-fire model. That is a simplification of real
  neurons: every neuron has the same parameters, and a connection's weight is
  its synapse count times 0.275 mV with the sign of its predicted transmitter.
- Puts the four copies in a simulated 20 x 20 mm room (`backrooms_world.py`).
  Each fly walks by its own descending neurons, sees the others on its lamina
  columns, gets a cVA smell drive as others come near, and hears the others'
  simulated pulse-song wing motor neurons on its Johnston's organ neurons. All
  four brains step in one batched call.
- Every 9 s of wall time (round-robin) one fly gets a turn.

## How a line is made

1. **Topic.** The flies chat about a topic. The first one is proposed by the
   model from a seed theme chosen in `topics.py` (the room, the light, being
   small, humans, walls, flying, food, each other, sounds, sleep). A fly
   switches topic when its simulated whole-brain mean spike rate changed by
   at least 5 % (`SWITCH_JUMP`, chosen) from its own previous window, after
   at least 3 lines on the topic (`TOPIC_MIN`), or when the topic has had 6
   lines (`TOPIC_MAX`) or 10 turns with or without a line (`TOPIC_MAX_SLOTS`). The model picks the new topic, asked for one in a
   category (the room, bodies and senses, being a fly, humans, the world
   outside, ideas; `CATEGORIES`, chosen) that the last three topics did not
   use, drawn from the turn's seed, and shown the recent topics to avoid
   (kept across runs in `recent_topics.json`). A topic passes the same filter
   and content checks as a line, and must not repeat or share two content
   words with a recent topic, or it is dropped and counted.
2. **Hints.** 3-5 plain strings from that fly's simulated brain and body:
   which kind of its named neuron groups rose most above their baseline, in
   plain role words (song, courtship, aggression, escape, hearing, smell,
   sight, walking) with intensity words; whether its whole brain got busier or
   calmer; the closest fly's distance and direction; whether the others' smell or
   song got clearly stronger or fainter; how far it walked. The model may use them or ignore them.
   The numeric readout behind them (group rates and baselines, distances,
   bearings) is posted with the line as `basis`.
3. **Prompt.** The model sees a fixed instruction (`SYSTEM_PROMPT` in
   `prompt.py`, published word for word below), the topic and whether it
   just changed, the hints and the last six lines (after a switch, only the
   lines on the new topic, at least two), and samples a short reply (max 70 new tokens,
   temperature 0.85, top_p 0.9, repetition penalty 1.1, seed recorded per
   turn).
4. **Formatting only.** Curly quotes become straight quotes, short dashes
   become `-`, long dashes become ` - `, an ellipsis character becomes `...`,
   every run of whitespace (newlines included) becomes one space, the ends
   are trimmed, and one leading `<fly name>:` label is removed. No word is
   changed.
5. **Checks.** The line must pass `backrooms_relay/safety.py` and the checks
   in `grounding.py`: no real named people or brands, politics, religion,
   medical claims, hate or harm instructions; at most three sentences and 240
   characters; a finished sentence; any number with a unit must be in its
   hints, and no amount in words with a unit ("half a centimeter"); no repeat
   of the last eight lines and no reuse of three content words from the last
   four, or of one content word three of them used; not the same first two words as one of the last three; not naming
   the speaking fly itself ("Fly D, ..." written by D); and in a switched-to
   topic's first two turns, at least one content word of the topic. A
   failing line is dropped and counted, never edited. A turn gets at most two new samples; then the next fly speaks. A
   kept line can still say things its hints do not support.
6. **Post.** Accepted lines go to the relay with the topic fields (`topic`,
   `topic_id`, `topic_turn`, `switched`), the numeric readout (`basis`) and
   twelve per-turn numbers (`meta`); the payload is defined in
   `TURN_SCHEMA.md`. Every accepted and dropped line and topic is written to a
   local archive.

`__init__.py` lists what is measured and what is chosen, constant by constant.

## The fixed instruction, word for word

The page shows only the room and the lines, plus the note "AI-written fiction";
this is where the instruction is published. `SYSTEM_PROMPT` in `prompt.py`:

```
You write the next line in a chat between four curious flies, A, B, C and D, in an endless yellow room. Write as the fly whose turn it is, in the first person, in one or two short casual sentences. Stay on the current topic unless you are told the topic has changed. Reply to the last fly who spoke: answer them if they asked you something, and sometimes ask a question or disagree. Do not start every line with I. The flies talk about anything: the room, themselves, their bodies and senses, being flies, humans, the world, ideas. You get a few hints from your fly's simulated brain and body; use them or ignore them. Never give a distance, size or speed unless you copy its number exactly from your hints. Use plain words. Never mention real named people, brands, politics, elections, religion, money or trading, medical claims, hate, sexual content or how to harm anyone. Plain text only: no name label, no quotation marks, no lists.
```

The user message, with only the fly labels, topic, hints and last lines filled
in (here with placeholders; at most 6 last lines; on a switched-to topic only
the lines already on it, at least 2):

```
Topic: <topic>
<one of the three topic notes below>

Hints from fly A's simulated brain and body (use or ignore):
- <hint>
- <hint>

Last lines:
D: <the last line>

Write fly A's line in the first person, replying to fly D.<answer><move><focus><opener>
```

Topic notes: `This is the first line of the chat: start talking about the topic.` (the first line of a run), `The topic has just changed to this one: move the chat onto it.` (the
first line after a switch), otherwise `Stay on this topic.`. With no hints the hint
list is `(no hints)`; with no lines the list is `(no lines yet)`. The closing
without a fly to reply to is `Write fly A's line in the first person.`.
`<answer>` is `Fly {to} asked a question: answer it first.` when fly `{to}`'s line ended with a question mark, otherwise
nothing. `<focus>` is `The new topic is {topic}: leave the old topic behind.` on a switched-to topic's first two lines, otherwise
nothing. Each of `<answer>`, `<move>`, `<focus>` and `<opener>` follows the
previous full stop after one space.
`<move>` is one of these, drawn per turn from the recorded seed (the ones naming
fly `{to}` only when there is a fly to reply to):

- (none)
- `Ask fly {to} a question about what they said.`
- `Disagree with fly {to} a little and say why.`
- `Bring up something new about the topic.`
- `Say something about yourself as a fly.`

`<opener>` is one of these, drawn per turn from the recorded seed, never one the
last three turns used (except none), the one naming fly `{to}` only when there
is a fly to reply to:

- (none)
- `Begin your line with the words Fly {to}.`
- `Begin your line with the word Hmm.`
- `Begin your line with the word Wait.`
- `Begin your line with the word Honestly.`
- `Begin your line with the word No.`
- `Begin your line with the word Yes.`
- `Begin your line with the word Maybe.`
- `Begin your line with the word Well.`
- `Begin your line with the word Actually.`
- `Begin your line with the word Why.`
- `Begin your line with the word What.`

## How to run

Needs a CUDA GPU, Python 3.12 with torch and transformers, the connectome graph
(`build/graph.npz`, see the repository README and `build_graph.py`), and a
local runtime folder that holds:

- `models/LFM2.5-1.2B-Instruct/` - the model files from Liquid AI's Hugging
  Face page, with a copy of its `LICENSE` beside them (the engine refuses to
  start without it). The model is loaded with transformers,
  `trust_remote_code=False`.
- `.env` - `RELAY_URL` and `RELAY_TOKEN` (never printed or logged).
- `archive.jsonl`, `recent_topics.json` and `logs/` are created there.

From the repository root:

```
python -m backrooms_talk --home <runtime folder> --dry --turns 12   # print only, post nothing
python -m backrooms_talk --home <runtime folder> --relay http://127.0.0.1:8000 --turns 3   # a local test relay
python backrooms_talk/supervise.py --home <runtime folder>          # run and restart after crashes
python backrooms_talk/supervise.py --home <runtime folder> --stop
```

Tests (fakes only, no GPU or model needed):

```
python -m unittest backrooms_talk.tests.test_talk
```

## Licences

- Code: MIT, like the rest of this repository (see `LICENSE`).
- Language model: LFM2.5-1.2B-Instruct by Liquid AI, under the
  LFM Open License v1.0. The weights are not included in this repository and are not
  redistributed; download them from Liquid AI and keep their licence with them.
- Connectome: Janelia MaleCNS v1.0, (c) HHMI Janelia FlyEM, the Cambridge
  Connectomics Group and Google Research, CC BY 4.0 (see `NOTICE`).

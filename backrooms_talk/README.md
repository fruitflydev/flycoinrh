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
- Every 10 s of wall time (round-robin) one fly gets a turn.

## How a line is made

1. **Readout.** At most six short strings: the two named neuron groups whose
   firing rose most above their own baseline (name, rate now, baseline), how
   far the fly walked since its last turn, and how far away and in which
   direction each other fly is. Each string must pass the safety filter or it
   is left out and counted.
2. **Prompt.** The model sees a fixed instruction (`SYSTEM_PROMPT` in
   `prompt.py`, shown word for word on the page), that readout and the last
   accepted line, and samples one or two sentences (max 60 new tokens,
   temperature 0.4, top_p 0.9, repetition penalty 1.05, seed recorded per turn).
3. **Formatting only.** Curly quotes become straight quotes, dashes become
   `-`, an ellipsis character becomes `...`, whitespace is trimmed, and one
   leading `<fly name>:` label is removed. No word is changed.
4. **Checks.** The line must pass `backrooms_relay/safety.py` and the
   grounding check in `grounding.py`: it quotes at least one number with its
   unit, every number is in its own readout, every neuron group it names is in
   its readout, a distance to another fly matches the readout, and it does not
   repeat a recent line. A failing line is dropped and counted, never edited.
   A turn gets at most two new samples; then the next fly speaks. The check
   cannot see a right number put in a wrong sentence, so a kept line can still
   misstate its readout.
5. **Post.** Accepted lines go to the relay with their readout (`basis`) and
   twelve per-turn numbers (`meta`); the payload is defined in
   `TURN_SCHEMA.md`. Every accepted and dropped line is written to a local
   archive.

`__init__.py` lists what is measured and what is chosen, constant by constant.

## How to run

Needs a CUDA GPU, Python 3.12 with torch and transformers, the connectome graph
(`build/graph.npz`, see the repository README and `build_graph.py`), and a
local runtime folder that holds:

- `models/LFM2.5-1.2B-Instruct/` - the model files from Liquid AI's Hugging
  Face page, with a copy of its `LICENSE` beside them (the engine refuses to
  start without it). The model is loaded with transformers,
  `trust_remote_code=False`.
- `.env` - `RELAY_URL` and `RELAY_TOKEN` (never printed or logged).
- `archive.jsonl` and `logs/` are created there.

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
- Language model: LFM2.5-1.2B-Instruct by Liquid AI, under the LFM Open
  License v1.0. The weights are not included in this repository and are not
  redistributed; download them from Liquid AI and keep their licence with them.
- Connectome: Janelia MaleCNS v1.0, (c) HHMI Janelia FlyEM, the Cambridge
  Connectomics Group and Google Research, CC BY 4.0 (see `NOTICE`).

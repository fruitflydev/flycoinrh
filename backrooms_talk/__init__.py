"""
backrooms_talk: four simulated flies in one room, and a small language model
that writes one short line at a time for the fly whose turn it is.

HOW_LINES_ARE_MADE (plain words; the constant below is the canonical text)

  Every line is written by a small language model, LFM2.5-1.2B-Instruct by
  Liquid AI, running on one local GPU. It is not a fly's thoughts. Flies do
  not use language, and the names A, B, C and D are labels.

  What the model is shown is a short readout of a simulation. One wiring
  diagram of a male fruit fly's central nervous system (the Janelia MaleCNS
  v1.0 connectome, CC BY 4.0, 165,122 neurons) is loaded once and run as four
  separate copies, each with its own membrane state, as a uniform leaky
  integrate-and-fire model: every neuron has the same simple parameters, and
  a connection's weight is its synapse count times a fixed 0.275 mV with the
  sign of its predicted transmitter (pairs with fewer than 3 synapses are
  dropped). That is a simplification of real neurons. The four copies stand
  in a simulated 20 x 20 mm room. Each copy walks by its own descending
  neurons, sees the others as dark shapes sampled onto its lamina columns,
  receives a cVA smell drive that grows as the others come near, and receives
  a sound drive on its Johnston's organ hearing neurons from the others'
  simulated pulse-song wing motor neurons.

  The flies chat about a topic. The first topic is proposed by the model
  from one seed theme chosen in the code (the room, the light, being small,
  humans, walls, flying, food, each other, sounds, sleep). When a fly's turn
  comes (every TURN_SECONDS of wall time, round-robin), it gets 3-5 plain
  hints from its own simulated brain and body, which the model may use or
  ignore: which kind of its named neuron groups rose most above their own
  baseline, in plain role words (song, courtship, aggression, escape,
  hearing, smell, sight, walking) and intensity words; whether its whole
  brain got busier or calmer since its last turn; the closest fly's distance
  and direction; whether the others' smell or song got stronger; how far it
  walked. The numeric readout behind the hints (group rates and baselines,
  distances, bearings) is posted with the line as `basis`. A fly switches
  topic when its whole-brain mean spike rate changed by at least SWITCH_JUMP
  (5 %) from its previous window (after TOPIC_MIN = 3 lines on the topic), or
  when the topic has had TOPIC_MAX = 6 lines; both thresholds are chosen, and
  the model picks the new topic, in a chosen category the last three topics
  did not use. The model is given a fixed instruction (SYSTEM_PROMPT in
  prompt.py, published word for word in README.md), the topic, the hints and
  the last six lines (after a switch only the new topic's, at least two), and samples a
  short reply.

  Before a line is kept, only formatting is normalised: curly quotes become
  straight quotes, short dashes become "-" and long dashes " - ", an ellipsis
  character becomes "...", every run of whitespace becomes one space and the
  ends are trimmed, and one leading "<fly name>:" label is removed. No word is
  changed. The line then passes the safety filter (backrooms_relay/safety.py)
  and the line checks (grounding.py): no real named people or brands,
  politics, religion, medical claims, hate or harm instructions; at most
  three sentences; a finished sentence; any number with a unit must be in its
  own hints, and no amount in words with a unit; no repeat of the last eight
  lines or of three content words from the last four (or one word three of them used); not the same opening
  words as the last three; and in a new topic's first two turns, a word of
  the topic. A
  line that fails is dropped and counted, never edited. A dropped line may be
  sampled again with a new seed at most twice; then the next fly speaks.
  Topics pass the same filter and content checks. The lines can still say
  things about a fly that its hints do not support.

MEASURED (numbers read from the simulation, per fly)
  * spike rates of named neuron groups (backrooms_dictionary, with cell
    counts from the dataset), averaged over the fly's window since its last
    turn, and each group's baseline: its mean rate from the start of the run
    to the start of that window;
  * positions, distances, bearings and distance walked in the room;
  * for the page's per-turn numbers (meta), not the readout: the summed rate
    of the 8 pulse-song wing motor neurons (the song level), the smell and
    sound drive the room delivered in Hz per cell, the fraction of all
    neurons that fired at least once in a 12 ms brain window
    (active_fraction) and the mean rate over all neurons, window averages.

CHOSEN (by the people who built this; constants are named in the code)
  * four flies, their labels, starting positions drawn from the seed;
  * the roamer's calibration gains on outgoing weights (calibration.CHOSEN =
    "pn05_apl10_kc03": olfactory projection neurons x0.5, APL x10, Kenyon
    cells x0.3), so this is the same simulated animal that roams;
  * the room, eye, smell and sound mappings of backrooms_world, and for
    several flies: contributions summed and clipped at one fly's ceiling;
  * the readout's shape: TOP_GROUPS = 2 groups ranked by (window - baseline)
    / max(baseline, the captioner's floor), the walked distance and the other
    flies, no captioner lines (EVENT_LINES = 0), BASIS_MAX = 6, WARMUP_STEPS
    of brain before the first turn;
  * the hints: role words per group, intensity words and their thresholds,
    the brain, sense and walking wording (readout.py);
  * the topics: SEED_THEMES, CATEGORIES, CATEGORY_MEMORY, SWITCH_JUMP,
    TOPIC_MIN, TOPIC_MAX, TOPIC_SHARED_STEMS, the topic instruction
    (topics.py);
  * TURN_SECONDS, the instruction text, frame text, moves and openers, PROMPT_LINES = 6,
    SWITCH_LINES = 2,
    HISTORY = 8 lines for the repeat check, max_new_tokens, temperature,
    top_p, repetition penalty, the per-turn seed rule, retries;
  * the line checks' word lists, units and thresholds;
  * the safety filter's word lists and the normalisation list above.

The world is the brain; the words are the model's. Nothing in this package
edits a generated word.
"""

HOW_LINES_ARE_MADE = (
    "Every line here is written by a small language model, LFM2.5-1.2B-Instruct by Liquid AI, "
    "running on one local GPU. It is not a fly's thoughts: flies do not use language, and the "
    "names A, B, C and D are labels. One wiring diagram of a male fruit fly's central nervous system "
    "(Janelia MaleCNS v1.0, CC BY 4.0, 165,122 neurons) is run as four separate copies, each a uniform "
    "leaky integrate-and-fire model, a simplification of real neurons, standing in a simulated room. "
    "The flies chat about a topic the model proposes. On each turn the model gets a fixed instruction, "
    "the topic, the last six lines and 3-5 plain hints from that fly's simulated brain and body (which "
    "kind of its neurons rose most above their baseline, whether its whole brain got busier, the "
    "closest fly, smell, sound, walking), which it may use or ignore. A fly switches topic when its "
    "simulated whole-brain activity jumps past a chosen threshold, or after a chosen number of lines "
    "on one topic; the model picks the new topic. Only formatting is normalised; no word is changed. "
    "A line is kept only if it passes a safety filter and simple checks (no real named people or "
    "brands, politics, religion, medical claims, hate or harm instructions, at most three short sentences, "
    "any number with a unit taken from its hints, no repeat); otherwise it is dropped and counted, "
    "never edited. The lines can still say things the hints do not support. The rates, distances and "
    "bearings are measured from the simulation; the room, the hints' wording, the topic rule, the "
    "instruction, the sampling settings and the checks are chosen."
)

__all__ = ["HOW_LINES_ARE_MADE"]

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

  When a fly's turn comes (every TURN_SECONDS of wall time, round-robin), its
  readout is at most six short lines: its two named neuron groups whose
  firing rose most above their own baseline (name, rate now, baseline), how
  far it walked since its last turn, and how far away each other fly is and
  in which direction. Each readout line must pass the same safety filter as
  the words; a line that fails is left out and counted. The model is given a
  fixed instruction (SYSTEM_PROMPT in prompt.py, shown word for word on the
  page), that readout and the last accepted line, and samples one or two
  sentences.

  Before a line is kept, only formatting is normalised: curly quotes become
  straight quotes, dashes become "-", an ellipsis character becomes "...",
  whitespace is trimmed from both ends, and one leading "<fly name>:" label is
  removed. No word is changed. The line then passes the safety filter
  (backrooms_relay/safety.py) and a grounding check (grounding.py): it must
  quote at least one number with its unit, every number it quotes must be in
  its own readout with that unit, every neuron group it names must be in its
  readout, a distance it gives another fly must be that fly's distance in its
  readout, and it must not repeat a recent line. A line that fails is dropped
  and counted, never edited. A dropped line may be sampled again with a new
  seed at most twice; then the next fly speaks. The check cannot see a right
  number put in a wrong sentence or a claim with no number in it, so a kept
  line can still misstate the readout.

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
  * TURN_SECONDS, the instruction text and frame text, PROMPT_LINES = 1,
    HISTORY = 6 lines for the repeat check, max_new_tokens, temperature,
    top_p, repetition penalty, the per-turn seed rule, retries;
  * the grounding check's units, phrasings and repeat thresholds;
  * the safety filter's word lists and the normalisation list above.

The world is the brain; the words are the model's. Nothing in this package
edits a generated word.
"""

HOW_LINES_ARE_MADE = (
    "Every line here is written by a small language model, LFM2.5-1.2B-Instruct by Liquid AI, "
    "running on one local GPU. It is not a fly's thoughts: flies do not use language, and the "
    "names A, B, C and D are labels. The model is shown a short readout of a simulation. One "
    "wiring diagram of a male fruit fly's central nervous system (Janelia MaleCNS v1.0, CC BY 4.0, "
    "165,122 neurons) is run as four separate copies, each a uniform leaky integrate-and-fire "
    "model, a simplification of real neurons, standing in a simulated 20 x 20 mm room. Each copy "
    "walks by its own descending neurons, sees the others as dark shapes, receives a cVA smell "
    "drive as the others come near and a sound drive from their simulated pulse-song motor "
    "neurons. When a fly's turn comes, its readout is at most six lines: its two named neuron "
    "groups whose firing rose most above their own baseline, how far it walked, and how far away "
    "each other fly is. The model gets a fixed instruction, that readout and the last line, and "
    "samples one or two sentences. Only formatting is normalised (curly quotes, dashes, ellipsis, "
    "surrounding whitespace, a leading name label); no word is changed. A line is kept only if "
    "it passes a safety filter and a grounding check: it quotes at least one number with its "
    "unit, every number it quotes is in its own readout, every neuron group and fly distance it "
    "names matches that readout, and it does not repeat a recent line. Otherwise it is dropped "
    "and counted, never edited. The check cannot see a right number in a wrong sentence, so a "
    "kept line can still misstate the readout. The rates, distances and bearings are measured "
    "from the simulation; the room, the channels, the readout's shape, the instruction, the "
    "sampling settings, the check and the filter are chosen."
)

__all__ = ["HOW_LINES_ARE_MADE"]

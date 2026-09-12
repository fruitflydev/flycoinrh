"""
How the mushroom body is set, and how like and dislike are read from it.

WHY A CALIBRATION EXISTS AT ALL
The simulator (flysim.py, after Shiu et al. 2024) gives every neuron the same
parameters and every synapse 0.275 mV per contact. Measured on this connectome
(2026-09-12), that stock brain is not a fly's mushroom body: any odour ignites
about 35,000 neurons within a millisecond, fires 99.9-100% of the 4,064
Kenyon cells, and makes every odour's pattern identical. Sugar and shock then
depress every KC->MBON synapse alike, so "learning" dulls everything at once.

flysim leaves one free efficacy per cell type (`gains`). The settings below
change only antennal-lobe and mushroom-body types: the APL feedback neuron (1
type, 2 cells), olfactory receptor neurons (53 types, 2,635 cells), antennal-lobe
projection neurons (141 types, 575 cells, which include 13 thermo- and
hygrosensory VP projection neuron types, since the measurements were made with
them in the group) and Kenyon cells (15 types, 4,064 cells). They were chosen
by measurement, not by fitting behaviour: they move Kenyon-cell coding toward the sparse, distinct patterns
measured in real flies (about 5-10% of KCs per odour; Turner et al. 2008,
Honegger et al. 2011) and away from the ignition regime.

WHAT CALIBRATION DOES AND DOES NOT ACHIEVE (measured, 11 settings)
Learning was tested by pairing 3-octanol with shock and 4-methylcyclohexanol
with sugar, 12 times each, and reading each odour's KC firing pattern against
the KC->MBON weights. "Leak" is how much an untouched odour lost relative to
the trained one: 0 is perfectly specific, 1 is fully global.
  * stock brain: leak 1.00
  * best settings: leak 0.45-0.48
  * leak tracks overlap between odour patterns (r = +0.72 over 44 comparisons)
    and the trained odour's Kenyon-cell share (r = +0.53), not pattern
    repeatability (r = +0.01)
  * shock can stay specific (leak ~0.2) when the shocked odour is sparse and
    distinct; sugar spreads (0.6-1.0), since reward-side compartments hold
    twice the synapses
So the honest claim is: sugar and shock change the fly's response partly to
the paired smell and partly to similar smells. Not clean one-smell learning.

ROAMING IS UNCHANGED (measured)
The same brain roams the web, so every setting was checked on the roaming
pilot: 4 real screenshots x 3 cursor positions x 20 seeds = 240 steps each.
The stock brain against itself over five seed blocks gives the noise floor:
clicks 14-25 per 240 steps (mean 18.2, sd 4.2), per-step cursor difference
between two seed blocks 43.9 px across and 18.8 px down. The calibrated
settings on stock's own seeds click 7, 17 and 12 times, stay within 1-3 px of
stock's mean moves, and put the cursor 27-28 px and 11-12 px from stock per
step, less than a reseed does. The one consistent change is right-steering
drive 17-23 Hz lower (2-3x its block noise), with turning asymmetry inside
stock's own spread.

READING LIKE AND DISLIKE
mushroom.py names MBONs by which dopamine cluster innervates their compartment,
counted from the synapses each MBON type receives (build/mb_sides.json):
`reward_side` (PAM) and `punish_side` (PPL1). Those names are about dopamine,
not behaviour. Behaviourally (Aso et al. 2014, eLife 3:e04580), MBONs in PAM
compartments drive AVOIDANCE and MBONs in PPL1 compartments drive APPROACH.
Reward depresses KC input to avoidance MBONs, so a rewarded smell is approached
more. The old lander.py read this the other way round.

The backroom reads a look off the mushroom body's OUTPUT SYNAPSES rather than
off those MBONs' firing rates, and it reads it against the other cards in the
room. Three functions, in this one file because the room (backroom.py) and the
offline gate (backroom_screen.py) must never drift apart:
  syn_drive(mb, fired)    the synapses whose Kenyon cell fired, summed per side
  leaning(reading)        (A - V) / (A + V), scale-free
  relative(own, others)   leaning(own) minus the mean leaning of the other cards

MEASURED 2026-09-12 (backroom_screen.py, build/backroom_screen.json), and this
is why the rule is that and not something simpler:
  * with the MBON firing rates as the readout, BOTH sugar and shock lowered the
    score - in a whole-brain simulation those cells also carry recurrent input
    from everything else - so a rewarded coin came out backwards (-0.0042 on
    its own coin, 2.8 SE, trained alone);
  * a blank control run on a uniform ground-grey frame is not blank: that frame
    fired more approach MBONs (13,075) than a card did (6,796), training moved
    the control more than it moved the cards, and shock inverted;
  * comparing raw sums instead of leanings let the coin that fires five times
    as many Kenyon cells decide every other coin's score, which is the other
    way sugar came out backwards;
  * with the synaptic readout, per-card leanings and the other cards as the
    reference, over 60 paired seeds with both signs trained: the coin paired
    with profit +0.0359 (5.6 SE), the coin paired with loss -0.0399 (5.8 SE),
    an untouched coin +0.0041 (0.8 SE), leak 0.10. One-sided runs
    (build/backroom_screen_sugar.json, _shock.json) are 66% and 63%
    coin-specific.
valence() and contrast() below are the rate readout and the blank control. The
room uses neither; they stay for the offline screen's other modes, which is
where they were measured and failed.
"""
import re

import numpy as np

# Cell-type groups a setting may scale, matched against FlyBrain.type_names.
GROUPS = {
    "APL": r"^APL",                   # the GABAergic feedback neuron onto Kenyon cells
    "ORN": r"^ORN_",                  # olfactory receptor neurons
    "PN": r"_(l|v|ad|il|lv)PN",       # antennal-lobe projection neurons (DM1_lPN, DA1_vPN, ...), incl. 13 thermo/hygro VP types
    "KC": r"^KC",                     # Kenyon cells (their outgoing synapses)
}

# The three best-measured settings. Each value multiplies that group's outgoing
# weights. Numbers are the measured sweep and roaming results, kept here as
# documentation. roam_clicks is per 240 roaming steps on stock seed block 0
# (stock: 14; stock across five seed blocks: 18.2 +- 4.2); roam_max_dn_shift_sd
# is the largest descending-neuron rate change in units of stock's per-step SD.
SETTINGS = {
    "stock": {"gains": {}, "odour_max_hz": 200, "mean_leak": 1.00,
              "kc_pct": (99.9, 100.0), "pattern_overlap": (1.00, 1.00),
              "roam_clicks": 14, "roam_max_dn_shift_sd": 0.0},
    "pn03_apl10_kc03": {"gains": {"PN": 0.3, "APL": 10, "KC": 0.3}, "odour_max_hz": 200, "mean_leak": 0.45,
                        "kc_pct": (0.4, 1.9), "pattern_overlap": (0.22, 0.72),
                        "roam_clicks": 7, "roam_max_dn_shift_sd": 0.27},
    "pn05_apl10_kc03": {"gains": {"PN": 0.5, "APL": 10, "KC": 0.3}, "odour_max_hz": 200, "mean_leak": 0.46,
                        "kc_pct": (2.8, 10.7), "pattern_overlap": (0.42, 0.80),
                        "roam_clicks": 17, "roam_max_dn_shift_sd": 0.13},
    "pn03_apl10": {"gains": {"PN": 0.3, "APL": 10}, "odour_max_hz": 200, "mean_leak": 0.48,
                   "kc_pct": (0.6, 2.0), "pattern_overlap": (0.17, 0.59),
                   "roam_clicks": 12, "roam_max_dn_shift_sd": 0.25},
}

# The setting the fly runs on. The three best tie on learning leak (0.45-0.48);
# this one is the only one inside the measured 5-10% Kenyon-cell band, gives
# learning the most signal (shock cut the trained odour's approach drive 15%),
# and moves roaming least (largest DN shift 0.13 SD, clicks 17 vs 18.2 +- 4.2).
CHOSEN = "pn05_apl10_kc03"


def group_mask(fb, group):
    """Boolean mask over fb.type_names for one group."""
    if group not in GROUPS:
        raise KeyError(f"unknown group {group!r}; known: {sorted(GROUPS)}")
    rx = re.compile(GROUPS[group])
    return np.array([bool(rx.search(str(n))) for n in fb.type_names])


def matched_types(fb, group):
    """The cell-type names a group scales, for disclosure."""
    return [str(n) for n, m in zip(fb.type_names, group_mask(fb, group)) if m]


def gains_for(fb, setting):
    """
    A per-type gains vector for FlyBrain.run, or None for the stock brain.
    `setting` is a name from SETTINGS or a {group: factor} dict.
    """
    cfg = SETTINGS[setting]["gains"] if isinstance(setting, str) else dict(setting)
    if not cfg:
        return None
    g = np.ones(fb.n_types, dtype=np.float32)
    for group, factor in cfg.items():
        if not factor > 0:
            raise ValueError(f"gain for {group} must be positive, got {factor}")
        g[group_mask(fb, group)] *= np.float32(factor)
    return g


def readout(mb):
    """
    Populations to record for a like/dislike reading: approach vs avoidance MBONs.

    Their rates are recorded and reported. Nothing decides on them: what a look
    is worth is read off the synapses below.
    """
    return {"approach": np.asarray(mb.punish_side), "avoid": np.asarray(mb.reward_side)}


def valence(run_result):
    """
    Approach minus avoidance, in Hz, from a FlyBrain.run recorded with readout(mb).

    The first readout, kept for the offline screen and for lander.py. MEASURED
    2026-09-12: it cannot carry a lesson (see the module docstring), so the room
    does not use it.
    """
    return float(np.asarray(run_result["approach"]).mean() - np.asarray(run_result["avoid"]).mean())


def syn_drive(mb, fired):
    """
    What the mushroom body's output synapses carry for one look: (approach, avoid).

    MEASURED: which Kenyon cells fired in that run, every KC->MBON synapse in
    the connectome, and which dopamine cluster innervates each MBON's
    compartment (build/mb_sides.json). CHOSEN: that a look is worth the sum of
    base * gain over the synapses whose Kenyon cell fired, split by side -
    approach for the PPL1-input MBONs, avoidance for the PAM-input ones.

    This is the quantity the plasticity rule actually moves, and depressing one
    side can only lower that side. MBON firing rates were the first readout and
    are not usable for it: in a whole-brain simulation those cells also receive
    recurrent input from everything else, and measured 2026-09-12 both sugar and
    shock lowered the rate-based score, so sugar came out backwards
    (backroom_screen.py, build/backroom_screen.json).
    """
    w = np.asarray(mb.base, dtype=np.float64) * np.asarray(mb.gain, dtype=np.float64)
    active = np.zeros(int(mb.fb.n), dtype=bool)
    if fired is not None and len(fired):
        active[np.asarray(fired, dtype=np.int64)] = True
    hot = active[np.asarray(mb.pre, dtype=np.int64)]
    side = np.asarray(mb.side)
    return float(w[hot & (side == -1)].sum()), float(w[hot & (side == 1)].sum())


def has_reading(reading):
    """
    True when a look measured something at all: some Kenyon cell's synapses
    were counted.

    syn_drive returns (0.0, 0.0) for a run in which no Kenyon cell fired, and
    leaning() scores that 0.0 - which sits about a quarter of the range above
    every card this project has actually measured (every leaning in the first
    paper run was between -0.18 and -0.32). An empty reading is not a
    measurement of a card, so it may not be one card's drive nor another
    card's reference. MEASURED 2026-09-12: one such reading in the reference
    of look 1789229112324-0009 moved its drive from -0.068 to -0.171, and the
    drive is the order size. Both callers check this: the room keeps an empty
    reading out of the dwell and out of the room behind the fly, and the
    offline gate never had one.
    """
    return float(reading[0]) + float(reading[1]) > 0.0


def leaning(reading):
    """
    How far one look leans toward approach, in [-1, 1]: (A - V) / (A + V).

    `reading` is one (approach, avoid) pair from syn_drive. A card that fires
    no Kenyon cell at all leans nowhere, so an empty reading is 0 - which is a
    number to discard (has_reading), not a score to compare. The formula
    is CHOSEN; that it has to be a ratio and not a difference is MEASURED: a
    coin that fires five times as many Kenyon cells as another would otherwise
    dominate every comparison it takes part in, which is what made sugar look
    backwards when the offline screen first averaged raw sums (2026-09-12,
    build/backroom_screen.json).
    """
    a, v = (float(x) for x in reading)
    den = a + v
    if not den > 0:
        return 0.0
    return float(np.clip((a - v) / den, -1.0, 1.0))


def relative(own, others):
    """
    How far one card leans above the rest of the room, in [-1, 1]. CHOSEN.

    `own` is this look's (approach, avoid) reading; `others` are the readings of
    the other cards being compared with it - in the offline gate the same seed's
    look at every other coin, in the room the last look the fly took at each
    other card during this visit. drive = clip(leaning(own) - mean(leanings of
    others), -1, 1). It has no constant.

    A fly in a T-maze is offered two arms and chooses between them, and this is
    the same comparison - which is also the only form the offline gate could
    measure a lesson in (build/backroom_screen.json). MEASURED 2026-09-12:
    learning here is partly global, so a lesson about one coin moves every
    coin's absolute reading the same way and a card read against a blank grey
    frame carries almost none of it - an untouched coin moved 0.88 to 0.92 as
    far as a trained one, and the shocked coin moved the wrong way, whether the
    reading was the output synapses or the MBON firing rates. The difference
    between two coins carries it.

    With no other card there is no comparison and this is 0. A caller that must
    tell "nothing to compare with" from "compared and indifferent" - the room
    does, because a stop with no reference commits nothing - has to check
    `others` itself.
    """
    return relative_leaning(leaning(own), [leaning(x) for x in others])


def relative_leaning(own, others):
    """
    The same rule, when the other cards' leanings are what is in hand.

    relative() is the way to call this while the readings are still there. A
    replay of a stored look has only the leanings the other cards gave at the
    time - they were read at other rectangles and other moments, and re-running
    them now would be a different measurement - so it calls this instead, and
    the subtraction and the clip stay in one place either way.
    """
    others = [float(x) for x in others]
    if not others:
        return 0.0
    return float(np.clip(float(own) - float(np.mean(others)), -1.0, 1.0))


def contrast(stim, blank):
    """
    Like or dislike of one look against a blank look, in [-1, 1].

    stim and blank are (approach Hz, avoid Hz), each summed over its MBON
    population: one from the run that saw the card and smelled the coin, one
    from a run with the same seed and cursor that saw only the room's ground
    and smelled nothing. drive = ((A - V) - (A0 - V0)) / (A + V + A0 + V0).
    Subtracting the blank removes any fixed imbalance between the two MBON
    populations, so a coin is liked or disliked for what it looks and smells
    like rather than because one population is larger. The formula is CHOSEN.

    The room does not use this: measured 2026-09-12, a look read against a
    blank control does not carry a lesson at all, whichever way the two
    populations are read. `relative()` is what backroom.py asks for. This stays
    for the offline screen's blank modes, which is where that was measured.
    """
    a, v = (float(x) for x in stim)
    a0, v0 = (float(x) for x in blank)
    den = a + v + a0 + v0
    if not den > 0:
        return 0.0
    return float(np.clip(((a - v) - (a0 - v0)) / den, -1.0, 1.0))

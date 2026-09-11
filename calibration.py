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
mushroom.py names MBONs by which dopamine cluster innervates their compartment:
`reward_side` (PAM) and `punish_side` (PPL1). Those names are about dopamine,
not behaviour. Behaviourally (Aso et al. 2014, eLife 3:e04580), MBONs in PAM
compartments drive AVOIDANCE and MBONs in PPL1 compartments drive APPROACH.
Reward depresses KC input to avoidance MBONs, so a rewarded smell is approached
more. Hence: valence = mean rate of PPL1-compartment MBONs minus mean rate of
PAM-compartment MBONs. The old lander.py read this the other way round.
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
    """Populations to record for a like/dislike reading: approach vs avoidance MBONs."""
    return {"approach": np.asarray(mb.punish_side), "avoid": np.asarray(mb.reward_side)}


def valence(run_result):
    """Approach minus avoidance, in Hz, from a FlyBrain.run recorded with readout(mb)."""
    return float(np.asarray(run_result["approach"]).mean() - np.asarray(run_result["avoid"]).mean())

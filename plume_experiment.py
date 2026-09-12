"""
The plume experiment: does the connectome, with the roamer's calibrated type
gains and no learning or fitting to this task, track an odour plume?

WHY THIS FILE EXISTS
A walking fly in a wind tunnel (after Alvarez-Salvado et al. 2018 eLife and
van Breugel and Dickinson 2014 Curr Biol) surges upwind when it meets odour
and casts crosswind when it loses it. Those behaviours were measured in real
flies. This runner asks whether the same behaviours EMERGE from the simulated
connectome when its receptor neurons smell ethyl acetate (plume.py builds the
plume; plume_fly.py couples odour and wind to the brain and reads the walking
descending neurons). Nothing here is fitted to the task: the wiring is the
connectome's, the per-type gains are calibration.CHOSEN (PN x 0.5, APL x 10,
KC x 0.3, chosen earlier for the roamer and disclosed in the output with the
type counts they touch). The predictions and the metrics below were fixed
before any trial was run, and whatever comes out is the result, reported as
such.

TWO PROTOCOLS, ONE RUNNER
  v1 is the published run (build/plume_experiment.json, 2026-09-12). Its code
  path is unchanged and its outputs are never written by a new run: a new
  run is refused if it would land on the published paths.
  v2 is a new protocol, fixed on 2026-09-13 before any v2 trial, that removes
  the confounds v1's own report named as follow-up work. It is not a retry
  of v1: the conditions, the start, the brain's memory, the antennal drive,
  the control and the window rules all differ, and PROTOCOL_V2_CHANGES below
  says, for each change, what it removes and where v1's report asked for it.
  Selected with --protocol v2; outputs go to build/plume_v2_*.

WHAT PROTOCOL V1 COULD NOT SHOW (found in review after the first run; none of
these changed a trial, so the first run stands, and each is disclosed in the
JSON next to the number it qualifies)
  * The controller is memoryless: FlyBrain.run restarts every neuron from
    rest at each 50 ms world step and nothing is carried over, so the motor
    command is a stochastic function of (c, phi, seed) at that step. Surge
    and cast are history-defined; here they can only show as a static
    difference between the motor map above and below the odour threshold.
  * The fly starts inside the plume (plume.py explains the arithmetic), so
    the "encounters" P1 and P2 count are re-entries after the meander swept
    the plume off a nearly stationary fly, not odour onsets from clean air.
  * Windows are clipped at the trace end; an encounter in the last second of
    a trial gets a one-step "after" window. Full-window-only and
    post-encounter-only versions were reported as non-preregistered
    sensitivity checks; the preregistered verdicts were never replaced.
  * P4 is not a discriminating test: reaching the source needs 92.5 % of top
    speed straight upwind for the whole trial.
  * The readout is quantised (one DNa02 cell per side counted over 12 ms:
    1.67 deg of heading per world step per spike difference); the JO drive is
    asymmetric by population (203 left : 132 right cells), JO-C and JO-E are
    co-activated, and the constant-rate control delivered about twice the
    total JO drive the wind-sense flies received.

PROTOCOL V2 (fixed before any v2 data; letters as in the preregistration)
  A  continuous brain: the fly carries membrane potentials, refractory
     counters and the random stream across the world steps of a trial and
     resets them at each trial start (a fresh seed per trial); 120 LIF steps
     = 24 ms of brain per 50 ms of world (CHOSEN, disclosed; v1 was 60 steps
     restarted from rest). The runner tells the fly when a trial begins.
  B  motor low-pass: the turn and speed commands the world receives are
     exponentially smoothed with a 150 ms time constant (3 world steps),
     CHOSEN, standing in for leg and body inertia the readout has none of.
     The runner does the smoothing, from rest at the trial start, and
     records the raw per-step commands beside the smoothed ones.
  C  start outside the plume: x = 0.25 (0.20 m downwind), at least 0.08 m
     off the centreline on a random side, heading uniform (plume.World with
     protocol="v2"). The runner checks c < 0.05 at t = 0 for every trial and
     reports the count of trials that violate it; a violation is reported,
     never discarded or redrawn. plume.start_rule_v2_check() measured 9 of
     the first 50 seeds above the threshold at t = 0 (the meander), so the
     expected count for 10 seeds is 1 (seed 0), and the record says so.
  D  reachable source: 0.20 m in 20 s needs 42.5 % of top speed; reach
     radius 0.03 m; a trial ends when the source is reached.
  E  antennal drive: only JO-E cells, per-side rates scaled so both antennae
     deliver the same total cell-Hz at the same deflection. That lives in
     plume_fly.PlumeFly(protocol="v2"); the runner records the per-side cell
     counts from describe() and the realised per-side and total cell-Hz from
     what the fly reports each step.
  F  matched control "shuffled": odour on, wind drive computed exactly as in
     E but from a wind angle drawn uniformly at random each step by the
     runner, independent of the heading, so the drive statistics match and
     the direction information is zero. The fly is given phi_fly; the true
     phi is recorded too. Conditions, paired by seed: odour (odour + wind),
     blank (wind, no odour), shuffled (odour + shuffled wind). The realised
     mean total JO cell-Hz per condition is reported, with whether odour and
     shuffled are within 15 % of each other.
  G  metrics: encounter and loss as v1 (0.05 with 0.5 s below / 0.25 s
     above); P1 counts only encounters with a full 1 s window on both sides;
     P2 counts only losses that follow an encounter and have a full 2 s
     window on both sides (the "before" window is the last 2 s inside, so
     the fly must have been inside for 2 s); P3 odour > blank and odour >
     shuffled by more than 2 SE of the paired difference; P4 fraction
     reaching the source, odour > blank. Also reported: baseline speed and
     turn with no odour, the fraction of steps facing upwind (cos phi > 0),
     encounters per trial, time in plume, wall contacts, the smoothed and
     raw command distributions, and v1's numbers beside v2's.
  H  budget: 10 seeds x 3 conditions x up to 400 steps at 120 LIF steps per
     world step; if the first seed projects more than 150 minutes the seed
     count drops to 8 and never lower. Outputs build/plume_v2_experiment.json,
     build/plume_v2_trajectories.npz/.png; v1's outputs are never touched.

MEASURED, by this runner
  * per trial: encounters and losses of the plume, the surge after an
    encounter, the cast after a loss, upwind progress, whether the source was
    reached, wall contacts, time in plume, the fraction of steps facing
    upwind, the mean motor commands (raw and as received by the world);
  * across seeds: means, standard errors, paired differences between
    conditions on the same seeds, and each prediction's verdict.

CHOSEN, and said so in the output
  * the concentration threshold 0.05 and the hysteresis (0.5 s below before an
    encounter, 0.25 s above before a loss), the 1 s surge window, the 2 s cast
    window, and "more than 2 standard errors" as the bar every prediction has
    to clear (SE = sd / sqrt(n), sd with one degree of freedom lost);
  * v1: windows are clipped at the ends of the trace and use the samples that
    exist; v2: only full windows count (G above); under both, the history a
    hysteresis rule needs must lie inside the trace, so nothing can be an
    encounter before sample 10 or a loss before sample 5;
  * a trial with zero encounters contributes nothing to P1 or P2 and is
    counted; inside such a trial a loss is never used;
  * heading change is the wrapped difference of successive headings, in
    radians internally, reported in degrees; its spread within a window is
    the population standard deviation;
  * the budget ladder: if the first seed's trials say the whole run would
    exceed the budget, the seed count walks down the protocol's ladder
    (v1: 12 -> 10 -> 8; v2: 10 -> 8) and never below 8, and the steps per
    trial are never cut; seeds are the outer loop so every kept seed has all
    of its conditions.

THE PREDICTIONS (fixed before data; the v2 statements are in PROTOCOLS)
  P1 surge   mean upwind velocity (-vx) in the 1 s after an encounter minus
             the 1 s before it, odour condition: > 0 by more than 2 SE.
  P2 cast    mean crosswind speed |vy| and the spread of heading change in the
             2 s after a loss versus the last 2 s inside the plume: both larger
             after the loss by more than 2 SE.
  P3 upwind  x_start - x_end: odour > blank and odour > the second control
             (v1 nowind, v2 shuffled), each by more than 2 SE of the paired
             difference.
  P4 source  fraction of trials ending within the source radius: odour > blank.

INTERFACES (plume.py and plume_fly.py; the runner reads them defensively)
  World(seed, odour=True[, protocol=...]): concentration(x, y);
    wind_direction_relative(heading) (radians, the angle the wind comes FROM
    relative to the heading, 0 = headwind, positive = from the fly's left);
    step(turn, speed) -> {x, y, heading, c, t, reached, wall_contacts, ...};
    `state` (a property, or a method) for the state before the first step,
    else the attributes x, y, heading, t are read; heading in radians unless
    a heading_unit attribute says "deg"; dt is read off the world's clock (t
    after one step), else 0.05; start_c_field, if present, is the plume at
    the start pose whatever the odour flag. turn > 0 is a RIGHT turn. v2
    requires the protocol argument and refuses a World without it.
  PlumeFly(fb, gains, odorant=..., odour_hz=..., wind_hz=..., sim_steps=...,
    seed=...[, protocol=...]): step(c, phi, seed=..., wind_sense=True) ->
    (turn, speed, info) where info holds scalar rates (six motor rates, mean
    ORN rate, JO rate per side, jo_total_cell_hz and, if the fly reports
    them, jo_left_cell_hz / jo_right_cell_hz, fired count); a dict with turn
    and speed keys, or a bare (turn, speed) pair, are accepted too.
    describe() lists its constants. v2 requires PlumeFly to accept
    protocol="v2" (the antennal drive E and the state carry A live there)
    and to have a begin_trial(seed) method (or start_trial / reset_trial /
    reset), which the runner calls once at the start of every trial with
    that trial's brain seed; the per-step seed the runner still passes is
    only meaningful to a fly that does not carry state, and v2 refuses one.

KNOWN SIMULATOR LIMITS, disclosed in the output: uniform 0.275 mV synapses,
no conduction delays, no receptor adaptation, although adaptation matters
for real plume tracking, and under v1 the restart from rest at every step.

  py plume_experiment.py --quick 1            smoke test, 2 seeds x 80 steps (v1, build/plume_quick_*)
  py plume_experiment.py protocol=v2 quick=1  v2 smoke test, build/plume_v2_quick_*
  py plume_experiment.py protocol=v2          10 seeds x 3 conditions x 400 steps, build/plume_v2_*
  py plume_experiment.py seeds=10 out=build/plume_rerun.json log=run.log
                                              key=value tokens are accepted too, and an
                                              --out that names the JSON itself is read as
                                              the prefix (build/plume_experiment.json ->
                                              build/plume_{experiment.json,trajectories.npz,.png});
                                              the published v1 prefix build/plume is refused
  py plume_experiment.py --reanalyse build/plume_v2_experiment.json
                                              recompute the summary and the picture from the
                                              saved trajectories, no brain; the run record is
                                              kept and a reanalysis entry is added. The
                                              preregistered numbers are checked against the
                                              old JSON and the check is recorded. The published
                                              v1 files need --allow-published 1 to be rewritten.
"""
import argparse
import inspect
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
BUILD = ROOT / "build"

# ---- CHOSEN constants, every one of them, in one place ----------------------
DT = 0.05                  # s of world time per brain run (the roamer's convention)
THRESHOLD = 0.05           # concentration that counts as "in the plume"
MIN_BELOW_S = 0.5          # below the threshold for at least this long before an encounter
MIN_ABOVE_S = 0.25         # above the threshold for at least this long before a loss
SURGE_WINDOW_S = 1.0       # P1: 1 s before and after an encounter
CAST_WINDOW_S = 2.0        # P2: 2 s after a loss and the last 2 s inside
SE_MULTIPLE = 2.0          # a prediction holds when the effect exceeds this many SE
ARENA = (0.6, 0.3)         # m, length (x) by width (y); the render's default, the world owns it
SOURCE = (0.05, 0.15)      # m; render default
SOURCE_RADIUS = 0.03       # m; render default, the world decides "reached"
CONDITIONS = {             # v1: name -> (odour in the world, wind sense in the fly)
    "odour": (True, True),
    "blank": (False, True),
    "nowind": (True, False),
}
CONDITION_ORDER = ("odour", "blank", "nowind")
PAIRS = (("odour", "blank"), ("odour", "nowind"))
DEFAULT_SEEDS = 12
DEFAULT_STEPS = 400
SEED_LADDER = (12, 10, 8)  # drop through this if the first trial says the run is too slow
MIN_SEEDS = 8
BUDGET_S = 3600.0          # the whole run should fit in an hour (design: ~50 min at 0.2 s/run)
QUICK_SEEDS = 2
QUICK_STEPS = 80
SIM_STEPS = 60             # brain steps per world step (12 ms of brain time)
ODORANT = "ethyl acetate"
RAM_FLOOR_GB = 6.0         # never load a brain with less free memory than this
SNAPSHOT_STEPS = 0         # world steps before the plume picture: plume.World warms its plume up before the trial clock, so the picture is the plume the fly starts in
SNAPSHOT_GRID = (120, 60)  # samples across x and y for that picture
RUNNER_OWNED = ("turn", "speed", "turn_raw", "speed_raw", "c", "phi", "seed", "wind_sense", "t", "step")  # echoed by the fly's info; recorded by the runner itself

# ---- protocol v2 (fixed before any v2 trial) ---------------------------------
SIM_STEPS_V2 = 120         # LIF steps per world step: 24 ms of brain per 50 ms of world (CHOSEN, disclosed)
SMOOTH_TAU_S = 0.15        # motor low-pass time constant, 3 world steps (CHOSEN; leg and body inertia the readout lacks)
DEFAULT_SEEDS_V2 = 10
SEED_LADDER_V2 = (10, 8)
BUDGET_S_V2 = 150 * 60.0   # 150 minutes for the whole run
JO_MATCH_TOLERANCE = 0.15  # odour and shuffled should deliver total JO cell-Hz within this relative difference
CONDITIONS_V2 = {          # name -> (odour in the world, wind sense in the fly, wind angle shuffled by the runner)
    "odour": (True, True, False),
    "blank": (False, True, False),
    "shuffled": (True, True, True),
}
CONDITION_ORDER_V2 = ("odour", "blank", "shuffled")
PAIRS_V2 = (("odour", "blank"), ("odour", "shuffled"))
V1_PREFIX = BUILD / "plume"                                   # the published run: read, never written
V1_JSON = V1_PREFIX.with_name(V1_PREFIX.name + "_experiment.json")
PUBLISHED_V1 = tuple(V1_PREFIX.with_name(V1_PREFIX.name + s)
                     for s in ("_experiment.json", "_trajectories.npz", "_trajectories.png", "_report.md"))
BEGIN_TRIAL_NAMES = ("begin_trial", "start_trial", "reset_trial", "reset", "reset_state")   # what a state-carrying fly may call it

PROTOCOLS = {
    "v1": {
        "conditions": {k: (v[0], v[1], False) for k, v in CONDITIONS.items()},
        "order": CONDITION_ORDER, "pairs": PAIRS,
        "default_seeds": DEFAULT_SEEDS, "seed_ladder": SEED_LADDER, "min_seeds": MIN_SEEDS, "budget_s": BUDGET_S,
        "sim_steps": SIM_STEPS, "smooth_tau_s": None, "state_carry": False,
        "p1_rule": "any_window", "p2_rule": "any_window",
        "default_out": "plume", "world_protocol": "v1",
    },
    "v2": {
        "conditions": dict(CONDITIONS_V2),
        "order": CONDITION_ORDER_V2, "pairs": PAIRS_V2,
        "default_seeds": DEFAULT_SEEDS_V2, "seed_ladder": SEED_LADDER_V2, "min_seeds": MIN_SEEDS, "budget_s": BUDGET_S_V2,
        "sim_steps": SIM_STEPS_V2, "smooth_tau_s": SMOOTH_TAU_S, "state_carry": True,
        "p1_rule": "full_window", "p2_rule": "full_window_after_encounter",
        "default_out": "plume_v2", "world_protocol": "v2",
    },
}
PROTOCOL_DEFAULT = "v1"

PROTOCOL_V2_CHANGES = (   # each change, what it removes, and where v1's report asked for it (build/plume_report.md)
    {"letter": "A", "change": "continuous brain: membrane potentials, refractory counters and the random stream are carried "
                              "across the world steps of a trial and reset at each trial start; 120 LIF steps (24 ms) of "
                              "brain per 50 ms of world",
     "removes": "(a) memoryless brain: every v1 world step restarted the brain from rest",
     "named_in_v1_report": "'No memory across steps' under simulator limits, and 'carry membrane state across steps' under Design"},
    {"letter": "B", "change": f"motor low-pass: the commands the world receives are exponentially smoothed with a "
                              f"{SMOOTH_TAU_S * 1000:.0f} ms time constant; raw commands recorded too",
     "removes": "(g) mitigates the readout quantisation (1.67 deg heading steps, |heading rate| at the readout floor)",
     "named_in_v1_report": "'the heading readout is quantised at 1.67 deg per step' under Design"},
    {"letter": "C", "change": "start outside the plume: x = 0.25, at least 0.08 m off the centreline on a random side",
     "removes": "(b) 9 of 10 v1 odour trials began above the threshold, so encounters were re-entries",
     "named_in_v1_report": "'start the fly outside the plume' under Design"},
    {"letter": "D", "change": "reachable source: 0.20 m in 20 s needs 42.5 % of top speed",
     "removes": "(c) v1 needed 92.5 % of top speed straight upwind for the whole trial",
     "named_in_v1_report": "'the source needs 92.5 % of top speed' and 'give it time' under Design"},
    {"letter": "E", "change": "antennal drive: JO-E cells only, per-side rates scaled so both antennae deliver the same total cell-Hz",
     "removes": "(d) the 203 : 132 rootSide split read a headwind as wind from about +31 deg; (f) JO-C and JO-E co-activated",
     "named_in_v1_report": "'Chosen wind encoding' under simulator limits and 'equalise JO drive per side' under Design"},
    {"letter": "F", "change": "matched control 'shuffled': odour on, the same wind drive from a random angle each step; the constant-rate control is dropped",
     "removes": "(e) the v1 nowind control delivered about twice the total JO drive the wind-sense flies received",
     "named_in_v1_report": "'a control whose total drive matches the heading average, not what a downwind-facing fly received'"},
    {"letter": "G", "change": "window rules preregistered as primary: full windows only for P1; losses after an encounter with full windows for P2",
     "removes": "v1 clipped windows at the trace end and counted losses of a plume the fly was born in (disclosed, sensitivity-checked)",
     "named_in_v1_report": "the sensitivity checks in v1's JSON and 'each preregistered separately' under Design"},
    {"letter": "H", "change": "budget: 10 seeds, 150 minutes, ladder 10 -> 8, never below 8",
     "removes": "nothing; the budget follows from A (twice the brain time per step)",
     "named_in_v1_report": "the seed ladder in v1's run record"},
)

SIMULATOR_LIMITS = (
    "uniform 0.275 mV synapses",
    "no conduction delays",
    "no receptor adaptation, although adaptation matters for real plume tracking",
    "the brain is restarted from rest at every 50 ms world step (FlyBrain.run resets membrane potentials "
    "and refractory state and nothing is fed back), so the controller has no memory beyond the fly's pose; "
    "surge and cast are history-defined and can only appear here as a static difference between the motor "
    "map above and below the odour threshold",
)
SIMULATOR_LIMITS_V2 = (
    "uniform 0.275 mV synapses",
    "no conduction delays",
    "no receptor adaptation, although adaptation matters for real plume tracking",
    f"the brain runs {SIM_STEPS_V2} LIF steps ({SIM_STEPS_V2 * 0.2:.0f} ms) per 50 ms of world and carries its state "
    "across the steps of a trial (protocol v2 A); it is still less than real time, and the motor rates are "
    "counts over that window",
)
DESIGN_LIMITS = (   # v1: properties of the protocol, disclosed next to the numbers they qualify
    "the start band y in [0.10, 0.20] at x = 0.45 lies inside the plume (meander-free concentration 0.18-0.52 "
    "against the 0.05 threshold) and the plume is warmed up for 6 s before the trial (plume.py, not in the "
    "design text), so encounters and losses are re-entries as the meander sweeps over a nearly stationary "
    "fly, not odour onsets from clean air",
    "P1/P2 windows are clipped at the trace end; an event in the last second gets a short 'after' window; "
    "per-event window lengths are recorded and full-window-only and post-encounter-only versions are "
    "reported as non-preregistered sensitivity checks",
    "P4 is not a discriminating test: reaching the source needs 0.37 m in 20 s at 0.02 m/s, i.e. 92.5 % of "
    "top speed straight upwind for the whole trial",
    "the steering readout is one DNa02 cell per side counted over a 12 ms window: rates move in 83.3 Hz "
    "quanta, the turn command in multiples of 0.185 and the heading in multiples of 1.67 deg per world "
    "step, so heading-change spreads and |heading rate| are readout-limited",
    "JO-C/E cells root 203 left : 132 right, so the summed drive is left-heavy at every heading (a headwind "
    "carries the left-minus-right total of wind from about +31 deg on equal populations; the control about "
    "+17 deg); JO-C and JO-E are co-activated; the control's 50 Hz per cell matches the heading-averaged "
    "input, not the ~15 Hz per cell a downwind-facing fly received, so odour-versus-nowind differences mix "
    "direction with total JO drive (realised totals are reported per condition)",
)
DESIGN_LIMITS_V2 = (
    "the start rule puts the fly 3.2 sd outside the meander-free plume, but the meander swings the plume onto "
    "the start band on some seeds (9 of the first 50, measured before the run and kept as fixed); trials that "
    "start above the threshold are counted and reported in summary.start_rule, never discarded",
    f"the motor low-pass ({SMOOTH_TAU_S * 1000:.0f} ms) is a chosen stand-in for body inertia: it spreads one "
    "spike's 1.67 deg heading quantum over several steps but does not add resolution to the readout; the raw "
    "commands are recorded so the smoothing can be undone in analysis",
    "only JO-E cells are driven (a choice between two opposite-direction classes); the per-side equalisation "
    "makes the summed drive symmetric but the cosine encoding and the 45 deg antenna offset remain chosen",
    "the shuffled control matches the wind drive's per-step statistics, not the exact distribution of angles "
    "the odour fly experienced; the realised total JO cell-Hz per condition is reported and the report says "
    "whether odour and shuffled were within 15 %",
    "24 ms of brain per 50 ms of world is still not real time; no receptor adaptation, no delays, uniform "
    "synapses, the roamer's motor mapping and 450 Hz scale, calibration.CHOSEN gains and the DoOR odour "
    "drive are kept from v1 unchanged",
)

CONSTANTS = {
    "DT": DT, "THRESHOLD": THRESHOLD, "MIN_BELOW_S": MIN_BELOW_S,
    "MIN_ABOVE_S": MIN_ABOVE_S, "SURGE_WINDOW_S": SURGE_WINDOW_S,
    "CAST_WINDOW_S": CAST_WINDOW_S, "SE_MULTIPLE": SE_MULTIPLE,
    "ARENA": ARENA, "SOURCE": SOURCE, "SOURCE_RADIUS": SOURCE_RADIUS,
    "CONDITIONS": {k: {"odour": v[0], "wind_sense": v[1]} for k, v in CONDITIONS.items()},
    "PAIRS": PAIRS, "DEFAULT_SEEDS": DEFAULT_SEEDS, "DEFAULT_STEPS": DEFAULT_STEPS,
    "SEED_LADDER": SEED_LADDER, "MIN_SEEDS": MIN_SEEDS, "BUDGET_S": BUDGET_S,
    "QUICK_SEEDS": QUICK_SEEDS, "QUICK_STEPS": QUICK_STEPS, "SIM_STEPS": SIM_STEPS,
    "ODORANT": ODORANT, "RAM_FLOOR_GB": RAM_FLOOR_GB,
    "SNAPSHOT_STEPS": SNAPSHOT_STEPS, "SNAPSHOT_GRID": SNAPSHOT_GRID,
    "V2": {
        "CONDITIONS": {k: {"odour": v[0], "wind_sense": v[1], "shuffled_wind": v[2]} for k, v in CONDITIONS_V2.items()},
        "PAIRS": PAIRS_V2, "DEFAULT_SEEDS": DEFAULT_SEEDS_V2, "SEED_LADDER": SEED_LADDER_V2, "MIN_SEEDS": MIN_SEEDS,
        "BUDGET_S": BUDGET_S_V2, "SIM_STEPS": SIM_STEPS_V2, "SMOOTH_TAU_S": SMOOTH_TAU_S,
        "JO_MATCH_TOLERANCE": JO_MATCH_TOLERANCE, "P1_RULE": PROTOCOLS["v2"]["p1_rule"], "P2_RULE": PROTOCOLS["v2"]["p2_rule"],
    },
}


# ---- small helpers ----------------------------------------------------------

def wrap(a):
    """Angle(s) wrapped into (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def stats(values):
    """n, mean and SE of the finite values. SE is nan below two values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(len(v))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    se = float(v.std(ddof=1) / math.sqrt(n)) if n >= 2 else float("nan")
    return {"n": n, "mean": float(v.mean()), "se": se}


def verdict(effect, se, n=None):
    """
    The preregistered bar: the effect must exceed SE_MULTIPLE standard errors
    in the predicted (positive) direction. Callers flip the sign for a
    prediction of "smaller". Fewer than two values, or a non-finite SE, is
    undetermined, never supported.
    """
    if n is not None and n < 2:
        return "undetermined"
    if effect is None or se is None or not np.isfinite(effect) or not np.isfinite(se):
        return "undetermined"
    return "supported" if effect > SE_MULTIPLE * se else "not supported"


def verdict_fraction(a, b):
    """P4 has no SE bar: the odour fraction simply has to exceed the blank one."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return "undetermined"
    return "supported" if a > b else "not supported"


def _clean(obj):
    """JSON-ready: numpy scalars to Python, nan/inf to None, arrays to lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def module_constants(mod):
    """Uppercase, simple-typed attributes of a module: its CHOSEN constants, for disclosure."""
    out = {}
    for name in dir(mod):
        if not name.isupper() or name.startswith("_"):
            continue
        v = getattr(mod, name)
        if isinstance(v, (int, float, str, bool, tuple, list, dict)):
            out[name] = _clean(v)
    return out


def free_ram_gb():
    """Free physical memory in GB on Windows (GlobalMemoryStatusEx); None elsewhere."""
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return None
        return s.ullAvailPhys / 2 ** 30
    except Exception:
        return None


def protocol_spec(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {tuple(PROTOCOLS)}, not {protocol!r}")
    return PROTOCOLS[protocol]


def condition_flags(protocol, cond):
    """(odour, wind_sense, shuffled) for a condition of the protocol."""
    return protocol_spec(protocol)["conditions"][cond]


def smooth_alpha(dt, tau_s):
    """
    The per-step gain of an exponential low-pass with time constant tau_s
    sampled every dt: 1 - exp(-dt / tau). None or a non-positive tau means
    no smoothing (gain 1: the world gets the raw command).
    """
    if tau_s is None or not np.isfinite(tau_s) or tau_s <= 0.0:
        return 1.0
    return float(1.0 - math.exp(-float(dt) / float(tau_s)))


def _accepts(callable_, name):
    """Whether callable_ takes a keyword argument called name (or **kwargs)."""
    try:
        params = inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def begin_trial_name(fly):
    """The name of the fly's trial-start method (BEGIN_TRIAL_NAMES), or None."""
    for name in BEGIN_TRIAL_NAMES:
        if callable(getattr(fly, name, None)):
            return name
    return None


# ---- output-path guard --------------------------------------------------------

def output_paths(out_prefix):
    """The three files a run writes for a prefix."""
    p = Path(out_prefix)
    return tuple(p.with_name(p.name + s) for s in ("_experiment.json", "_trajectories.npz", "_trajectories.png"))


def assert_not_published(out_prefix):
    """
    v1's outputs are published and stay: refuse any prefix whose files would
    land on them (compared as resolved paths, so relative spellings count).
    """
    published = {str(Path(p).resolve()) for p in PUBLISHED_V1}
    for path in output_paths(out_prefix):
        if str(Path(path).resolve()) in published:
            raise ValueError(f"refusing to write {path}: the v1 outputs {[p.name for p in PUBLISHED_V1]} are published "
                             f"and are never overwritten; use another prefix (v2 runs go to build/plume_v2)")


# ---- event detection ---------------------------------------------------------

def events(c, dt=DT, threshold=THRESHOLD, min_below_s=MIN_BELOW_S, min_above_s=MIN_ABOVE_S):
    """
    Encounter and loss sample indices by the hysteresis rule.

    above[k] is c[k] > threshold (exactly the threshold counts as below). An
    encounter is a sample that is above when the previous sample was not,
    with the run of below-samples ending at the previous sample at least
    min_below_s long. A loss is the mirror image with min_above_s. The runs
    must lie inside the trace: the fly's history before sample 0 is unknown.
    Returns (encounters, losses, run_before) where run_before[k] is the
    length of the run of same-state samples ending at k-1 for each event k.
    """
    c = np.asarray(c, dtype=float)
    above = c > threshold
    n_below = int(round(min_below_s / dt))
    n_above = int(round(min_above_s / dt))
    enc, loss, run_before = [], [], {}
    run = 0
    for k in range(len(c)):
        if k > 0 and above[k] != above[k - 1]:
            if above[k] and run >= n_below:
                enc.append(k)
                run_before[k] = run
            if (not above[k]) and run >= n_above:
                loss.append(k)
                run_before[k] = run
            run = 1
        else:
            run += 1
    return enc, loss, run_before


# ---- one trial -----------------------------------------------------------------

def _initial_state(world):
    st = getattr(world, "state", None)
    if callable(st):
        st = st()
    if not isinstance(st, dict):
        x, y, h = float(world.x), float(world.y), float(world.heading)
        st = {"x": x, "y": y, "heading": h, "t": float(getattr(world, "t", 0.0)),
              "c": float(world.concentration(x, y)),
              "reached": bool(getattr(world, "reached", False)),
              "wall_contacts": int(getattr(world, "wall_contacts", 0))}
    return st


def _fly_step(fly, c, phi, wind_sense, seed):
    out = fly.step(c, phi, wind_sense=wind_sense, seed=seed)
    if isinstance(out, dict):
        info = dict(out)
        turn, speed = info.pop("turn"), info.pop("speed")
    elif len(out) >= 3:
        turn, speed, info = out[0], out[1], out[2]
    else:
        turn, speed = out[0], out[1]
        info = {}
    return float(turn), float(speed), (info or {})


def step_seed(seed, k):
    """The brain seed for world step k of trial `seed`: paired across conditions."""
    return int((int(seed) * 1_000_003 + int(k)) % (2 ** 31 - 1))


def trial_seed(seed):
    """The brain seed a state-carrying fly is given at the start of trial `seed` (v2): step 0's seed."""
    return step_seed(seed, 0)


def shuffle_seed(seed):
    """The seed of the shuffled condition's random wind angles for trial `seed`: paired by seed, distinct from the brain's."""
    return int((int(seed) * 1_000_003 + 999_983) % (2 ** 31 - 1))


def run_trial(world, fly, steps, seed, wind_sense, condition=None, shuffled=False,
              smooth_tau_s=None, state_carry=False):
    """
    Drive one world with one fly for up to `steps` world steps, stopping when
    the world says the source is reached. Returns a trial dict with the
    per-sample arrays (steps_run + 1 samples: the start and every step after)
    and the per-step commands and brain rates.

    shuffled      the fly is given a wind angle drawn uniformly on (-pi, pi]
                  each step (its own seeded stream, shuffle_seed) instead of
                  the true one; both are recorded (phi_fly, phi).
    smooth_tau_s  the commands the world receives are the fly's raw commands
                  passed through an exponential low-pass of this time
                  constant, from rest (0, 0) at the trial start; None gives
                  the world the raw commands. Both are recorded.
    state_carry   the fly's begin_trial (or a synonym) is called once with
                  trial_seed(seed) before the first step; a fly without one
                  is refused, since the protocol needs the reset.
    """
    st = _initial_state(world)
    heading_unit = str(getattr(world, "heading_unit", "rad")).lower()
    to_rad = math.pi / 180.0 if heading_unit.startswith("deg") else 1.0
    dt = float(getattr(world, "dt", DT))
    alpha = smooth_alpha(dt, smooth_tau_s)
    begin = None
    if state_carry:
        begin = begin_trial_name(fly)
        if begin is None:
            raise TypeError(f"state carry needs the fly to have one of {BEGIN_TRIAL_NAMES}")
        getattr(fly, begin)(trial_seed(seed))
    shuffle_rng = np.random.default_rng(shuffle_seed(seed)) if shuffled else None

    t = [float(st.get("t", 0.0))]
    x, y = [float(st["x"])], [float(st["y"])]
    h = [float(st["heading"]) * to_rad]
    c = [float(st.get("c", world.concentration(x[0], y[0])))]
    phi, phi_fly, turn_cmd, speed_cmd, turn_raw, speed_raw = [], [], [], [], [], []
    rates = {}
    reached = bool(st.get("reached", False))
    wall = int(st.get("wall_contacts", 0))
    s_turn, s_speed = 0.0, 0.0
    t0 = time.perf_counter()
    k = 0
    while k < steps and not reached:
        p = float(world.wind_direction_relative(h[-1] / to_rad))
        p_fly = float(shuffle_rng.uniform(-math.pi, math.pi)) if shuffled else p
        turn, speed, info = _fly_step(fly, c[-1], p_fly, wind_sense, step_seed(seed, k))
        s_turn = s_turn + alpha * (turn - s_turn)
        s_speed = s_speed + alpha * (speed - s_speed)
        s = world.step(s_turn, s_speed)
        phi.append(p)
        phi_fly.append(p_fly)
        turn_raw.append(turn)
        speed_raw.append(speed)
        turn_cmd.append(s_turn)
        speed_cmd.append(s_speed)
        for name, v in info.items():
            if name in RUNNER_OWNED or isinstance(v, (bool, np.bool_)) \
                    or not isinstance(v, (int, float, np.integer, np.floating)):
                continue
            rates.setdefault(name, []).append(float(v))
        x.append(float(s["x"]))
        y.append(float(s["y"]))
        h.append(float(s["heading"]) * to_rad)
        c.append(float(s.get("c", world.concentration(s["x"], s["y"]))))
        t.append(float(s.get("t", t[-1] + dt)))
        reached = bool(s.get("reached", False))
        wall = int(s.get("wall_contacts", wall))
        k += 1
    n_run = len(x) - 1
    if n_run >= 1 and t[1] - t[0] > 0:
        dt = float(t[1] - t[0])          # the world's own clock wins
    c_field = getattr(world, "start_c_field", None)
    return {
        "seed": int(seed), "condition": condition, "wind_sense": bool(wind_sense),
        "shuffled": bool(shuffled), "smooth_tau_s": smooth_tau_s, "smooth_alpha": alpha,
        "state_carry": bool(state_carry), "begin_trial_method": begin,
        "dt": dt, "steps": int(steps), "steps_run": n_run,
        "reached": bool(reached), "wall_contacts": int(wall),
        "c_start_field": float(c_field) if c_field is not None else float("nan"),
        "t": np.asarray(t), "x": np.asarray(x), "y": np.asarray(y),
        "heading": np.asarray(h), "c": np.asarray(c), "phi": np.asarray(phi), "phi_fly": np.asarray(phi_fly),
        "turn": np.asarray(turn_cmd), "speed": np.asarray(speed_cmd),
        "turn_raw": np.asarray(turn_raw), "speed_raw": np.asarray(speed_raw),
        "rates": {name: np.asarray(v) for name, v in rates.items()},
        "elapsed_s": time.perf_counter() - t0,
        "sec_per_step": (time.perf_counter() - t0) / n_run if n_run else float("nan"),
    }


# ---- per-trial metrics --------------------------------------------------------

def metrics(trial, protocol=PROTOCOL_DEFAULT):
    """
    The per-trial quantities, exactly as preregistered for the protocol.
    Velocities belong to steps: step j runs from sample j to sample j+1, so
    vx[j] = (x[j+1]-x[j])/dt. An event at sample k has its "after" window on
    steps k.., and its "before" window on the steps ending at sample k.

    Every event is recorded with its window lengths and flags; the PRIMARY
    p1_surge / p2_* values follow the protocol's rule (v1: every event,
    clipped windows; v2: full windows only, and for P2 only losses that
    follow an encounter), and the other rule's values are kept under their
    own names so the two protocols can be read side by side.
    """
    spec = protocol_spec(protocol)
    dt = float(trial.get("dt", DT))
    x, y = np.asarray(trial["x"], float), np.asarray(trial["y"], float)
    h, c = np.asarray(trial["heading"], float), np.asarray(trial["c"], float)
    n_steps = len(x) - 1
    vx, vy = np.diff(x) / dt, np.diff(y) / dt
    dh = wrap(np.diff(h))
    w1 = int(round(SURGE_WINDOW_S / dt))
    w2 = int(round(CAST_WINDOW_S / dt))

    enc, loss, run_before = events(c, dt)
    first_enc = enc[0] if enc else None

    # P1: surge after each encounter. Each event's window lengths are
    # recorded so a reader can see a one-step window without reloading the
    # trajectories.
    p1_events = []
    for k in enc:
        before = -vx[max(0, k - w1):k]
        after = -vx[k:min(n_steps, k + w1)]
        if len(before) and len(after):
            s = float(after.mean() - before.mean())
            p1_events.append({"k": int(k), "t_s": float(k * dt), "n_before": int(len(before)),
                              "n_after": int(len(after)), "full_window": bool(len(before) == w1 and len(after) == w1),
                              "surge": s})

    # P2: cast after each loss, against the last 2 s inside the plume; only
    # in trials that had an encounter. Recorded per event: whether the loss
    # follows the trial's first encounter (a loss of a plume the fly was born
    # in is structurally different from losing one it found), and how much
    # of the "after" window was back above threshold.
    p2_events = []
    if enc:
        for k in loss:
            inside = run_before[k]                    # samples k-inside..k-1 were above
            lo = max(0, k - min(inside, w2))
            before_vy, before_dh = np.abs(vy[lo:k]), dh[lo:k]
            hi = min(n_steps, k + w2)
            after_vy, after_dh = np.abs(vy[k:hi]), dh[k:hi]
            if len(before_dh) >= 2 and len(after_dh) >= 2:
                dvy = float(after_vy.mean() - before_vy.mean())
                ddh = float(np.degrees(after_dh.std() - before_dh.std()))
                after_c = c[k + 1:hi + 1]
                p2_events.append({"k": int(k), "t_s": float(k * dt), "n_before": int(len(before_dh)),
                                  "n_after": int(len(after_dh)),
                                  "full_window": bool(len(before_dh) == w2 and len(after_dh) == w2),
                                  "after_first_encounter": bool(k > first_enc),
                                  "after_above_fraction": float(np.mean(after_c > THRESHOLD)) if len(after_c) else float("nan"),
                                  "vy": dvy, "dh_deg": ddh})

    def mean_or_nan(v):
        return float(np.mean(v)) if len(v) else float("nan")

    p1_any = [e["surge"] for e in p1_events]
    p1_full = [e["surge"] for e in p1_events if e["full_window"]]
    p2_any = list(p2_events)
    p2_full = [e for e in p2_events if e["full_window"]]
    p2_post = [e for e in p2_events if e["after_first_encounter"]]
    p2_full_post = [e for e in p2_events if e["full_window"] and e["after_first_encounter"]]
    p1_primary = p1_full if spec["p1_rule"] == "full_window" else p1_any
    p2_primary = p2_full_post if spec["p2_rule"] == "full_window_after_encounter" else p2_any

    ground = np.hypot(vx, vy)
    turn = np.asarray(trial.get("turn", []), float)
    speed = np.asarray(trial.get("speed", []), float)
    turn_raw = np.asarray(trial.get("turn_raw", []), float)
    speed_raw = np.asarray(trial.get("speed_raw", []), float)
    # facing upwind: cos(phi) > 0 with phi the true wind angle at the start of
    # each step; from the headings (the wind comes from -x), so a reanalysis
    # from saved trajectories gets the same number
    upwind = (-np.cos(h[:-1]) > 0.0) if n_steps else np.zeros(0, dtype=bool)
    c_field = trial.get("c_start_field")
    c_field = float(c_field) if c_field is not None else float("nan")
    out = {
        "seed": int(trial["seed"]), "condition": trial.get("condition"),
        "steps_run": int(n_steps), "reached": bool(trial.get("reached", False)),
        "wall_contacts": int(trial.get("wall_contacts", 0)),
        "x_start": float(x[0]), "x_end": float(x[-1]),
        "progress": float(x[0] - x[-1]),
        "n_encounters": len(enc), "n_losses": len(loss),
        "first_encounter_s": float(enc[0] * dt) if enc else float("nan"),
        "time_in_plume_s": float((c > THRESHOLD).sum() * dt),
        # PRIMARY under this protocol's rule
        "p1_rule": spec["p1_rule"], "p2_rule": spec["p2_rule"],
        "p1_surge": mean_or_nan(p1_primary), "n_p1_events": len(p1_primary),
        "p2_vy": mean_or_nan([e["vy"] for e in p2_primary]),
        "p2_dh_deg": mean_or_nan([e["dh_deg"] for e in p2_primary]),
        "n_p2_events": len(p2_primary),
        "mean_speed_cmd": mean_or_nan(speed), "mean_turn_cmd": mean_or_nan(turn),
        "mean_speed_raw": mean_or_nan(speed_raw), "mean_turn_raw": mean_or_nan(turn_raw),
        "mean_ground_speed_m_s": mean_or_nan(ground),
        "mean_heading_rate_deg_s": float(np.degrees(dh.mean()) / dt) if n_steps else float("nan"),
        "abs_heading_rate_deg_s": float(np.degrees(np.abs(dh).mean()) / dt) if n_steps else float("nan"),
        "facing_upwind_fraction": float(upwind.mean()) if n_steps else float("nan"),
        # disclosure: where the fly started relative to the plume, and the events in detail
        "c_start": float(c[0]) if len(c) else float("nan"),
        "starts_above_threshold": bool(len(c) and c[0] > THRESHOLD),
        "c_start_field": c_field,
        "starts_above_threshold_field": bool(c_field > THRESHOLD) if np.isfinite(c_field) else bool(len(c) and c[0] > THRESHOLD),
        "fraction_in_plume": float(np.mean(c > THRESHOLD)) if len(c) else float("nan"),
        "p1_events": p1_events, "p2_events": p2_events,
        "n_p1_events_any_window": len(p1_any), "n_p2_events_any_window": len(p2_any),
        "n_p1_events_clipped": int(sum(not e["full_window"] for e in p1_events)),
        "n_p2_events_clipped": int(sum(not e["full_window"] for e in p2_events)),
        "n_p2_before_first_encounter": int(sum(not e["after_first_encounter"] for e in p2_events)),
        "p2_after_above_fraction": mean_or_nan([e["after_above_fraction"] for e in p2_events]),
        # the other rules, named, so v1 and v2 can be read side by side
        "p1_surge_any_window": mean_or_nan(p1_any),
        "p2_vy_any_window": mean_or_nan([e["vy"] for e in p2_any]),
        "p2_dh_deg_any_window": mean_or_nan([e["dh_deg"] for e in p2_any]),
        "p1_surge_full": mean_or_nan(p1_full), "n_p1_events_full": len(p1_full),
        "p2_vy_full": mean_or_nan([e["vy"] for e in p2_full]),
        "p2_dh_deg_full": mean_or_nan([e["dh_deg"] for e in p2_full]), "n_p2_events_full": len(p2_full),
        "p2_vy_post_encounter": mean_or_nan([e["vy"] for e in p2_post]),
        "p2_dh_deg_post_encounter": mean_or_nan([e["dh_deg"] for e in p2_post]),
        "n_p2_events_post_encounter": len(p2_post),
        "p2_vy_full_post_encounter": mean_or_nan([e["vy"] for e in p2_full_post]),
        "p2_dh_deg_full_post_encounter": mean_or_nan([e["dh_deg"] for e in p2_full_post]),
        "n_p2_events_full_post_encounter": len(p2_full_post),
    }
    for name, v in (trial.get("rates") or {}).items():
        out[f"mean_{name}"] = mean_or_nan(np.asarray(v, float))
    # a reanalysis from saved trajectories has no per-step rates, only the
    # means the original run recorded; they are carried through unchanged
    for name, v in (trial.get("mean_rates") or {}).items():
        out[name] = float(v) if v is not None else float("nan")
    return out


# ---- across seeds ----------------------------------------------------------------

REPORTED = ("progress", "n_encounters", "n_losses", "wall_contacts", "time_in_plume_s",
            "steps_run", "mean_speed_cmd", "mean_turn_cmd", "mean_speed_raw", "mean_turn_raw",
            "mean_ground_speed_m_s", "mean_heading_rate_deg_s", "abs_heading_rate_deg_s",
            "facing_upwind_fraction", "p1_surge", "p2_vy", "p2_dh_deg", "first_encounter_s")
SENSITIVITY = ("p1_surge_full", "p2_vy_full", "p2_dh_deg_full", "p2_vy_post_encounter", "p2_dh_deg_post_encounter",
               "p1_surge_any_window", "p2_vy_any_window", "p2_dh_deg_any_window",
               "p2_vy_full_post_encounter", "p2_dh_deg_full_post_encounter")
CARRIED_COMMANDS = ("mean_speed_cmd", "mean_turn_cmd", "mean_speed_raw", "mean_turn_raw")   # a reanalysis carries their means when the NPZ lacks the per-step arrays
CARRIED_MEAN_OF = {"turn": "mean_turn_cmd", "speed": "mean_speed_cmd", "turn_raw": "mean_turn_raw", "speed_raw": "mean_speed_raw"}
DISCLOSED = ("c_start", "c_start_field", "fraction_in_plume", "p2_after_above_fraction")
COMMAND_KEYS = ("turn_raw", "speed_raw", "turn", "speed")
COMMAND_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
P4_DESIGN_NOTE = ("not a discriminating test under these constants: the source is 0.40 m upwind and the "
                  "reach radius 0.03 m, so a fly must cover 0.37 m in 20 s at a top speed of 0.02 m/s, i.e. "
                  "18.5 s (92.5 %) of straight full-speed upwind walking; 0/n in every condition is what "
                  "any imperfect tracker gives")
P4_DESIGN_NOTE_V2 = ("reachable by design: the source is 0.20 m upwind and the reach radius 0.03 m, so the upwind leg "
                     "is 0.17 m in 20 s at 0.02 m/s, i.e. 8.5 s (42.5 %) of full-speed walking; the start is also "
                     "0.08-0.12 m off the centreline, so the straight line to the reach radius is 0.185-0.203 m "
                     "(46-51 % of top speed) and at least 0.05 m of crosswind travel is needed: a fly that only walks "
                     "upwind passes the source (plume.start_rule_v2_check)")
P1_P2_PROTOCOL_NOTE = ("the brain is restarted from rest at every 50 ms world step, so surge and cast can only "
                       "appear as a static difference between the motor map above and below the threshold; "
                       "the fly also starts inside the plume on most seeds, so the events are re-entries, not "
                       "onsets from clean air; per-event window lengths are in per_trial[*].p1_events/p2_events "
                       "and full-window / post-encounter versions are in summary.sensitivity (not preregistered)")
P1_P2_PROTOCOL_NOTE_V2 = ("membrane state is carried across the steps of a trial (v2 A) and the commands the world "
                          "receives are low-passed (v2 B); the fly starts outside the meander-free plume, and "
                          "summary.start_rule says on how many seeds the meander put odour on the start pose "
                          "anyway; only events with full windows count (P1), and for P2 only losses that follow "
                          "an encounter; per-event details are in per_trial[*].p1_events/p2_events and v1's "
                          "window rule is in summary.sensitivity for comparison (not preregistered for v2)")


def command_distributions(trials):
    """
    The per-step command distributions pooled over the given trials: for
    turn_raw / speed_raw (the fly's outputs) and turn / speed (what the world
    received; smoothed under v2, identical to raw under v1): n, mean, sd,
    min, max, the quantiles COMMAND_QUANTILES, the fraction below zero
    (backing, or a left turn) and the fraction beyond the world's clip.
    """
    out = {}
    for key in COMMAND_KEYS:
        parts = [np.asarray(t.get(key, []), float) for t in trials]
        parts = [p for p in parts if p.size]
        if not parts:
            continue
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        q = np.quantile(v, COMMAND_QUANTILES)
        out[key] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size >= 2 else float("nan"),
                    "min": float(v.min()), "max": float(v.max()),
                    **{f"q{int(round(p * 100)):02d}": float(x) for p, x in zip(COMMAND_QUANTILES, q)},
                    "fraction_negative": float(np.mean(v < 0.0)), "fraction_beyond_clip": float(np.mean(np.abs(v) > 1.0))}
    if out:
        out["note"] = ("turn/speed are the commands the world received (low-passed under v2, equal to raw under v1); "
                       "turn_raw/speed_raw are the fly's per-step outputs; turn > 0 is a right turn, speed < 0 is "
                       "MDN above DNa01 under the roamer's mapping")
    return out


def jo_match(conditions, a="odour", b="shuffled", tolerance=JO_MATCH_TOLERANCE):
    """
    Whether two conditions delivered the same total JO drive: the relative
    difference of their mean total cell-Hz, |a - b| / mean(a, b), against
    the tolerance. The protocol says the odour and shuffled flies should be
    within 15 % of each other; this is the number the report quotes.
    """
    va = (conditions.get(a, {}).get("mean_jo_total_cell_hz") or {}).get("mean")
    vb = (conditions.get(b, {}).get("mean_jo_total_cell_hz") or {}).get("mean")
    out = {"a": a, "b": b, "a_mean_total_cell_hz": va, "b_mean_total_cell_hz": vb, "tolerance": tolerance,
           "definition": "|a - b| / mean(a, b)"}
    if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb) or (va + vb) == 0:
        out.update({"relative_difference": None, "within_tolerance": None,
                    "statement": f"{a} versus {b} total JO drive: not available"})
        return out
    rel = abs(va - vb) / ((va + vb) / 2.0)
    ok = bool(rel <= tolerance)
    out.update({"relative_difference": float(rel), "within_tolerance": ok,
                "statement": f"{a} delivered {va:.0f} and {b} {vb:.0f} cell-Hz of total JO drive on average, "
                             f"{rel * 100:.1f} % apart: {'within' if ok else 'NOT within'} the {tolerance * 100:.0f} % the protocol asked for"})
    return out


def summarise(trials_by_condition, jo_cells=None, protocol=PROTOCOL_DEFAULT):
    """
    Means, SEs, paired differences and verdicts. Pure function of its input
    and deterministic: trials are ordered by seed, and every number comes
    from the per-trial metrics. jo_cells, if given as (n_left, n_right),
    lets the realised summed JO drive per condition be computed from the
    recorded per-side means when the trials did not record it themselves.
    The protocol picks the conditions, the pairs and the window rules.
    """
    spec = protocol_spec(protocol)
    rows = {}
    for cond, trials in trials_by_condition.items():
        rows[cond] = sorted((metrics(dict(t, condition=cond), protocol) for t in trials), key=lambda r: r["seed"])
        if jo_cells is not None:
            n_l, n_r = float(jo_cells[0]), float(jo_cells[1])
            for r in rows[cond]:
                if "mean_jo_total_cell_hz" not in r and "mean_jo_left_hz" in r and "mean_jo_right_hz" in r:
                    r["mean_jo_total_cell_hz"] = r["mean_jo_left_hz"] * n_l + r["mean_jo_right_hz"] * n_r
                if "mean_jo_left_cell_hz" not in r and "mean_jo_left_hz" in r:
                    r["mean_jo_left_cell_hz"] = r["mean_jo_left_hz"] * n_l
                if "mean_jo_right_cell_hz" not in r and "mean_jo_right_hz" in r:
                    r["mean_jo_right_cell_hz"] = r["mean_jo_right_hz"] * n_r

    order = list(spec["order"]) + sorted(set(rows) - set(spec["order"]))
    conditions = {}
    for cond in order:
        if cond not in rows:
            continue
        rr = rows[cond]
        keys = list(REPORTED) + sorted(k for k in (rr[0].keys() if rr else ()) if k.startswith("mean_") and k not in REPORTED)
        summary = {k: stats([r.get(k, float("nan")) for r in rr]) for k in keys}
        summary["reached_fraction"] = float(np.mean([r["reached"] for r in rr])) if rr else float("nan")
        summary["n_reached"] = int(sum(r["reached"] for r in rr))
        summary["n_trials"] = len(rr)
        summary["n_no_encounter"] = int(sum(r["n_encounters"] == 0 for r in rr))
        summary["n_with_p2"] = int(sum(r["n_p2_events"] > 0 for r in rr))
        # disclosure: start pose relative to the plume, and the events' anatomy
        summary["n_start_above_threshold"] = int(sum(r.get("starts_above_threshold", False) for r in rr))
        summary["n_start_above_threshold_field"] = int(sum(r.get("starts_above_threshold_field", False) for r in rr))
        for k in DISCLOSED:
            summary[k] = stats([r.get(k, float("nan")) for r in rr])
        summary["p1_rule"], summary["p2_rule"] = spec["p1_rule"], spec["p2_rule"]
        summary["n_p1_events"] = int(sum(r["n_p1_events"] for r in rr))
        summary["n_p1_events_any_window"] = int(sum(r.get("n_p1_events_any_window", 0) for r in rr))
        summary["n_p1_events_clipped"] = int(sum(r.get("n_p1_events_clipped", 0) for r in rr))
        summary["n_p2_events"] = int(sum(r["n_p2_events"] for r in rr))
        summary["n_p2_events_any_window"] = int(sum(r.get("n_p2_events_any_window", 0) for r in rr))
        summary["n_p2_events_clipped"] = int(sum(r.get("n_p2_events_clipped", 0) for r in rr))
        summary["n_p2_before_first_encounter"] = int(sum(r.get("n_p2_before_first_encounter", 0) for r in rr))
        summary["n_p2_events_full_post_encounter"] = int(sum(r.get("n_p2_events_full_post_encounter", 0) for r in rr))
        # the other window rules' values, for the side-by-side reading
        for k in SENSITIVITY:
            summary[k] = stats([r.get(k, float("nan")) for r in rr])
        summary["commands"] = command_distributions(trials_by_condition[cond])
        conditions[cond] = summary

    # paired differences on shared seeds
    paired = {}
    for a, b in spec["pairs"]:
        if a not in rows or b not in rows:
            continue
        ra = {r["seed"]: r for r in rows[a]}
        rb = {r["seed"]: r for r in rows[b]}
        seeds = sorted(set(ra) & set(rb))
        diffs = [ra[s]["progress"] - rb[s]["progress"] for s in seeds]
        st = stats(diffs)
        paired[f"{a}-{b}"] = dict(st, seeds=seeds, diffs=[float(d) for d in diffs],
                                  metric="progress", verdict=verdict(st["mean"], st["se"], st["n"]))

    odour = conditions.get("odour", {})
    blank = conditions.get("blank", {})
    control = spec["pairs"][1][1] if len(spec["pairs"]) > 1 else None     # v1 nowind, v2 shuffled
    p1 = odour.get("p1_surge", stats([]))
    p2v = odour.get("p2_vy", stats([]))
    p2h = odour.get("p2_dh_deg", stats([]))
    p3 = {f"{a}-{b}": paired.get(f"{a}-{b}", {}) for a, b in spec["pairs"]}
    p4_o = odour.get("reached_fraction", float("nan"))
    p4_b = blank.get("reached_fraction", float("nan"))
    v_p2 = [verdict(p2v["mean"], p2v["se"], p2v["n"]), verdict(p2h["mean"], p2h["se"], p2h["n"])]
    v_p3 = [p3[k].get("verdict", "undetermined") for k in p3]

    def both(vs):
        if all(v == "supported" for v in vs):
            return "supported"
        if any(v == "undetermined" for v in vs) and not any(v == "not supported" for v in vs):
            return "undetermined"
        return "not supported"

    def se_multiple(st):
        return (st["mean"] / st["se"]) if st["n"] >= 2 and st["se"] and np.isfinite(st["se"]) and st["se"] > 0 else None

    def block(st, extra=None):
        d = {"effect": st["mean"], "se": st["se"], "n_trials": st["n"], "effect_in_se": se_multiple(st),
             "verdict": verdict(st["mean"], st["se"], st["n"])}
        if extra:
            d.update(extra)
        return d

    v2 = protocol == "v2"
    p1_statement = ("upwind velocity rises in the 1 s after an encounter (odour), > 2 SE" if not v2 else
                    "upwind velocity rises in the 1 s after an encounter (odour), > 2 SE; only encounters with a full 1 s window on both sides count")
    p2_statement = ("crosswind speed and heading-change spread both larger in the 2 s after a loss than the last 2 s inside, > 2 SE" if not v2 else
                    "crosswind speed and heading-change spread both larger in the 2 s after a loss than the last 2 s inside, > 2 SE; "
                    "only losses that follow an encounter and have full 2 s windows on both sides count")
    p3_statement = (f"x_start - x_end: odour > blank and odour > {control}, each > 2 SE of the paired difference" if control else
                    "x_start - x_end: odour > blank, > 2 SE of the paired difference")
    predictions = {
        "P1_surge": {"statement": p1_statement,
                     "rule": spec["p1_rule"],
                     "effect": p1["mean"], "se": p1["se"], "n_trials": p1["n"],
                     "effect_in_se": se_multiple(p1),
                     "n_events": odour.get("n_p1_events"), "n_events_any_window": odour.get("n_p1_events_any_window"),
                     "n_events_clipped": odour.get("n_p1_events_clipped"),
                     "verdict": verdict(p1["mean"], p1["se"], p1["n"]),
                     "protocol_note": P1_P2_PROTOCOL_NOTE_V2 if v2 else P1_P2_PROTOCOL_NOTE},
        "P2_cast": {"statement": p2_statement,
                    "rule": spec["p2_rule"],
                    "vy": {"effect": p2v["mean"], "se": p2v["se"], "n_trials": p2v["n"], "effect_in_se": se_multiple(p2v), "verdict": v_p2[0]},
                    "heading_change_deg": {"effect": p2h["mean"], "se": p2h["se"], "n_trials": p2h["n"], "effect_in_se": se_multiple(p2h), "verdict": v_p2[1]},
                    "n_events": odour.get("n_p2_events"), "n_events_any_window": odour.get("n_p2_events_any_window"),
                    "n_events_clipped": odour.get("n_p2_events_clipped"),
                    "n_events_before_first_encounter": odour.get("n_p2_before_first_encounter"),
                    "after_window_above_threshold_fraction": odour.get("p2_after_above_fraction", {}).get("mean"),
                    "verdict": both(v_p2),
                    "protocol_note": P1_P2_PROTOCOL_NOTE_V2 if v2 else P1_P2_PROTOCOL_NOTE},
        "P3_upwind_progress": {"statement": p3_statement,
                               **{k: {kk: p3[k].get(kk) for kk in ("mean", "se", "n", "verdict")} for k in p3},
                               "verdict": both(v_p3)},
        "P4_source_reached": {"statement": "fraction within the source radius: odour > blank",
                              "odour": p4_o, "blank": p4_b,
                              **({control: conditions.get(control, {}).get("reached_fraction", float("nan"))} if control else {}),
                              "verdict": verdict_fraction(p4_o, p4_b),
                              "design_note": P4_DESIGN_NOTE_V2 if v2 else P4_DESIGN_NOTE},
    }
    if v2:
        predictions["P3_upwind_progress"]["control_note"] = (
            "the shuffled control receives the same wind drive statistics from a random angle each step, so "
            "odour-shuffled differs in direction information only; summary.jo_drive reports the realised totals "
            "and whether odour and shuffled were within 15 %")
    else:
        predictions["P3_upwind_progress"]["confound_note"] = (
            "odour-nowind also differs in total JO drive (see conditions[*].mean_jo_total_cell_hz): "
            "the control holds 50 Hz per cell, a downwind-facing fly gets about 15 Hz per cell")

    # NOT preregistered for this protocol: the other window rules' values,
    # so a reader can see how much of P1/P2 rests on the rule. The verdicts
    # here are what the preregistered bar would say; they do not replace the
    # preregistered verdicts above.
    g = lambda k: odour.get(k, stats([]))
    if v2:
        sensitivity = {
            "note": "not preregistered for v2; v1's window rule (clipped windows, every event, losses of a plume the "
                    "fly was born in) and the single-condition rules are shown for comparison; the preregistered "
                    "verdicts in predictions stand",
            "P1_any_window_v1_rule": {"effect": g("p1_surge_any_window")["mean"], "se": g("p1_surge_any_window")["se"],
                                      "n_trials": g("p1_surge_any_window")["n"], "effect_in_se": se_multiple(g("p1_surge_any_window")),
                                      "would_be_verdict": verdict(g("p1_surge_any_window")["mean"], g("p1_surge_any_window")["se"], g("p1_surge_any_window")["n"])},
            "P2_any_window_v1_rule": {
                "vy": {"effect": g("p2_vy_any_window")["mean"], "se": g("p2_vy_any_window")["se"], "n_trials": g("p2_vy_any_window")["n"],
                       "would_be_verdict": verdict(g("p2_vy_any_window")["mean"], g("p2_vy_any_window")["se"], g("p2_vy_any_window")["n"])},
                "heading_change_deg": {"effect": g("p2_dh_deg_any_window")["mean"], "se": g("p2_dh_deg_any_window")["se"],
                                       "n_trials": g("p2_dh_deg_any_window")["n"],
                                       "would_be_verdict": verdict(g("p2_dh_deg_any_window")["mean"], g("p2_dh_deg_any_window")["se"], g("p2_dh_deg_any_window")["n"])}},
            "P2_full_windows_only": {
                "vy": {"effect": g("p2_vy_full")["mean"], "se": g("p2_vy_full")["se"], "n_trials": g("p2_vy_full")["n"],
                       "would_be_verdict": verdict(g("p2_vy_full")["mean"], g("p2_vy_full")["se"], g("p2_vy_full")["n"])},
                "heading_change_deg": {"effect": g("p2_dh_deg_full")["mean"], "se": g("p2_dh_deg_full")["se"], "n_trials": g("p2_dh_deg_full")["n"],
                                       "would_be_verdict": verdict(g("p2_dh_deg_full")["mean"], g("p2_dh_deg_full")["se"], g("p2_dh_deg_full")["n"])}},
            "P2_losses_after_an_encounter_only": {
                "vy": {"effect": g("p2_vy_post_encounter")["mean"], "se": g("p2_vy_post_encounter")["se"], "n_trials": g("p2_vy_post_encounter")["n"],
                       "would_be_verdict": verdict(g("p2_vy_post_encounter")["mean"], g("p2_vy_post_encounter")["se"], g("p2_vy_post_encounter")["n"])},
                "heading_change_deg": {"effect": g("p2_dh_deg_post_encounter")["mean"], "se": g("p2_dh_deg_post_encounter")["se"],
                                       "n_trials": g("p2_dh_deg_post_encounter")["n"],
                                       "would_be_verdict": verdict(g("p2_dh_deg_post_encounter")["mean"], g("p2_dh_deg_post_encounter")["se"], g("p2_dh_deg_post_encounter")["n"])}},
        }
    else:
        p1_full, p2v_full, p2h_full = g("p1_surge_full"), g("p2_vy_full"), g("p2_dh_deg_full")
        p2v_post, p2h_post = g("p2_vy_post_encounter"), g("p2_dh_deg_post_encounter")
        sensitivity = {
            "note": "not preregistered; sensitivity checks added in review after the first run; the preregistered "
                    "verdicts in predictions stand",
            "P1_full_windows_only": {"effect": p1_full["mean"], "se": p1_full["se"], "n_trials": p1_full["n"],
                                     "effect_in_se": se_multiple(p1_full),
                                     "would_be_verdict": verdict(p1_full["mean"], p1_full["se"], p1_full["n"])},
            "P2_full_windows_only": {
                "vy": {"effect": p2v_full["mean"], "se": p2v_full["se"], "n_trials": p2v_full["n"],
                       "would_be_verdict": verdict(p2v_full["mean"], p2v_full["se"], p2v_full["n"])},
                "heading_change_deg": {"effect": p2h_full["mean"], "se": p2h_full["se"], "n_trials": p2h_full["n"],
                                       "would_be_verdict": verdict(p2h_full["mean"], p2h_full["se"], p2h_full["n"])}},
            "P2_losses_after_an_encounter_only": {
                "vy": {"effect": p2v_post["mean"], "se": p2v_post["se"], "n_trials": p2v_post["n"],
                       "would_be_verdict": verdict(p2v_post["mean"], p2v_post["se"], p2v_post["n"])},
                "heading_change_deg": {"effect": p2h_post["mean"], "se": p2h_post["se"], "n_trials": p2h_post["n"],
                                       "would_be_verdict": verdict(p2h_post["mean"], p2h_post["se"], p2h_post["n"])}},
        }
    baseline = {k: blank.get(k) for k in ("mean_speed_cmd", "mean_turn_cmd", "mean_speed_raw", "mean_turn_raw",
                                          "mean_ground_speed_m_s", "mean_heading_rate_deg_s", "abs_heading_rate_deg_s",
                                          "facing_upwind_fraction")} if blank else {}
    if baseline:
        baseline["note"] = ("mean_speed_cmd is the roamer's mapping (mean DNa01 minus MDN rate over 450 Hz), a "
                            "convention, not an observed gait; " +
                            ("abs_heading_rate_deg_s is readout-limited: with one DNa02 cell per side over 12 ms the "
                             "heading moves in 1.67 deg quanta per step" if not v2 else
                             "under v2 the world's commands are low-passed (150 ms), so abs_heading_rate_deg_s is no "
                             "longer pinned to the readout floor; the raw commands' distribution is in conditions[*].commands"))
    disclosures = {cond: {
        "n_trials": conditions[cond]["n_trials"],
        "n_start_above_threshold": conditions[cond]["n_start_above_threshold"],
        "n_start_above_threshold_field": conditions[cond]["n_start_above_threshold_field"],
        "fraction_in_plume": conditions[cond]["fraction_in_plume"]["mean"],
        "facing_upwind_fraction": conditions[cond]["facing_upwind_fraction"]["mean"],
        "n_p1_events": conditions[cond]["n_p1_events"], "n_p1_events_any_window": conditions[cond]["n_p1_events_any_window"],
        "n_p1_events_clipped": conditions[cond]["n_p1_events_clipped"],
        "n_p2_events": conditions[cond]["n_p2_events"], "n_p2_events_any_window": conditions[cond]["n_p2_events_any_window"],
        "n_p2_events_clipped": conditions[cond]["n_p2_events_clipped"],
        "n_p2_before_first_encounter": conditions[cond]["n_p2_before_first_encounter"],
        "p2_after_window_above_threshold_fraction": conditions[cond]["p2_after_above_fraction"]["mean"],
        "mean_jo_total_cell_hz": conditions[cond].get("mean_jo_total_cell_hz", {}).get("mean"),
    } for cond in conditions}
    jo_drive = {cond: {
        "mean_total_cell_hz": conditions[cond].get("mean_jo_total_cell_hz", {}).get("mean"),
        "mean_left_cell_hz": conditions[cond].get("mean_jo_left_cell_hz", {}).get("mean"),
        "mean_right_cell_hz": conditions[cond].get("mean_jo_right_cell_hz", {}).get("mean"),
        "mean_left_hz_per_cell": conditions[cond].get("mean_jo_left_hz", {}).get("mean"),
        "mean_right_hz_per_cell": conditions[cond].get("mean_jo_right_hz", {}).get("mean"),
    } for cond in conditions}
    if control:
        jo_drive[f"odour_vs_{control}"] = jo_match(conditions, "odour", control)
    # the start rule, checked on every trial: c at t = 0 from the plume field
    # (so blank trials count too) and from what the nose got
    all_rows = [r for cond in conditions for r in rows[cond]]
    above_field = sorted({r["seed"] for r in all_rows if r.get("starts_above_threshold_field")})
    above_nose = sorted({r["seed"] for r in all_rows if r.get("starts_above_threshold")})
    start_rule = {
        "threshold": THRESHOLD, "n_trials": len(all_rows),
        "n_trials_above_threshold_field": int(sum(bool(r.get("starts_above_threshold_field")) for r in all_rows)),
        "n_trials_above_threshold_nose": int(sum(bool(r.get("starts_above_threshold")) for r in all_rows)),
        "seeds_above_threshold_field": above_field, "seeds_above_threshold_nose": above_nose,
        "expected_violations": 0 if v2 else None,
        "note": ("v2 asks for c < 0.05 at t = 0 on every trial; a violation is reported here, never discarded; "
                 "'field' is the plume at the start pose whatever the odour flag, 'nose' is what the receptors got"
                 if v2 else "v1 started the fly inside the plume by design; counts shown for the side-by-side reading"),
    }
    return {
        "protocol": protocol,
        "conditions": conditions,
        "paired": paired,
        "predictions": predictions,
        "sensitivity": sensitivity,
        "disclosures": disclosures,
        "jo_drive": jo_drive,
        "start_rule": start_rule,
        "baseline_blank": baseline,
        "no_encounter": {cond: conditions[cond]["n_no_encounter"] for cond in conditions},
        "per_trial": [r for cond in conditions for r in rows[cond]],
    }


# ---- v1 beside v2 ------------------------------------------------------------------

COMPARISON_NOTE = ("v1 (the published build/plume_experiment.json, unchanged) beside this run. The second control "
                   "differs (v1 nowind held every JO cell at 50 Hz; v2 shuffled gives the wind drive a random angle "
                   "each step), the P1/P2 window rules differ (v1 clipped windows and every loss; v2 full windows and "
                   "losses after an encounter), the start, the brain's memory and the antennal drive differ; the rows "
                   "are what each protocol measured under its own rule and are not the same experiment")


def facing_upwind_from_npz(npz_path):
    """Per condition, the mean over trials of the fraction of steps facing upwind, from saved headings."""
    out = {}
    with np.load(npz_path) as z:
        for i in range(len(z["length"])):
            L = int(z["length"][i])
            if L < 2:
                continue
            h = np.asarray(z["heading"][i, :L - 1], float)
            out.setdefault(str(z["condition"][i]), []).append(float(np.mean(-np.cos(h) > 0.0)))
    return {cond: float(np.mean(v)) for cond, v in out.items()}


def v1_comparison(v1_json_path, summary, run_info=None):
    """
    The side-by-side table: every headline number of the published v1 run
    read from its JSON (and its facing-upwind fraction from the NPZ next to
    it, which the v1 JSON did not record) beside the same quantity of this
    run's summary. Rows carry the protocol each number was measured under;
    a v1 number that does not exist is None, never invented.
    """
    p = Path(v1_json_path)
    if not p.exists():
        return {"source": str(p), "available": False, "rows": [], "note": "v1 JSON not found; no comparison"}
    old = json.loads(p.read_text(encoding="utf-8"))
    os_, ns = old.get("summary", {}), summary
    control_old = "nowind"
    pairs_new = [k for k in ns.get("paired", {})]
    control_new = next((k.split("-", 1)[1] for k in pairs_new if k != "odour-blank"), None)
    rows = []

    def row(metric, a, b, note=None):
        r = {"metric": metric, "v1": _clean(a), "v2": _clean(b)}
        if note:
            r["note"] = note
        rows.append(r)

    op, np_ = os_.get("predictions", {}), ns.get("predictions", {})
    for key, label in (("effect", "effect (m/s upwind, after - before)"), ("se", "SE"), ("n_trials", "n trials"),
                       ("n_events", "n events"), ("verdict", "verdict")):
        row(f"P1 surge: {label}", _dig(op, ["P1_surge", key]), _dig(np_, ["P1_surge", key]),
            "v1 rule: every encounter, clipped windows; v2 rule: full windows only" if key in ("effect", "n_events") else None)
    for part, label in (("vy", "crosswind speed (m/s, after - before)"), ("heading_change_deg", "heading-change spread (deg, after - before)")):
        for key in ("effect", "se", "verdict"):
            row(f"P2 cast: {label}: {key}", _dig(op, ["P2_cast", part, key]), _dig(np_, ["P2_cast", part, key]),
                "v1 rule: every loss, clipped windows; v2 rule: losses after an encounter with full windows" if key == "effect" else None)
    row("P2 cast: n events", _dig(op, ["P2_cast", "n_events"]), _dig(np_, ["P2_cast", "n_events"]))
    row("P2 cast: verdict", _dig(op, ["P2_cast", "verdict"]), _dig(np_, ["P2_cast", "verdict"]))
    for key in ("mean", "se", "n", "verdict"):
        row(f"P3 progress odour - blank (m): {key}", _dig(op, ["P3_upwind_progress", "odour-blank", key]),
            _dig(np_, ["P3_upwind_progress", "odour-blank", key]))
    for key in ("mean", "se", "n", "verdict"):
        row(f"P3 progress odour - second control (m): {key}", _dig(op, ["P3_upwind_progress", f"odour-{control_old}", key]),
            _dig(np_, ["P3_upwind_progress", f"odour-{control_new}", key]) if control_new else None,
            f"v1 control: {control_old} (50 Hz per JO cell, no direction); v2 control: {control_new} (same drive statistics, random direction)")
    row("P3: verdict", _dig(op, ["P3_upwind_progress", "verdict"]), _dig(np_, ["P3_upwind_progress", "verdict"]))
    for key in ("odour", "blank", "verdict"):
        row(f"P4 source reached: {key}", _dig(op, ["P4_source_reached", key]), _dig(np_, ["P4_source_reached", key]))
    row("P4 source reached: second control", _dig(op, ["P4_source_reached", control_old]),
        _dig(np_, ["P4_source_reached", control_new]) if control_new else None)

    oc, nc = os_.get("conditions", {}), ns.get("conditions", {})
    upwind_old = {}
    npz_old = p.with_name(p.name.replace("_experiment.json", "_trajectories.npz"))
    if npz_old.exists():
        try:
            upwind_old = facing_upwind_from_npz(npz_old)
        except Exception:
            upwind_old = {}
    cond_pairs = [("odour", "odour"), ("blank", "blank"), (control_old, control_new)]
    for co, cn in cond_pairs:
        label = co if co == cn else f"second control ({co} / {cn})"
        a, b = oc.get(co, {}), (nc.get(cn, {}) if cn else {})
        for key, name in (("progress", "progress (m upwind)"), ("n_encounters", "encounters per trial"),
                          ("n_losses", "losses per trial"), ("time_in_plume_s", "time in plume (s)"),
                          ("wall_contacts", "wall contacts"), ("mean_ground_speed_m_s", "ground speed (m/s)"),
                          ("mean_heading_rate_deg_s", "heading rate (deg/s)"), ("abs_heading_rate_deg_s", "|heading rate| (deg/s)"),
                          ("mean_speed_cmd", "speed command to the world"), ("mean_turn_cmd", "turn command to the world"),
                          ("mean_jo_total_cell_hz", "total JO drive (cell-Hz)")):
            row(f"{label}: {name}: mean", _dig(a, [key, "mean"]), _dig(b, [key, "mean"]))
            row(f"{label}: {name}: SE", _dig(a, [key, "se"]), _dig(b, [key, "se"]))
        row(f"{label}: facing upwind (fraction of steps)", upwind_old.get(co), _dig(b, ["facing_upwind_fraction", "mean"]),
            "v1 from the published trajectories' headings (not in the v1 JSON)")
        row(f"{label}: trials starting above threshold", a.get("n_start_above_threshold"), b.get("n_start_above_threshold"))
        row(f"{label}: n trials", a.get("n_trials"), b.get("n_trials"))
        row(f"{label}: reached", a.get("n_reached"), b.get("n_reached"))
    orun = old.get("run", {})
    nrun = run_info or {}
    for key in ("seeds", "steps", "sim_steps", "sec_per_brain_run", "elapsed_s"):
        a, b = orun.get(key), nrun.get(key)
        if key == "seeds":
            a, b = (len(a) if isinstance(a, list) else a), (len(b) if isinstance(b, list) else b)
        row(f"run: {key}", a, b)
    return {"source": str(p), "available": True, "note": COMPARISON_NOTE,
            "v1_control": control_old, "v2_control": control_new, "rows": rows}


# ---- picture ----------------------------------------------------------------------

def _ordered(trials_by_condition, protocol=None):
    order = protocol_spec(protocol)["order"] if protocol else CONDITION_ORDER
    known = [c for c in order if c in trials_by_condition]
    # a condition set from another protocol still renders, in its own order
    for o in PROTOCOLS.values():
        for c in o["order"]:
            if c in trials_by_condition and c not in known:
                known.append(c)
    return known + sorted(c for c in trials_by_condition if c not in known)


DEFAULT_PLUME_LABEL = "plume: seed 0 at t = 0 (each seed's plume meanders differently and moves during the trial)"


def render(trials_by_condition, path, plume=None, arena=ARENA, source=SOURCE,
           source_radius=SOURCE_RADIUS, plume_label=DEFAULT_PLUME_LABEL, protocol=None):
    """
    One PNG: the arena, a plume picture (an (ny, nx) array over the arena, if
    given) captioned by plume_label so nobody reads one seed's snapshot as
    the plume every fly saw, and every trial's path, one panel per condition.
    matplotlib if it imports, else PIL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conds = _ordered(trials_by_condition, protocol)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        return _render_pil(trials_by_condition, path, plume, arena, source, source_radius, conds, plume_label)

    L, W = arena
    fig, axes = plt.subplots(1, max(1, len(conds)), figsize=(5.2 * max(1, len(conds)), 3.7), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for ax, cond in zip(axes[0], conds or [None]):
        if plume is not None:
            ax.imshow(np.asarray(plume, float), extent=(0, L, 0, W), origin="lower",
                      cmap="Greens", vmin=0, vmax=1, alpha=0.85, aspect="equal", interpolation="bilinear")
        ax.add_patch(Rectangle((0, 0), L, W, fill=False, lw=1.0, ec="black"))
        ax.add_patch(Circle(source, source_radius, fill=False, lw=1.2, ec="red"))
        ax.plot([source[0]], [source[1]], "r*", ms=8)
        ax.annotate("wind", xy=(0.14, W - 0.03), xytext=(0.03, W - 0.03),
                    arrowprops=dict(arrowstyle="->", color="0.3"), color="0.3", fontsize=8, va="center")
        trials = trials_by_condition.get(cond, []) if cond else []
        n_reached = 0
        for i, tr in enumerate(sorted(trials, key=lambda t: t.get("seed", 0))):
            xs, ys = np.asarray(tr["x"], float), np.asarray(tr["y"], float)
            col = cmap(i % 20)
            ax.plot(xs, ys, lw=0.9, color=col, alpha=0.9)
            ax.plot(xs[:1], ys[:1], "o", ms=3, color=col)
            if tr.get("reached"):
                n_reached += 1
                ax.plot(xs[-1:], ys[-1:], "*", ms=7, color=col, mec="black")
            else:
                ax.plot(xs[-1:], ys[-1:], "s", ms=3, color=col, mec="black")
        ax.set_xlim(-0.01, L + 0.01)
        ax.set_ylim(-0.01, W + 0.01)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m); wind blows toward +x")
        ax.set_title(f"{cond}: {len(trials)} trials, {n_reached} reached", fontsize=10)
    axes[0][0].set_ylabel("y (m)")
    if plume is not None and plume_label:
        fig.text(0.5, 0.01, plume_label, ha="center", va="bottom", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _render_pil(trials_by_condition, path, plume, arena, source, source_radius, conds, plume_label=DEFAULT_PLUME_LABEL):
    from PIL import Image, ImageDraw
    L, W = arena
    scale = 800
    pw, ph = int(L * scale), int(W * scale)
    margin = 20
    n = max(1, len(conds))
    img = Image.new("RGB", (n * (pw + margin) + margin, ph + 2 * margin + 36), "white")
    draw = ImageDraw.Draw(img)
    if plume is not None and plume_label:
        draw.text((margin, margin + ph + 20), plume_label, fill=(60, 60, 60))

    def px(xv, yv, ox):
        return (ox + xv * scale, margin + (W - yv) * scale)

    if plume is not None:
        p = np.clip(np.asarray(plume, float), 0, 1)
        tile = Image.fromarray((255 - 120 * p[::-1]).astype(np.uint8)).resize((pw, ph)).convert("RGB")
    for i, cond in enumerate(conds or [None]):
        ox = margin + i * (pw + margin)
        if plume is not None:
            img.paste(tile, (ox, margin))
        draw.rectangle([ox, margin, ox + pw, margin + ph], outline="black")
        sx, sy = px(source[0], source[1], ox)
        r = source_radius * scale
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline="red")
        for tr in trials_by_condition.get(cond, []) if cond else []:
            pts = [px(float(a), float(b), ox) for a, b in zip(tr["x"], tr["y"])]
            if len(pts) >= 2:
                draw.line(pts, fill=(30, 60, 200), width=1)
        draw.text((ox, margin + ph + 4), f"{cond}: {len(trials_by_condition.get(cond, []))} trials", fill="black")
    img.save(path, "PNG")
    return path


def plume_snapshot(world, steps=SNAPSHOT_STEPS, grid=SNAPSHOT_GRID, arena=ARENA):
    """Concentration over the arena after `steps` world steps with the fly standing still: (ny, nx)."""
    for _ in range(steps):
        world.step(0.0, 0.0)
    nx, ny = grid
    if callable(getattr(world, "snapshot", None)):
        _, _, field = world.snapshot(nx, ny)
        return np.clip(np.asarray(field, float), 0, 1)
    xs = (np.arange(nx) + 0.5) / nx * arena[0]
    ys = (np.arange(ny) + 0.5) / ny * arena[1]
    try:
        X, Y = np.meshgrid(xs, ys)
        field = np.asarray(world.concentration(X, Y), float)
        if field.shape != (ny, nx):
            raise ValueError
    except Exception:
        field = np.array([[float(world.concentration(xv, yv)) for xv in xs] for yv in ys])
    return np.clip(field, 0, 1)


AVERAGE_EVERY = 10   # world steps between the samples of the time-averaged plume picture


def plume_average(world_factory, seeds, steps, every=AVERAGE_EVERY, grid=SNAPSHOT_GRID, arena=ARENA):
    """
    The plume the flies of this run actually saw, on average: mean over the
    given seeds and over the trial (sampled every `every` world steps, the
    fly standing still) of clip(field, 0, 1). One seed's snapshot at t = 0
    misleads because every seed meanders differently and the plume moves
    during the 20 s; this picture is honest about where odour was, and its
    label says what it is. Returns (field (ny, nx), label).
    """
    seeds = list(seeds)
    acc, n = None, 0
    for seed in seeds:
        world = world_factory(seed, odour=True)
        for k in range(0, int(steps) + 1):
            if k % every == 0:
                f = plume_snapshot(world, steps=0, grid=grid, arena=arena)
                acc = f if acc is None else acc + f
                n += 1
            if k < steps:
                world.step(0.0, 0.0)
    if acc is None:
        return None, ""
    label = (f"plume: mean of clip(c, 0, 1) over seeds {seeds[0]}..{seeds[-1]} and trial time "
             f"(sampled every {every * DT:.1f} s); each fly saw its own seed's meander" if len(seeds) > 1 else
             f"plume: mean of clip(c, 0, 1) over trial time for seed {seeds[0]} (sampled every {every * DT:.1f} s)")
    return acc / n, label


# ---- outputs ---------------------------------------------------------------------------

PER_STEP_SAVED = ("turn", "speed", "turn_raw", "speed_raw", "phi", "phi_fly")   # per-step arrays saved beside the samples when present


def save_trajectories(trials_by_condition, path, protocol=None):
    """
    Padded arrays (nan past each trial's end) with a length, seed and
    condition per row. Per-sample arrays (x, y, heading, c, t) always; the
    per-step commands and wind angles (PER_STEP_SAVED, one shorter) when the
    trials carry them, so a reanalysis can rebuild the command distributions.
    """
    conds = _ordered(trials_by_condition, protocol)
    trials = [(c, t) for c in conds for t in sorted(trials_by_condition[c], key=lambda t: t["seed"])]
    n = len(trials)
    m = max((len(t["x"]) for _, t in trials), default=0)
    arrs = {k: np.full((n, m), np.nan) for k in ("x", "y", "heading", "c", "t")}
    lengths = np.zeros(n, dtype=np.int64)
    for i, (_, t) in enumerate(trials):
        L = len(t["x"])
        lengths[i] = L
        for k in arrs:
            arrs[k][i, :L] = np.asarray(t[k], float)
    extra = {}
    for k in PER_STEP_SAVED:
        if any(np.asarray(t.get(k, []), float).size for _, t in trials):
            a = np.full((n, max(m, 1)), np.nan)
            for i, (_, t) in enumerate(trials):
                v = np.asarray(t.get(k, []), float)
                a[i, :v.size] = v
            extra[k] = a
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, length=lengths,
                        seed=np.array([t["seed"] for _, t in trials], dtype=np.int64),
                        condition=np.array([c for c, _ in trials], dtype=str),
                        reached=np.array([bool(t["reached"]) for _, t in trials]),
                        c_start_field=np.array([float(t.get("c_start_field", float("nan"))) if t.get("c_start_field") is not None
                                                else float("nan") for _, t in trials]),
                        **arrs, **extra)
    return path


class Log:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "a", encoding="utf-8")
        else:
            self.f = None

    def __call__(self, line):
        line = f"{datetime.now().strftime('%H:%M:%S')} {line}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def ladder(requested, sec_per_step, steps, n_conditions, budget_s=BUDGET_S, seed_ladder=SEED_LADDER, min_seeds=MIN_SEEDS):
    """
    How many seeds the budget allows, walking seed_ladder downward from the
    requested count and never below min_seeds. Steps are never cut. A request
    at or below min_seeds is left alone.
    """
    if requested <= min_seeds or not np.isfinite(sec_per_step):
        return requested, "kept"
    for n in [k for k in seed_ladder if k <= requested]:
        if sec_per_step * steps * n_conditions * n <= budget_s:
            return n, "kept" if n == requested else f"dropped {requested} -> {n}"
    return min_seeds, f"dropped {requested} -> {min_seeds} (floor)"


EXPERIMENT_TITLE = ("plume tracking by the connectome with the roamer's calibrated type gains "
                    "(calibration.CHOSEN; no learning, no fitting to this task), walking fly, wind tunnel")


def jo_cells_of(run_info):
    """
    (n_left, n_right) driven JO cells from the run record's fly description,
    or None: the v2 fly names its JO-E counts (jo_e_left / jo_e_right, or
    jo_driven_left / jo_driven_right) and v1's fly names jo_left / jo_right.
    """
    fi = (run_info.get("constants") or {}).get("fly_instance") or {}
    for l, r in (("jo_e_left", "jo_e_right"), ("jo_driven_left", "jo_driven_right"), ("jo_left", "jo_right")):
        if l in fi and r in fi:
            return int(fi[l]), int(fi[r])
    return None


def write_outputs(trials_by_condition, run_info, out_prefix, plume=None, partial=False, render_kw=None,
                  plume_label=DEFAULT_PLUME_LABEL, protocol=None, v1_json=None, allow_published=False):
    """
    The JSON, the NPZ and the PNG for a prefix. The protocol defaults to the
    run record's (else v1). The published v1 prefix is refused unless
    allow_published is set (a reanalysis that keeps the numbers). A v2 run
    carries the v1 comparison table and the list of protocol changes.
    """
    out_prefix = Path(out_prefix)
    if not allow_published:
        assert_not_published(out_prefix)
    protocol = protocol or run_info.get("protocol") or PROTOCOL_DEFAULT
    spec = protocol_spec(protocol)
    v2 = protocol == "v2"
    summary = summarise(trials_by_condition, jo_cells=jo_cells_of(run_info), protocol=protocol)
    payload = {
        "experiment": EXPERIMENT_TITLE,
        "protocol": protocol,
        "partial": bool(partial),
        "constants": run_info.get("constants", {"runner": CONSTANTS}),
        "run": {k: v for k, v in run_info.items() if k != "constants"},
        "simulator_limits": list(SIMULATOR_LIMITS_V2 if v2 else SIMULATOR_LIMITS),
        "design_limits": list(DESIGN_LIMITS_V2 if v2 else DESIGN_LIMITS),
        "conventions": {
            "progress": "x_start - x_end in metres; positive is upwind",
            "turn": "the turn command the world received; positive is a right turn",
            "turn_raw": "the fly's per-step turn output before the low-pass (equal to turn under v1)",
            "speed_raw": "the fly's per-step speed output before the low-pass (equal to speed under v1)",
            "phi": "the true angle the wind comes from relative to the heading, radians, 0 = headwind, positive = from the left",
            "phi_fly": "the angle the fly's antennae were given (differs from phi only in the shuffled condition)",
            "facing_upwind_fraction": "fraction of steps with cos(phi) > 0, from the headings",
            "heading_rate": "degrees per second of the world's heading, counter-clockwise positive, so a right turn is negative",
            "p2_dh_deg": "population standard deviation of the per-step wrapped heading change, in degrees, after minus before",
            "se": "sample standard deviation (one degree of freedom lost) over the square root of n",
            "rates": f"every motor, ORN and JO rate is a spike count over one {spec['sim_steps'] * 0.2:.0f} ms brain window divided by "
                     f"its duration, so a single cell's rate moves in {1000.0 / (spec['sim_steps'] * 0.2):.1f} Hz quanta; "
                     "see constants.fly_instance.readout_quanta",
            "mean_jo_total_cell_hz": "sum over all driven JO cells of their drive rate (left mean x n_left + right mean x n_right): "
                                     "the total mechanosensory input the brain received, comparable across conditions",
            "sensitivity": "summary.sensitivity is NOT preregistered; the verdicts in summary.predictions are",
        },
        "plume_picture": plume_label if plume is not None else None,
        "summary": summary,
    }
    if v2:
        payload["protocol_changes"] = [dict(c) for c in PROTOCOL_V2_CHANGES]
        payload["v1_comparison"] = v1_comparison(v1_json or V1_JSON, summary, payload["run"])
    json_path, npz_path, png_path = output_paths(out_prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_clean(payload), indent=1), encoding="utf-8")
    save_trajectories(trials_by_condition, npz_path, protocol=protocol)
    try:
        render(trials_by_condition, png_path, plume=plume, plume_label=plume_label, protocol=protocol, **(render_kw or {}))
    except Exception as e:  # a picture must never lose the numbers
        print(f"render failed: {e!r}", flush=True)
        png_path = None
    return summary, json_path, npz_path, png_path


def load_trials(npz_path, json_path=None):
    """
    The trials of a finished run, rebuilt from its trajectories file so the
    summary can be recomputed without a brain. The per-step brain rates are
    not in the NPZ; the per-trial means the run recorded are taken from the
    JSON's per_trial rows (if given) and carried through as mean_rates. The
    per-step commands and wind angles are read back when the NPZ has them
    (v2 runs); otherwise their means are carried like the rates.
    """
    means, walls = {}, {}
    if json_path is not None and Path(json_path).exists():
        old = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for r in old.get("summary", {}).get("per_trial", []):
            key = (r["condition"], int(r["seed"]))
            # brain-rate means, and the command means whose per-step values
            # may not be in the NPZ either; everything else is recomputed
            means[key] = {k: v for k, v in r.items()
                          if k.startswith("mean_") and (k not in REPORTED or k in CARRIED_COMMANDS)}
            walls[key] = int(r.get("wall_contacts", 0))
    trials_by_condition = {}
    with np.load(npz_path) as z:
        have = set(z.files)
        for i in range(len(z["length"])):
            L = int(z["length"][i])
            cond, seed = str(z["condition"][i]), int(z["seed"][i])
            t = np.asarray(z["t"][i, :L], float)
            dt = float(t[1] - t[0]) if L >= 2 else DT
            carried = dict(means.get((cond, seed), {}))
            tr = {
                "seed": seed, "condition": cond, "dt": dt, "steps": L - 1, "steps_run": L - 1,
                "reached": bool(z["reached"][i]), "wall_contacts": walls.get((cond, seed), 0),
                "c_start_field": float(z["c_start_field"][i]) if "c_start_field" in have else None,
                "t": t, "x": np.asarray(z["x"][i, :L], float), "y": np.asarray(z["y"][i, :L], float),
                "heading": np.asarray(z["heading"][i, :L], float), "c": np.asarray(z["c"][i, :L], float),
                "phi": np.zeros(L - 1), "turn": np.array([]), "speed": np.array([]), "rates": {},
            }
            for k in PER_STEP_SAVED:
                if k in have:
                    tr[k] = np.asarray(z[k][i, :max(L - 1, 0)], float)
                    carried.pop(CARRIED_MEAN_OF.get(k, ""), None)     # recomputed from the array instead
            tr["mean_rates"] = carried
            trials_by_condition.setdefault(cond, []).append(tr)
    return trials_by_condition


PREREGISTERED_KEYS = (("P1_surge", "effect"), ("P1_surge", "se"), ("P1_surge", "verdict"),
                      ("P2_cast", "vy", "effect"), ("P2_cast", "vy", "se"), ("P2_cast", "heading_change_deg", "effect"),
                      ("P2_cast", "heading_change_deg", "se"), ("P2_cast", "verdict"),
                      ("P3_upwind_progress", "odour-blank", "mean"), ("P3_upwind_progress", "odour-blank", "se"),
                      ("P3_upwind_progress", "verdict"), ("P4_source_reached", "odour"), ("P4_source_reached", "blank"),
                      ("P4_source_reached", "verdict"))
PREREGISTERED_KEYS_CONTROL = {   # the second pair's keys, by protocol
    "v1": (("P3_upwind_progress", "odour-nowind", "mean"), ("P3_upwind_progress", "odour-nowind", "se")),
    "v2": (("P3_upwind_progress", "odour-shuffled", "mean"), ("P3_upwind_progress", "odour-shuffled", "se")),
}


def _dig(d, keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def compare_preregistered(old_predictions, new_predictions, tol=1e-9, protocol=PROTOCOL_DEFAULT):
    """Every preregistered number and verdict, old against new: (all_equal, max_abs_diff, mismatches)."""
    worst, mismatches = 0.0, []
    for keys in PREREGISTERED_KEYS + PREREGISTERED_KEYS_CONTROL.get(protocol, ()):
        a, b = _dig(old_predictions, keys), _dig(new_predictions, keys)
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                mismatches.append(".".join(keys))
        elif a is None and b is None:
            continue
        elif a is None or b is None or not np.isfinite(float(a)) or not np.isfinite(float(b)):
            if not ((a is None or not np.isfinite(float(a))) and (b is None or not np.isfinite(float(b)))):
                mismatches.append(".".join(keys))
        else:
            d = abs(float(a) - float(b))
            worst = max(worst, d)
            if d > tol:
                mismatches.append(".".join(keys))
    return (not mismatches), worst, mismatches


def ladder_note(run_info):
    """
    What the run record says about how the seed count was decided. The runner's
    own ladder only acts on the seeds it was asked for; if the operator asked
    for fewer seeds or a bigger budget than the design's, that decision was
    made outside the runner, and the record says so, with what the runner's
    rule would have done at the design budget. The design is the protocol's.
    """
    spec = protocol_spec(run_info.get("protocol") or PROTOCOL_DEFAULT)
    d_seeds, d_budget, d_ladder = spec["default_seeds"], spec["budget_s"], spec["seed_ladder"]
    req, budget = run_info.get("requested_seeds"), run_info.get("budget_s")
    steps = run_info.get("steps", DEFAULT_STEPS)
    rate = run_info.get("sec_per_brain_run")
    n_cond = len(spec["order"])
    if run_info.get("quick") or req is None:
        return None
    parts = []
    if req != d_seeds:
        parts.append(f"requested_seeds {req} differs from the design's {d_seeds}: set by hand before launch, outside the runner's ladder")
    if budget is not None and abs(float(budget) - d_budget) > 1e-6:
        parts.append(f"budget_s {float(budget):.0f} differs from the design's {d_budget:.0f}: set by hand before launch")
    if rate is not None and np.isfinite(rate):
        n_design, why_design = ladder(d_seeds, float(rate), int(steps), n_cond, d_budget, d_ladder, spec["min_seeds"])
        parts.append(f"at the measured {float(rate):.3f} s/run the runner's ladder from the design's {d_seeds} seeds "
                     f"and {d_budget / 60:.0f} min budget would have given {n_design} seeds ({why_design})")
        if req is not None and req != d_seeds:
            n_req, why_req = ladder(int(req), float(rate), int(steps), n_cond, d_budget, d_ladder, spec["min_seeds"])
            parts.append(f"from the requested {req} at the design budget it would have given {n_req} ({why_req})")
    return "; ".join(parts) if parts else "as designed"


def make_world(plume_mod, seed, odour, protocol=PROTOCOL_DEFAULT):
    """
    plume.World for a trial. v1 calls World(seed, odour=...) as it always did
    (a World that takes protocol= is told "v1"); v2 needs the protocol
    argument and refuses a World without it, because the start rule is the
    protocol.
    """
    World = plume_mod.World
    world_protocol = protocol_spec(protocol)["world_protocol"]
    if _accepts(World, "protocol"):
        return World(seed, odour=odour, protocol=world_protocol)
    if protocol == "v2":
        raise TypeError("protocol v2 needs plume.World(seed, odour=..., protocol=...); this World has no protocol argument")
    return World(seed, odour=odour)


def make_fly(plume_fly_mod, fb, gains, protocol=PROTOCOL_DEFAULT, **fly_kw):
    """
    plume_fly.PlumeFly for the protocol. v2 passes protocol="v2" (the JO-E
    equalised drive and the state carry live in the fly) and refuses a fly
    that does not accept it or has no trial-start method.
    """
    PlumeFly = plume_fly_mod.PlumeFly
    if protocol == "v2":
        if not _accepts(PlumeFly, "protocol"):
            raise TypeError("protocol v2 needs plume_fly.PlumeFly(..., protocol='v2'); this PlumeFly has no protocol argument")
        fly = PlumeFly(fb, gains, protocol="v2", **fly_kw)
        if begin_trial_name(fly) is None:
            raise TypeError(f"protocol v2 needs the fly to reset its carried state at each trial start via one of {BEGIN_TRIAL_NAMES}")
        return fly
    if _accepts(PlumeFly, "protocol"):
        # a fly whose defaults are v2's must be told v1 to reproduce the published coupling
        return PlumeFly(fb, gains, protocol="v1", **fly_kw)
    return PlumeFly(fb, gains, **fly_kw)


def reanalyse(prefix, note=None, allow_published=False):
    """
    Recompute the summary and the picture of a finished run from its saved
    trajectories, without a brain, and rewrite the JSON with the original run
    record kept and a reanalysis entry added. The preregistered numbers and
    verdicts are compared with the old JSON and the comparison is recorded:
    a reanalysis that changed them would be a different experiment. The
    protocol is the one the run was made under.
    """
    prefix = out_prefix_from(prefix)
    if not allow_published:
        assert_not_published(prefix)
    json_path, npz_path, _ = output_paths(prefix)
    old = json.loads(json_path.read_text(encoding="utf-8"))
    protocol = old.get("protocol") or PROTOCOL_DEFAULT
    spec = protocol_spec(protocol)
    trials = load_trials(npz_path, json_path)
    run_info = dict(old.get("run", {}))
    run_info["constants"] = old.get("constants", {"runner": CONSTANTS})
    run_info["protocol"] = protocol
    run_info["design_seeds"] = spec["default_seeds"]
    run_info["design_budget_s"] = spec["budget_s"]
    run_info["design_steps"] = DEFAULT_STEPS
    ln = ladder_note(run_info)
    if ln:
        run_info["seed_decision"] = ln
    seeds = sorted({int(t["seed"]) for c in trials for t in trials[c]})
    steps = max(int(t["steps_run"]) for c in trials for t in trials[c])
    world_c = (old.get("constants") or {}).get("world") or {}
    render_kw = {"arena": (float(world_c.get("ARENA_X", ARENA[0])), float(world_c.get("ARENA_Y", ARENA[1]))),
                 "source": tuple(float(v) for v in world_c.get("SOURCE", SOURCE)),
                 "source_radius": float(world_c.get("REACH_RADIUS", SOURCE_RADIUS))}
    pic, label = None, DEFAULT_PLUME_LABEL
    try:
        import plume
        pic, label = plume_average(lambda seed, odour=True: make_world(plume, seed, odour, protocol), seeds, steps)
    except Exception as e:
        print(f"plume picture skipped: {e!r}", flush=True)
    # the summary first, so the comparison can be recorded in the same JSON
    new_summary = summarise(trials, jo_cells=jo_cells_of(run_info), protocol=protocol)
    same, worst, mismatches = compare_preregistered(old.get("summary", {}).get("predictions", {}),
                                                    new_summary["predictions"], protocol=protocol)
    history = list(run_info.get("reanalyses", []))
    history.append({
        "when": datetime.now().isoformat(timespec="seconds"),
        "from": str(npz_path), "runner": "plume_experiment.reanalyse",
        "why": note or "disclosures and sensitivity checks added in review; no trial was rerun",
        "preregistered_metrics_unchanged": bool(same), "max_abs_difference": float(worst),
        "mismatches": mismatches,
    })
    run_info["reanalyses"] = history
    return write_outputs(trials, run_info, prefix, plume=pic, plume_label=label, render_kw=render_kw,
                         protocol=protocol, allow_published=allow_published)


# ---- main -------------------------------------------------------------------------------

def normalise_argv(argv):
    """
    Accept key=value tokens (quick=1 out=path seeds=10) next to --key value
    flags: each becomes --key value, with underscores read as dashes.
    """
    out = []
    for a in argv:
        if "=" in a and not a.startswith("-"):
            k, v = a.split("=", 1)
            out.extend([f"--{k.strip().replace('_', '-')}", v])
        else:
            out.append(a)
    return out


def out_prefix_from(path, quick=False):
    """
    The output prefix behind --out. A bare prefix (build/plume) is used as is;
    a path that names the JSON itself (build/plume_experiment.json) is cut
    back to the prefix so the outputs land at exactly that JSON and its
    sibling _trajectories.npz/.png. Quick mode appends _quick unless the
    prefix already ends with it, so quick outputs never overwrite a real run.
    """
    p = Path(path)
    name = p.name
    if name.lower().endswith(".json"):
        name = name[:-5]
    if name.endswith("_experiment"):
        name = name[: -len("_experiment")]
    if quick and not name.endswith("_quick"):
        name = name + "_quick"
    return p.with_name(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", default=PROTOCOL_DEFAULT, choices=sorted(PROTOCOLS),
                    help="v1: the published protocol (default, unchanged); v2: continuous brain, motor low-pass, start outside the plume, JO-E equalised drive, shuffled control")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (0..n-1), paired across conditions (default: the protocol's)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="world steps per trial (0.05 s each)")
    ap.add_argument("--out", default=None, help="output prefix: <out>_experiment.json, <out>_trajectories.npz/.png (default build/plume or build/plume_v2)")
    ap.add_argument("--quick", type=int, default=0, help="1: smoke test, 2 seeds x 80 steps, outputs under <out>_quick")
    ap.add_argument("--odorant", default=ODORANT)
    ap.add_argument("--log", default=None, help="progress log path (default <out>_experiment.log)")
    ap.add_argument("--odour-hz", type=float, default=None, help="passed to PlumeFly if given")
    ap.add_argument("--wind-hz", type=float, default=None, help="passed to PlumeFly if given")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="wall-clock budget the seed ladder is judged against after the first seed (default: the protocol's)")
    ap.add_argument("--reanalyse", default=None, metavar="PREFIX_OR_JSON",
                    help="recompute the summary and picture of a finished run from its saved trajectories (no brain)")
    ap.add_argument("--note", default=None, help="why the reanalysis was done; recorded in the JSON")
    ap.add_argument("--allow-published", type=int, default=0,
                    help="1: let --reanalyse rewrite the published v1 files (numbers are checked unchanged); new runs never may")
    args = ap.parse_args(normalise_argv(sys.argv[1:] if argv is None else argv))
    protocol = args.protocol
    spec = protocol_spec(protocol)
    budget_s = float(args.budget_min) * 60.0 if args.budget_min is not None else float(spec["budget_s"])

    if args.reanalyse:
        try:
            summary, json_path, npz_path, png_path = reanalyse(args.reanalyse, args.note, allow_published=bool(args.allow_published))
        except ValueError as e:
            print(str(e), flush=True)
            return 4
        for name, p in summary["predictions"].items():
            print(f"{name}: {p['verdict']}", flush=True)
        print(f"rewrote {json_path}, {npz_path}, {png_path}", flush=True)
        return 0

    seeds_n = args.seeds if args.seeds is not None else spec["default_seeds"]
    steps = args.steps
    out_prefix = out_prefix_from(args.out if args.out is not None else str(BUILD / spec["default_out"]), quick=bool(args.quick))
    try:
        assert_not_published(out_prefix)          # before anything is written or loaded
    except ValueError as e:
        print(str(e), flush=True)
        return 4
    if args.quick:
        seeds_n, steps = QUICK_SEEDS, QUICK_STEPS
    log = Log(args.log or out_prefix.with_name(out_prefix.name + "_experiment.log"))
    log(f"protocol {protocol}: conditions {list(spec['order'])}, sim_steps {spec['sim_steps']}, "
        f"smoothing tau {spec['smooth_tau_s']}, state carry {spec['state_carry']}, outputs {out_prefix}_*")

    free = free_ram_gb()
    if free is not None and free < RAM_FLOOR_GB:
        log(f"refusing to load a brain: {free:.1f} GB free, floor {RAM_FLOOR_GB} GB")
        log.close()
        return 2
    log(f"free RAM {free:.1f} GB" if free is not None else "free RAM unknown (not Windows); proceeding")

    import calibration
    import flysim
    import plume
    import plume_fly

    t_load = time.perf_counter()
    fb = flysim.FlyBrain()
    gains = calibration.gains_for(fb, calibration.CHOSEN)
    fly_kw = dict(odorant=args.odorant, sim_steps=spec["sim_steps"], seed=0)
    if args.odour_hz is not None:
        fly_kw["odour_hz"] = args.odour_hz
    if args.wind_hz is not None:
        fly_kw["wind_hz"] = args.wind_hz
    # the motor low-pass (B) is applied once, here in run_trial, which records
    # the raw and the smoothed commands; a fly with a smoother of its own is
    # asked to pass its commands through so they are not smoothed twice
    smoothing_by = "none" if spec["smooth_tau_s"] is None else "runner"
    if spec["smooth_tau_s"] is not None and _accepts(plume_fly.PlumeFly, "smooth_tau_s"):
        fly_kw["smooth_tau_s"] = 0.0
    try:
        fly = make_fly(plume_fly, fb, gains, protocol, **fly_kw)
        make_world(plume, 0, True, protocol)      # the world must speak the protocol too
    except TypeError as e:
        log(f"refusing to run: {e}")
        log.close()
        return 3
    log(f"brain loaded in {time.perf_counter() - t_load:.1f} s: {fb.n} neurons, setting {calibration.CHOSEN}, odorant {args.odorant!r}")

    if callable(getattr(fly, "describe", None)):
        fly_instance = _clean(fly.describe())
    else:
        fly_instance = {k: _clean(v) for k, v in vars(fly).items()
                        if isinstance(v, (int, float, str, bool)) and not k.startswith("_")}
    constants = {"runner": CONSTANTS, "protocol": protocol, "protocol_spec": _clean(spec),
                 "world": module_constants(plume), "fly": module_constants(plume_fly),
                 "fly_instance": fly_instance}
    if callable(getattr(plume, "start_band_straight_plume", None)):
        try:
            constants["world_start_band"] = _clean(plume.start_band_straight_plume())
        except Exception as e:
            log(f"start band disclosure skipped: {e!r}")
    if protocol == "v2" and callable(getattr(plume, "start_rule_v2_check", None)):
        try:
            constants["world_start_rule_v2"] = _clean(plume.start_rule_v2_check(50))
            log(f"start rule v2 over 50 seeds: {constants['world_start_rule_v2']['n_start_above_threshold']} start above "
                f"threshold at t = 0 (seeds {constants['world_start_rule_v2']['seeds_above_threshold']}); kept as fixed, reported")
        except Exception as e:
            log(f"start rule disclosure skipped: {e!r}")
    render_kw = {
        "arena": (float(getattr(plume, "ARENA_X", ARENA[0])), float(getattr(plume, "ARENA_Y", ARENA[1]))),
        "source": tuple(float(v) for v in getattr(plume, "SOURCE", SOURCE)),
        "source_radius": float(getattr(plume, "REACH_RADIUS", SOURCE_RADIUS)),
    }
    # the brain setting is a choice: record the multipliers and what they touch,
    # not only the label
    brain_gains, gain_types = None, None
    settings = getattr(calibration, "SETTINGS", None)
    if isinstance(settings, dict) and calibration.CHOSEN in settings:
        brain_gains = dict(settings[calibration.CHOSEN].get("gains", {}))
        if callable(getattr(calibration, "matched_types", None)):
            try:
                gain_types = {g: len(calibration.matched_types(fb, g)) for g in brain_gains}
            except Exception as e:
                log(f"gain type counts skipped: {e!r}")
    run_info = {
        "constants": constants, "protocol": protocol,
        "requested_seeds": seeds_n, "steps": steps, "odorant": args.odorant,
        "design_seeds": spec["default_seeds"], "design_steps": DEFAULT_STEPS, "design_budget_s": spec["budget_s"],
        "brain_setting": calibration.CHOSEN, "brain_gains": brain_gains, "brain_gain_type_counts": gain_types,
        "brain_gains_note": "per-type multipliers on outgoing synaptic weights, chosen for the roamer before this "
                            "experiment (calibration.CHOSEN); wiring is the connectome's, nothing is fitted here",
        "sim_steps": spec["sim_steps"], "smooth_tau_s": spec["smooth_tau_s"], "smooth_alpha": smooth_alpha(DT, spec["smooth_tau_s"]),
        "smoothing_applied_by": smoothing_by,
        "smoothing_note": "the low-pass is applied once, by the runner (run_trial), which records raw and smoothed commands; "
                          "the fly's own Smoother is set to pass through (smooth_tau_s=0) so fly_instance.smoothing reads "
                          "passthrough by design, not by omission",
        "shuffled_wind_applied_by": "runner",
        "shuffled_wind_note": "the shuffled condition's wind angle is drawn by the runner (shuffle_seed(seed), uniform on "
                              "(-pi, pi] each step) and handed to the fly as its phi in wind mode, so the fly's own "
                              "shuffled mode is not used; the fly's phi_drive equals the runner's phi_fly",
        "state_carry": spec["state_carry"], "begin_trial_method": begin_trial_name(fly) if spec["state_carry"] else None,
        "quick": bool(args.quick),
        "budget_s": budget_s, "started": datetime.now().isoformat(timespec="seconds"),
        "seed_ladder": "pending",
    }
    if protocol == "v2":
        run_info["protocol_changes"] = [dict(c) for c in PROTOCOL_V2_CHANGES]

    plume_pic = None
    try:
        plume_pic = plume_snapshot(make_world(plume, 0, True, protocol))
    except Exception as e:
        log(f"plume snapshot skipped: {e!r}")

    order = list(spec["order"])
    trials_by_condition = {c: [] for c in order}
    seeds = list(range(seeds_n))
    t_start = time.perf_counter()
    total = len(seeds) * len(order)
    done = 0
    i_seed = 0
    n_start_violations = 0
    while i_seed < len(seeds):
        seed = seeds[i_seed]
        for cond in order:
            odour, wind_sense, shuffled = spec["conditions"][cond]
            world = make_world(plume, seed, odour, protocol)
            tr = run_trial(world, fly, steps, seed, wind_sense, condition=cond, shuffled=shuffled,
                           smooth_tau_s=spec["smooth_tau_s"], state_carry=spec["state_carry"])
            trials_by_condition[cond].append(tr)
            done += 1
            m = metrics(tr, protocol)
            above = m["starts_above_threshold_field"]
            n_start_violations += int(bool(above))
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done)
            log(f"trial {done}/{total} seed={seed} cond={cond} steps={tr['steps_run']} reached={int(tr['reached'])} "
                f"x_end={tr['x'][-1]:.3f} progress={m['progress']:+.3f} encounters={m['n_encounters']} "
                f"losses={m['n_losses']} walls={tr['wall_contacts']} in_plume={m['time_in_plume_s']:.1f}s "
                f"upwind={m['facing_upwind_fraction']:.2f} c0={m['c_start_field']:.3f}{' ABOVE-THRESHOLD-AT-START' if above else ''} "
                f"{tr['sec_per_step']:.3f}s/step elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")
        if i_seed == 0:
            # the budget decision, once, on the first seed's trials
            rate = float(np.mean([trials_by_condition[c][0]["sec_per_step"] for c in order]))
            run_info["sec_per_brain_run"] = rate
            n_keep, why = ladder(len(seeds), rate, steps, len(order), budget_s, spec["seed_ladder"], spec["min_seeds"])
            run_info["seed_ladder"] = why
            decision = ladder_note(run_info)
            if decision:
                run_info["seed_decision"] = decision
            log(f"budget: {rate:.3f} s/run projects {rate * steps * len(order) * seeds_n / 60:.0f} min "
                f"for {seeds_n} seeds against {budget_s / 60:.0f} min; seeds {why}"
                + (f"; {decision}" if decision and decision != "as designed" else ""))
            if n_keep != len(seeds):
                seeds = seeds[:n_keep]
                total = len(seeds) * len(order)
        i_seed += 1
        run_info["seeds"] = seeds[:i_seed]
        run_info["elapsed_s"] = time.perf_counter() - t_start
        run_info["n_start_above_threshold_so_far"] = n_start_violations
        if i_seed < len(seeds):
            write_outputs(trials_by_condition, run_info, out_prefix, plume=plume_pic, partial=True,
                          render_kw=render_kw, protocol=protocol)

    run_info["seeds"] = seeds
    run_info["finished"] = datetime.now().isoformat(timespec="seconds")
    run_info["elapsed_s"] = time.perf_counter() - t_start
    run_info.pop("n_start_above_threshold_so_far", None)
    # the final picture shows the plume the run's flies actually saw on average,
    # not one seed's snapshot; the partial pictures above were captioned as such
    plume_label = DEFAULT_PLUME_LABEL
    try:
        avg, label = plume_average(lambda s, odour=True: make_world(plume, s, odour, protocol), seeds, steps)
        if avg is not None:
            plume_pic, plume_label = avg, label
    except Exception as e:
        log(f"time-averaged plume skipped: {e!r}")
    summary, json_path, npz_path, png_path = write_outputs(trials_by_condition, run_info, out_prefix,
                                                           plume=plume_pic, render_kw=render_kw,
                                                           plume_label=plume_label, protocol=protocol)
    for name, p in summary["predictions"].items():
        log(f"{name}: {p['verdict']}")
    log(f"no encounter: {summary['no_encounter']}")
    sr = summary["start_rule"]
    log(f"start rule: {sr['n_trials_above_threshold_field']} of {sr['n_trials']} trials began above the threshold "
        f"(seeds {sr['seeds_above_threshold_field']})")
    for key, block in summary["jo_drive"].items():
        if key.startswith("odour_vs_"):
            log(f"JO drive: {block['statement']}")
    log(f"wrote {json_path}, {npz_path}, {png_path}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

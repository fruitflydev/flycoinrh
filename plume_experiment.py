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

WHAT THIS PROTOCOL CANNOT SHOW (found in review after the first run; none of
these changes a trial, so the first run stands, and each is disclosed in the
JSON next to the number it qualifies)
  * The controller is memoryless: FlyBrain.run restarts every neuron from
    rest at each 50 ms world step and nothing is carried over, so the motor
    command is a stochastic function of (c, phi, seed) at that step. Surge
    and cast are history-defined; here they can only show as a static
    difference between the motor map above and below the odour threshold.
    A negative P1/P2 is therefore a result about this protocol as much as
    about the connectome.
  * The fly starts inside the plume (plume.py explains the arithmetic), so
    the "encounters" P1 and P2 count are re-entries after the meander swept
    the plume off a nearly stationary fly, not odour onsets from clean air;
    the JSON reports how many trials began above threshold, the fraction of
    samples in plume, and how many P2 losses came before any encounter.
  * Windows are clipped at the trace end (a preregistered rule, meant for
    early termination at the source, which never happened); an encounter in
    the last second of a trial gets a one-step "after" window. The JSON
    records every event's window lengths and, as a labelled non-preregistered
    sensitivity check, the same metrics on full windows only and on losses
    that follow an encounter. The preregistered verdicts are never replaced.
  * P4 is not a discriminating test: reaching the source needs 92.5 % of top
    speed straight upwind for the whole trial.
  * The readout is quantised (one DNa02 cell per side counted over 12 ms:
    1.67 deg of heading per world step per spike difference) and the JO
    drive is asymmetric by population (203 left : 132 right cells) and, in
    the control, matched to the heading-averaged input rather than to what a
    downwind-facing fly received; plume_fly.py has the details, describe()
    the numbers, and the JSON carries the realised summed JO drive per
    condition.

MEASURED, by this runner
  * per trial: encounters and losses of the plume, the surge after an
    encounter, the cast after a loss, upwind progress, whether the source was
    reached, wall contacts, time in plume, the mean motor commands;
  * across seeds: means, standard errors, paired differences between
    conditions on the same seeds, and each prediction's verdict.

CHOSEN, and said so in the output
  * the concentration threshold 0.05 and the hysteresis (0.5 s below before an
    encounter, 0.25 s above before a loss), the 1 s surge window, the 2 s cast
    window, and "more than 2 standard errors" as the bar every prediction has
    to clear (SE = sd / sqrt(n), sd with one degree of freedom lost);
  * windows are clipped at the ends of the trace and use the samples that
    exist (a trial ends early when the fly reaches the source); the history a
    hysteresis rule needs must lie inside the trace, so nothing can be an
    encounter before sample 10 or a loss before sample 5;
  * a trial with zero encounters contributes nothing to P1 or P2 and is
    counted; inside such a trial a loss is never used;
  * heading change is the wrapped difference of successive headings, in
    radians internally, reported in degrees; its spread within a window is
    the population standard deviation;
  * the budget ladder: if the first seed's three trials say the whole run
    would exceed the budget (BUDGET_S, or --budget-min), the seed count drops
    12 -> 10 -> 8 and never lower, and the steps per trial are never cut;
    seeds are the outer loop so every kept seed has all three conditions.

THE PREDICTIONS (fixed before data)
  P1 surge   mean upwind velocity (-vx) in the 1 s after an encounter minus
             the 1 s before it, odour condition: > 0 by more than 2 SE.
  P2 cast    mean crosswind speed |vy| and the spread of heading change in the
             2 s after a loss versus the last 2 s inside the plume: both larger
             after the loss by more than 2 SE.
  P3 upwind  x_start - x_end: odour > blank and odour > nowind, each by more
             than 2 SE of the paired difference.
  P4 source  fraction of trials ending within the source radius: odour > blank.
Reported without prediction: baseline speed and turning bias with no odour,
encounters per trial, wall contacts, time in plume, and how many trials had no
encounter at all.

INTERFACES (plume.py and plume_fly.py; the runner was written against the
design while they were being written, so it reads them defensively)
  World(seed, odour=True): concentration(x, y); wind_direction_relative(heading)
    (radians, the angle the wind comes FROM relative to the heading, 0 =
    headwind, positive = from the fly's left); step(turn, speed) -> {x, y,
    heading, c, t, reached, wall_contacts, ...}; `state` (a property, or a
    method) for the state before the first step, else the attributes x, y,
    heading, t are read; heading in radians unless a heading_unit attribute
    says "deg"; dt is read off the world's clock (t after one step), else
    0.05. turn > 0 is a RIGHT turn (heading, counter-clockwise from +x,
    decreases), so a right-turning bias shows as a positive mean turn command
    and a negative mean heading rate.
  PlumeFly(fb, gains, odorant=..., odour_hz=..., wind_hz=..., sim_steps=...,
    seed=...): step(c, phi, seed=..., wind_sense=True) -> (turn, speed, info)
    where info holds scalar rates (six motor rates, mean ORN rate, JO rate per
    side, fired count); a dict with turn and speed keys, or a bare (turn,
    speed) pair, are accepted too. describe() lists its constants.

KNOWN SIMULATOR LIMITS, disclosed in the output: uniform 0.275 mV synapses,
no conduction delays, no receptor adaptation, although adaptation matters
for real plume tracking, and the restart from rest at every world step.

  py plume_experiment.py --quick 1            smoke test, 2 seeds x 80 steps
  py plume_experiment.py                      12 seeds x 3 conditions x 400 steps
  py plume_experiment.py seeds=10 out=build/plume_experiment.json log=run.log
                                              key=value tokens are accepted too, and an
                                              --out that names the JSON itself is read as
                                              the prefix (build/plume_experiment.json ->
                                              build/plume_{experiment.json,trajectories.npz,.png})
  py plume_experiment.py --reanalyse build/plume_experiment.json
                                              recompute the summary and the picture from the
                                              saved trajectories, no brain; the run record is
                                              kept and a reanalysis entry is added. The
                                              preregistered numbers are checked against the
                                              old JSON and the check is recorded.
"""
import argparse
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
CONDITIONS = {             # name -> (odour in the world, wind sense in the fly)
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
RUNNER_OWNED = ("turn", "speed", "c", "phi", "seed", "wind_sense", "t", "step")  # echoed by the fly's info; recorded by the runner itself
SIMULATOR_LIMITS = (
    "uniform 0.275 mV synapses",
    "no conduction delays",
    "no receptor adaptation, although adaptation matters for real plume tracking",
    "the brain is restarted from rest at every 50 ms world step (FlyBrain.run resets membrane potentials "
    "and refractory state and nothing is fed back), so the controller has no memory beyond the fly's pose; "
    "surge and cast are history-defined and can only appear here as a static difference between the motor "
    "map above and below the odour threshold",
)
DESIGN_LIMITS = (   # properties of the protocol, disclosed next to the numbers they qualify
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


def run_trial(world, fly, steps, seed, wind_sense, condition=None):
    """
    Drive one world with one fly for up to `steps` world steps, stopping when
    the world says the source is reached. Returns a trial dict with the
    per-sample arrays (steps_run + 1 samples: the start and every step after)
    and the per-step commands and brain rates.
    """
    st = _initial_state(world)
    heading_unit = str(getattr(world, "heading_unit", "rad")).lower()
    to_rad = math.pi / 180.0 if heading_unit.startswith("deg") else 1.0
    dt = float(getattr(world, "dt", DT))

    t = [float(st.get("t", 0.0))]
    x, y = [float(st["x"])], [float(st["y"])]
    h = [float(st["heading"]) * to_rad]
    c = [float(st.get("c", world.concentration(x[0], y[0])))]
    phi, turn_cmd, speed_cmd = [], [], []
    rates = {}
    reached = bool(st.get("reached", False))
    wall = int(st.get("wall_contacts", 0))
    t0 = time.perf_counter()
    k = 0
    while k < steps and not reached:
        p = float(world.wind_direction_relative(h[-1] / to_rad))
        turn, speed, info = _fly_step(fly, c[-1], p, wind_sense, step_seed(seed, k))
        s = world.step(turn, speed)
        phi.append(p)
        turn_cmd.append(turn)
        speed_cmd.append(speed)
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
    return {
        "seed": int(seed), "condition": condition, "wind_sense": bool(wind_sense),
        "dt": dt, "steps": int(steps), "steps_run": n_run,
        "reached": bool(reached), "wall_contacts": int(wall),
        "t": np.asarray(t), "x": np.asarray(x), "y": np.asarray(y),
        "heading": np.asarray(h), "c": np.asarray(c), "phi": np.asarray(phi),
        "turn": np.asarray(turn_cmd), "speed": np.asarray(speed_cmd),
        "rates": {name: np.asarray(v) for name, v in rates.items()},
        "elapsed_s": time.perf_counter() - t0,
        "sec_per_step": (time.perf_counter() - t0) / n_run if n_run else float("nan"),
    }


# ---- per-trial metrics --------------------------------------------------------

def metrics(trial):
    """
    The per-trial quantities, exactly as preregistered. Velocities belong to
    steps: step j runs from sample j to sample j+1, so vx[j] = (x[j+1]-x[j])/dt.
    An event at sample k has its "after" window on steps k.., and its
    "before" window on the steps ending at sample k.
    """
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

    # P1: surge after each encounter (preregistered: clipped windows, every
    # event counts). Each event's window lengths are recorded so a reader can
    # see a one-step window without reloading the trajectories.
    surges, p1_events = [], []
    for k in enc:
        before = -vx[max(0, k - w1):k]
        after = -vx[k:min(n_steps, k + w1)]
        if len(before) and len(after):
            s = float(after.mean() - before.mean())
            surges.append(s)
            p1_events.append({"k": int(k), "t_s": float(k * dt), "n_before": int(len(before)),
                              "n_after": int(len(after)), "full_window": bool(len(before) == w1 and len(after) == w1),
                              "surge": s})

    # P2: cast after each loss, against the last 2 s inside the plume; only
    # in trials that had an encounter (preregistered). Recorded per event:
    # whether the loss follows the trial's first encounter (a loss of a plume
    # the fly was born in is structurally different from losing one it found),
    # and how much of the "after" window was back above threshold.
    cast_vy, cast_dh, p2_events = [], [], []
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
                cast_vy.append(dvy)
                cast_dh.append(ddh)
                after_c = c[k + 1:hi + 1]
                p2_events.append({"k": int(k), "t_s": float(k * dt), "n_before": int(len(before_dh)),
                                  "n_after": int(len(after_dh)),
                                  "full_window": bool(len(before_dh) == w2 and len(after_dh) == w2),
                                  "after_first_encounter": bool(k > first_enc),
                                  "after_above_fraction": float(np.mean(after_c > THRESHOLD)) if len(after_c) else float("nan"),
                                  "vy": dvy, "dh_deg": ddh})

    def mean_or_nan(v):
        return float(np.mean(v)) if len(v) else float("nan")

    # NOT preregistered: the same quantities restricted to full windows, and
    # P2 restricted to losses that follow an encounter. Disclosed as
    # sensitivity checks; they never replace the preregistered values above.
    p1_full = [e["surge"] for e in p1_events if e["full_window"]]
    p2_full = [e for e in p2_events if e["full_window"]]
    p2_post = [e for e in p2_events if e["after_first_encounter"]]

    ground = np.hypot(vx, vy)
    turn = np.asarray(trial.get("turn", []), float)
    speed = np.asarray(trial.get("speed", []), float)
    out = {
        "seed": int(trial["seed"]), "condition": trial.get("condition"),
        "steps_run": int(n_steps), "reached": bool(trial.get("reached", False)),
        "wall_contacts": int(trial.get("wall_contacts", 0)),
        "x_start": float(x[0]), "x_end": float(x[-1]),
        "progress": float(x[0] - x[-1]),
        "n_encounters": len(enc), "n_losses": len(loss),
        "first_encounter_s": float(enc[0] * dt) if enc else float("nan"),
        "time_in_plume_s": float((c > THRESHOLD).sum() * dt),
        "p1_surge": mean_or_nan(surges), "n_p1_events": len(surges),
        "p2_vy": mean_or_nan(cast_vy), "p2_dh_deg": mean_or_nan(cast_dh),
        "n_p2_events": len(cast_vy),
        "mean_speed_cmd": mean_or_nan(speed), "mean_turn_cmd": mean_or_nan(turn),
        "mean_ground_speed_m_s": mean_or_nan(ground),
        "mean_heading_rate_deg_s": float(np.degrees(dh.mean()) / dt) if n_steps else float("nan"),
        "abs_heading_rate_deg_s": float(np.degrees(np.abs(dh).mean()) / dt) if n_steps else float("nan"),
        # disclosure: where the fly started relative to the plume, and the events in detail
        "c_start": float(c[0]) if len(c) else float("nan"),
        "starts_above_threshold": bool(len(c) and c[0] > THRESHOLD),
        "fraction_in_plume": float(np.mean(c > THRESHOLD)) if len(c) else float("nan"),
        "p1_events": p1_events, "p2_events": p2_events,
        "n_p1_events_clipped": int(sum(not e["full_window"] for e in p1_events)),
        "n_p2_events_clipped": int(sum(not e["full_window"] for e in p2_events)),
        "n_p2_before_first_encounter": int(sum(not e["after_first_encounter"] for e in p2_events)),
        "p2_after_above_fraction": mean_or_nan([e["after_above_fraction"] for e in p2_events]),
        # NOT preregistered, sensitivity only
        "p1_surge_full": mean_or_nan(p1_full), "n_p1_events_full": len(p1_full),
        "p2_vy_full": mean_or_nan([e["vy"] for e in p2_full]),
        "p2_dh_deg_full": mean_or_nan([e["dh_deg"] for e in p2_full]), "n_p2_events_full": len(p2_full),
        "p2_vy_post_encounter": mean_or_nan([e["vy"] for e in p2_post]),
        "p2_dh_deg_post_encounter": mean_or_nan([e["dh_deg"] for e in p2_post]),
        "n_p2_events_post_encounter": len(p2_post),
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
            "steps_run", "mean_speed_cmd", "mean_turn_cmd", "mean_ground_speed_m_s",
            "mean_heading_rate_deg_s", "abs_heading_rate_deg_s", "p1_surge", "p2_vy",
            "p2_dh_deg", "first_encounter_s")
SENSITIVITY = ("p1_surge_full", "p2_vy_full", "p2_dh_deg_full", "p2_vy_post_encounter", "p2_dh_deg_post_encounter")
CARRIED_COMMANDS = ("mean_speed_cmd", "mean_turn_cmd")   # per-step commands are not saved; a reanalysis carries their means
DISCLOSED = ("c_start", "fraction_in_plume", "p2_after_above_fraction")
P4_DESIGN_NOTE = ("not a discriminating test under these constants: the source is 0.40 m upwind and the "
                  "reach radius 0.03 m, so a fly must cover 0.37 m in 20 s at a top speed of 0.02 m/s, i.e. "
                  "18.5 s (92.5 %) of straight full-speed upwind walking; 0/n in every condition is what "
                  "any imperfect tracker gives")
P1_P2_PROTOCOL_NOTE = ("the brain is restarted from rest at every 50 ms world step, so surge and cast can only "
                       "appear as a static difference between the motor map above and below the threshold; "
                       "the fly also starts inside the plume on most seeds, so the events are re-entries, not "
                       "onsets from clean air; per-event window lengths are in per_trial[*].p1_events/p2_events "
                       "and full-window / post-encounter versions are in summary.sensitivity (not preregistered)")


def summarise(trials_by_condition, jo_cells=None):
    """
    Means, SEs, paired differences and verdicts. Pure function of its input
    and deterministic: trials are ordered by seed, and every number comes
    from the per-trial metrics. jo_cells, if given as (n_left, n_right),
    lets the realised summed JO drive per condition be computed from the
    recorded per-side means when the trials did not record it themselves.
    """
    rows = {}
    for cond, trials in trials_by_condition.items():
        rows[cond] = sorted((metrics(dict(t, condition=cond)) for t in trials), key=lambda r: r["seed"])
        if jo_cells is not None:
            n_l, n_r = float(jo_cells[0]), float(jo_cells[1])
            for r in rows[cond]:
                if "mean_jo_total_cell_hz" not in r and "mean_jo_left_hz" in r and "mean_jo_right_hz" in r:
                    r["mean_jo_total_cell_hz"] = r["mean_jo_left_hz"] * n_l + r["mean_jo_right_hz"] * n_r

    conditions = {}
    for cond in list(CONDITION_ORDER) + sorted(set(rows) - set(CONDITION_ORDER)):
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
        for k in DISCLOSED:
            summary[k] = stats([r.get(k, float("nan")) for r in rr])
        summary["n_p1_events"] = int(sum(r["n_p1_events"] for r in rr))
        summary["n_p1_events_clipped"] = int(sum(r.get("n_p1_events_clipped", 0) for r in rr))
        summary["n_p2_events"] = int(sum(r["n_p2_events"] for r in rr))
        summary["n_p2_events_clipped"] = int(sum(r.get("n_p2_events_clipped", 0) for r in rr))
        summary["n_p2_before_first_encounter"] = int(sum(r.get("n_p2_before_first_encounter", 0) for r in rr))
        # NOT preregistered: sensitivity versions of P1 and P2
        for k in SENSITIVITY:
            summary[k] = stats([r.get(k, float("nan")) for r in rr])
        conditions[cond] = summary

    # paired differences on shared seeds
    paired = {}
    for a, b in PAIRS:
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
    p1 = odour.get("p1_surge", stats([]))
    p2v = odour.get("p2_vy", stats([]))
    p2h = odour.get("p2_dh_deg", stats([]))
    p3a = paired.get("odour-blank", {})
    p3b = paired.get("odour-nowind", {})
    p4_o = odour.get("reached_fraction", float("nan"))
    p4_b = blank.get("reached_fraction", float("nan"))
    v_p2 = [verdict(p2v["mean"], p2v["se"], p2v["n"]), verdict(p2h["mean"], p2h["se"], p2h["n"])]
    v_p3 = [p3a.get("verdict", "undetermined"), p3b.get("verdict", "undetermined")]

    def both(vs):
        if all(v == "supported" for v in vs):
            return "supported"
        if any(v == "undetermined" for v in vs) and not any(v == "not supported" for v in vs):
            return "undetermined"
        return "not supported"

    def se_multiple(st):
        return (st["mean"] / st["se"]) if st["n"] >= 2 and st["se"] and np.isfinite(st["se"]) and st["se"] > 0 else None

    p1_full = odour.get("p1_surge_full", stats([]))
    p2v_full = odour.get("p2_vy_full", stats([]))
    p2h_full = odour.get("p2_dh_deg_full", stats([]))
    p2v_post = odour.get("p2_vy_post_encounter", stats([]))
    p2h_post = odour.get("p2_dh_deg_post_encounter", stats([]))
    predictions = {
        "P1_surge": {"statement": "upwind velocity rises in the 1 s after an encounter (odour), > 2 SE",
                     "effect": p1["mean"], "se": p1["se"], "n_trials": p1["n"],
                     "effect_in_se": se_multiple(p1),
                     "n_events": odour.get("n_p1_events"), "n_events_clipped": odour.get("n_p1_events_clipped"),
                     "verdict": verdict(p1["mean"], p1["se"], p1["n"]),
                     "protocol_note": P1_P2_PROTOCOL_NOTE},
        "P2_cast": {"statement": "crosswind speed and heading-change spread both larger in the 2 s after a loss than the last 2 s inside, > 2 SE",
                    "vy": {"effect": p2v["mean"], "se": p2v["se"], "n_trials": p2v["n"], "effect_in_se": se_multiple(p2v), "verdict": v_p2[0]},
                    "heading_change_deg": {"effect": p2h["mean"], "se": p2h["se"], "n_trials": p2h["n"], "effect_in_se": se_multiple(p2h), "verdict": v_p2[1]},
                    "n_events": odour.get("n_p2_events"), "n_events_clipped": odour.get("n_p2_events_clipped"),
                    "n_events_before_first_encounter": odour.get("n_p2_before_first_encounter"),
                    "after_window_above_threshold_fraction": odour.get("p2_after_above_fraction", {}).get("mean"),
                    "verdict": both(v_p2),
                    "protocol_note": P1_P2_PROTOCOL_NOTE},
        "P3_upwind_progress": {"statement": "x_start - x_end: odour > blank and odour > nowind, each > 2 SE of the paired difference",
                               "odour-blank": {k: p3a.get(k) for k in ("mean", "se", "n", "verdict")},
                               "odour-nowind": {k: p3b.get(k) for k in ("mean", "se", "n", "verdict")},
                               "verdict": both(v_p3),
                               "confound_note": "odour-nowind also differs in total JO drive (see conditions[*].mean_jo_total_cell_hz): "
                                                "the control holds 50 Hz per cell, a downwind-facing fly gets about 15 Hz per cell"},
        "P4_source_reached": {"statement": "fraction within the source radius: odour > blank",
                              "odour": p4_o, "blank": p4_b,
                              "nowind": conditions.get("nowind", {}).get("reached_fraction", float("nan")),
                              "verdict": verdict_fraction(p4_o, p4_b),
                              "design_note": P4_DESIGN_NOTE},
    }
    # NOT preregistered. Reported so a reader can see how much of P1/P2 rests on
    # clipped windows or on losses of a plume the fly was born in. The verdicts
    # here are what the preregistered bar would say; they do not replace the
    # preregistered verdicts above.
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
    baseline = {k: blank.get(k) for k in ("mean_speed_cmd", "mean_turn_cmd", "mean_ground_speed_m_s",
                                          "mean_heading_rate_deg_s", "abs_heading_rate_deg_s")} if blank else {}
    if baseline:
        baseline["note"] = ("mean_speed_cmd is the roamer's mapping (mean DNa01 minus MDN rate over 450 Hz), a "
                            "convention, not an observed gait; abs_heading_rate_deg_s is readout-limited: with one "
                            "DNa02 cell per side over 12 ms the heading moves in 1.67 deg quanta per step")
    disclosures = {cond: {
        "n_trials": conditions[cond]["n_trials"],
        "n_start_above_threshold": conditions[cond]["n_start_above_threshold"],
        "fraction_in_plume": conditions[cond]["fraction_in_plume"]["mean"],
        "n_p1_events": conditions[cond]["n_p1_events"], "n_p1_events_clipped": conditions[cond]["n_p1_events_clipped"],
        "n_p2_events": conditions[cond]["n_p2_events"], "n_p2_events_clipped": conditions[cond]["n_p2_events_clipped"],
        "n_p2_before_first_encounter": conditions[cond]["n_p2_before_first_encounter"],
        "p2_after_window_above_threshold_fraction": conditions[cond]["p2_after_above_fraction"]["mean"],
        "mean_jo_total_cell_hz": conditions[cond].get("mean_jo_total_cell_hz", {}).get("mean"),
    } for cond in conditions}
    return {
        "conditions": conditions,
        "paired": paired,
        "predictions": predictions,
        "sensitivity": sensitivity,
        "disclosures": disclosures,
        "baseline_blank": baseline,
        "no_encounter": {cond: conditions[cond]["n_no_encounter"] for cond in conditions},
        "per_trial": [r for cond in conditions for r in rows[cond]],
    }


# ---- picture ----------------------------------------------------------------------

def _ordered(trials_by_condition):
    return [c for c in CONDITION_ORDER if c in trials_by_condition] + \
           sorted(c for c in trials_by_condition if c not in CONDITION_ORDER)


DEFAULT_PLUME_LABEL = "plume: seed 0 at t = 0 (each seed's plume meanders differently and moves during the trial)"


def render(trials_by_condition, path, plume=None, arena=ARENA, source=SOURCE,
           source_radius=SOURCE_RADIUS, plume_label=DEFAULT_PLUME_LABEL):
    """
    One PNG: the arena, a plume picture (an (ny, nx) array over the arena, if
    given) captioned by plume_label so nobody reads one seed's snapshot as
    the plume every fly saw, and every trial's path, one panel per condition.
    matplotlib if it imports, else PIL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conds = _ordered(trials_by_condition)
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

def save_trajectories(trials_by_condition, path):
    """Padded arrays (nan past each trial's end) with a length, seed and condition per row."""
    conds = _ordered(trials_by_condition)
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, length=lengths,
                        seed=np.array([t["seed"] for _, t in trials], dtype=np.int64),
                        condition=np.array([c for c, _ in trials], dtype=str),
                        reached=np.array([bool(t["reached"]) for _, t in trials]),
                        **arrs)
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


def ladder(requested, sec_per_step, steps, n_conditions, budget_s=BUDGET_S):
    """
    How many seeds the budget allows, walking SEED_LADDER downward from the
    requested count and never below MIN_SEEDS. Steps are never cut. A request
    at or below MIN_SEEDS is left alone.
    """
    if requested <= MIN_SEEDS or not np.isfinite(sec_per_step):
        return requested, "kept"
    for n in [k for k in SEED_LADDER if k <= requested]:
        if sec_per_step * steps * n_conditions * n <= budget_s:
            return n, "kept" if n == requested else f"dropped {requested} -> {n}"
    return MIN_SEEDS, f"dropped {requested} -> {MIN_SEEDS} (floor)"


EXPERIMENT_TITLE = ("plume tracking by the connectome with the roamer's calibrated type gains "
                    "(calibration.CHOSEN; no learning, no fitting to this task), walking fly, wind tunnel")


def jo_cells_of(run_info):
    """(n_left, n_right) JO-C/E cells from the run record's fly description, or None."""
    fi = (run_info.get("constants") or {}).get("fly_instance") or {}
    if "jo_left" in fi and "jo_right" in fi:
        return int(fi["jo_left"]), int(fi["jo_right"])
    return None


def write_outputs(trials_by_condition, run_info, out_prefix, plume=None, partial=False, render_kw=None,
                  plume_label=DEFAULT_PLUME_LABEL):
    out_prefix = Path(out_prefix)
    summary = summarise(trials_by_condition, jo_cells=jo_cells_of(run_info))
    payload = {
        "experiment": EXPERIMENT_TITLE,
        "partial": bool(partial),
        "constants": run_info.get("constants", {"runner": CONSTANTS}),
        "run": {k: v for k, v in run_info.items() if k != "constants"},
        "simulator_limits": list(SIMULATOR_LIMITS),
        "design_limits": list(DESIGN_LIMITS),
        "conventions": {
            "progress": "x_start - x_end in metres; positive is upwind",
            "turn": "the fly's turn command; positive is a right turn",
            "heading_rate": "degrees per second of the world's heading, counter-clockwise positive, so a right turn is negative",
            "p2_dh_deg": "population standard deviation of the per-step wrapped heading change, in degrees, after minus before",
            "se": "sample standard deviation (one degree of freedom lost) over the square root of n",
            "rates": "every motor, ORN and JO rate is a spike count over one 12 ms brain window divided by 0.012 s, "
                     "so a single cell's rate moves in 83.3 Hz quanta; see constants.fly_instance.readout_quanta",
            "mean_jo_total_cell_hz": "sum over all JO-C/E cells of their drive rate (left mean x n_left + right mean x n_right): "
                                     "the total mechanosensory input the brain received, comparable across conditions",
            "sensitivity": "summary.sensitivity is NOT preregistered; the verdicts in summary.predictions are",
        },
        "plume_picture": plume_label if plume is not None else None,
        "summary": summary,
    }
    json_path = out_prefix.with_name(out_prefix.name + "_experiment.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_clean(payload), indent=1), encoding="utf-8")
    npz_path = out_prefix.with_name(out_prefix.name + "_trajectories.npz")
    save_trajectories(trials_by_condition, npz_path)
    png_path = out_prefix.with_name(out_prefix.name + "_trajectories.png")
    try:
        render(trials_by_condition, png_path, plume=plume, plume_label=plume_label, **(render_kw or {}))
    except Exception as e:  # a picture must never lose the numbers
        print(f"render failed: {e!r}", flush=True)
        png_path = None
    return summary, json_path, npz_path, png_path


def load_trials(npz_path, json_path=None):
    """
    The trials of a finished run, rebuilt from its trajectories file so the
    summary can be recomputed without a brain. The per-step brain rates are
    not in the NPZ; the per-trial means the run recorded are taken from the
    JSON's per_trial rows (if given) and carried through as mean_rates.
    """
    means, walls = {}, {}
    if json_path is not None and Path(json_path).exists():
        old = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for r in old.get("summary", {}).get("per_trial", []):
            key = (r["condition"], int(r["seed"]))
            # brain-rate means, and the two command means whose per-step
            # values are not in the NPZ either; everything else is recomputed
            means[key] = {k: v for k, v in r.items()
                          if k.startswith("mean_") and (k not in REPORTED or k in CARRIED_COMMANDS)}
            walls[key] = int(r.get("wall_contacts", 0))
    trials_by_condition = {}
    with np.load(npz_path) as z:
        for i in range(len(z["length"])):
            L = int(z["length"][i])
            cond, seed = str(z["condition"][i]), int(z["seed"][i])
            t = np.asarray(z["t"][i, :L], float)
            dt = float(t[1] - t[0]) if L >= 2 else DT
            tr = {
                "seed": seed, "condition": cond, "dt": dt, "steps": L - 1, "steps_run": L - 1,
                "reached": bool(z["reached"][i]), "wall_contacts": walls.get((cond, seed), 0),
                "t": t, "x": np.asarray(z["x"][i, :L], float), "y": np.asarray(z["y"][i, :L], float),
                "heading": np.asarray(z["heading"][i, :L], float), "c": np.asarray(z["c"][i, :L], float),
                "phi": np.zeros(L - 1), "turn": np.array([]), "speed": np.array([]), "rates": {},
                "mean_rates": means.get((cond, seed), {}),
            }
            trials_by_condition.setdefault(cond, []).append(tr)
    return trials_by_condition


PREREGISTERED_KEYS = (("P1_surge", "effect"), ("P1_surge", "se"), ("P1_surge", "verdict"),
                      ("P2_cast", "vy", "effect"), ("P2_cast", "vy", "se"), ("P2_cast", "heading_change_deg", "effect"),
                      ("P2_cast", "heading_change_deg", "se"), ("P2_cast", "verdict"),
                      ("P3_upwind_progress", "odour-blank", "mean"), ("P3_upwind_progress", "odour-blank", "se"),
                      ("P3_upwind_progress", "odour-nowind", "mean"), ("P3_upwind_progress", "odour-nowind", "se"),
                      ("P3_upwind_progress", "verdict"), ("P4_source_reached", "odour"), ("P4_source_reached", "blank"),
                      ("P4_source_reached", "verdict"))


def _dig(d, keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def compare_preregistered(old_predictions, new_predictions, tol=1e-9):
    """Every preregistered number and verdict, old against new: (all_equal, max_abs_diff, mismatches)."""
    worst, mismatches = 0.0, []
    for keys in PREREGISTERED_KEYS:
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
    rule would have done at the design budget.
    """
    req, budget = run_info.get("requested_seeds"), run_info.get("budget_s")
    steps = run_info.get("steps", DEFAULT_STEPS)
    rate = run_info.get("sec_per_brain_run")
    if run_info.get("quick") or req is None:
        return None
    parts = []
    if req != DEFAULT_SEEDS:
        parts.append(f"requested_seeds {req} differs from the design's {DEFAULT_SEEDS}: set by hand before launch, outside the runner's ladder")
    if budget is not None and abs(float(budget) - BUDGET_S) > 1e-6:
        parts.append(f"budget_s {float(budget):.0f} differs from the design's {BUDGET_S:.0f}: set by hand before launch")
    if rate is not None and np.isfinite(rate):
        n_design, why_design = ladder(DEFAULT_SEEDS, float(rate), int(steps), len(CONDITION_ORDER), BUDGET_S)
        parts.append(f"at the measured {float(rate):.3f} s/run the runner's ladder from the design's {DEFAULT_SEEDS} seeds "
                     f"and {BUDGET_S / 60:.0f} min budget would have given {n_design} seeds ({why_design})")
        if req is not None and req != DEFAULT_SEEDS:
            n_req, why_req = ladder(int(req), float(rate), int(steps), len(CONDITION_ORDER), BUDGET_S)
            parts.append(f"from the requested {req} at the design budget it would have given {n_req} ({why_req})")
    return "; ".join(parts) if parts else "as designed"


def reanalyse(prefix, note=None):
    """
    Recompute the summary and the picture of a finished run from its saved
    trajectories, without a brain, and rewrite the JSON with the original run
    record kept and a reanalysis entry added. The preregistered numbers and
    verdicts are compared with the old JSON and the comparison is recorded:
    a reanalysis that changed them would be a different experiment.
    """
    prefix = out_prefix_from(prefix)
    json_path = prefix.with_name(prefix.name + "_experiment.json")
    npz_path = prefix.with_name(prefix.name + "_trajectories.npz")
    old = json.loads(json_path.read_text(encoding="utf-8"))
    trials = load_trials(npz_path, json_path)
    run_info = dict(old.get("run", {}))
    run_info["constants"] = old.get("constants", {"runner": CONSTANTS})
    run_info["design_seeds"] = DEFAULT_SEEDS
    run_info["design_budget_s"] = BUDGET_S
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
        pic, label = plume_average(plume.World, seeds, steps)
    except Exception as e:
        print(f"plume picture skipped: {e!r}", flush=True)
    # the summary first, so the comparison can be recorded in the same JSON
    new_summary = summarise(trials, jo_cells=jo_cells_of(run_info))
    same, worst, mismatches = compare_preregistered(old.get("summary", {}).get("predictions", {}),
                                                    new_summary["predictions"])
    history = list(run_info.get("reanalyses", []))
    history.append({
        "when": datetime.now().isoformat(timespec="seconds"),
        "from": str(npz_path), "runner": "plume_experiment.reanalyse",
        "why": note or "disclosures and sensitivity checks added in review; no trial was rerun",
        "preregistered_metrics_unchanged": bool(same), "max_abs_difference": float(worst),
        "mismatches": mismatches,
    })
    run_info["reanalyses"] = history
    return write_outputs(trials, run_info, prefix, plume=pic, plume_label=label, render_kw=render_kw)


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
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="number of seeds (0..n-1), paired across conditions")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="world steps per trial (0.05 s each)")
    ap.add_argument("--out", default=str(BUILD / "plume"), help="output prefix: <out>_experiment.json, <out>_trajectories.npz/.png")
    ap.add_argument("--quick", type=int, default=0, help="1: smoke test, 2 seeds x 80 steps, outputs under <out>_quick")
    ap.add_argument("--odorant", default=ODORANT)
    ap.add_argument("--log", default=None, help="progress log path (default <out>_experiment.log)")
    ap.add_argument("--odour-hz", type=float, default=None, help="passed to PlumeFly if given")
    ap.add_argument("--wind-hz", type=float, default=None, help="passed to PlumeFly if given")
    ap.add_argument("--budget-min", type=float, default=BUDGET_S / 60.0,
                    help="wall-clock budget the seed ladder (12 -> 10 -> 8) is judged against, after the first seed")
    ap.add_argument("--reanalyse", default=None, metavar="PREFIX_OR_JSON",
                    help="recompute the summary and picture of a finished run from its saved trajectories (no brain)")
    ap.add_argument("--note", default=None, help="why the reanalysis was done; recorded in the JSON")
    args = ap.parse_args(normalise_argv(sys.argv[1:] if argv is None else argv))
    budget_s = float(args.budget_min) * 60.0

    if args.reanalyse:
        summary, json_path, npz_path, png_path = reanalyse(args.reanalyse, args.note)
        for name, p in summary["predictions"].items():
            print(f"{name}: {p['verdict']}", flush=True)
        print(f"rewrote {json_path}, {npz_path}, {png_path}", flush=True)
        return 0

    seeds_n, steps = args.seeds, args.steps
    out_prefix = out_prefix_from(args.out, quick=bool(args.quick))
    if args.quick:
        seeds_n, steps = QUICK_SEEDS, QUICK_STEPS
    log = Log(args.log or out_prefix.with_name(out_prefix.name + "_experiment.log"))

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
    fly_kw = dict(odorant=args.odorant, sim_steps=SIM_STEPS, seed=0)
    if args.odour_hz is not None:
        fly_kw["odour_hz"] = args.odour_hz
    if args.wind_hz is not None:
        fly_kw["wind_hz"] = args.wind_hz
    fly = plume_fly.PlumeFly(fb, gains, **fly_kw)
    log(f"brain loaded in {time.perf_counter() - t_load:.1f} s: {fb.n} neurons, setting {calibration.CHOSEN}, odorant {args.odorant!r}")

    if callable(getattr(fly, "describe", None)):
        fly_instance = _clean(fly.describe())
    else:
        fly_instance = {k: _clean(v) for k, v in vars(fly).items()
                        if isinstance(v, (int, float, str, bool)) and not k.startswith("_")}
    constants = {"runner": CONSTANTS, "world": module_constants(plume), "fly": module_constants(plume_fly),
                 "fly_instance": fly_instance}
    if callable(getattr(plume, "start_band_straight_plume", None)):
        try:
            constants["world_start_band"] = _clean(plume.start_band_straight_plume())
        except Exception as e:
            log(f"start band disclosure skipped: {e!r}")
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
        "constants": constants, "requested_seeds": seeds_n, "steps": steps, "odorant": args.odorant,
        "design_seeds": DEFAULT_SEEDS, "design_steps": DEFAULT_STEPS, "design_budget_s": BUDGET_S,
        "brain_setting": calibration.CHOSEN, "brain_gains": brain_gains, "brain_gain_type_counts": gain_types,
        "brain_gains_note": "per-type multipliers on outgoing synaptic weights, chosen for the roamer before this "
                            "experiment (calibration.CHOSEN); wiring is the connectome's, nothing is fitted here",
        "sim_steps": SIM_STEPS, "quick": bool(args.quick),
        "budget_s": budget_s, "started": datetime.now().isoformat(timespec="seconds"),
        "seed_ladder": "pending",
    }

    plume_pic = None
    try:
        plume_pic = plume_snapshot(plume.World(0, odour=True))
    except Exception as e:
        log(f"plume snapshot skipped: {e!r}")

    trials_by_condition = {c: [] for c in CONDITION_ORDER}
    seeds = list(range(seeds_n))
    t_start = time.perf_counter()
    total = len(seeds) * len(CONDITION_ORDER)
    done = 0
    i_seed = 0
    while i_seed < len(seeds):
        seed = seeds[i_seed]
        for cond in CONDITION_ORDER:
            odour, wind_sense = CONDITIONS[cond]
            world = plume.World(seed, odour=odour)
            tr = run_trial(world, fly, steps, seed, wind_sense, condition=cond)
            trials_by_condition[cond].append(tr)
            done += 1
            m = metrics(tr)
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done)
            log(f"trial {done}/{total} seed={seed} cond={cond} steps={tr['steps_run']} reached={int(tr['reached'])} "
                f"x_end={tr['x'][-1]:.3f} progress={m['progress']:+.3f} encounters={m['n_encounters']} "
                f"losses={m['n_losses']} walls={tr['wall_contacts']} in_plume={m['time_in_plume_s']:.1f}s "
                f"{tr['sec_per_step']:.3f}s/step elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")
        if i_seed == 0:
            # the budget decision, once, on the first seed's three trials
            rate = float(np.mean([trials_by_condition[c][0]["sec_per_step"] for c in CONDITION_ORDER]))
            run_info["sec_per_brain_run"] = rate
            n_keep, why = ladder(len(seeds), rate, steps, len(CONDITION_ORDER), budget_s)
            run_info["seed_ladder"] = why
            decision = ladder_note(run_info)
            if decision:
                run_info["seed_decision"] = decision
            log(f"budget: {rate:.3f} s/run projects {rate * steps * len(CONDITION_ORDER) * seeds_n / 60:.0f} min "
                f"for {seeds_n} seeds against {budget_s / 60:.0f} min; seeds {why}"
                + (f"; {decision}" if decision and decision != "as designed" else ""))
            if n_keep != len(seeds):
                seeds = seeds[:n_keep]
                total = len(seeds) * len(CONDITION_ORDER)
        i_seed += 1
        run_info["seeds"] = seeds[:i_seed]
        run_info["elapsed_s"] = time.perf_counter() - t_start
        if i_seed < len(seeds):
            write_outputs(trials_by_condition, run_info, out_prefix, plume=plume_pic, partial=True,
                          render_kw=render_kw)

    run_info["seeds"] = seeds
    run_info["finished"] = datetime.now().isoformat(timespec="seconds")
    run_info["elapsed_s"] = time.perf_counter() - t_start
    # the final picture shows the plume the run's flies actually saw on average,
    # not one seed's snapshot; the partial pictures above were captioned as such
    plume_label = DEFAULT_PLUME_LABEL
    try:
        avg, label = plume_average(plume.World, seeds, steps)
        if avg is not None:
            plume_pic, plume_label = avg, label
    except Exception as e:
        log(f"time-averaged plume skipped: {e!r}")
    summary, json_path, npz_path, png_path = write_outputs(trials_by_condition, run_info, out_prefix,
                                                           plume=plume_pic, render_kw=render_kw,
                                                           plume_label=plume_label)
    for name, p in summary["predictions"].items():
        log(f"{name}: {p['verdict']}")
    log(f"no encounter: {summary['no_encounter']}")
    log(f"wrote {json_path}, {npz_path}, {png_path}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

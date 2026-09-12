"""
The plume fly: one connectome with an odour on its nose and wind on its
antennae, read out through its walking descending neurons.

This is the coupling layer of the plume experiment (plume.py is the wind
tunnel, plume_experiment.py is the runner). It turns two world numbers, the
odour concentration at the fly and the direction the wind comes from, into
spike rates on the sensory neurons that actually carry those signals in a real
fly, runs the brain for one control step, and reads (turn, speed) off the same
descending neurons the roamer uses. Nothing here decides anything: the
constants are input scalings and the conversions are the roamer's.

MEASURED (this connectome, build/graph.npz and data/body-annotations.feather,
checked 2026-09-12). Only existence, counts and sides are measured here; the
code matches type-name regexes and a side column, nothing more.
  * Odour: 53 ORN_<glomerulus> types, 2,635 receptor neurons. DoOR 2.0 gives
    ethyl acetate's response above spontaneous firing on 32 of those glomeruli
    (20 at or above 0.05), via olfaction.Door.profile.
  * Wind: the types matching ^JO-(C|E) (JO-CA1/CA2/CL/CM and JO-ED1/ED2_a/b/c/
    EV1..EV6) number 335 cells. Their cell bodies are in the antenna, so they
    have no somaSide, but the annotations' rootSide names the antenna: L for
    203, R for 132, none missing. 13 of the 14 types are left-heavy, so the
    split is 1.54 : 1, not 1 : 1.
  * Motor: the types DNa02 (one cell per somaSide), DNa01 (one per side), MDN
    (4 cells) and DNp09 (2 cells) exist and are selected by pumpui.FlyPilot
    with a type regex plus somaSide. That is all the connectome check does.

CHOSEN, and said so (none of these is measured in this repo)
  * Which neuron stands for which output, and the 450 Hz scale, are inherited
    unchanged from pumpui.FlyPilot.motor (the roamer's cursor convention) and
    rest on the literature, not on anything checked here: JO-C/E as the
    antennal wind and gravity mechanosensors (Yorozu et al. 2009; Kamikouchi
    et al. 2009), wind direction read downstream as the difference between
    the two antennae (Suver et al. 2019), DNa02 asymmetry as steering
    (Rayshubskiy et al. 2020), MDN as backward walking (Bidaye et al. 2014),
    DNa01 as "forward" and DNp09 as "stop". The last two are the roamer's
    reading; the literature also describes DNa01 as a steering neuron and
    DNp09 as a forward-walking neuron that freezes the fly at strong
    activation (Bidaye et al. 2020). A negative speed command therefore means
    "MDN rate above DNa01 rate under the roamer's mapping", not an observed
    gait.
  * The brain setting is calibration.CHOSEN (pn05_apl10_kc03: projection-
    neuron output x 0.5, APL x 10, Kenyon-cell output x 0.3), the same brain
    as the roamer. It is a setting, so it is a choice; the odour pathway the
    experiment depends on (ORN -> PN) runs at half efficacy under it. The
    experiment discloses the multipliers and the type counts they touch.
  * Odour drive: DoOR profile of the odorant times clip(c, 0, 1) times
    odour_hz (200 Hz, the receptor ceiling of Hallem and Carlson 2006). One
    odorant, one concentration axis, no equal-sniff scaling.
  * Wind encoding: with phi the angle the wind comes FROM relative to the
    heading (0 = headwind, positive = from the fly's LEFT), the left antenna
    fires at wind_hz * clip(0.5 + 0.5 cos(phi - 45 deg), 0, 1) and the right
    at wind_hz * clip(0.5 + 0.5 cos(phi + 45 deg), 0, 1), one rate for every
    cell on that side. A headwind drives both sides equally per cell; wind
    from the left drives the left antenna harder. The 45 degree offset is a
    chosen stand-in for two antennae angled apart; real JO tuning is not this
    cosine. Any JO-C/E cell without a rootSide gets no drive (none in this
    connectome, but the rule is kept and tested).
  * Consequence of the 203 : 132 split, disclosed by describe(): the per-cell
    rates are symmetric but the summed drive is not. A headwind (85.4 Hz per
    cell on both sides) delivers 17,328 cell-Hz on the left and 11,267 on the
    right, the same left-minus-right total that wind from +31 deg would give
    on equal populations; the control (50 Hz both sides) still carries a
    fixed asymmetry worth about +17 deg. The brain never receives a
    symmetric headwind, and the control is not direction-free. This is the
    preregistered encoding and it was not changed after the data; a future
    run could equalise the summed drive per side (scale each side's per-cell
    rate by 167.5 / n_side) or subsample the left population to 132 cells.
  * JO-C and JO-E are driven identically on each side. In the fly they are
    reported to respond to opposite directions of static deflection
    (Kamikouchi et al. 2009), so the C-versus-E contrast a downstream circuit
    could read is held at zero at every heading here; the only directional
    cue the brain gets is the left/right total, the very cue the population
    imbalance biases.
  * The no-wind-sense control holds both antennae at 0.5 * wind_hz whatever
    the heading. That matches the encoding's mean over uniformly distributed
    headings, not the input a fly actually receives at its actual headings:
    a fly facing downwind gets 14.6 Hz per cell (about 4,900 cell-Hz), the
    control gets 50 Hz per cell (16,750 cell-Hz). An odour-versus-nowind
    difference therefore mixes the loss of direction with a change of total
    mechanosensory drive; the experiment reports the realised totals per
    condition next to any such comparison.
  * Motor conversions are the roamer's, unchanged: turn = (R - L) / 450,
    forward = mean(DNa01 L, R) / 450, back = MDN / 450, stop = DNp09 / 450,
    speed = clip(forward - back, -1, 1) * (1 - clip(stop, 0, 1)). turn > 0 is
    a right turn. turn is returned unclipped, as the roamer keeps it, and the
    world clips it.
  * One brain run of sim_steps LIF steps (60 steps x 0.2 ms = 12 ms) per world
    step, seeded by the caller. Every rate is a spike count in that window
    divided by 0.012 s, so a single cell's rate moves in quanta of 83.3 Hz:
    "DNa02 at 263 Hz" means about three spikes. With one DNa02 cell per side
    the turn command takes values in multiples of 83.3 / 450 = 0.185, i.e.
    the heading moves in multiples of 1.67 deg per 50 ms world step, and the
    spread of heading change is set by this readout, not by the fly.
    describe() reports the quanta.
  * FlyBrain.run restarts every neuron from rest at each call and PlumeFly
    keeps no state, so the motor command at a step is a stochastic function
    of (c, phi, seed) at that step only: the controller has no memory beyond
    the fly's pose. Surge and cast are history-defined behaviours; under this
    protocol they can only appear as a static difference between the motor
    map above and below the odour threshold. Carrying membrane state across
    the steps of a trial would be a new, separately preregistered experiment.

Simulator limits that matter here and are disclosed by the experiment:
uniform 0.275 mV synapses, no conduction delays, no receptor adaptation, and
the restart from rest at every world step described above.
"""
import numpy as np

import olfaction

ODORANT = "ethyl acetate"
ODOUR_HZ = 200.0            # receptor ceiling, olfaction.MAX_HZ
WIND_HZ = 100.0             # JO-C/E rate for a full-on antenna
SIM_STEPS = 60              # LIF steps per world step: 60 x 0.2 ms = 12 ms
LIF_DT_MS = 0.2             # flysim.Params.dt; pinned here so the quanta below are honest (tested)
MOTOR_SCALE_HZ = 450.0      # the roamer's descending-neuron rate scale
ANTENNA_OFFSET_DEG = 45.0   # how far each antenna's tuning is angled off the heading
JO_TYPE_RE = r"^JO-(C|E)"
ANNOTATIONS = "data/body-annotations.feather"
MOTOR_NAMES = ("steer_L", "steer_R", "fwd_L", "fwd_R", "back", "stop")


def readout_quanta(sim_steps=SIM_STEPS, motor=None, lif_dt_ms=LIF_DT_MS):
    """
    The granularity of the readout, for disclosure: every rate is a spike
    count over one window of sim_steps x lif_dt_ms, so one spike of one cell
    is rate_quantum_hz, and a group of n cells moves its mean in steps of
    rate_quantum_hz / n. The turn command is (mean_R - mean_L) / 450, so its
    quantum is rate_quantum_hz / (n_steer x 450); the speed parts likewise.
    """
    window_ms = float(sim_steps) * float(lif_dt_ms)
    q = 1000.0 / window_ms
    out = {"window_ms": window_ms, "rate_quantum_hz": q}
    if motor:
        n = {k: int(len(motor[k])) for k in MOTOR_NAMES if k in motor}
        steer = max(1, min(n.get("steer_L", 1), n.get("steer_R", 1)))
        fwd = max(1, n.get("fwd_L", 1) + n.get("fwd_R", 1))
        out.update({
            "turn_quantum": q / (steer * MOTOR_SCALE_HZ),
            "forward_quantum": q / (fwd * MOTOR_SCALE_HZ),
            "back_quantum": q / (max(1, n.get("back", 1)) * MOTOR_SCALE_HZ),
            "stop_quantum": q / (max(1, n.get("stop", 1)) * MOTOR_SCALE_HZ),
        })
    return out


def jo_drive_totals(n_left, n_right, wind_hz=WIND_HZ):
    """
    What the 203 : 132 rootSide split does to the summed drive, for disclosure.
    For a few named wind directions and the control: the per-cell rate on
    each side, the summed cell-Hz per side, their difference, and the wind
    angle (degrees, positive = from the left) that would produce the same
    left-minus-right total on two EQUAL populations of (n_left + n_right) / 2
    cells. On equal populations the difference is (n/2) x wind_hz x sin(phi)
    x sin(offset), so the equivalent angle is asin of the ratio, clipped.
    """
    n_left, n_right = int(n_left), int(n_right)
    n_half = (n_left + n_right) / 2.0
    off = np.deg2rad(ANTENNA_OFFSET_DEG)
    scale = n_half * wind_hz * np.sin(off)     # the equal-population difference at phi = 90 deg
    cases = {"headwind": 0.0, "wind_from_left": np.pi / 2, "wind_from_right": -np.pi / 2, "tailwind": np.pi}
    rows = {}
    for name, phi in cases.items():
        left, right = wind_rates(phi, wind_hz, True)
        rows[name] = _jo_row(phi, left, right, n_left, n_right, scale)
    left, right = wind_rates(0.0, wind_hz, False)
    rows["control_no_wind_sense"] = _jo_row(None, left, right, n_left, n_right, scale)
    return {
        "n_left": n_left, "n_right": n_right,
        "population_ratio_left_over_right": (n_left / n_right) if n_right else float("inf"),
        "equal_population_reference": n_half,
        "cases": rows,
        "note": "per-cell rates are symmetric; the summed cell-Hz is not, because the annotations root "
                "more JO-C/E cells in the left antenna; the 'equivalent_phi_deg' is the wind angle that "
                "would give the same left-minus-right total on equal populations",
    }


def _jo_row(phi, left, right, n_left, n_right, scale):
    tl, tr = left * n_left, right * n_right
    ratio = (tl - tr) / scale if scale else float("nan")
    equiv = float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))) if np.isfinite(ratio) else float("nan")
    return {"phi_deg": None if phi is None else float(np.degrees(phi)),
            "per_cell_hz_left": float(left), "per_cell_hz_right": float(right),
            "total_cell_hz_left": float(tl), "total_cell_hz_right": float(tr),
            "total_cell_hz": float(tl + tr), "left_minus_right_cell_hz": float(tl - tr),
            "equivalent_phi_deg": equiv}


def wind_rates(phi, wind_hz=WIND_HZ, wind_sense=True):
    """
    (left_hz, right_hz) for wind coming from angle phi (radians) relative to
    the heading: 0 is a headwind, positive is from the fly's left.

    With wind_sense=False both antennae sit at 0.5 * wind_hz for every phi.
    """
    if not wind_sense:
        return 0.5 * wind_hz, 0.5 * wind_hz
    off = np.deg2rad(ANTENNA_OFFSET_DEG)
    left = wind_hz * float(np.clip(0.5 + 0.5 * np.cos(phi - off), 0.0, 1.0))
    right = wind_hz * float(np.clip(0.5 + 0.5 * np.cos(phi + off), 0.0, 1.0))
    return left, right


def root_side_of(fb, path=ANNOTATIONS):
    """Per-neuron rootSide ("L", "R" or "") aligned with fb's neuron indices."""
    import pandas as pd
    a = pd.read_feather(path).drop_duplicates("bodyId").set_index("bodyId")
    return a["rootSide"].reindex(fb.bodies).fillna("").to_numpy().astype(str)


def motor_groups(fb, sim_steps=SIM_STEPS):
    """The roamer's motor index groups, the six walking ones."""
    import pumpui
    motor = pumpui.FlyPilot(fb, sim_steps=sim_steps).motor
    return {k: np.asarray(motor[k], dtype=np.int64) for k in MOTOR_NAMES}


class PlumeFly:
    """
    fb needs where(type_re=) and run(drive, steps, gains, record, seed), as
    FlyBrain does. motor and root_side can be given directly so the coupling
    is testable on a fake brain; by default they come from pumpui.FlyPilot and
    the annotations file.
    """

    def __init__(self, fb, gains=None, odorant=ODORANT, odour_hz=ODOUR_HZ,
                 wind_hz=WIND_HZ, sim_steps=SIM_STEPS, seed=0, motor=None,
                 root_side=None, nose=None, annotations_path=ANNOTATIONS):
        self.fb = fb
        self.gains = gains
        self.odorant = odorant
        self.odour_hz = float(odour_hz)
        self.wind_hz = float(wind_hz)
        self.sim_steps = int(sim_steps)
        self.seed = int(seed)

        # odour: the odorant's DoOR profile on the glomeruli this brain has
        self.nose = nose or olfaction.Nose(fb, max_hz=self.odour_hz)
        key = self.nose.door.key_of.get(odorant.lower())
        if key is None:
            raise KeyError(f"{odorant!r} is not in DoOR")
        self.profile = {g: float(v) for g, v in self.nose.door.profile(key).items()
                        if g in self.nose.orn and v > 0.0}
        self.orn = np.concatenate([self.nose.orn[g] for g in sorted(self.profile)]) \
            if self.profile else np.array([], dtype=np.int64)
        # every receptor neuron in the connectome, for the recorded mean ORN
        # rate (DoOR maps 2,524 of the 2,635 to a glomerulus it knows)
        self.orn_all = np.asarray(fb.where(type_re=r"^ORN_"), dtype=np.int64)

        # wind: JO-C/E split by the antenna they root in
        jo = np.asarray(fb.where(type_re=JO_TYPE_RE), dtype=np.int64)
        side = np.asarray(root_side if root_side is not None
                          else root_side_of(fb, annotations_path)).astype(str)
        self.jo_left = jo[side[jo] == "L"]
        self.jo_right = jo[side[jo] == "R"]
        self.jo_unsided = jo[(side[jo] != "L") & (side[jo] != "R")]

        # motor: the roamer's groups
        self.motor = {k: np.asarray(v, dtype=np.int64) for k, v in
                      (motor if motor is not None else motor_groups(fb, sim_steps)).items()
                      if k in MOTOR_NAMES}
        missing = [k for k in MOTOR_NAMES if k not in self.motor]
        if missing:
            raise KeyError(f"motor groups missing: {missing}")

    # ---- inputs ------------------------------------------------------------

    def odour_drive(self, c):
        """ORN drive for concentration c: profile x clip(c, 0, 1) x odour_hz."""
        cc = float(np.clip(c, 0.0, 1.0))
        return self.nose.drive({"profile": {g: v * cc for g, v in self.profile.items()}})

    def wind_drive(self, phi, wind_sense=True):
        """JO-C/E drive for wind from angle phi; cells without a rootSide get none."""
        left, right = wind_rates(phi, self.wind_hz, wind_sense)
        d = {}
        if self.jo_left.size:
            d[tuple(self.jo_left.tolist())] = left
        if self.jo_right.size:
            d[tuple(self.jo_right.tolist())] = right
        return d

    def drives(self, c, phi, wind_sense=True):
        """The full FlyBrain drive dict for one world step."""
        d = self.odour_drive(c)
        for k, v in self.wind_drive(phi, wind_sense).items():
            if k in d:
                raise ValueError("wind drive overlaps the odour drive")
            d[k] = v
        return d

    # ---- readout -----------------------------------------------------------

    @staticmethod
    def motor_from_rates(rates):
        """
        The roamer's conversions on a dict of the six motor rates (Hz).
        Returns (turn, speed, parts); turn is unclipped, parts holds the
        normalised forward_n, back_n and stop_n (rate / 450) for the record,
        named so they never shadow the Hz rates "back" and "stop".
        """
        turn = (rates["steer_R"] - rates["steer_L"]) / MOTOR_SCALE_HZ
        forward = (rates["fwd_L"] + rates["fwd_R"]) / 2.0 / MOTOR_SCALE_HZ
        back = rates["back"] / MOTOR_SCALE_HZ
        stop = rates["stop"] / MOTOR_SCALE_HZ
        speed = float(np.clip(forward - back, -1.0, 1.0) * (1.0 - np.clip(stop, 0.0, 1.0)))
        return float(turn), speed, {"forward_n": float(forward), "back_n": float(back), "stop_n": float(stop)}

    def step(self, c, phi, seed, wind_sense=True):
        """
        One brain run for one world step: returns (turn, speed, info) with the
        six motor rates, the mean ORN rate, the mean JO rate per side and the
        number of neurons that fired.
        """
        drive = self.drives(c, phi, wind_sense)
        record = dict(self.motor)
        record["orn"] = self.orn_all
        record["jo_left"] = self.jo_left
        record["jo_right"] = self.jo_right
        r = self.fb.run(drive, steps=self.sim_steps, gains=self.gains,
                        record=record, seed=int(seed))

        def mean_hz(name):
            v = np.asarray(r[name], dtype=np.float64)
            return float(v.mean()) if v.size else 0.0

        rates = {k: mean_hz(k) for k in MOTOR_NAMES}
        turn, speed, parts = self.motor_from_rates(rates)
        fired = r.get("_fired")
        info = dict(rates)
        info.update(parts)
        jo_l, jo_r = mean_hz("jo_left"), mean_hz("jo_right")
        info.update({
            "turn": turn, "speed": speed,
            "orn_hz": mean_hz("orn"),
            "jo_left_hz": jo_l,
            "jo_right_hz": jo_r,
            # the summed JO drive the brain actually got this step (cell-Hz),
            # so conditions can be compared on total input, not per-side means
            "jo_total_cell_hz": jo_l * float(self.jo_left.size) + jo_r * float(self.jo_right.size),
            "fired": int(len(fired)) if fired is not None else 0,
            "c": float(c), "phi": float(phi), "seed": int(seed),
            "wind_sense": bool(wind_sense),
        })
        return turn, speed, info

    def describe(self):
        """Every constant and population size, for the experiment's JSON."""
        return {
            "odorant": self.odorant, "odour_hz": self.odour_hz, "wind_hz": self.wind_hz,
            "sim_steps": self.sim_steps, "motor_scale_hz": MOTOR_SCALE_HZ,
            "antenna_offset_deg": ANTENNA_OFFSET_DEG, "jo_type_re": JO_TYPE_RE,
            "profile": dict(sorted(self.profile.items())),
            "orn_driven": int(self.orn.size), "orn_total": int(self.orn_all.size),
            "jo_left": int(self.jo_left.size), "jo_right": int(self.jo_right.size),
            "jo_unsided": int(self.jo_unsided.size),
            "motor_cells": {k: int(v.size) for k, v in self.motor.items()},
            "gains": "custom" if self.gains is not None else "stock",
            # disclosed, not measured: what the readout and the encoding can resolve
            "lif_dt_ms": LIF_DT_MS,
            "readout_quanta": readout_quanta(self.sim_steps, self.motor),
            "jo_drive_totals": jo_drive_totals(self.jo_left.size, self.jo_right.size, self.wind_hz),
            "jo_ce_coactivated": True,
            "memoryless": "FlyBrain.run restarts from rest every world step; no state is carried between steps",
        }

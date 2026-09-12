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

Protocol v2 (fixed 2026-09-13, before any v2 trial) changes four things about
this coupling. Each was named in v1's own report (build/plume_report.md,
"Simulator and design limits") as follow-up work before v2 ran, and each is
selectable back to v1 through the V1 settings dict so the published v1
coupling stays reproducible:
  A. the brain keeps its membrane potentials, refractory counters and rng
     across the world steps of a trial (carry_state=True) instead of
     restarting from rest every 50 ms, and runs 120 LIF steps (24 ms) per
     world step instead of 60 (12 ms);
  B. the turn and speed commands are low-passed with a 150 ms time constant
     (Smoother) before the world sees them; the raw commands stay in info;
  E. only the JO-E class is driven, and the per-cell rate on each side is
     scaled so both antennae deliver the same total cell-Hz at the same
     deflection (equalise_sides=True);
  F. a "shuffled" wind mode computes the same drive from a wind angle drawn
     uniformly at random each step, from the trial's own rng, so the drive
     statistics match the real wind sense and the direction information is
     zero. v1's constant-rate control ("none") is kept only so v1 can be
     reproduced.

MEASURED (this connectome, build/graph.npz and data/body-annotations.feather,
checked 2026-09-12; the JO classes recounted per side 2026-09-13). Only
existence, counts and sides are measured here; the code matches type-name
regexes and a side column, nothing more.
  * Odour: 53 ORN_<glomerulus> types, 2,635 receptor neurons. DoOR 2.0 gives
    ethyl acetate's response above spontaneous firing on 32 of those glomeruli
    (20 at or above 0.05), via olfaction.Door.profile.
  * Wind: the types matching ^JO-(C|E) (JO-CA1/CA2/CL/CM and JO-ED1/ED2_a/b/c/
    EV1..EV6) number 335 cells. Their cell bodies are in the antenna, so they
    have no somaSide, but the annotations' rootSide names the antenna: L for
    203, R for 132, none missing. By class: JO-E is 267 cells, 157 left and
    110 right; JO-C is 68 cells, 46 left and 22 right. 13 of the 14 types are
    left-heavy, so the split is 1.54 : 1 over both classes and 1.43 : 1 within
    JO-E, not 1 : 1.
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
  * Wind encoding, the base cosine (unchanged from v1): with phi the angle
    the wind comes FROM relative to the heading (0 = headwind, positive =
    from the fly's LEFT), the left antenna's base rate is wind_hz x clip(0.5
    + 0.5 cos(phi - 45 deg), 0, 1) and the right's wind_hz x clip(0.5 + 0.5
    cos(phi + 45 deg), 0, 1). A headwind gives both sides the same base
    rate; wind from the left gives the left antenna more. The 45 degree
    offset is a chosen stand-in for two antennae angled apart; real JO
    tuning is not this cosine. Any driven cell without a rootSide gets no
    drive (none in this connectome, but the rule is kept and tested).
  * Which wind class is driven (v2): JO-E, the larger class. Wind deflects
    both antennae the same way, and JO-C and JO-E are reported to respond to
    opposite directions of static deflection (Kamikouchi et al. 2009; Yorozu
    et al. 2009), so v1, which drove both classes identically, held their
    contrast at zero at every heading. One class is chosen and named; JO-C
    stays silent (never driven; it can still fire from the network, and its
    rate is recorded as jo_silent_hz). Which class a headwind excites in the
    fly is not settled here; the choice is the larger population.
  * Per-side equalisation (v2): rate_side = base(phi_side) x (mean JO-E count
    per side / that side's JO-E count), i.e. x 133.5 / 157 = 0.850 on the
    left and x 133.5 / 110 = 1.214 on the right, so that at equal deflection
    both antennae deliver the same total cell-Hz (a headwind: 85.4 x 133.5 =
    11,395 cell-Hz per side) and the left-minus-right total is zero at phi =
    0. The per-cell rate on the right can therefore exceed wind_hz; the spike
    probability per 0.2 ms LIF step stays far below one. v1's summed drive
    (no scaling) read every headwind as wind from about +31 deg; describe()
    reports the scales and the summed drive at named directions.
  * Wind modes (v2): "wind" computes the drive from the true wind angle
    relative to the heading; "shuffled" computes the same drive from an
    angle drawn uniformly on [-pi, pi) each step from the trial's own rng,
    independent of the heading, so the drive statistics match "wind" and
    the direction information is zero; "none" is v1's control, 0.5 x wind_hz
    per cell on both sides whatever the heading, kept for reproducing v1
    only. wind_sense=False selects "none". The drawn angle is recorded as
    phi_drive next to the true phi.
  * Motor conversions are the roamer's, unchanged: turn = (R - L) / 450,
    forward = mean(DNa01 L, R) / 450, back = MDN / 450, stop = DNp09 / 450,
    speed = clip(forward - back, -1, 1) * (1 - clip(stop, 0, 1)). turn > 0 is
    a right turn. turn is returned unclipped, as the roamer keeps it, and the
    world clips it.
  * Brain window and state (v2): sim_steps LIF steps per world step, 120 x
    0.2 ms = 24 ms of brain per 50 ms of world, and the brain's state
    (membrane potentials, refractory counters, rng) is carried from one
    window to the next within a trial; reset_state() at each trial start
    gives the trial a fresh seed, and the world-step seeds v1 used are then
    ignored (recorded, not used). 24 ms per 50 ms is a chosen ratio: brain
    time still runs slower than world time. Every rate is a spike count in
    the window divided by 0.024 s, so a single cell's rate moves in quanta
    of 41.7 Hz and the turn command in multiples of 41.7 / 450 = 0.0926,
    0.83 deg per world step before smoothing (v1's 12 ms window: 83.3 Hz and
    1.67 deg). describe() reports the quanta.
  * Command smoothing (v2): the world receives turn and speed passed through
    a first-order low-pass with a 150 ms time constant, three world steps
    (y += a (x - y), a = 1 - exp(-0.05 / 0.15) = 0.283), starting from zero
    at each trial start because the fly starts at rest. It stands in for leg
    and body inertia, which the descending-neuron readout has none of; it
    is not a fit to anything. The raw per-step commands are recorded next to
    the smoothed ones (turn_raw, speed_raw). The world clips turn after
    smoothing, as it clipped it before.
  * V1 (sim_steps=60, JO-C and JO-E driven alike, no equalisation, no state
    carry, no smoothing) reproduces the published coupling exactly. Its
    consequences (a memoryless controller that restarts every neuron from
    rest, the 203 : 132 summed-drive asymmetry, 1.67 deg turn quanta, a
    control that is not direction-free) are in v1's report.

Simulator limits that matter here and are disclosed by the experiment:
uniform 0.275 mV synapses, no conduction delays, no receptor adaptation.
"""
import math

import numpy as np

import olfaction

ODORANT = "ethyl acetate"
ODOUR_HZ = 200.0            # receptor ceiling, olfaction.MAX_HZ
WIND_HZ = 100.0             # JO base rate for a full-on antenna, before per-side scaling
SIM_STEPS = 120             # v2: LIF steps per world step, 120 x 0.2 ms = 24 ms
SIM_STEPS_V1 = 60           # v1: 60 x 0.2 ms = 12 ms
LIF_DT_MS = 0.2             # flysim.Params.dt; pinned here so the quanta below are honest (tested)
MOTOR_SCALE_HZ = 450.0      # the roamer's descending-neuron rate scale
ANTENNA_OFFSET_DEG = 45.0   # how far each antenna's tuning is angled off the heading
JO_TYPE_RE = r"^JO-(C|E)"   # every wind cell the brain has: counted and recorded
JO_DRIVEN_RE = r"^JO-E"     # v2: the one class the wind drives
JO_DRIVEN_RE_V1 = JO_TYPE_RE
SMOOTH_TAU_S = 0.150        # v2 command low-pass, three world steps
WORLD_DT_S = 0.05           # plume.DT; pinned here so the smoother's arithmetic is checkable without the world
WIND_MODES = ("wind", "shuffled", "none")
SHUFFLE_STREAM = 0x5EED0002  # second seed word of a trial's shuffle rng, so it is never the brain's stream
ANNOTATIONS = "data/body-annotations.feather"
MOTOR_NAMES = ("steer_L", "steer_R", "fwd_L", "fwd_R", "back", "stop")

# the published v1 coupling, for reproducing it: PlumeFly(fb, gains, **V1)
V1 = dict(sim_steps=SIM_STEPS_V1, jo_driven_re=JO_DRIVEN_RE_V1, equalise_sides=False,
          carry_state=False, smooth_tau_s=0.0)
# the v2 coupling, the defaults: PlumeFly(fb, gains) and PlumeFly(fb, gains, protocol="v2") are the same fly
V2 = dict(sim_steps=SIM_STEPS, jo_driven_re=JO_DRIVEN_RE, equalise_sides=True,
          carry_state=True, smooth_tau_s=SMOOTH_TAU_S)
PROTOCOL_SETTINGS = {"v1": V1, "v2": V2}
PROTOCOL_KEYS = tuple(V2)
_DEFAULT = object()         # "not given": the protocol's value applies


def protocol_of(settings):
    """The protocol name whose coupling settings these are, or "custom"."""
    for name, base in PROTOCOL_SETTINGS.items():
        if all(settings.get(k) == base[k] for k in PROTOCOL_KEYS):
            return name
    return "custom"


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


def side_scales(n_left, n_right, equalise=True):
    """
    (scale_left, scale_right): the per-cell multipliers that make both sides
    deliver the same total cell-Hz at the same deflection, mean count per
    side over the side's own count. (1, 1) when not equalising (v1); a side
    with no cells gets 0 rather than a division by zero.
    """
    if not equalise:
        return 1.0, 1.0
    n_left, n_right = int(n_left), int(n_right)
    n_mean = (n_left + n_right) / 2.0
    return (n_mean / n_left if n_left else 0.0), (n_mean / n_right if n_right else 0.0)


def jo_drive_totals(n_left, n_right, wind_hz=WIND_HZ, scale_left=1.0, scale_right=1.0):
    """
    What the rootSide split does to the summed drive, for disclosure. For a
    few named wind directions and the v1 control: the per-cell rate on each
    side after scaling, the summed cell-Hz per side, their difference, and
    the wind angle (degrees, positive = from the left) that would produce the
    same left-minus-right total on two EQUAL, unscaled populations of
    (n_left + n_right) / 2 cells. On equal populations the difference is
    (n/2) x wind_hz x sin(phi) x sin(offset), so the equivalent angle is asin
    of the ratio, clipped. With v2's scales the headwind row reads zero.
    """
    n_left, n_right = int(n_left), int(n_right)
    n_half = (n_left + n_right) / 2.0
    off = np.deg2rad(ANTENNA_OFFSET_DEG)
    scale = n_half * wind_hz * np.sin(off)     # the equal-population difference at phi = 90 deg
    cases = {"headwind": 0.0, "wind_from_left": np.pi / 2, "wind_from_right": -np.pi / 2, "tailwind": np.pi}
    rows = {}
    for name, phi in cases.items():
        left, right = wind_rates(phi, wind_hz, True)
        rows[name] = _jo_row(phi, left * scale_left, right * scale_right, n_left, n_right, scale)
    left, right = wind_rates(0.0, wind_hz, False)
    rows["control_no_wind_sense"] = _jo_row(None, left * scale_left, right * scale_right, n_left, n_right, scale)
    return {
        "n_left": n_left, "n_right": n_right,
        "population_ratio_left_over_right": (n_left / n_right) if n_right else float("inf"),
        "equal_population_reference": n_half,
        "scale_left": float(scale_left), "scale_right": float(scale_right),
        "cases": rows,
        "note": "per-cell rates are the base cosine times the side's scale; the summed cell-Hz per side "
                "is rate x count; the 'equivalent_phi_deg' is the wind angle that would give the same "
                "left-minus-right total on equal unscaled populations",
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
    Base (left_hz, right_hz) for wind coming from angle phi (radians)
    relative to the heading: 0 is a headwind, positive is from the fly's
    left. Per-side scaling is applied by the fly, not here.

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
    try:
        import pumpui
    except ImportError:                     # the public copy calls it flyeye
        import flyeye as pumpui
    motor = pumpui.FlyPilot(fb, sim_steps=sim_steps).motor
    return {k: np.asarray(motor[k], dtype=np.int64) for k in MOTOR_NAMES}


class Smoother:
    """
    First-order low-pass on the command channels the world receives, a
    stand-in for leg and body inertia that the descending-neuron readout has
    none of. y += alpha (x - y) with alpha = 1 - exp(-dt / tau), so after k
    steps of a unit step the output is 1 - exp(-k dt / tau): 0.632 at k =
    tau / dt. tau = 0 passes the input through unchanged (v1). It starts at
    zero and reset() puts it back there, because a trial starts with the
    fly at rest.
    """

    def __init__(self, tau_s=SMOOTH_TAU_S, dt_s=WORLD_DT_S, channels=2):
        self.tau_s = float(tau_s)
        self.dt_s = float(dt_s)
        if self.dt_s <= 0.0:
            raise ValueError("dt must be positive")
        self.alpha = 1.0 if self.tau_s <= 0.0 else 1.0 - math.exp(-self.dt_s / self.tau_s)
        self.y = np.zeros(int(channels), dtype=np.float64)

    def reset(self):
        self.y[:] = 0.0

    def update(self, *x):
        if len(x) != self.y.size:
            raise ValueError(f"{self.y.size} channels, got {len(x)}")
        self.y += self.alpha * (np.asarray(x, dtype=np.float64) - self.y)
        return tuple(float(v) for v in self.y)

    def describe(self):
        return {"tau_s": self.tau_s, "dt_s": self.dt_s, "alpha": self.alpha,
                "steps_per_tau": (self.tau_s / self.dt_s) if self.tau_s > 0 else 0.0,
                "start": 0.0, "passthrough": self.alpha >= 1.0}


class PlumeFly:
    """
    fb needs where(type_re=), types, and run(drive, steps, gains, record,
    seed, state) returning a '_state' entry, as FlyBrain does. motor and
    root_side can be given directly so the coupling is testable on a fake
    brain; by default they come from pumpui.FlyPilot and the annotations
    file. The v2 settings are the defaults; PlumeFly(fb, gains, **V1) is the
    published v1 coupling.

    A trial: reset_state(seed) (or reset_state() and let the first step's
    seed name the trial), then step(...) once per world step. Within a trial
    the brain state and the smoother carry over; across trials nothing does.
    The runner calls reset_state by its own name, begin_trial; both are the
    same method.

    protocol="v1" or "v2" names the coupling (PROTOCOL_SETTINGS); the five
    coupling arguments (sim_steps, jo_driven_re, equalise_sides,
    carry_state, smooth_tau_s) take the protocol's values unless given
    explicitly, and an explicit value wins so a runner that does its own
    smoothing can ask a v2 fly for smooth_tau_s=0. With no protocol the
    defaults are v2's. describe() reports the name, the resolved settings
    and which of them differ from the named protocol.
    """

    def __init__(self, fb, gains=None, odorant=ODORANT, odour_hz=ODOUR_HZ,
                 wind_hz=WIND_HZ, sim_steps=_DEFAULT, seed=0, motor=None,
                 root_side=None, nose=None, annotations_path=ANNOTATIONS,
                 jo_driven_re=_DEFAULT, equalise_sides=_DEFAULT, carry_state=_DEFAULT,
                 smooth_tau_s=_DEFAULT, world_dt_s=WORLD_DT_S, protocol=None):
        if protocol is not None and protocol not in PROTOCOL_SETTINGS:
            raise ValueError(f"protocol must be one of {tuple(PROTOCOL_SETTINGS)} or None, not {protocol!r}")
        base = PROTOCOL_SETTINGS[protocol or "v2"]
        given = dict(sim_steps=sim_steps, jo_driven_re=jo_driven_re, equalise_sides=equalise_sides,
                     carry_state=carry_state, smooth_tau_s=smooth_tau_s)
        settings = {k: (base[k] if v is _DEFAULT else v) for k, v in given.items()}
        self.protocol = protocol or protocol_of(settings)
        self.protocol_overrides = sorted(k for k in PROTOCOL_KEYS if settings[k] != base[k])
        sim_steps, jo_driven_re, equalise_sides, carry_state, smooth_tau_s = (settings[k] for k in PROTOCOL_KEYS)
        self.fb = fb
        self.gains = gains
        self.odorant = odorant
        self.odour_hz = float(odour_hz)
        self.wind_hz = float(wind_hz)
        self.sim_steps = int(sim_steps)
        self.seed = int(seed)
        self.jo_driven_re = str(jo_driven_re)
        self.equalise_sides = bool(equalise_sides)
        self.carry_state = bool(carry_state)

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

        # wind: the driven class split by the antenna it roots in; the rest of
        # the wind cells are never driven but are counted and recorded
        self.jo_all = np.asarray(fb.where(type_re=JO_TYPE_RE), dtype=np.int64)
        driven = np.asarray(fb.where(type_re=self.jo_driven_re), dtype=np.int64)
        side = np.asarray(root_side if root_side is not None
                          else root_side_of(fb, annotations_path)).astype(str)
        self.jo_left = driven[side[driven] == "L"]
        self.jo_right = driven[side[driven] == "R"]
        self.jo_unsided = driven[(side[driven] != "L") & (side[driven] != "R")]
        self.jo_silent = self.jo_all[~np.isin(self.jo_all, driven)]
        self.scale_left, self.scale_right = side_scales(
            self.jo_left.size, self.jo_right.size, self.equalise_sides)

        # motor: the roamer's groups
        self.motor = {k: np.asarray(v, dtype=np.int64) for k, v in
                      (motor if motor is not None else motor_groups(fb, sim_steps)).items()
                      if k in MOTOR_NAMES}
        missing = [k for k in MOTOR_NAMES if k not in self.motor]
        if missing:
            raise KeyError(f"motor groups missing: {missing}")

        # per-trial state: the brain's, the shuffle rng's and the smoother's
        self.smoother = Smoother(smooth_tau_s, world_dt_s)
        self._state = None
        self._trial_seed = None
        self._shuffle_rng = None

    # ---- trial state ---------------------------------------------------------

    def reset_state(self, seed=None):
        """
        Start a trial: forget the brain state, the shuffle rng and the
        smoother. With a seed, that seed names the trial: it seeds the
        brain's first window and the shuffle stream. Without one, the seed
        of the first step after the reset does.
        """
        self._state = None
        self._trial_seed = None if seed is None else int(seed)
        self._shuffle_rng = None
        self.smoother.reset()

    begin_trial = reset_state       # the runner's name for the same thing

    @property
    def trial_seed(self):
        return self._trial_seed

    @property
    def state_carried(self):
        """True once a window's state is held for the next step."""
        return self._state is not None

    def _begin_trial_if_needed(self, seed):
        if self._trial_seed is None:
            self._trial_seed = int(seed)
        if self._shuffle_rng is None:
            self._shuffle_rng = np.random.default_rng([self._trial_seed, SHUFFLE_STREAM])

    # ---- inputs ------------------------------------------------------------

    @staticmethod
    def _mode(wind_sense=True, wind_mode=None):
        mode = wind_mode if wind_mode is not None else ("wind" if wind_sense else "none")
        if mode not in WIND_MODES:
            raise ValueError(f"wind_mode {mode!r} is not one of {WIND_MODES}")
        return mode

    def odour_drive(self, c):
        """ORN drive for concentration c: profile x clip(c, 0, 1) x odour_hz."""
        cc = float(np.clip(c, 0.0, 1.0))
        return self.nose.drive({"profile": {g: v * cc for g, v in self.profile.items()}})

    def wind_rates_sides(self, phi, wind_mode="wind"):
        """(left_hz, right_hz) per driven cell: the base cosine times the side's scale."""
        left, right = wind_rates(phi, self.wind_hz, wind_sense=(wind_mode != "none"))
        return left * self.scale_left, right * self.scale_right

    def wind_drive(self, phi, wind_sense=True, wind_mode=None):
        """Drive on the driven wind class for wind from angle phi; cells without a rootSide get none."""
        left, right = self.wind_rates_sides(phi, self._mode(wind_sense, wind_mode))
        d = {}
        if self.jo_left.size:
            d[tuple(self.jo_left.tolist())] = left
        if self.jo_right.size:
            d[tuple(self.jo_right.tolist())] = right
        return d

    def drives(self, c, phi, wind_sense=True, wind_mode=None):
        """The full FlyBrain drive dict for one world step (phi is the angle the drive is computed from)."""
        d = self.odour_drive(c)
        for k, v in self.wind_drive(phi, wind_sense, wind_mode).items():
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

    def step(self, c, phi, seed=0, wind_sense=True, wind_mode=None):
        """
        One brain window for one world step: returns (turn, speed, info),
        the commands the world should receive (smoothed under v2) and, in
        info, the raw commands, the six motor rates, the mean ORN rate, the
        JO rate per side with the realised and delivered cell-Hz, the
        undriven wind cells' rate, the number of neurons that fired, and the
        inputs as they reached the brain (c, phi, phi_drive, the mode).

        With carry_state the brain continues from the previous step's state;
        the first step of a trial is seeded by the trial seed (reset_state's,
        else this step's `seed`) and later `seed`s are recorded but unused.
        Without it every step is a fresh run seeded by `seed`, as in v1.
        """
        mode = self._mode(wind_sense, wind_mode)
        self._begin_trial_if_needed(seed)
        phi = float(phi)
        phi_drive = float(self._shuffle_rng.uniform(-np.pi, np.pi)) if mode == "shuffled" else phi
        drive = self.drives(c, phi_drive, wind_mode=mode)
        record = dict(self.motor)
        record["orn"] = self.orn_all
        record["jo_left"] = self.jo_left
        record["jo_right"] = self.jo_right
        record["jo_silent"] = self.jo_silent

        carried = self.carry_state and self._state is not None
        brain_seed = self._trial_seed if self.carry_state else int(seed)
        r = self.fb.run(drive, steps=self.sim_steps, gains=self.gains,
                        record=record, seed=brain_seed,
                        state=self._state if self.carry_state else None)
        if self.carry_state:
            self._state = r["_state"]          # a brain that cannot carry state is an error, not a silent restart

        def mean_hz(name):
            v = np.asarray(r[name], dtype=np.float64)
            return float(v.mean()) if v.size else 0.0

        rates = {k: mean_hz(k) for k in MOTOR_NAMES}
        turn_raw, speed_raw, parts = self.motor_from_rates(rates)
        turn, speed = self.smoother.update(turn_raw, speed_raw)
        fired = r.get("_fired")
        left_hz, right_hz = self.wind_rates_sides(phi_drive, mode)
        n_l, n_r = float(self.jo_left.size), float(self.jo_right.size)
        jo_l, jo_r = mean_hz("jo_left"), mean_hz("jo_right")
        info = dict(rates)
        info.update(parts)
        info.update({
            "turn": turn, "speed": speed,
            "turn_raw": turn_raw, "speed_raw": speed_raw,
            "orn_hz": mean_hz("orn"),
            "jo_left_hz": jo_l,
            "jo_right_hz": jo_r,
            # what the driven wind cells actually did this step, summed per
            # side and in total (cell-Hz), so conditions compare on input
            "jo_left_cell_hz": jo_l * n_l,
            "jo_right_cell_hz": jo_r * n_r,
            "jo_total_cell_hz": jo_l * n_l + jo_r * n_r,
            # and what was delivered to them, the same way
            "jo_left_drive_hz": left_hz,
            "jo_right_drive_hz": right_hz,
            "jo_left_drive_cell_hz": left_hz * n_l,
            "jo_right_drive_cell_hz": right_hz * n_r,
            "jo_total_drive_cell_hz": left_hz * n_l + right_hz * n_r,
            "jo_silent_hz": mean_hz("jo_silent"),
            "fired": int(len(fired)) if fired is not None else 0,
            "c": float(c), "phi": phi, "phi_drive": phi_drive,
            "seed": int(seed), "trial_seed": int(self._trial_seed),
            "wind_sense": mode != "none", "wind_mode": mode,
            "state_carried": bool(carried),
        })
        return turn, speed, info

    def describe(self):
        """Every constant, setting and population size, for the experiment's JSON."""
        types = getattr(self.fb, "types", None)
        driven = np.concatenate([self.jo_left, self.jo_right, self.jo_unsided])
        return {
            "protocol": self.protocol,
            "protocol_overrides": list(self.protocol_overrides),
            "odorant": self.odorant, "odour_hz": self.odour_hz, "wind_hz": self.wind_hz,
            "sim_steps": self.sim_steps, "lif_dt_ms": LIF_DT_MS,
            "brain_ms_per_world_step": self.sim_steps * LIF_DT_MS,
            "world_dt_s": self.smoother.dt_s,
            "carry_state": self.carry_state,
            "motor_scale_hz": MOTOR_SCALE_HZ,
            "antenna_offset_deg": ANTENNA_OFFSET_DEG,
            "jo_type_re": JO_TYPE_RE, "jo_driven_re": self.jo_driven_re,
            "profile": dict(sorted(self.profile.items())),
            "orn_driven": int(self.orn.size), "orn_total": int(self.orn_all.size),
            "jo_all": int(self.jo_all.size),
            "jo_left": int(self.jo_left.size), "jo_right": int(self.jo_right.size),
            "jo_unsided": int(self.jo_unsided.size), "jo_silent": int(self.jo_silent.size),
            "jo_driven_types": sorted(set(types[driven].tolist())) if types is not None else None,
            "jo_silent_types": sorted(set(types[self.jo_silent].tolist())) if types is not None else None,
            "equalise_sides": self.equalise_sides,
            "side_scale_left": float(self.scale_left), "side_scale_right": float(self.scale_right),
            "motor_cells": {k: int(v.size) for k, v in self.motor.items()},
            "gains": "custom" if self.gains is not None else "stock",
            # disclosed, not measured: what the readout and the encoding can resolve
            "readout_quanta": readout_quanta(self.sim_steps, self.motor),
            "jo_drive_totals": jo_drive_totals(self.jo_left.size, self.jo_right.size, self.wind_hz,
                                               self.scale_left, self.scale_right),
            "jo_ce_coactivated": bool(self.jo_silent.size == 0),
            "wind_modes": list(WIND_MODES),
            "smoothing": self.smoother.describe(),
            "memoryless": not self.carry_state,
            "state": ("membrane potentials, refractory counters and rng carried across the world steps "
                      "of a trial; reset_state() at each trial start with a fresh seed per trial")
                     if self.carry_state else
                     "FlyBrain.run restarts from rest every world step; no state is carried between steps",
        }

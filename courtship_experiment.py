"""Seed-paired courtship: a male sings and a female graph answers."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

import backrooms_world as bw
import backrooms_dictionary as bd
from courtship import BlindEye, HerBody, female_groups, female_motor, load_female
from flysim import FlyBrain

# CHOSEN: raw female weights match the male loading convention.
FEMALE_EXC_SCALE = 1.0
# CHOSEN: uniform grey luminance carries no information and floods her brain.
FEMALE_EYE = "blind"
CONDITIONS = {"song": (True, False), "silence": (False, False), "shuffled": (True, True), "dark": (False, False)}
DEFAULT_STEPS = 400
ACCEPT_WINDOWS = 3
PROTOCOL_TEXT = """Conditions, same seed and same arena start for all four (paired):
- `song`: the room as built; his pulse and sine motor sums reach her JO-A and JO-B separately through `sound_hz` with distance attenuation.
- `silence`: her incoming `sound_hz` is forced to (0.0, 0.0) every step (he still sings; she does not hear). Implement without touching `FlyBody`: wrap/patch the value handed to her body at the Room level (a `listen` flag on the room or a small subclass of `Room`; the male path stays byte-identical).
- `shuffled`: her incoming `sound_hz` sequence is the `song` trial's sequence for the same seed, permuted as paired rows in time with a seed-derived permutation (run `song` first for that seed, keep its per-step sound series, then feed the permutation). This matches total sound energy; only the timing/coupling is broken.

- `dark`: her eye is blind and incoming sound is forced to (0.0, 0.0); this is the baseline. With FEMALE_EYE blind, dark and silence coincide; both are retained because they differ if FEMALE_EYE changes.

CHOSEN before data: female raw weights (FEMALE_EXC_SCALE = 1.0) give parity with the male. Her eye is blind (FEMALE_EYE = "blind"): uniform grey carries no information and floods her brain (measured silence vpoDN 54-206 Hz with eye, 0 Hz with blind). Contrast/motion vision is a later step.

CHOSEN: pulse-motor sum drives JO-A and sine-motor sum drives JO-B through sound_hz(rate, distance), each normalised by song_full_hz(n_cells=8) and song_full_hz(n_cells=2) respectively; separate references preserve each population's fraction of its ceiling.
CHOSEN: female_scent_hz = SMELL_MAX_HZ x falloff(distance) drives ORN_VA1v (Or47b); volatile female scent supplies the male's female-presence input.
CHOSEN: the same female scent reaches putative_ppk23 only at distance <= CONTACT_MM = 2.0 mm; contact chemosensation needs touch.
CHOSEN: ORN_DA1 cVA input is zero in courtship because there is no other male; this is delivered input, not a claim that those neurons cannot fire.
CHOSEN: annotation-backed columnar vision and soma sides are used when available; the ellipse floor is one measured column spacing so her silhouette can reach retinal samples.
CHOSEN: missing annotations use luminance and unknown soma sides; no column or side identities are invented.
CHOSEN: P1 mean > 0 Hz counts active windows descriptively; it decides nothing.
CHOSEN: P6 compares pulse[1:] and sine[1:] with distance[:-1] and her_speed[:-1]; a one-window lag describes his next output against her previous state without adding adaptation.
MEASURED: P1 mean, pulse/sine motor sums, lagged correlations, retinal visibility and restored motor group counts are readouts; none gates either brain.

Predictions (fixed before data, verdict rule as in plume: paired difference > 2 SE across seeds):
- P0 baseline (descriptive, no verdict): her vpoDN active windows per seed in dark and silence, expected 0.
- P1 answer: her mean vpoDN rate, song > silence.
- P2 pC1: her mean pC1 rate, song > silence.
- P3 coupling: song > shuffled on vpoDN (the timing matters, not only the energy).
- P4 approach: her mean distance to him in the last quarter, song < silence.
- P5 his adaptation (descriptive, no verdict): Spearman correlation across steps between his `song_hz` and her distance, and between his `song_hz` and her speed, reported per seed; state in the report that this measures whether his song already depends on her, and that no adaptation mechanism was added.
- P6 his adaptation (descriptive): P1 active windows and per-seed lagged pulse/sine correlations; no verdict.
- Seed spread (maintainer's requirement, descriptive): number of seeds with accept / approach / retreat per condition; if every seed gives the same outcome in a condition, the report says so in one plain sentence and does not soften it."""
PROTOCOLS = {"v3": {"text": PROTOCOL_TEXT, "conditions": CONDITIONS,
    "default_seeds": 10, "steps": DEFAULT_STEPS, "seed_ladder": (10, 8),
    "budget_s": 9000, "accept_windows": ACCEPT_WINDOWS, "sim_steps": bw.SIM_STEPS,
    "world_dt_s": bw.WORLD_DT_S, "state_carry": True,
    "trial_text": "Trial: `steps` world steps (default 400 = 20 s at 0.05 s; `--quick 1` = 2 seeds x 80 steps). Both brains carry state across steps (the room already does). Record per step: positions and headings of both, distance, her speed, her `vpodn_hz`, `pc1_hz`, her delivered `sound_hz`, his `song_hz`, `p1_hz`, `pulse_hz`, `sine_hz`, both her sound channels, his delivered `sound_hz`; trial eye choice, sides restored, and sight_ok.",
    "outcome_text": "Per-trial outcome (CHOSEN, disclosed): `accept` = her vpoDN window rate exceeds 0 Hz in at least `ACCEPT_WINDOWS` = 3 windows of the trial; `approach` = mean distance in the last quarter of the trial is smaller than in the first quarter; `retreat` = the opposite. Report all three per seed and per condition in a table in the JSON and in the report; never only the pooled mean.",
    "permutation": "numpy default_rng(SeedSequence([seed, 731])).permutation(song_sound)",
    "quarter_rule": "floor(steps / 4) windows at each end; geometry before the window",
    "budget_note": "First trial measures both graphs per step; choose largest fitting seed count, with floor 8 (requests below 8 kept). Setup and rendering excluded from estimate."}}
LIMITATIONS = [
    "Pulse and sine are separate motor population sums, not acoustic waveforms.",
    "Her ear receives a rate, not a waveform.",
    "12 ms brain per 50 ms world.",
    "vpoDN identified as DNp37 by alias, 2 cells.",
    "pC1a-e 10 cells.",
    "No pheromone channel to her.",
    "Male P1 membership is the dictionary's uncertain 86-cell group.",
    "No adaptation mechanism was added on his side.",
    "CHOSEN before data: female loaded raw (FEMALE_EXC_SCALE = 1.0) for parity with the male.",
    "CHOSEN before data: female eye blind; uniform grey carries no information and floods the brain (measured silence vpoDN 54-206 Hz with eye, 0 Hz with blind).",
    "Dark is blind and silent; with FEMALE_EYE blind it coincides with silence. Contrast/motion vision is a later step.",
    "Male eye and soma-side availability are recorded per trial; gains remain uncalibrated.",
    "Distance changes include both bodies; approach is not an isolated female command.",
    "Shuffling matches the input rate multiset, not the downstream neural response."]
PUBLISHED = Path("build/courtship_published")


class LuminanceEye:
    """CHOSEN: the real-brain integration test's luminance encoding."""
    def __init__(self, fb):
        self.on_idx = fb.where(type_re="^L1$")
        self.off_idx = fb.where(type_re="^L2$")

    def look(self, img, cx, cy):
        h, w = img.shape
        x0, x1 = int(cx - bw.EYE_FOV_W / 2), int(cx + bw.EYE_FOV_W / 2)
        y0, y1 = int(cy - bw.EYE_FOV_H / 2), int(cy + bw.EYE_FOV_H / 2)
        m = float(img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].mean())
        return {tuple(self.on_idx): np.full(len(self.on_idx), m * 180, np.float32),
                tuple(self.off_idx): np.full(len(self.off_idx), (1-m) * 108, np.float32)}


class ExperimentRoom(bw.Room):
    def configure(self, seed, condition, song_sound=None):
        self.condition = condition
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        self.replay = None
        if condition == "shuffled":
            if song_sound is None:
                raise ValueError("shuffled requires the paired song series")
            self.replay = iter(np.random.default_rng(np.random.SeedSequence([seed, 731])).permutation(song_sound))

    def listener_sound(self, sound_hz):
        if self.condition in ("silence", "dark"):
            return 0.0 if np.isscalar(sound_hz) else (0.0, 0.0)
        if self.condition == "shuffled":
            value = next(self.replay)
            return float(value) if np.isscalar(value) else tuple(map(float, value))
        return sound_hz


def build_room(seed, condition, song_sound=None, brains=None,
               annotations_path="data/body-annotations.feather", brain_class=FlyBrain):
    """One construction for every condition; new bodies reset both states."""
    cls = bw.brain_class(brain_class) if isinstance(brain_class, str) else brain_class
    male, female = brains if brains is not None else (cls(), load_female(exc_scale=FEMALE_EXC_SCALE, brain_class=cls))
    eye = BlindEye(female) if condition == "dark" or FEMALE_EYE == "blind" else LuminanceEye(female)
    body = HerBody("B", female, eye, female_groups(female),
                   female_motor(female), seed=seed * 2 + 2)
    available = Path(annotations_path).is_file()
    if available:
        from flyeye import FlyEye
        male_eye = FlyEye(male, annotations_path=str(annotations_path))
        sides = bw.soma_sides(male, annotations_path)
    else:
        male_eye = LuminanceEye(male)
        sides = np.full(male.n, "", dtype=str)
    room = ExperimentRoom(male, male_eye, bd.present_groups(male),
        bw.motor_groups(male, sides), seed=seed, body_b=body,
        min_radius_px=bw.column_spacing_px(bw.distinct_columns(male_eye)))
    room.male_setup = dict(male_eye="columnar" if available else "luminance",
        sides_restored=available,
        disclosure=("MEASURED: annotation columns and soma sides loaded by bodyId." if available
                    else "CHOSEN: missing annotations use luminance and unknown soma sides; no column or side identities are invented."))
    room.configure(seed, condition, song_sound)
    return room


def correlation(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def outcome(seed, condition, trace, start):
    q = max(1, len(trace["distance_mm"]) // 4)
    first = float(np.mean(trace["distance_mm"][:q]))
    last = float(np.mean(trace["distance_mm"][-q:]))
    active = int(np.count_nonzero(trace["vpodn_hz"] > 0))
    result = dict(seed=seed, condition=condition, start=start, accept=active >= ACCEPT_WINDOWS,
        approach=last < first, retreat=last > first, active_windows=active,
        vpodn_hz=float(np.mean(trace["vpodn_hz"])), pc1_hz=float(np.mean(trace["pc1_hz"])),
        first_distance_mm=first, last_distance_mm=last,
        song_distance_rho=correlation(trace["song_hz"], trace["distance_mm"]),
        song_speed_rho=correlation(trace["song_hz"], trace["her_speed_mm_s"]))
    # CHOSEN: P1 mean > 0 Hz counts active windows descriptively; it decides nothing.
    p1 = trace.get("p1_hz")
    result["p1_active_windows"] = (int(np.count_nonzero(p1 > 0))
                                    if p1 is not None and np.all(np.isfinite(p1)) else None)
    for channel in ("pulse", "sine"):
        values = trace.get(channel + "_hz")
        for label, key in (("distance", "distance_mm"), ("speed", "her_speed_mm_s")):
            result[f"{channel}_{label}_lagged_rho"] = (
                correlation(values[1:], trace[key][:-1]) if values is not None else None)
    return result


def run_trial(room, steps, seed, condition):
    if steps < 4:
        raise ValueError("at least four steps required")
    start = room.arena.geometry()
    sight = room.sight_check()
    # MEASURED: compare the actual starting silhouette with a ground-only frame.
    eye, ch = room.bodies["A"].eye, room.channels
    blank = np.full((ch.h, ch.w), ch.ground, dtype=np.float32)
    baseline = eye.look(blank, *ch.gaze)
    image = eye.look(ch.sight_frame(room.arena.A, room.arena.B), *ch.gaze)
    sight_ok = any(np.any(np.abs(np.asarray(image[k]) - v) > 1e-6)
                   for k, v in baseline.items())
    rows = []
    for _ in range(steps):
        r = room.step()
        g = r["before"]
        # MEASURED: pre-window geometry; speed is realised displacement in this window.
        row = {f"{name}_{key}": g[name][key] for name in ("A", "B")
               for key in ("x", "y", "heading_rad")}
        displacement = np.hypot(r["after"]["B"]["x"]-g["B"]["x"],
                                r["after"]["B"]["y"]-g["B"]["y"])
        sound_a, sound_b = HerBody.sound_pair(r["B"]["in"]["sound_hz"])
        row.update(time_s=g["time_s"], distance_mm=g["distance_mm"],
            p1_hz=r["A"].get("p1_hz"), pulse_hz=r["A"].get("pulse_hz", 0.),
            sine_hz=r["A"].get("sine_hz", 0.),
            her_speed_mm_s=float(displacement / bw.WORLD_DT_S),
            **r["B"]["her_answer"], her_sound_hz=max(sound_a, sound_b),
            her_sound_a_hz=sound_a, her_sound_b_hz=sound_b,
            song_hz=r["A"]["song_hz"], his_sound_hz=r["A"]["in"]["sound_hz"])
        rows.append(row)
    trace = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}
    result = outcome(seed, condition, trace, start)
    result.update(sight_ok=bool(sight_ok), sight_check=sight, **room.male_setup)
    return result, trace


def verdict(mean, se, n):
    if n < 2 or se is None:
        return "undetermined"
    return "supported" if mean > 2 * se else "not supported"


def budget_ladder(requested, first_seconds, budget_s):
    candidates = [requested] if requested < 8 else sorted({requested, *[v for v in (10, 8) if v <= requested]}, reverse=True)
    estimates = {str(v): first_seconds * len(CONDITIONS) * v for v in candidates}
    selected = next((v for v in candidates if estimates[str(v)] <= budget_s), min(candidates))
    return dict(requested=requested, candidates=candidates, budget_s=budget_s,
                selected=selected, estimated_seconds=estimates,
                budget_exceeded=estimates[str(selected)] > budget_s)


def summarise(rows):
    indexed = {(r["seed"], r["condition"]): r for r in rows}
    predictions = {}
    for label, key, control, sign in (("P1", "vpodn_hz", "silence", 1),
        ("P2", "pc1_hz", "silence", 1), ("P3", "vpodn_hz", "shuffled", 1),
        ("P4", "last_distance_mm", "silence", -1)):
        diffs = [sign * (indexed[s, "song"][key] - indexed[s, control][key])
                 for s in sorted({r["seed"] for r in rows})]
        mean = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else None
        predictions[label] = dict(mean=mean, se=se, n=len(diffs), paired_differences=diffs,
            metric=key, control=control, direction="song - control" if sign == 1 else "control - song",
            verdict=verdict(mean, se, len(diffs)))
    spread = {}
    for c in CONDITIONS:
        rr = [r for r in rows if r["condition"] == c]
        counts = {k: sum(r[k] for r in rr) for k in ("accept", "approach", "retreat")}
        same = len({tuple(r[k] for k in counts) for r in rr}) == 1
        sentence = f"{c}: every seed gives the same outcome." if same else f"{c}: outcomes differ across seeds."
        spread[c] = dict(n=len(rr), **counts, sentence=sentence)
    p0 = dict(expected_active_windows=0, per_seed=[
        dict(seed=r["seed"], condition=r["condition"], active_windows=r["active_windows"])
        for r in rows if r["condition"] in ("dark", "silence")])
    return dict(P0=p0, predictions=predictions, seed_spread=spread)


def paths(prefix):
    return [Path(str(prefix) + s) for s in ("_experiment.json", "_trajectories.npz", "_trajectories.png", "_report.md")]


def assert_not_published(prefix):
    for target, protected in ((t, p) for t in paths(prefix) for p in paths(PUBLISHED)):
        # Resolve the parent identity, including aliases, before the files exist.
        same_parent = target.parent.exists() and protected.parent.exists() and os.path.samefile(target.parent, protected.parent)
        same_file = target.exists() and protected.exists() and os.path.samefile(target, protected)
        if same_file or (same_parent and target.name.casefold() == protected.name.casefold()):
            raise ValueError("published prefix is refused")


def write_report(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    environment = data.get("environment", {"brain_class": "flysim.FlyBrain", "torch_devices": {"male": None, "female": None}})
    lines = ["# Courtship experiment", "", f"Run: {len(data['seeds'])} seeds x {len(data['protocol_spec']['conditions'])} conditions x {data['steps']} steps; quick={data['quick']}.",
        "Quick runs are smoke tests; their two-seed verdicts are not the full ten-seed experiment.", "",
        "## Question", "Does his song change her graph's answer and their distance?", "",
        "## Measured versus chosen", "MEASURED: positions and headings, realised female speed, delivered sound, song, pC1 and vpoDN window rates.",
        f"IMPLEMENTATION: brain class {environment['brain_class']}; torch devices {environment['torch_devices']}. Device is not a scientific choice. Required equivalence: same numbers on either device, verified by test (COURTSHIP_REAL_BRAIN=1); a failing test invalidates this claim.",
        "CHOSEN: accept means vpoDN > 0 Hz in at least 3 windows; approach/retreat compare last and first quarter mean distance. Equal distance is neither. Geometry is sampled before each window; speed is displacement during it.",
        "CHOSEN: paired seeds, annotation-backed male eye when available, uncalibrated gains, 12 ms brain windows, and the rate-to-motion mapping. No outcomes were tuned to differ across seeds.", "",
        f"CHOSEN: FEMALE_EXC_SCALE = {data['female_exc_scale']}; FEMALE_EYE = {data['female_eye']}.",
        data["protocol_spec"]["text"], "", "## Predictions", "| Prediction | Paired difference | SE | n | Verdict |", "|---|---:|---:|---:|---|"]
    for k, p in data["summary"]["predictions"].items():
        lines.append(f"| {k}: {p['metric']}, {p['direction']} ({p['control']}) | {p['mean']:.6g} | {p['se']} | {p['n']} | {p['verdict']} |")
    lines += ["", "## P0: baseline (descriptive, no verdict)", "Expected active windows: 0."]
    for r in data["summary"]["P0"]["per_seed"]:
        lines.append(f"Seed {r['seed']}, {r['condition']}: {r['active_windows']} active windows.")
    lines += ["", "## Per-seed outcomes", "| Seed | Condition | Accept | Approach | Retreat | Active windows |", "|---:|---|---|---|---|---:|"]
    for r in data["outcomes"]:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['accept']} | {r['approach']} | {r['retreat']} | {r['active_windows']} |")
    lines += ["", "## Seed spread"]
    for c, s in data["summary"]["seed_spread"].items():
        lines += [f"{c}: accept {s['accept']}/{s['n']}, approach {s['approach']}/{s['n']}, retreat {s['retreat']}/{s['n']}. {s['sentence']}"]
    lines += ["", "## P5: his song", "This measures whether his song already depends on her; no adaptation mechanism was added. Correlation does not establish causation. None means a constant series or fewer than two samples.",
              "| Seed | Condition | Song-distance Spearman | Song-speed Spearman |", "|---:|---|---:|---:|"]
    for r in data["outcomes"]:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['song_distance_rho']} | {r['song_speed_rho']} |")
    lines += ["", "## P6 his adaptation (descriptive)",
        "His next pulse/sine window against her previous distance and speed; no verdict or adaptation mechanism. None means unavailable or constant data.",
        "| Seed | Condition | P1 active windows | Pulse-distance lagged rho | Pulse-speed lagged rho | Sine-distance lagged rho | Sine-speed lagged rho |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for r in data["outcomes"]:
        values = [r.get(k) for k in ("p1_active_windows", "pulse_distance_lagged_rho", "pulse_speed_lagged_rho", "sine_distance_lagged_rho", "sine_speed_lagged_rho")]
        lines.append(f"| {r['seed']} | {r['condition']} | " + " | ".join(map(str, values)) + " |")
    interpretation = "; ".join(f"{k} was {p['verdict']}" for k, p in data["summary"]["predictions"].items())
    lines += ["", "## What it means", f"Under this protocol, {interpretation}. These comparisons concern this simulator and input encoding.",
        "<!-- interpretation: to be written after the run -->", "", "## Limitations"]
    lines += ["- " + s for s in data["limitations"]]
    destination = Path(str(json_path).replace("_experiment.json", "_report.md"))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def render_pil(path, data, traces):
    """The plume runner's existing Pillow fallback, with rate traces."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (420 * len(CONDITIONS), 740), "white")
    draw = ImageDraw.Draw(img)
    colors = ("blue", "red", "green", "purple", "orange", "brown", "teal", "magenta", "navy", "gray")
    for col, c in enumerate(CONDITIONS):
        left = 50 + col * 420
        draw.text((left, 10), c + ": her solid / his dashed", fill="black")
        draw.rectangle((left, 50, left+320, 370), outline="black")
        draw.text((left, 375), "x, y: 0 to 20 mm; y increases upward", fill="black")
        selected = [r for r in data["outcomes"] if r["condition"] == c]
        peak = max(1., max(float(traces[f"s{r['seed']}_{c}_vpodn_hz"].max()) for r in selected))
        draw.rectangle((left, 470, left+320, 670), outline="black")
        duration = data["steps"] * bw.WORLD_DT_S
        draw.text((left, 685), f"time 0 to {duration:g} s; vpoDN 0 to {peak:.4g} Hz", fill="black")
        for r in selected:
            color = colors[r["seed"] % len(colors)]
            key = f"s{r['seed']}_{c}_"
            draw.text((left + (r["seed"] % 5)*65, 405 + (r["seed"]//5)*18), f"seed {r['seed']}", fill=color)
            for name in ("A", "B"):
                xy = [(left + x*16, 370-y*16) for x, y in zip(traces[key+name+"_x"], traces[key+name+"_y"])]
                if name == "B":
                    draw.line(xy, fill=color, width=2)
                else:
                    for i in range(0, len(xy)-1, 2):
                        draw.line(xy[i:i+2], fill=color, width=1)
                x, y = xy[0]
                draw.ellipse((x-3, y-3, x+3, y+3), outline=color)
            draw.line([(left + t/duration*320, 670-v/peak*200) for t, v in
                       zip(traces[key+"time_s"], traces[key+"vpodn_hz"])], fill=color, width=2)
    img.save(path)


def save(prefix, data, traces):
    jp, npz, png, _ = paths(prefix)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(npz, **traces)
    try:
        import matplotlib
    except ImportError:
        render_pil(png, data, traces)
        write_report(jp)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 7))
    for col, c in enumerate(CONDITIONS):
        for r in data["outcomes"]:
            if r["condition"] != c:
                continue
            key = f"s{r['seed']}_{c}_"
            line, = axes[0, col].plot(traces[key+"B_x"], traces[key+"B_y"], label=f"her, seed {r['seed']}")
            axes[0, col].plot(traces[key+"A_x"], traces[key+"A_y"], "--", color=line.get_color(), label=f"his, seed {r['seed']}")
            axes[1, col].plot(traces[key+"time_s"], traces[key+"vpodn_hz"], color=line.get_color())
        axes[0, col].set(title=c, xlim=(0, bw.ARENA_MM), ylim=(0, bw.ARENA_MM), xlabel="x (mm)", ylabel="y (mm)")
        axes[0, col].legend(fontsize=6)
        axes[1, col].set(xlabel="time (s)", ylabel="her vpoDN (Hz)")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    write_report(jp)


def main(argv=None, room_factory=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="v3")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--quick", type=int, choices=(0, 1), default=0)
    ap.add_argument("--brain", default="flysim.FlyBrain")
    ap.add_argument("--out", default="build/courtship_v3")
    ap.add_argument("--budget-min", type=float, default=150)
    ap.add_argument("--log")
    ap.add_argument("--reanalyse")
    ap.add_argument("--note")
    args = ap.parse_args(argv)
    prefix = Path(args.out)
    if args.reanalyse:
        jp = Path(args.reanalyse)
        if not str(jp).endswith("_experiment.json"):
            jp = paths(jp)[0]
        prefix = Path(str(jp)[:-len("_experiment.json")])
    try:
        assert_not_published(prefix)
    except ValueError as exc:
        print(exc)
        return 4
    if args.reanalyse:
        data = json.loads(jp.read_text(encoding="utf-8"))
        with np.load(paths(prefix)[1]) as z:
            traces = dict(z)
        data["outcomes"] = [dict(r, **outcome(r["seed"], r["condition"],
            {k.split(f"s{r['seed']}_{r['condition']}_", 1)[1]: v for k, v in traces.items()
             if k.startswith(f"s{r['seed']}_{r['condition']}_")}, r["start"])) for r in data["outcomes"]]
        data["summary"] = summarise(data["outcomes"])
        data["reanalysis_note"] = args.note
        save(prefix, data, traces)
        return 0
    n, steps = (2, 80) if args.quick else (args.seeds, args.steps)
    if n < 1 or steps < 4 or not np.isfinite(args.budget_min) or args.budget_min <= 0:
        ap.error("positive seeds and budget, and at least four steps required")
    factory = room_factory or build_room
    cls = bw.brain_class(args.brain)
    brains = None if room_factory else (cls(), load_female(exc_scale=FEMALE_EXC_SCALE, brain_class=cls))
    environment = dict(brain_class=args.brain, torch_devices={
        name: str(fb.device) if getattr(fb, "device", None) is not None else None
        for name, fb in zip(("male", "female"), brains or (None, None))})
    rows, traces, timings = [], {}, []
    ladder = None
    seed = 0
    while seed < n:
        sound = None
        for c in CONDITIONS:
            room = factory(seed, c, song_sound=sound, brains=brains)
            t0 = time.perf_counter()
            row, trace = run_trial(room, steps, seed, c)
            elapsed = time.perf_counter() - t0
            timings.append(dict(seed=seed, condition=c, seconds=elapsed, step_s=elapsed/steps))
            message = f"seed={seed} condition={c} step_s={elapsed/steps:.6f} active_windows={row['active_windows']}"
            print(message, flush=True)
            if args.log:
                with Path(args.log).open("a", encoding="utf-8") as log:
                    log.write(message + "\n")
            if seed == 0 and c == "song":
                ladder = budget_ladder(n, elapsed, args.budget_min * 60)
                ladder["first_step_s"] = elapsed / steps
                n = ladder["selected"]
            if c == "song":
                sound = np.column_stack((trace["her_sound_a_hz"], trace["her_sound_b_hz"]))
            rows.append(row)
            traces.update({f"s{seed}_{c}_{k}": v for k, v in trace.items()})
        seed += 1
    data = dict(protocol=args.protocol, protocol_spec=PROTOCOLS[args.protocol], steps=steps,
        seeds=list(range(n)), quick=bool(args.quick), outcomes=rows, summary=summarise(rows),
        female_exc_scale=FEMALE_EXC_SCALE, female_eye=FEMALE_EYE,
        limitations=LIMITATIONS, budget_ladder=ladder, timing=timings, environment=environment, note=args.note)
    save(prefix, data, traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

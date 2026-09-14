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
from song import Singer, BIOLOGY
from courtship import WaveEar

# CHOSEN: raw female weights match the male loading convention.
FEMALE_EXC_SCALE = 1.0
# CHOSEN: uniform grey luminance carries no information and floods her brain.
FEMALE_EYE = "blind"
CONDITIONS = {"song": (True, False), "jittered": (True, True),
              "silence": (False, False), "mute": (True, False), "dark": (False, False)}
DEFAULT_STEPS = 400
ACCEPT_WINDOWS = 3
PROTOCOL_TEXT = """CHOSEN before data: all five conditions share seed and arena start.
- song: his previous measured pIP10 mean sets amplitude; per-cell pulse and sine motor means set mode.
- jittered: same amplitude/mode mapping; each IPI is uniform 15-60 ms, seed-derived RNG (seed, 731). Closed-loop trajectories may differ; measured trial RMS ratios need not equal one.
- silence: her waveform is zero; his brain still runs.
- mute: a lesion, the way the tests already lesion; it asks whether the song we synthesise depends on P1. Outgoing gains zero exactly on every dictionary P1 type; all other gains one.
- dark: blind and silent; coincides with silence while FEMALE_EYE is blind.
CHOSEN: pIP10 is the descending song command; no P1, no pIP10, no song. Amplitude clips pIP10 mean / (1000 / refractory_ms). No explicit P1 gate is applied.
CHOSEN: mode = pulse per-cell mean / (pulse per-cell mean + sine per-cell mean), zero if both zero. Waveform = a * (m * pulse + (1-m) * sine), scaled by existing distance falloff.
CHOSEN: 22050 Hz waveform; 35 ms IPI, 4 ms Hann-windowed 250 Hz pulse, 150 Hz sine; phases and sample clock carry across windows.
CHOSEN: JO-A 100-500 Hz, JO-B 500-2500 Hz, Butterworth order 4, sosfiltfilt with padlen 27 per 50 ms waveform. Each of ten 5 ms band-RMS windows drives SOUND_MAX_HZ * clip(RMS / RMS_FULL, 0, 1), equalised per soma side.
CHOSEN: RMS_FULL is JO-A RMS of a one-second full-amplitude pure pulse train including filter edges, computed once at import and printed with ear settings.
CHOSEN: her brain runs in real time so pulse timing can reach it. Ten carried 25-step runs; full-window answers average all 250 steps. His constructor also uses 250 steps: both brains run 50 ms per world step.
CHOSEN: female raw weights (FEMALE_EXC_SCALE = 1.0), female blind eye. Uniform grey is not contrast vision. Male annotation-backed columns and soma sides when available; otherwise luminance and unknown sides, with no invented identities.
CHOSEN: female scent = SMELL_MAX_HZ * falloff(distance) into ORN_VA1v; putative_ppk23 only within CONTACT_MM = 2.0 mm. His ORN_DA1 cVA drive is zero because no other male is present.
CHOSEN: P1 > 0 counts active windows; P7 correlates a[1:], m[1:] with her distance[:-1], speed[:-1]. No adaptation mechanism is added. These readouts decide nothing.
MEASURED: brain rates, song amplitude and mode, delivered waveform RMS, distance, speed, LC10a and P1 activity. Nothing gates either brain.

Predictions fixed before data (paired difference > 2 SE across seeds; P5 requires both comparisons):
- P1 answer: her vpoDN, song > silence.
- P2 pC1: song > silence.
- P3 timing: her vpoDN, song > jittered (now meaningful: a real-time ear and a 35 ms rhythm).
- P4 approach: last-quarter distance, song < silence.
- P5 command: pIP10 mean rate, song > mute, and delivered song RMS, song > mute (the causal chain P1 → pIP10 → song).
- P6 answer follows command: her vpoDN, song > mute.
- P0 baselines (descriptive): silence/dark/mute active windows per seed.
- P7 (descriptive): his P1 active windows, LC10a mean rate per window (275 cells), and lagged correlations of his song amplitude a and mode m with her previous distance and speed.
- Seed spread: accept / approach / retreat per seed and condition; say plainly when every seed gives the same outcome.
""" + "\n" + BIOLOGY
PROTOCOLS = {"v4": {"text": PROTOCOL_TEXT, "conditions": CONDITIONS,
    "default_seeds": 10, "steps": DEFAULT_STEPS, "seed_ladder": (10, 8),
    "budget_s": 9000, "accept_windows": ACCEPT_WINDOWS, "sim_steps": 250,
    "world_dt_s": bw.WORLD_DT_S, "state_carry": True,
    "trial_text": "Trial: `steps` world steps (default 400 = 20 s at 0.05 s; `--quick 1` = 2 seeds x 80 steps). Both brains carry state across steps (the room already does). Record per step: positions and headings of both, distance, her speed, her `vpodn_hz`, `pc1_hz`, her delivered `sound_hz`, his `song_hz`, `p1_hz`, `pulse_hz`, `sine_hz`, both her sound channels, his delivered `sound_hz`; trial eye choice, sides restored, and sight_ok.",
    "outcome_text": "Per-trial outcome (CHOSEN, disclosed): `accept` = her vpoDN window rate exceeds 0 Hz in at least `ACCEPT_WINDOWS` = 3 windows of the trial; `approach` = mean distance in the last quarter of the trial is smaller than in the first quarter; `retreat` = the opposite. Report all three per seed and per condition in a table in the JSON and in the report; never only the pooled mean.",
    "jitter_rng": "numpy default_rng(SeedSequence([seed, 731])).uniform(.015, .060)",
    "quarter_rule": "floor(steps / 4) windows at each end; geometry before the window",
    "budget_note": "First trial measures both graphs per step; choose largest fitting seed count, with floor 8 (requests below 8 kept). Setup and rendering excluded from estimate."}}
LIMITATIONS = [
    "His wing motor neurons run near ceiling from background activity; the song is taken from the command neuron pIP10, not from the motor sum. Mode still uses the motor means.",
    "Her ear now hears a waveform in 5 ms sub-windows; the brain runs in real time for her. His brain also runs in real time.",
    "Pulse rhythm and carriers are chosen synthesis, not measured spike timing or a biomechanical wing model.",
    "The fork's zero-phase filter uses the whole current 50 ms block; boundary effects and within-block lookahead remain. Sample bins alternate lengths at 22050 Hz.",
    "Jitter changes pulse density as well as timing (mean IPI 37.5 ms versus 35 ms). Closed-loop amplitude, mode and distance can differ; realised RMS ratio is measured, not forced.",
    "The contact chemosensory cells are a putative receptor label (putative_ppk23), not verified ppk23 expression.",
    "His female-scent input is a chosen drive at the smell ceiling with distance falloff, not a measured pheromone plume; the cVA channel is held at zero because there is no other male.",
    "He is inside the loop: his trajectory and song can change when she moves differently. Mute additionally changes his P1 outgoing gains.",
    "vpoDN identified as DNp37 by alias, 2 cells; pC1a-e 10 cells. No pheromone channel to her.",
    "Male P1 membership is the dictionary's uncertain 86-cell group. Outgoing-gain lesion need not silence P1's own spikes.",
    "No adaptation mechanism was added on his side; correlation does not establish causation.",
    "CHOSEN: female raw weights for parity; female eye blind. Earlier luminance silence measured vpoDN 54-206 Hz; contrast/motion vision remains a later step.",
    "Dark and silence coincide with the blind female eye.",
    "Male eye and soma-side availability are recorded per trial; gains remain uncalibrated.",
    "Distance changes include both bodies; approach is not an isolated female command.",
    "Retired v3 limitations: rate-only ear, motor-sum song, 12 ms brain windows, separate motor-reference clipping and shuffled-rate multiset no longer describe this protocol."]
PUBLISHED = Path("build/courtship")


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
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        self.condition = condition
        body = self.bodies["A"]
        groups = getattr(body, "groups", {})
        self.singer = Singer(seed, condition == "jittered",
            pulse_cells=len(groups.get("song_pulse_mn", range(8))),
            sine_cells=len(groups.get("song_sine_hg1", range(2))),
            pip10_full=1000/getattr(getattr(getattr(body, "fb", None), "p", None), "refractory", 2.2))
        self.previous_rates = dict(pip10_hz=0., pulse_hz=0., sine_hz=0.)

    def listener_sound(self, sound_hz):
        wave = self.singer.render(**self.previous_rates,
                                  attenuation=self.channels.falloff(self.arena.distance()))
        if self.condition in ("silence", "dark"):
            wave[:] = 0.
            self.singer.record["delivered_rms"] = 0.
        return wave

    def step(self):
        result = super().step()
        male = result["A"]
        rates = male.get("rates", {})
        self.previous_rates = dict(pip10_hz=rates["pIP10"],
                                   pulse_hz=male["pulse_hz"], sine_hz=male["sine_hz"])
        result["song_wave"] = dict(self.singer.record)
        result["sound_clipped"] = result["B"].get("ear_clipped", (False, False))
        return result


def p1_lesion(fb, groups):
    gains = np.ones(len(fb.type_names), dtype=np.float32)
    gains[np.unique(fb.type_code[groups["P1"]])] = 0.
    return gains


def build_room(seed, condition, song_sound=None, brains=None,
               annotations_path="data/body-annotations.feather", brain_class=FlyBrain):
    """One construction for every condition; new bodies reset both states."""
    cls = bw.brain_class(brain_class) if isinstance(brain_class, str) else brain_class
    male, female = brains if brains is not None else (cls(), load_female(exc_scale=FEMALE_EXC_SCALE, brain_class=cls))
    eye = BlindEye(female) if condition == "dark" or FEMALE_EYE == "blind" else LuminanceEye(female)
    body = HerBody("B", female, eye, female_groups(female),
                   female_motor(female), seed=seed * 2 + 2, sim_steps=250)
    available = Path(annotations_path).is_file()
    if available:
        from flyeye import FlyEye
        male_eye = FlyEye(male, annotations_path=str(annotations_path))
        sides = bw.soma_sides(male, annotations_path)
    else:
        male_eye = LuminanceEye(male)
        sides = np.full(male.n, "", dtype=str)
    groups = bd.present_groups(male)
    groups["LC10a"] = male.where(type_re="^LC10a$")
    for key in ("P1", "pIP10", "LC10a"):
        if key not in groups or not len(groups[key]):
            raise ValueError(f"v4 requires measured male group {key}")
    gains = p1_lesion(male, groups) if condition == "mute" else None
    room = ExperimentRoom(male, male_eye, groups,
        bw.motor_groups(male, sides), gains=gains, seed=seed, body_b=body,
        sim_steps=250,
        min_radius_px=bw.column_spacing_px(bw.distinct_columns(male_eye)))
    room.male_setup = dict(male_eye="columnar" if available else "luminance",
        sides_restored=available,
        male_recorded_groups={k: len(groups[k]) for k in ("P1", "pIP10", "LC10a")},
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
    for channel in ("a", "b"):
        key = f"her_sound_{channel}_clipped"
        result[key] = int(np.count_nonzero(trace[key])) if key in trace else None
    result["p1_active_windows"] = (int(np.count_nonzero(p1 > 0))
                                    if p1 is not None and np.all(np.isfinite(p1)) else None)
    for channel in ("pulse", "sine"):
        values = trace.get(channel + "_hz")
        for label, key in (("distance", "distance_mm"), ("speed", "her_speed_mm_s")):
            result[f"{channel}_{label}_lagged_rho"] = (
                correlation(values[1:], trace[key][:-1]) if values is not None else None)
    for key in ("pip10_hz", "lc10a_hz", "a", "m"):
        result[key] = float(np.mean(trace[key])) if key in trace else None
    result["delivered_rms"] = float(np.sqrt(np.mean(trace["delivered_rms"]**2))) if "delivered_rms" in trace else None
    for channel in ("a", "m"):
        for label, key in (("distance", "distance_mm"), ("speed", "her_speed_mm_s")):
            result[f"{channel}_{label}_lagged_rho"] = correlation(trace[channel][1:], trace[key][:-1]) if channel in trace else None
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
        if any(r["A"].get(k) is None for k in ("pulse_hz", "sine_hz")):
            raise ValueError("experiment protocol requires both male song groups: song_pulse_mn and song_sine_hg1")
        g = r["before"]
        # MEASURED: pre-window geometry; speed is realised displacement in this window.
        row = {f"{name}_{key}": g[name][key] for name in ("A", "B")
               for key in ("x", "y", "heading_rad")}
        displacement = np.hypot(r["after"]["B"]["x"]-g["B"]["x"],
                                r["after"]["B"]["y"]-g["B"]["y"])
        sound_a, sound_b = HerBody.sound_pair(r["B"]["in"]["sound_hz"])
        row.update(time_s=g["time_s"], distance_mm=g["distance_mm"],
            p1_hz=r["A"].get("p1_hz"), pulse_hz=r["A"]["pulse_hz"],
            sine_hz=r["A"]["sine_hz"],
            her_sound_a_clipped=r["sound_clipped"][0],
            her_sound_b_clipped=r["sound_clipped"][1],
            her_speed_mm_s=float(displacement / bw.WORLD_DT_S),
            **r["B"]["her_answer"], her_sound_hz=max(sound_a, sound_b),
            her_sound_a_hz=sound_a, her_sound_b_hz=sound_b,
            song_hz=r["A"]["song_hz"], his_sound_hz=r["A"]["in"]["sound_hz"])
        wave = r.get("song_wave", {})
        row.update(pip10_hz=r["A"]["rates"]["pIP10"],
                   lc10a_hz=r["A"]["rates"]["LC10a"],
                   source_pip10_hz=wave.get("pip10_hz", 0.),
                   a=wave.get("a", 0.), m=wave.get("m", 0.),
                   delivered_rms=wave.get("delivered_rms", 0.),
                   her_brain_s=r["B"].get("brain_s", 0.))
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
        ("P2", "pc1_hz", "silence", 1), ("P3", "vpodn_hz", "jittered", 1),
        ("P4", "last_distance_mm", "silence", -1),
        ("P5_command", "pip10_hz", "mute", 1),
        ("P5_rms", "delivered_rms", "mute", 1),
        ("P6", "vpodn_hz", "mute", 1)):
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
    p0 = dict(per_seed=[
        dict(seed=r["seed"], condition=r["condition"], active_windows=r["active_windows"])
        for r in rows if r["condition"] in ("dark", "silence", "mute")])
    ratios = []
    for seed in sorted({r["seed"] for r in rows}):
        baseline = indexed[seed, "song"]["delivered_rms"]
        ratios.append(dict(seed=seed, jittered_song_rms_ratio=(indexed[seed, "jittered"]["delivered_rms"]/baseline if baseline else None)))
    p5 = "supported" if all(predictions[k]["verdict"] == "supported" for k in ("P5_command", "P5_rms")) else ("undetermined" if len(ratios) < 2 else "not supported")
    return dict(P0=p0, P5_verdict=p5, rms_ratios=ratios, predictions=predictions, seed_spread=spread)


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
        "CHOSEN: paired seeds, annotation-backed male eye when available, uncalibrated gains, 50 ms brain windows, and the rate-to-motion mapping. No outcomes were tuned to differ across seeds.", "",
        f"CHOSEN: FEMALE_EXC_SCALE = {data['female_exc_scale']}; FEMALE_EYE = {data['female_eye']}.",
        data["protocol_spec"]["text"], "", "## Predictions", "| Prediction | Paired difference | SE | n | Verdict |", "|---|---:|---:|---:|---|"]
    for k, p in data["summary"]["predictions"].items():
        lines.append(f"| {k}: {p['metric']}, {p['direction']} ({p['control']}) | {p['mean']:.6g} | {p['se']} | {p['n']} | {p['verdict']} |")
    lines += ["", "## P0: baseline (descriptive, no verdict)"]
    for r in data["summary"]["P0"]["per_seed"]:
        lines.append(f"Seed {r['seed']}, {r['condition']}: {r['active_windows']} active windows.")
    lines += ["", "## Per-seed outcomes", "MEASURED: clipped windows count actual ear sub-window clipping in each band.", "| Seed | Condition | Accept | Approach | Retreat | Active windows | vpoDN Hz | pC1 Hz | Last distance mm | pIP10 Hz | Delivered RMS |", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["outcomes"]:
        lines.append("| " + " | ".join(str(r[k]) for k in ("seed", "condition", "accept", "approach", "retreat", "active_windows", "vpodn_hz", "pc1_hz", "last_distance_mm", "pip10_hz", "delivered_rms")) + " |")
    lines += ["", "## Seed spread"]
    for c, s in data["summary"]["seed_spread"].items():
        lines += [f"{c}: accept {s['accept']}/{s['n']}, approach {s['approach']}/{s['n']}, retreat {s['retreat']}/{s['n']}. {s['sentence']}"]
    lines += ["", "## P7 his response (descriptive)",
        "MEASURED: P1 active windows, LC10a mean rate per window, lagged amplitude/mode correlations with her previous distance/speed. None means constant data. No adaptation mechanism was added.",
        "| Seed | Condition | P1 active windows | LC10a Hz | a-distance | a-speed | m-distance | m-speed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["outcomes"]:
        lines.append("| " + " | ".join(str(r.get(k)) for k in ("seed", "condition", "p1_active_windows", "lc10a_hz", "a_distance_lagged_rho", "a_speed_lagged_rho", "m_distance_lagged_rho", "m_speed_lagged_rho")) + " |")
    lines += ["", f"P5 joint verdict: {data['summary']['P5_verdict']}.",
              "MEASURED jittered/song RMS ratios: " + str(data["summary"]["rms_ratios"]),
              "CHOSEN ear settings: " + str(data.get("ear", WaveEar().describe()))]
    lines.append("The realised RMS ratios include condition-dependent command, mode and distance; "
                 "P3 cannot isolate timing when energy differs substantially. Mute tests the "
                 "command chain; its name does not guarantee silence.")
    if data.get("timing"):
        step = float(np.mean([t["step_s"] for t in data["timing"]]))
        female = float(np.mean([t["her_brain_s"] for t in data["timing"]]))
        lines += [f"MEASURED: mean world step {step:.6g} s; female brain step {female:.6g} s. "
                  f"Estimated ten-seed cost (5 x 400 windows each): {step*20000/3600:.3g} hours, excluding setup and rendering."]
        if female > .5:
            lines.append("MEASURED: female brain step exceeds 0.5 s; the full experiment was not run.")
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
    ap.add_argument("--protocol", choices=PROTOCOLS, default="v4")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--quick", type=int, choices=(0, 1), default=0)
    ap.add_argument("--brain", default="flysim.FlyBrain")
    ap.add_argument("--out", default="build/courtship_v4_local")
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
            timings.append(dict(seed=seed, condition=c, seconds=elapsed, step_s=elapsed/steps,
                                her_brain_s=float(trace["her_brain_s"].mean())))
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
        ear=WaveEar().describe(), limitations=LIMITATIONS, budget_ladder=ladder, timing=timings, environment=environment, note=args.note)
    save(prefix, data, traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

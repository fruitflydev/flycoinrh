"""Seed-paired courtship: a male sings and a female graph answers."""
import argparse
import hashlib
from datetime import datetime, timezone
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
from courtship import pc1_lesion, install_p1_drive, P1_DRIVE_HZ

# CHOSEN: raw female weights match the male loading convention.
FEMALE_EXC_SCALE = 1.0
# CHOSEN: uniform grey luminance carries no information and floods her brain.
FEMALE_EYE = "blind"
ENABLE_DARK = False
CONDITIONS = {"song": (True, False), "jittered": (True, True),
              "silence": (False, False), "mute": (True, False), "noscent": (True, False),
              "mated": (True, False)}
if ENABLE_DARK:
    CONDITIONS["dark"] = (False, False)
DEFAULT_STEPS = 400
ACCEPT_WINDOWS = 3
PROTOCOL_TEXT = """Protocol v5. CHOSEN before data: all six conditions share seed and arena start.
- virgin: SpsP driven at a tonic SPSN_HZ = 50 Hz every window (CHOSEN; "the sensory pathway that reports an unmated uterus is on").
- mated: SpsP driven at 0 Hz (CHOSEN; "sex peptide has silenced it").
CHOSEN: song, jittered, silence, mute and noscent are virgin; mated is identical to song except her state is mated. This adds a tonic SpsP drive relative to v4 and changes the physics: v5 predictions are fixed again before data. Today's prior experiment had no SpsP drive at all: it silently ran the mated encoding.
CHOSEN: courtship starts when he can see her. She starts 6 mm ahead of him, offset by a seed-derived angle within +/-30 degrees of his heading, facing a seed-derived random heading. All other arena rules are unchanged.
UNCERTAIN: SpsP identity is taken from the FlyWire name and not verified. Biology (not verified in session): active virgin SPSN, through SAG, keep pC1 receptive; sex peptide silences SPSN after mating and receptivity falls. Citations, not verified in session: Yapici et al. 2008 Nature 451:33 (sex peptide receptor); Feng et al. 2014 Neuron 83:135 (SPSN to SAG to pC1); Wang et al. 2021 Nature 589:577 (vpoDN); Wang et al. 2020 Nature 579:101 (oviDN, mating and egg laying).
- song: his previous measured pIP10 mean sets amplitude; per-cell pulse and sine motor means set mode.
- jittered: replay the paired song first-recorded a, m and distance series; each IPI is uniform 15-60 ms, seed-derived RNG (seed, 731). CHOSEN: one trial-wide gain matches delivered RMS to song, including pulse-density differences; his live brain still runs and is recorded.
- silence: her waveform is zero; his brain still runs.
- mute: a lesion, the way the tests already lesion; it asks whether the song we synthesise depends on P1. Outgoing gains zero exactly on every dictionary P1 type; all other gains one.
- noscent: identical to song with both Or47b and contact scent drives to him zero.
- dark is excluded because it duplicates silence while FEMALE_EYE is blind; ENABLE_DARK can re-enable it.
CHOSEN: pIP10 is the descending song command; zero pIP10 produces zero song. Dependence on P1 is tested, not assumed. Amplitude clips pIP10 mean / (1000 / refractory_ms). No explicit P1 gate is applied.
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
- P8 presence: his P1 mean rate, song > noscent.
- P8b (descriptive): his LC10a mean rate, song versus noscent; no verdict.
- P0 baselines (descriptive): silence/mute active windows per seed.
- P7 (descriptive): his P1 active windows, LC10a mean rate per window (275 cells), and lagged correlations of his song amplitude a and mode m with her previous distance and speed.
- P9 she can say no: her vpoDN mean rate, song (virgin) > mated.
- P10 rejection (descriptive, no verdict): oviDN mean rate and retreat fraction per seed, mated vs song.
- Seed spread: per condition, seeds with accept / no-accept; if `mated` gives accept on every seed the report says "the state did not produce a no" in one sentence.
- P11 he sees her (descriptive with a verdict rule): his LC10a mean rate in windows with sight > in windows without sight, paired within trial, across seeds, > 2 SE (song condition).
- P12 adaptation (descriptive, no verdict): lagged correlations already in P7, plus `m[1:]` vs `LC10a[:-1]` (does his song mode follow what his LC10a saw a window earlier).
MEASURED: per-window ovidn_hz (mean over six oviDN cells), spsp_hz and sight_ok (silhouette reaches his retinal samples); per-trial sight fraction. P11 uses only song trials with both sight and no-sight windows; missing pairs are excluded and fewer than two pairs is undetermined. P10 retreat fraction is the fraction of adjacent pre-window distances that increase, excluding the first window, which has no previous distance.
- Seed spread: accept / approach / retreat per seed and condition; say plainly when every seed gives the same outcome.
""" + "\n" + BIOLOGY
PROTOCOLS = {"v5": {"text": PROTOCOL_TEXT, "conditions": CONDITIONS,
    "default_seeds": 10, "steps": DEFAULT_STEPS, "seed_ladder": (10, 8),
    "budget_s": 9000, "accept_windows": ACCEPT_WINDOWS, "sim_steps": 250,
    "world_dt_s": bw.WORLD_DT_S, "state_carry": True,
    "trial_text": "Trial: `steps` world steps (default 400 = 20 s at 0.05 s; `--quick 1` = 2 seeds x 80 steps). Both brains carry state across steps (the room already does). Record per step: positions and headings of both, distance, her speed, her `vpodn_hz`, `pc1_hz`, her delivered `sound_hz`, his `song_hz`, `p1_hz`, `pulse_hz`, `sine_hz`, both her sound channels, his delivered `sound_hz`; trial eye choice, sides restored, and sight_ok.",
    "outcome_text": "Per-trial outcome (CHOSEN, disclosed): `accept` = her vpoDN window rate exceeds 0 Hz in at least `ACCEPT_WINDOWS` = 3 windows of the trial; `approach` = mean distance in the last quarter of the trial is smaller than in the first quarter; `retreat` = the opposite. Report all three per seed and per condition in a table in the JSON and in the report; never only the pooled mean.",
    "jitter_rng": "numpy default_rng(SeedSequence([seed, 731])).uniform(.015, .060)",
    "quarter_rule": "floor(steps / 4) windows at each end; geometry before the window",
    "budget_note": "First trial measures both graphs per step; choose largest fitting seed count, with floor 8 (requests below 8 kept). Setup and rendering excluded from estimate."}}
LIMITATIONS = [
    "her state is a chosen tonic drive on SpsP, an identity taken from the FlyWire name and not verified; oviDN is a readout of a descending command, she has no body to extrude an ovipositor with; the start geometry is chosen so that she is visible to him",
    "His wing motor neurons run near ceiling from background activity; the song is taken from the command neuron pIP10, not from the motor sum. Mode still uses the motor means.",
    "Her ear now hears a waveform in 5 ms sub-windows; the brain runs in real time for her. His brain also runs in real time.",
    "Pulse rhythm and carriers are chosen synthesis, not measured spike timing or a biomechanical wing model.",
    "The fork's zero-phase filter uses the whole current 50 ms block; boundary effects and within-block lookahead remain. Sample bins alternate lengths at 22050 Hz.",
    "Jitter changes pulse density as well as timing (mean IPI 37.5 ms versus 35 ms). Replay holds source amplitude, mode and distance fixed; one trial-wide gain matches energy. Local envelopes and spectral energy can still differ.",
    "The contact chemosensory cells are a putative receptor label (putative_ppk23), not verified ppk23 expression.",
    "His female-scent input is a chosen drive at the smell ceiling with distance falloff, not a measured pheromone plume; the cVA channel is held at zero because there is no other male.",
    "He is inside the loop: his trajectory and song can change when she moves differently. Mute additionally changes his P1 outgoing gains.",
    "vpoDN identified as DNp37 by alias, 2 cells; pC1a-e 10 cells. No pheromone channel to her.",
    "In closed-loop 50 ms windows, silencing P1 outputs did not reduce pIP10 in the v4 smoke run; pIP10 is driven mainly by AVLP717m and aIPg7 in this graph, so the song dependence on P1 is not established here.",
    "Male P1 membership is the dictionary's uncertain 86-cell group. Outgoing-gain lesion need not silence P1's own spikes.",
    "No adaptation mechanism was added on his side; correlation does not establish causation.",
    "CHOSEN: female raw weights for parity; female eye blind. Earlier luminance silence measured vpoDN 54-206 Hz; contrast/motion vision remains a later step.",
    "Dark and silence coincide with the blind female eye.",
    "Male eye and soma-side availability are recorded per trial; gains remain uncalibrated.",
    "Distance changes include both bodies; approach is not an isolated female command.",
    "Retired v3 limitations: rate-only ear, motor-sum song, 12 ms brain windows, separate motor-reference clipping and shuffled-rate multiset no longer describe this protocol."]
PUBLISHED = Path("build/courtship")
PUBLISHED_ADDENDUM = Path("build/courtship_addendum")
V6_CONDITIONS = {"gated": (True, False), "p1drive": (True, False)}
V6_PROTOCOL_TEXT = """Protocol v6 ADDENDUM. Only gated and p1drive are run; controls are the v5 song records paired by seed. Seed RNGs, start geometry, steps, brains and all other v5 song settings are retained.
CHOSEN gated: identical to v5 song (virgin), with outgoing gains 0.0 on exactly pC1a, pC1b, pC1c, pC1d, pC1e; other gains one. a mated female cannot be encoded through her own sex-peptide pathway in this map, because the SPSN and SAG axons carry no synapses here; the receptivity gate is closed by hand instead, as a lesion, the way the tests lesion.
CHOSEN p1drive: identical to v5 song, plus P1_DRIVE_HZ = 100 Hz to his P1 cells every window. asks whether the courtship command group can drive the song and her answer from above; it is an intervention, not a claim that P1 fires like this on its own.
Predictions, fixed before data:
- P9' she can be made to say no: her vpoDN mean rate, v5 song > gated (paired by seed), > 2 SE.
- P10' rejection (descriptive): oviDN mean and retreat fraction, gated vs v5 song.
- P13 the command can sing: his pIP10 mean rate and the delivered song RMS, p1drive > v5 song, both > 2 SE (joint verdict as P5).
- P14 the command reaches her: her vpoDN mean rate, p1drive > v5 song, > 2 SE.
- P15 (descriptive): his P1 active windows and LC10a in p1drive vs v5 song.
- Seed spread: per condition, seeds with accept / no-accept and approach / retreat; say plainly when every seed gives the same outcome.
"""
PROTOCOLS["v6"] = dict(PROTOCOLS["v5"], text=V6_PROTOCOL_TEXT,
    conditions=V6_CONDITIONS, budget_note="Run exactly the requested paired baseline seeds; no seed-count adaptation.")
V6_LIMITATIONS = [
    "Measured on the real graphs in GPU 250-step probes: the female brain-only FlyWire FAFB v783 map has no outgoing synapses from any of seven SpsP cells or either of two AN_SMP_2 (SAG) cells. Driving SpsP at 25-400 Hz or SAG at 25-200 Hz left pC1, vpoDN and oviDN unchanged within noise. v5 mated drives a dead switch: P9 cannot be supported for this anatomical reason. Gated is a manual lesion, not a mating-state encoding.",
    "Measured male probes: silence gave P1 0 Hz; Or47b scent gave P1 24-50 Hz, putative_ppk23 contact 24-46 Hz, and LC10a vision 26 Hz. LC10a drive gave pIP10 111 Hz; direct P1 drive at 100 Hz gave pIP10 76 Hz. P1 can drive pIP10, but vision also drives it, explaining why mute did not lower pIP10 in the loop. These are motivating probes, not v6 outcomes.",
    *LIMITATIONS,
    "The baseline is a separate v5 run; its path and byte SHA256 identify the controls. P1 drive is an imposed intervention, not spontaneous firing. The retained v5 state-drive limitation describes the disconnected tonic input; v6 closes pC1 by hand."]


def read_baseline(path, seeds, steps, brain):
    if not path:
        raise ValueError("v6 requires --baseline")
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("protocol") != "v5" or data.get("seeds") != seeds or data.get("steps") != steps:
        raise ValueError("baseline protocol, seeds or steps mismatch")
    env = data.get("environment", {})
    classes = env.get("brain_classes", {"male": env.get("brain_class"), "female": env.get("brain_class")})
    if classes != {"male": brain, "female": brain}:
        raise ValueError("baseline male/female brain classes mismatch")
    for key, expected in (("female_exc_scale", FEMALE_EXC_SCALE), ("female_eye", FEMALE_EYE)):
        if data.get(key) != expected:
            raise ValueError("baseline settings mismatch: " + key)
    song = [r for r in data["outcomes"] if r["condition"] == "song"]
    if sorted(r["seed"] for r in song) != seeds:
        raise ValueError("baseline song seeds mismatch")
    for key in ("sim_steps", "world_dt_s", "state_carry", "accept_windows", "quarter_rule"):
        if data.get("protocol_spec", {}).get(key) != PROTOCOLS["v5"][key]:
            raise ValueError("baseline protocol settings mismatch: " + key)
    for r in data["outcomes"]:
        check_baseline_start(data, r["seed"], r["start"])
    return data, dict(path=Path(os.path.relpath(path)).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                      date=data.get("date", "not recorded"))


def check_baseline_start(baseline, seed, start):
    expected = next(r["start"] for r in baseline["outcomes"] if r["seed"] == seed and r["condition"] == "song")
    if expected != start:
        raise ValueError(f"baseline start mismatch for seed {seed}")


def summarise_v6(rows, baseline_rows):
    indexed = {(r["seed"], r["condition"]): r for r in rows + baseline_rows}
    seeds = sorted({r["seed"] for r in rows})
    predictions = {}
    for label, metric, condition, sign in (("P9'", "vpodn_hz", "gated", -1),
        ("P13_command", "pip10_hz", "p1drive", 1), ("P13_rms", "delivered_rms", "p1drive", 1),
        ("P14", "vpodn_hz", "p1drive", 1)):
        diffs = [sign*(indexed[s, condition][metric]-indexed[s, "song"][metric]) for s in seeds]
        mean = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1)/np.sqrt(len(diffs))) if len(diffs) > 1 else None
        predictions[label] = dict(mean=mean, se=se, n=len(diffs), paired_differences=diffs,
            metric=metric, control="v5 song", direction=f"{condition} - v5 song" if sign == 1 else "v5 song - gated",
            verdict=verdict(mean, se, len(diffs)))
    spread = {}
    for c in V6_CONDITIONS:
        rr = [r for r in rows if r["condition"] == c]
        counts = {k: sum(r[k] for r in rr) for k in ("accept", "approach", "retreat")}
        same = len({tuple(r[k] for k in counts) for r in rr}) == 1
        spread[c] = dict(n=len(rr), **counts, no_accept=len(rr)-counts["accept"],
            accept_seeds=[r["seed"] for r in rr if r["accept"]], no_accept_seeds=[r["seed"] for r in rr if not r["accept"]],
            sentence=f"{c}: every seed gives the same outcome." if same else f"{c}: outcomes differ across seeds.")
    descriptive = {}
    for label, c, metrics in (("P10'", "gated", ("ovidn_hz", "retreat_fraction")),
                             ("P15", "p1drive", ("p1_active_windows", "lc10a_hz"))):
        descriptive[label] = [{k: indexed[s, cc][k] for k in ("seed", "condition", *metrics)}
                              for s in seeds for cc in (c, "song")]
    joint = [predictions[k]["verdict"] for k in ("P13_command", "P13_rms")]
    return dict(predictions=predictions, seed_spread=spread, **descriptive,
                P13_verdict="supported" if all(v == "supported" for v in joint) else
                "undetermined" if len(seeds) < 2 else "not supported")


def write_v6_report(json_path, data):
    lines = ["# Courtship v6 addendum", "", "Baseline: " + str(data["baseline"]),
        "Quick runs are smoke tests, not the full experiment.", data["protocol_spec"]["text"],
        "| Seed | Condition | Accept | Approach | Retreat | vpoDN Hz | pIP10 Hz | Delivered RMS |",
        "|---:|---|---|---|---|---:|---:|---:|"]
    for r in data["outcomes"]:
        lines.append("| " + " | ".join(str(r[k]) for k in ("seed", "condition", "accept", "approach", "retreat", "vpodn_hz", "pip10_hz", "delivered_rms")) + " |")
    for label, p in data["summary"]["predictions"].items():
        lines.append(f"{label}: {p}")
    lines.append("P13 joint verdict: " + data["summary"]["P13_verdict"])
    for label in ("P10'", "P15"):
        lines += [label + " (descriptive):", *[str(r) for r in data["summary"][label]]]
    for s in data["summary"]["seed_spread"].values():
        lines.append(str(s))
    lines += ["", "## Limitations", *["- " + s for s in data["limitations"]]]
    destination = Path(str(json_path).replace("_experiment.json", "_report.md"))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


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
        if condition not in CONDITIONS and condition not in V6_CONDITIONS and condition != "dark":
            raise ValueError("unknown condition")
        self.condition = condition
        rng = np.random.default_rng(np.random.SeedSequence([seed, 905]))
        angle = self.arena.A.heading + rng.uniform(-np.pi/6, np.pi/6)
        # Translate the pair together only if needed to retain the existing arena bounds.
        dx, dy = 6*np.cos(angle), 6*np.sin(angle)
        margin = bw.START_MARGIN_MM
        self.arena.A.x = float(np.clip(self.arena.A.x, margin-min(0., dx), bw.ARENA_MM-margin-max(0., dx)))
        self.arena.A.y = float(np.clip(self.arena.A.y, margin-min(0., dy), bw.ARENA_MM-margin-max(0., dy)))
        self.arena.B.x, self.arena.B.y = self.arena.A.x+dx, self.arena.A.y+dy
        self.arena.B.heading = float(rng.uniform(-np.pi, np.pi))
        self.bodies["B"].state = "mated" if condition == "mated" else "virgin"
        body = self.bodies["A"]
        groups = getattr(body, "groups", {})
        self.singer = Singer(seed, condition == "jittered",
            pulse_cells=len(groups.get("song_pulse_mn", range(8))),
            sine_cells=len(groups.get("song_sine_hg1", range(2))),
            pip10_full=1000/getattr(getattr(getattr(body, "fb", None), "p", None), "refractory", 2.2))
        self.previous_rates = dict(pip10_hz=0., pulse_hz=0., sine_hz=0.)
        self.replay = []
        self.replay_step = 0
        if condition == "jittered":
            if song_sound is None:
                raise ValueError("jittered requires the paired song trace first")
            for a, m, d in zip(song_sound["a"], song_sound["m"], song_sound["distance_mm"]):
                wave = self.singer.render(a*self.singer.pip10_full,
                    m*self.singer.pulse_cells, (1-m)*self.singer.sine_cells,
                    attenuation=self.channels.falloff(d))
                self.replay.append((wave, dict(self.singer.record, source_distance_mm=float(d))))
            raw = np.sqrt(np.mean([r[1]["delivered_rms"]**2 for r in self.replay]))
            target = np.sqrt(np.mean(song_sound["delivered_rms"]**2))
            if raw == 0 and target > 0:
                raise ValueError("cannot energy-match an empty jittered waveform")
            gain = float(target/raw) if raw else 1.
            for wave, record in self.replay:
                wave *= gain
                record.update(delivered_rms=float(np.sqrt(np.mean(wave**2))), replay_gain=gain)

    def listener_scent(self, smell):
        if self.condition == "noscent":
            return {"A": {"female_scent_orn": 0., "female_scent_contact": 0., bw.SMELL_KEY: 0.}, "B": 0.}
        return smell

    def listener_sound(self, sound_hz):
        if self.condition == "jittered":
            wave, self.singer.record = self.replay[self.replay_step]
            self.replay_step += 1
            return wave.copy()
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
            raise ValueError(f"v5 requires measured male group {key}")
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
    if condition == "gated":
        body.gains = pc1_lesion(female)
    if condition == "p1drive":
        install_p1_drive(room.bodies["A"])
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
        state="mated" if condition == "mated" else "virgin",
        ovidn_hz=float(np.mean(trace["ovidn_hz"])), spsp_hz=float(np.mean(trace["spsp_hz"])),
        sight_fraction=float(np.mean(trace["sight_ok"])),
        retreat_fraction=float(np.mean(np.diff(trace["distance_mm"]) > 0)),
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
    for key in ("p1_hz", "pip10_hz", "lc10a_hz", "a", "m"):
        result[key] = float(np.mean(trace[key])) if key in trace else None
    result["delivered_rms"] = float(np.sqrt(np.mean(trace["delivered_rms"]**2))) if "delivered_rms" in trace else None
    for channel in ("a", "m"):
        for label, key in (("distance", "distance_mm"), ("speed", "her_speed_mm_s")):
            result[f"{channel}_{label}_lagged_rho"] = correlation(trace[channel][1:], trace[key][:-1]) if channel in trace else None
    result["m_lc10a_lagged_rho"] = correlation(trace["m"][1:], trace["lc10a_hz"][:-1])
    seen = trace["sight_ok"].astype(bool)
    result["lc10a_sight_hz"] = float(trace["lc10a_hz"][seen].mean()) if seen.any() else None
    result["lc10a_no_sight_hz"] = float(trace["lc10a_hz"][~seen].mean()) if (~seen).any() else None
    result["lc10a_sight_difference"] = (result["lc10a_sight_hz"]-result["lc10a_no_sight_hz"]
                                        if seen.any() and (~seen).any() else None)
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
        image = eye.look(ch.sight_frame(room.arena.A, room.arena.B), *ch.gaze)
        window_sight = any(np.any(np.abs(np.asarray(image[k])-v) > 1e-6)
                           for k, v in baseline.items())
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
        row.update(time_s=g["time_s"], distance_mm=g["distance_mm"], sight_ok=bool(window_sight),
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
                   source_distance_mm=wave.get("source_distance_mm", g["distance_mm"]),
                   replay_gain=wave.get("replay_gain", 1.),
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
        ("P6", "vpodn_hz", "mute", 1),
        ("P8", "p1_hz", "noscent", 1),
        ("P8b", "lc10a_hz", "noscent", 1),
        ("P9", "vpodn_hz", "mated", 1)):
        diffs = [sign * (indexed[s, "song"][key] - indexed[s, control][key])
                 for s in sorted({r["seed"] for r in rows})]
        mean = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else None
        predictions[label] = dict(mean=mean, se=se, n=len(diffs), paired_differences=diffs,
            metric=key, control=control, direction="song - control" if sign == 1 else "control - song",
            verdict=verdict(mean, se, len(diffs)))
        if label == "P8b":
            predictions[label]["verdict"] = "descriptive; no verdict"
    song = [r for r in rows if r["condition"] == "song"]
    diffs = [r["lc10a_sight_difference"] for r in song if r["lc10a_sight_difference"] is not None]
    mean = float(np.mean(diffs)) if diffs else None
    se = float(np.std(diffs, ddof=1)/np.sqrt(len(diffs))) if len(diffs) > 1 else None
    predictions["P11"] = dict(mean=mean, se=se, n=len(diffs), paired_differences=diffs,
        excluded_seeds=[r["seed"] for r in song if r["lc10a_sight_difference"] is None],
        metric="lc10a_hz", control="no-sight windows within song trial", direction="sight - no sight",
        verdict=verdict(mean, se, len(diffs)))
    spread = {}
    for c in CONDITIONS:
        rr = [r for r in rows if r["condition"] == c]
        counts = {k: sum(r[k] for r in rr) for k in ("accept", "approach", "retreat")}
        same = len({tuple(r[k] for k in counts) for r in rr}) == 1
        sentence = f"{c}: every seed gives the same outcome." if same else f"{c}: outcomes differ across seeds."
        spread[c] = dict(n=len(rr), **counts, no_accept=len(rr)-counts["accept"],
                        accept_seeds=[r["seed"] for r in rr if r["accept"]],
                        no_accept_seeds=[r["seed"] for r in rr if not r["accept"]], sentence=sentence)
        if c == "mated" and rr and counts["accept"] == len(rr):
            spread[c]["state_sentence"] = "the state did not produce a no"
    p0 = dict(per_seed=[
        dict(seed=r["seed"], condition=r["condition"], active_windows=r["active_windows"])
        for r in rows if r["condition"] in ("dark", "silence", "mute")])
    ratios = []
    for seed in sorted({r["seed"] for r in rows}):
        baseline = indexed[seed, "song"]["delivered_rms"]
        ratios.append(dict(seed=seed, jittered_song_rms_ratio=(indexed[seed, "jittered"]["delivered_rms"]/baseline if baseline else None)))
    p5 = "supported" if all(predictions[k]["verdict"] == "supported" for k in ("P5_command", "P5_rms")) else ("undetermined" if len(ratios) < 2 else "not supported")
    p10 = [dict(seed=r["seed"], condition=r["condition"], ovidn_hz=r["ovidn_hz"],
                retreat_fraction=r["retreat_fraction"]) for r in rows if r["condition"] in ("song", "mated")]
    p12 = [{k: r[k] for k in ("seed", "condition", "a_distance_lagged_rho", "a_speed_lagged_rho",
            "m_distance_lagged_rho", "m_speed_lagged_rho", "m_lc10a_lagged_rho")} for r in rows]
    return dict(P0=p0, P5_verdict=p5, P10=p10, P12=p12, rms_ratios=ratios, predictions=predictions, seed_spread=spread)


def paths(prefix):
    return [Path(str(prefix) + s) for s in ("_experiment.json", "_trajectories.npz", "_trajectories.png", "_report.md")]


def assert_not_published(prefix):
    for target, protected in ((t, p) for t in paths(prefix)
                              for root in (PUBLISHED, PUBLISHED_ADDENDUM) for p in paths(root)):
        # Resolve the parent identity, including aliases, before the files exist.
        same_parent = target.parent.exists() and protected.parent.exists() and os.path.samefile(target.parent, protected.parent)
        same_file = target.exists() and protected.exists() and os.path.samefile(target, protected)
        if same_file or (same_parent and target.name.casefold() == protected.name.casefold()):
            raise ValueError("published prefix is refused")


def write_report(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if data["protocol"] == "v6":
        return write_v6_report(json_path, data)
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
        lines.append(f"| {k}: {p['metric']}, {p['direction']} ({p['control']}) | {p['mean']} | {p['se']} | {p['n']} | {p['verdict']} |")
    lines += ["", "## P0: baseline (descriptive, no verdict)"]
    for r in data["summary"]["P0"]["per_seed"]:
        lines.append(f"Seed {r['seed']}, {r['condition']}: {r['active_windows']} active windows.")
    lines += ["", "## Per-seed outcomes", "MEASURED: clipped windows count actual ear sub-window clipping in each band.", "| Seed | Condition | State | Accept | Approach | Retreat | Active windows | vpoDN Hz | pC1 Hz | oviDN Hz | Sight fraction | Last distance mm | pIP10 Hz | Delivered RMS |", "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["outcomes"]:
        lines.append("| " + " | ".join(str(r[k]) for k in ("seed", "condition", "state", "accept", "approach", "retreat", "active_windows", "vpodn_hz", "pc1_hz", "ovidn_hz", "sight_fraction", "last_distance_mm", "pip10_hz", "delivered_rms")) + " |")
    lines += ["", "## Seed spread"]
    for c, s in data["summary"]["seed_spread"].items():
        lines += [f"{c}: accept {s['accept']}/{s['n']}, approach {s['approach']}/{s['n']}, retreat {s['retreat']}/{s['n']}. {s['sentence']}"]
        lines.append(f"Accept seeds: {s['accept_seeds']}; no-accept seeds: {s['no_accept_seeds']}.")
        if "state_sentence" in s:
            lines.append(s["state_sentence"])
    lines += ["", "## P9 she can say no", str(data["summary"]["predictions"]["P9"]),
              "", "## P10 rejection (descriptive, no verdict)",
              "Retreat fraction: adjacent pre-window distances that increase / all adjacent pairs."]
    lines += [str(r) for r in data["summary"]["P10"]]
    lines += ["", "## P11 he sees her", str(data["summary"]["predictions"]["P11"]),
              "Paired within song trial; exclude trials missing either category; fewer than two pairs is undetermined."]
    lines += [str({k: r[k] for k in ("seed", "sight_fraction", "lc10a_sight_hz", "lc10a_no_sight_hz")})
              for r in data["outcomes"] if r["condition"] == "song"]
    lines += ["", "## P12 adaptation (descriptive, no verdict)",
              "Spearman correlations: a[1:] and m[1:] versus previous distance/speed, and m[1:] versus LC10a[:-1]; None means constant data."]
    lines += [str(r) for r in data["summary"]["P12"]]
    lines += ["", "## P7 his response (descriptive)",
        "MEASURED: P1 active windows, LC10a mean rate per window, lagged amplitude/mode correlations with her previous distance/speed. None means constant data. No adaptation mechanism was added.",
        "| Seed | Condition | P1 active windows | LC10a Hz | a-distance | a-speed | m-distance | m-speed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["outcomes"]:
        lines.append("| " + " | ".join(str(r.get(k)) for k in ("seed", "condition", "p1_active_windows", "lc10a_hz", "a_distance_lagged_rho", "a_speed_lagged_rho", "m_distance_lagged_rho", "m_speed_lagged_rho")) + " |")
    lines += ["", f"P5 joint verdict: {data['summary']['P5_verdict']}.",
              "MEASURED jittered/song RMS ratios: " + str(data["summary"]["rms_ratios"]),
              "CHOSEN ear settings: " + str(data.get("ear", WaveEar().describe()))]
    lines.append("Replay uses song command, mode and distance with a disclosed trial-wide energy gain; "
                 "P3 cannot isolate timing when energy differs substantially. Mute tests the "
                 "command chain; its name does not guarantee silence.")
    if data.get("timing"):
        step = float(np.mean([t["step_s"] for t in data["timing"]]))
        female = float(np.mean([t["her_brain_s"] for t in data["timing"]]))
        lines += [f"MEASURED: mean world step {step:.6g} s; female brain step {female:.6g} s. "
                  f"Estimated ten-seed cost ({len(CONDITIONS)} x 400 windows each): {step*len(CONDITIONS)*4000/3600:.3g} hours, excluding setup and rendering."]
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
    conditions = data["protocol_spec"]["conditions"]
    img = Image.new("RGB", (420 * len(conditions), 740), "white")
    draw = ImageDraw.Draw(img)
    colors = ("blue", "red", "green", "purple", "orange", "brown", "teal", "magenta", "navy", "gray")
    for col, c in enumerate(conditions):
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
    conditions = data["protocol_spec"]["conditions"]
    fig, axes = plt.subplots(2, len(conditions), figsize=(4 * len(conditions), 7))
    for col, c in enumerate(conditions):
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
    ap.add_argument("--protocol", choices=PROTOCOLS, default="v5")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--quick", type=int, choices=(0, 1), default=0)
    ap.add_argument("--brain", default="flysim.FlyBrain")
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    ap.add_argument("--budget-min", type=float, default=150)
    ap.add_argument("--log")
    ap.add_argument("--reanalyse")
    ap.add_argument("--note")
    args = ap.parse_args(argv)
    prefix = Path(args.out or f"build/courtship_{args.protocol}_quick")
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
        if data["protocol"] == "v6":
            baseline, identity = read_baseline(args.baseline or data["baseline"]["path"], data["seeds"], data["steps"], data["environment"]["brain_class"])
            if identity["sha256"] != data["baseline"]["sha256"]:
                raise ValueError("baseline SHA256 changed")
            for r in data["outcomes"]:
                check_baseline_start(baseline, r["seed"], r["start"])
            data["summary"] = summarise_v6(data["outcomes"], baseline["outcomes"])
        else:
            data["summary"] = summarise(data["outcomes"])
        data["reanalysis_note"] = args.note
        save(prefix, data, traces)
        return 0
    n, steps = (2, 80) if args.quick else (args.seeds, args.steps)
    if n < 1 or steps < 4 or not np.isfinite(args.budget_min) or args.budget_min <= 0:
        ap.error("positive seeds and budget, and at least four steps required")
    conditions = PROTOCOLS[args.protocol]["conditions"]
    baseline = baseline_identity = None
    if args.protocol == "v6":
        baseline, baseline_identity = read_baseline(args.baseline, list(range(n)), steps, args.brain)
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
        for c in conditions:
            room = factory(seed, c, song_sound=sound, brains=brains)
            if baseline is not None:
                check_baseline_start(baseline, seed, room.arena.geometry())
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
                sound = trace
            rows.append(row)
            traces.update({f"s{seed}_{c}_{k}": v for k, v in trace.items()})
        seed += 1
    data = dict(protocol=args.protocol, protocol_spec=PROTOCOLS[args.protocol], steps=steps,
        seeds=list(range(n)), quick=bool(args.quick), outcomes=rows,
        summary=summarise_v6(rows, baseline["outcomes"]) if baseline is not None else summarise(rows),
        female_exc_scale=FEMALE_EXC_SCALE, female_eye=FEMALE_EYE,
        ear=WaveEar().describe(), limitations=V6_LIMITATIONS if baseline is not None else LIMITATIONS,
        budget_ladder=ladder, timing=timings, environment=environment, note=args.note,
        date=datetime.now(timezone.utc).isoformat())
    if baseline_identity is not None:
        data["baseline"] = baseline_identity
    save(prefix, data, traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

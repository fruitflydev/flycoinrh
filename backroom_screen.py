"""
Offline gate for the backroom, on the real connectome with no network.

  py backroom_screen.py [seeds=40] [pairings=12] [out=build/backroom_screen.json]

Questions:
  1. does a coin paired with profit (sugar) become more liked?
  2. does a coin paired with loss (shock) become less liked?
  3. how far does a coin that was never paired move (leak)?
  4. how often does the fly commit while fixating a card, and is there a
     fixed like/dislike bias before any learning?

Setup (CHOSEN unless marked):
  brain       calibration.CHOSEN gains; MBON sides from build/mb_sides.json
              (MEASURED from DAN->MBON synapses)
  room frame  1280x800 at the room's ground grey with one card-sized block
              under the cursor: a logo square, two text bars, a graduation
              bar. Every coin gets the SAME picture, so smell is the only
              difference. That is the hardest case, since the fly cannot read.
  coins       A "Banana" (paired with sugar), B "Mud" (paired with shock),
              C "Moon Dog" (never paired); smells from olfaction.Nose
  look        FlyPilot.step on the room frame plus the coin's odour, recording
              approach (PPL1-input) and avoid (PAM-input) MBONs and the click
  blank       the control a look is measured against:
              "others"   the other cards in the room at the same seed, the way
                         a T-maze offers a fly two arms (default)
              "samelook" the same card, cursor and seed without the coin's smell
              "ground"   a uniform frame at the room's background grey
  drive       "others": calibration.relative(this look, the other coins' looks
              at the same seed); a blank mode: calibration.contrast(look, blank).
              Both live in calibration.py because backroom.py reads a look with
              the same two functions and the gate and the room must not drift.
  pairing     one look at the coin, mb.observe(fired), mb.dopamine(+1 or -1,
              amount 1), mb.apply(), then six empty steps so the eligibility
              trace decays before the next pairing

The same seeds are measured before and after training, so each seed gives a
paired difference. PASS needs drive(A) up by more than 2 SE and drive(B) down
by more than 2 SE. Learning goes to a throwaway file, never to the fly.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import calibration                      # noqa: E402
from flysim import FlyBrain             # noqa: E402
from mushroom import MushroomBody       # noqa: E402
from olfaction import Nose              # noqa: E402
try:
    from pumpui import FlyPilot         # noqa: E402
except ImportError:                     # the public copy calls it flyeye
    from flyeye import FlyPilot         # noqa: E402

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SEEDS = int(ARGS.get("seeds", 40))
PAIRINGS = int(ARGS.get("pairings", 12))
OUT = Path(ARGS.get("out", str(HERE / "build" / "backroom_screen.json")))
# what a look is measured against: "others" is the rest of the room at the
# same seed, "samelook" is the same card without the coin's smell, "ground" is
# a uniform frame at the room's background grey
BLANK_MODE = ARGS.get("blank", "others")
# which pairings to run: "both", or one side alone. Against an "others"
# reference the two sides confound each other - one coin falling lifts the
# rest - so a clean read of one side trains only that side.
TRAIN = ARGS.get("train", "both")
# how like and dislike are read out of a look:
#   "weight" the mushroom body's output synapses - the Kenyon cells that fired
#            times the weights dopamine changes, summed per side
#   "rate"   the firing rates of the approach and avoidance MBONs
READOUT = ARGS.get("readout", "weight")
# which calibration to run on: a name from calibration.SETTINGS. The sparser
# settings fire fewer Kenyon cells per odour, so coins share fewer of them and
# a lesson about one coin should spread less to the others.
CAL = ARGS.get("cal", calibration.CHOSEN)
# odour strength and an optional equal-sniff normalisation (0 = off): DoOR has
# no dose axis, so how loud a coin smells is a choice
HZ = float(ARGS.get("hz", 0))
# 2.0 is the setting that passed: every coin's smell scaled toward the same
# total. With norm=0 the loudest coin's lesson swamps the rest and sugar stops
# being distinguishable from drift. It is not equality - a response is capped at
# 1.0, so a coin landing on one glomerulus totals 1.0 where one spread over
# twenty totals 2.0 - and the room's nose does exactly the same arithmetic
# (olfaction.Nose equal_sniff), which is the point: the two must not drift.
NORM = float(ARGS.get("norm", 2.0))
T0 = time.time()

GROUND = 0.05                   # the room's background grey
CARD = (500, 300, 280, 200)     # x, y, w, h; the cursor sits at its centre
COIN_NAMES = {"A": ("Banana", "BNNA"), "B": ("Mud", "MUD"), "C": ("Moon Dog", "MDOG")}


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def room_frame():
    img = np.full((800, 1280), GROUND, dtype=np.float32)
    x, y, w, h = CARD
    img[y:y + h, x:x + w] = 0.14                       # card
    img[y + 16:y + 96, x + 16:x + 96] = 0.55           # logo
    img[y + 24:y + 34, x + 112:x + 250] = 0.85         # name
    img[y + 48:y + 56, x + 112:x + 200] = 0.60         # symbol
    img[y + 160:y + 172, x + 16:x + 264] = 0.25        # graduation track
    img[y + 160:y + 172, x + 16:x + 120] = 0.70        # graduation fill
    return img


def se(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    fb = FlyBrain()
    G = calibration.gains_for(fb, CAL)
    mb = MushroomBody(fb, calibration=CAL,
                      store=Path(tempfile.mkdtemp()) / "mb_gains.v2.npz")
    mb.gain[:] = 1.0
    mb.apply()
    pilot = FlyPilot(fb, sim_steps=60)
    nose = Nose(fb)
    nose.max_hz = HZ or calibration.SETTINGS[CAL]["odour_max_hz"]
    rec = calibration.readout(mb)
    img = room_frame()
    blank_img = np.full((800, 1280), GROUND, dtype=np.float32)
    cx, cy = CARD[0] + CARD[2] / 2.0, CARD[1] + CARD[3] / 2.0
    kc = np.asarray(mb.kc)

    def sniff(name, symbol):
        """One coin's smell, optionally scaled toward one total (never above a response of 1.0)."""
        sm = nose.smell(name, symbol)
        total = sum(sm["profile"].values())
        if NORM > 0 and total > 0:
            sm["profile"] = {g: min(1.0, v * NORM / total) for g, v in sm["profile"].items()}
        return sm

    coins = {k: sniff(n, s) for k, (n, s) in COIN_NAMES.items()}
    for k, sm in coins.items():
        say(f"coin {k} {COIN_NAMES[k][0]!r}: {[o['name'] for o in sm['odorants']]} "
            f"({len(sm['profile'])} glomeruli)")
    assert all(sm["profile"] for sm in coins.values()), "a coin has no smell"
    say(f"approach MBONs {len(rec['approach'])}, avoid MBONs {len(rec['avoid'])}, "
        f"KC->MBON synapses {len(mb.pos):,}, unassigned MBON types {mb.unassigned}")

    def syn_drive(fired):
        """
        What the mushroom body's output synapses carry for this look.

        One definition, in calibration.py, because the room reads a look the
        same way and the two must not drift apart.
        """
        return calibration.syn_drive(mb, fired)

    def read(info):
        if READOUT == "rate":
            ex = info["extra"]
            return float(np.sum(ex["approach"])), float(np.sum(ex["avoid"]))
        return syn_drive(info["fired"])

    def look(smell, seed):
        _, _, click, _, info = pilot.step(img, cx, cy, gains=G, seed=seed, detail=True,
                                          extra_drive=nose.drive(smell), extra_record=rec)
        a, v = read(info)
        return a, v, bool(click), info["fired"]

    def blank(seed):
        """
        The control run a look is measured against.

        "samelook" shows the same card at the same cursor with the same seed
        and no smell, so everything the eye contributes cancels and only the
        smell can move the reading. "ground" shows a uniform frame at the
        room's background grey, which is what the plan first proposed.
        Measured 2026-09-12: that ground frame is itself a strong stimulus (it
        drives the OFF channel hard and fires more approach MBONs than a card
        does), and learning moves it more than it moves any card, which
        inverted the sign of shock. Both are CHOSEN conventions.
        """
        if BLANK_MODE == "ground":
            r = fb.run(pilot.eye.look(blank_img, cx, cy), steps=pilot.sim_steps,
                       gains=G, record=rec, seed=seed)
            if READOUT == "rate":
                return float(np.sum(r["approach"])), float(np.sum(r["avoid"]))
            return syn_drive(r.get("_fired"))
        _, _, _, _, info = pilot.step(img, cx, cy, gains=G, seed=seed, detail=True,
                                      extra_record=rec)
        return read(info)

    seeds = [100_003 + 7919 * i for i in range(SEEDS)]

    def measure(tag):
        # every coin is looked at on every seed first, because the "others"
        # reference needs the rest of the room at that same seed
        looks = {k: [] for k in coins}
        clicks = {k: 0 for k in coins}
        kcf = {k: [] for k in coins}
        for s in seeds:
            for k, sm in coins.items():
                a, v, c, fired = look(sm, s)
                looks[k].append((a, v))
                clicks[k] += c
                kcf[k].append(float(np.isin(kc, fired).mean()) if fired is not None else 0.0)
        b = [(0.0, 0.0)] * len(seeds) if BLANK_MODE == "others" else [blank(s) for s in seeds]
        out = {"blank_A0": float(np.mean([x[0] for x in b])),
               "blank_V0": float(np.mean([x[1] for x in b]))}
        for k in coins:
            drives = []
            for i in range(len(seeds)):
                if BLANK_MODE == "others":
                    # calibration.relative compares normalised leanings, not raw
                    # sums: a coin that fires five times as many Kenyon cells
                    # otherwise dominates the reference and decides every other
                    # coin's score, which is what made sugar look backwards
                    # (measured 2026-09-12). The room calls the same function.
                    drives.append(calibration.relative(
                        looks[k][i], [looks[j][i] for j in coins if j != k]))
                else:
                    drives.append(calibration.contrast(looks[k][i], b[i]))
            A = [x[0] for x in looks[k]]
            V = [x[1] for x in looks[k]]
            out[k] = {"drive": drives, "clicks": clicks[k], "A": float(np.mean(A)),
                      "V": float(np.mean(V)), "kc_frac": float(np.mean(kcf[k]))}
            if BLANK_MODE == "others":
                ref_txt = "ref the other cards"
            else:
                ref_txt = f"ref {BLANK_MODE} A0 {out['blank_A0']:.0f} V0 {out['blank_V0']:.0f}"
            say(f"{tag:6} {k}: drive {np.mean(drives):+.4f} +- {se(drives):.4f}"
                f"  A {np.mean(A):8.0f} V {np.mean(V):8.0f} | {ref_txt}"
                f"  clicks {clicks[k]}/{len(seeds)}  KC {100 * np.mean(kcf[k]):.1f}%")
        return out

    before = measure("before")

    rng = np.random.default_rng(20260912)
    empty = np.array([], dtype=np.int64)
    pairs = {"both": (("A", +1), ("B", -1)), "sugar": (("A", +1),), "shock": (("B", -1),)}[TRAIN]
    hits = {"sugar": 0, "shock": 0}
    for _ in range(PAIRINGS):
        for k, sign in pairs:
            _, _, _, fired = look(coins[k], int(rng.integers(1 << 30)))
            mb.observe(fired)
            hits["sugar" if sign > 0 else "shock"] += mb.dopamine(sign, 1.0)
            mb.apply()
            for _ in range(6):
                mb.observe(empty)
    say("trained:", hits, mb.stats())

    after = measure("after")

    result = {"calibration": CAL, "sides_sha": mb.sides_sha, "seeds": SEEDS,
              "pairings": PAIRINGS, "ground": GROUND, "card": list(CARD), "synapses_hit": hits,
              "blank_mode": BLANK_MODE, "odour_max_hz": nose.max_hz, "equal_sniff_total": NORM,
              "readout": READOUT,
              "blank": {"before": [before["blank_A0"], before["blank_V0"]],
                        "after": [after["blank_A0"], after["blank_V0"]]},
              "coins": {}}
    for k in coins:
        d0, d1 = np.asarray(before[k]["drive"]), np.asarray(after[k]["drive"])
        diff = d1 - d0
        result["coins"][k] = {
            "name": COIN_NAMES[k][0], "paired_with": {"A": "sugar", "B": "shock"}.get(k, "nothing"),
            "odorants": [o["name"] for o in coins[k]["odorants"]],
            "drive_before": float(d0.mean()), "drive_before_se": se(d0),
            "drive_after": float(d1.mean()), "delta": float(diff.mean()), "delta_se": se(diff),
            "clicks_before": before[k]["clicks"], "clicks_after": after[k]["clicks"],
            "A_before": before[k]["A"], "V_before": before[k]["V"],
            "A_after": after[k]["A"], "V_after": after[k]["V"], "kc_frac": before[k]["kc_frac"]}
    c = result["coins"]
    sugar_ok = c["A"]["delta"] > 2 * c["A"]["delta_se"]
    shock_ok = c["B"]["delta"] < -2 * c["B"]["delta_se"]
    trained = max(abs(c["A"]["delta"]), abs(c["B"]["delta"]))
    result.update({
        "sugar_raises_paired_drive": bool(sugar_ok),
        "shock_lowers_paired_drive": bool(shock_ok),
        "leak_untouched_over_trained": float(abs(c["C"]["delta"]) / trained) if trained else None,
        "sugar_beats_untouched": bool(c["A"]["delta"] > c["C"]["delta"]),
        "shock_beats_untouched": bool(c["B"]["delta"] < c["C"]["delta"]),
        "commits_per_look_before": float(np.mean([c[k]["clicks_before"] for k in c]) / SEEDS),
        "trained_with": TRAIN,
        "pass": bool((sugar_ok or TRAIN == "shock") and (shock_ok or TRAIN == "sugar")),
        "mb": mb.stats(),
        "runtime_s": round(time.time() - T0)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    say(f"VERDICT pass={result['pass']}: sugar on A {c['A']['delta']:+.4f} (se {c['A']['delta_se']:.4f}), "
        f"shock on B {c['B']['delta']:+.4f} (se {c['B']['delta_se']:.4f}), "
        f"untouched C {c['C']['delta']:+.4f} (se {c['C']['delta_se']:.4f}); "
        f"leak {result['leak_untouched_over_trained']}; wrote {OUT}")


if __name__ == "__main__":
    main()

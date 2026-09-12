# Plume tracking: what the wiring did on its own

Run 2026-09-12: 10 seeds x 3 conditions x 20 s, 12,000 brain runs, 65 min. Record `build/plume_experiment.json`, paths `build/plume_trajectories.npz`, picture `build/plume_trajectories.png`. Predictions and metrics were fixed before the first trial and nothing was tuned afterwards; a reanalysis on 2026-09-13 added disclosures and left every preregistered number identical.

## What was asked

A walking fly that meets odour in wind surges upwind, and when it loses the odour it casts crosswind until it finds it again (Alvarez-Salvado et al. 2018; van Breugel and Dickinson 2014). The question was whether that comes out of the connectome by itself: the 165,122-neuron wiring, the roamer's existing type gains, ethyl acetate on the receptor neurons, wind on the Johnston's organ, and the walking descending neurons read out as turn and speed. No learning, no fitting, no hand-written controller. The fly was placed in a 0.6 x 0.3 m tunnel 0.40 m downwind of the source, in a meandering puff plume, for 20 s per trial, under three conditions on the same seeds and wind: **odour** (odour + wind sense), **blank** (wind sense, no odour), **nowind** (odour, antennae held at a fixed rate).

## Measured versus chosen

Measured: that the neurons exist and have sides (2,635 ORNs in 53 types, 32 responding to ethyl acetate in DoOR 2.0; 335 JO-C/E wind cells, 203 rooting left and 132 right; DNa02, DNa01, MDN and DNp09 with soma sides, 10 motor cells); and every number below, i.e. positions, headings, concentration along the path and spike counts, per trial and per 50 ms step.

Chosen, before the data and disclosed in the record: which neuron means which output (DNa02 difference = turn, DNa01 = forward, MDN = backward, DNp09 = stop) and the 450 Hz scale, inherited from the roamer; the calibration setting pn05_apl10_kc03; the odour drive (DoOR profile x clip(c, 0, 1) x 200 Hz); the wind encoding (a cosine per antenna, antennae 45 deg apart, 100 Hz peak, JO-C and JO-E driven alike) and the nowind control at 50 Hz per cell; the world (wind 0.12 m/s, plume 0.06 m wide at 0.3 m, 6 s warm-up, top speed 0.02 m/s, start 0.40 m downwind inside the plume's band); the metrics (threshold 0.05 with hysteresis, 1 s surge window, 2 s cast window, more than 2 SE as the bar); one 12 ms brain run per 50 ms step, restarted from rest. By hand before launch: 10 seeds instead of 12, because a brain run measured 0.31 s, not the 0.2 s estimated.

## The four predictions

| | Prediction (fixed before data) | Result (mean +/- SE, n) | Verdict |
|---|---|---|---|
| P1 surge | upwind speed rises in the 1 s after an encounter, by > 2 SE | +2.2 +/- 1.4 mm/s, n = 9 trials (16 events, 1 end-clipped); 1.5 SE | not supported |
| P2 cast | crosswind speed and heading-change spread both larger in the 2 s after a loss, by > 2 SE | crosswind -0.85 +/- 0.26 mm/s (3.3 SE, the wrong way); spread +0.06 +/- 0.19 deg; n = 9 (21 events, 9 before any encounter) | not supported |
| P3 upwind progress | odour > blank and odour > nowind, each by > 2 SE of the paired difference | odour - blank = -12.8 +/- 4.8 mm (blank went further); odour - nowind = +37.9 +/- 13.0 mm; n = 10 pairs | not supported (one half passes; confounded, see below) |
| P4 source reached | fraction ending within 0.03 m of the source: odour > blank | 0/10 odour, 0/10 blank, 0/10 nowind | not supported, and not discriminating |

Progress itself (start x minus end x): odour 45.8 +/- 5.1 mm, blank 58.6 +/- 7.1 mm, nowind 7.9 +/- 14.2 mm. The non-preregistered sensitivity checks (full windows only; losses after an encounter only) give the same verdicts.

## Baseline behaviours (no prediction was made)

- Blank: speed command -0.190 +/- 0.004 (MDN rate above DNa01 rate under the roamer's mapping: "backward" by convention, not an observed gait), turn command +0.015 +/- 0.007, ground speed 4.5 mm/s, heading rate -2.6 +/- 1.2 deg/s, |heading rate| 77 deg/s, which is the readout floor (one DNa02 cell per side over 12 ms moves the heading in 1.67 deg quanta).
- Motor rates, odour / blank / nowind (Hz): DNa01 L 112 / 61 / 82, R 70 / 67 / 77; DNa02 L 263 / 203 / 246, R 264 / 210 / 246; MDN 183 / 175 / 177; DNp09 70 / 99 / 75; ORN 6.7 / 1.8 / 6.4; JO L/R 35/12, 25/15, 46/46. Odour reaches the descending neurons; it does not become tracking.
- Encounters per trial: odour 1.6 +/- 0.4 (losses 2.3), nowind 1.5, blank 0; about 17 s of 20 in plume; wall contacts 0. Nine of ten odour trials began above threshold, so the events are the meander sweeping over a nearly stationary fly, not onsets from clean air.
- What produced the upwind drift: with wind sense the fly faces downwind on 91 % (odour) / 94 % (blank) of steps and, the net command being backward, backs upwind at 2.3 / 2.9 mm/s; without wind sense there is no heading preference (58 % downwind) and no progress (0.4 mm/s). The P3 odour-versus-nowind gap is a wind-sense effect, and the nowind control also delivered about twice the total JO drive (15,361 vs 8,696 cell-Hz), so it is confounded either way.

## What the result means

Under this protocol, cast-and-surge did not emerge from the wiring plus the roamer's gains. Odour changes the motor neurons' rates, but the change is not organised into a surge or a cast and does not increase upwind progress over a fly that smells nothing. The one consistent behaviour, backing upwind while facing downwind, is there with odour and without it, so it belongs to the wind encoding and the motor mapping, not to olfaction. Nothing was tuned to make it look better, and nothing here is evidence against the real fly's circuit: the sentence the numbers support is "this simulator, driven this way, does not track a plume", not "the connectome cannot".

## Simulator and design limits that could explain a failure

- **No receptor adaptation.** Real ORNs adapt within hundreds of milliseconds; encounter and loss are changes, and this brain sees only levels.
- **No memory across steps.** The brain restarts from rest every 50 ms; surge and cast are defined by history, and a memoryless controller can show only a static difference between the motor map above and below threshold.
- **No conduction delays, uniform 0.275 mV synapses.** Every synapse weighs the same and no spike takes time to travel.
- **Chosen wind encoding.** A cosine per antenna with a 45 deg offset; JO-C and JO-E co-activated although they respond oppositely in the fly; a 203:132 split that makes every headwind read like wind from about +31 deg; a control whose total drive matches the heading average, not what a downwind-facing fly received.
- **Chosen motor mapping and calibration.** "Forward" and "stop" follow the roamer's reading of DNa01 and DNp09, which the literature also describes as steering and forward-walking neurons; the ORN-to-PN pathway runs at half efficacy under pn05_apl10_kc03.
- **Design.** The fly starts inside the plume; the source needs 92.5 % of top speed straight upwind for all 20 s; the heading readout is quantised at 1.67 deg per step. A follow-up would start the fly outside the plume, give it time, carry membrane state across steps and equalise JO drive per side, each preregistered separately.

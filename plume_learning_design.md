# Plume learning: preregistered design

Written before any learning data exists. The innate plume-tracking run
(`plume_experiment.py`) is the baseline this experiment is measured against.
Nothing below is tuned after a result is seen; if a constant has to change to
make the pieces fit, the change and its reason are recorded in the report.

## The question

A real fly that finds sugar at the end of an odour plume comes back to that
odour faster. The plasticity behind that is known: reward dopamine depresses
Kenyon-cell synapses onto the avoidance-side mushroom body output neurons for
the odour that was present, so the odour drives less avoidance and more
approach. This connectome has every one of those cells and synapses. The
question is whether that measured change, delivered at the moment of success,
makes the simulated fly track the plume better on the next trial. Nobody has
asked it of a whole-brain simulation.

## What learns

`mushroom.MushroomBody`, the same circuit and the same rule the backroom uses:
dopamine-gated depression of KC to MBON synapses, sides from
`build/mb_sides.json`, calibrated gains `pn05_apl10_kc03`. It keeps its own
gain store, `build/plume_mb_gains.v2.npz`, so this experiment never touches
what the live fly has learned and the live fly never touches this. CHOSEN,
disclosed: one animal, two separate memories, for the duration of the
experiment.

## The trial

Identical to the innate experiment: the same arena, wind, plume, odorant
(ethyl acetate), wind encoding, motor readout and 400-step limit. One
difference: the mushroom body is loaded and applied, so `observe(fired)` and
`forget()` run every step, exactly as they do when the fly roams.

## The reward

- **Sugar** when the fly comes within 0.03 m of the source: `dopamine(+1, 1.0)`
  once, then the trial ends. Amount 1.0 is CHOSEN: one full appetitive
  pairing, as in a Tully and Quinn training session.
- **Nothing** when the trial times out. A fly that never finds the sugar is
  not punished; that is how the real assay works.
- Eligibility comes from the existing trace (0.55 per step, cutoff 0.05):
  the Kenyon cells that fired in the last five steps before the reward, which
  is the odour representation at the source, where the concentration is
  highest. No change to the trace for this experiment.

## Conditions, paired by seed

- **trained**: sugar at the source, as above.
- **yoked**: the same trials, same seeds, the mushroom body loaded and
  observed but never given dopamine. Any improvement here is drift or
  chance, not learning.
- **reversed**: `dopamine(-1, 1.0)` at the source instead. Punishment for
  finding the sugar. Predicted not to help, and to hurt if the circuit is
  doing what it is measured to do.

Each condition starts from a fresh, untrained gain vector.

## Schedule

3 flies (seeds) x 15 trials per fly per condition, in three blocks of five.
Trials that reach the source end early, so the budget is at most
3 x 15 x 3 x 400 = 54,000 brain runs, about three hours at 0.2 s each. If the
measured run time is slower, drop to 12 trials per fly before touching seeds.
The memory half-life stays at its default six hours; within a session of
this length it removes less than 5% of a lesson, which is reported.

## Predictions and thresholds (fixed now)

Standard errors are over trials within a condition, paired by seed where the
comparison is across conditions.

- **L1 finding the sugar.** Fraction of trials that reach the source, block
  three minus block one. PREDICTION: trained rises by more than 2 SE, and by
  more than the yoked change by more than 2 SE of the paired difference.
- **L2 getting upwind.** Upwind progress per trial regressed on trial index.
  PREDICTION: trained slope above 0 by more than 2 SE; yoked slope not.
- **L3 the wrong sign does not help.** PREDICTION: reversed shows no rise in
  L1 beyond 2 SE, and its block-three progress is at or below yoked.
- **L4 the mechanism is where it should be.** After training, in the trained
  fly only: the odour's synaptic leaning (`calibration.leaning` of
  `calibration.syn_drive` on a standard look at ethyl acetate) is higher
  than before training by more than 2 SE over 20 looks, and the depressed
  synapses sit mostly on Kenyon cells that fire to ethyl acetate rather than
  to a control odorant (2,3-butanedione), reported as a fraction.

Reported without a prediction: encounters per trial, time to source when
reached, wall contacts, the number of synapses depressed and the mean gain
after each block.

## Measured versus chosen

Measured: the connectome, the KC to MBON synapses and their dopamine sides,
DoOR receptor responses, the Johnston's organ cells and their root side, the
descending neurons, every rate the simulation produces.

Chosen: the plume model, the wind encoding, the kinematic scaling, 12 ms of
brain time per 50 ms of world time, the odorant, the reward amount, the
eligibility trace, the separate memory store, the schedule.

## Limits that could explain a failure

No receptor adaptation, no conduction delays, uniform synapses, and the
learning rule reaches the descending neurons only through whatever path the
wiring gives it. If training changes the mushroom body (L4 passes) but not
the tracking (L1 and L2 fail), that says the MBON to descending-neuron path is
too weak in this simulation to steer, which is itself a finding.

## What comes after

If L1 or L2 pass, the trial becomes a room in the live fly: a plume it walks
continuously, sugar at the source, its memory on the persistent volume, the
learning curve on the stream. That is the version that goes on.

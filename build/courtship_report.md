# Courtship experiment

Run: 10 seeds x 6 conditions x 400 steps; quick=False.
Quick runs are smoke tests; their two-seed verdicts are not the full ten-seed experiment.

## Question
Does his song change her graph's answer and their distance?

## Measured versus chosen
MEASURED: positions and headings, realised female speed, delivered sound, song, pC1 and vpoDN window rates.
IMPLEMENTATION: brain class flysim_gpu.FlyBrainGPU; torch devices {'male': 'cuda', 'female': 'cuda'}. Device is not a scientific choice. Required equivalence: same numbers on either device, verified by test (COURTSHIP_REAL_BRAIN=1); a failing test invalidates this claim.
CHOSEN: accept means vpoDN > 0 Hz in at least 3 windows; approach/retreat compare last and first quarter mean distance. Equal distance is neither. Geometry is sampled before each window; speed is displacement during it.
CHOSEN: paired seeds, annotation-backed male eye when available, uncalibrated gains, 50 ms brain windows, and the rate-to-motion mapping. No outcomes were tuned to differ across seeds.

CHOSEN: FEMALE_EXC_SCALE = 1.0; FEMALE_EYE = blind.
Protocol v5. CHOSEN before data: all six conditions share seed and arena start.
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

CHOSEN: D. melanogaster pulse IPI approximately 35 ms, carrier approximately 250 Hz, sine approximately 150 Hz; von Philipsborn et al. 2011 Neuron 69:509 (descending circuit; https://pubmed.ncbi.nlm.nih.gov/21315261/); Fast intensity adaptation enhances the encoding of sound in Drosophila, Nat Commun 2018 (carriers; https://doi.org/10.1038/s41467-017-02453-9); Zhou et al. 2015 eLife 4:e08477 (IPI; https://elifesciences.org/articles/08477). These numbers are synthesis choices, not brain measurements.

## Predictions
| Prediction | Paired difference | SE | n | Verdict |
|---|---:|---:|---:|---|
| P1: vpodn_hz, song - control (silence) | 13.645 | 4.110249384161501 | 10 | supported |
| P2: pc1_hz, song - control (silence) | 0.595 | 0.17356714998972458 | 10 | supported |
| P3: vpodn_hz, song - control (jittered) | -4.889999999999999 | 4.657224376051564 | 10 | not supported |
| P4: last_distance_mm, control - song (silence) | -2.492682688737323 | 2.739130645884937 | 10 | not supported |
| P5_command: pip10_hz, song - control (mute) | 1.4449999999999998 | 2.203610673417607 | 10 | not supported |
| P5_rms: delivered_rms, song - control (mute) | 0.0025127199486981206 | 0.004030346677678607 | 10 | not supported |
| P6: vpodn_hz, song - control (mute) | 3.1875000000000004 | 3.9692537956704714 | 10 | not supported |
| P8: p1_hz, song - control (noscent) | -4.600697674418605 | 2.638242758214443 | 10 | not supported |
| P8b: lc10a_hz, song - control (noscent) | 0.18949090909090907 | 0.12180941865058158 | 10 | descriptive; no verdict |
| P9: vpodn_hz, song - control (mated) | -2.1449999999999996 | 3.4233568645734582 | 10 | not supported |
| P11: lc10a_hz, sight - no sight (no-sight windows within song trial) | 0.09909998498580877 | 0.1739563528005651 | 10 | not supported |

## P0: baseline (descriptive, no verdict)
Seed 0, silence: 0 active windows.
Seed 0, mute: 8 active windows.
Seed 1, silence: 0 active windows.
Seed 1, mute: 10 active windows.
Seed 2, silence: 0 active windows.
Seed 2, mute: 20 active windows.
Seed 3, silence: 0 active windows.
Seed 3, mute: 8 active windows.
Seed 4, silence: 0 active windows.
Seed 4, mute: 1 active windows.
Seed 5, silence: 0 active windows.
Seed 5, mute: 50 active windows.
Seed 6, silence: 0 active windows.
Seed 6, mute: 26 active windows.
Seed 7, silence: 0 active windows.
Seed 7, mute: 31 active windows.
Seed 8, silence: 0 active windows.
Seed 8, mute: 66 active windows.
Seed 9, silence: 0 active windows.
Seed 9, mute: 60 active windows.

## Per-seed outcomes
MEASURED: clipped windows count actual ear sub-window clipping in each band.
| Seed | Condition | State | Accept | Approach | Retreat | Active windows | vpoDN Hz | pC1 Hz | oviDN Hz | Sight fraction | Last distance mm | pIP10 Hz | Delivered RMS |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | song | virgin | True | False | True | 11 | 4.025 | 0.02 | 0.0 | 0.48 | 14.799610643161254 | 0.2 | 0.001649539175473776 |
| 0 | jittered | virgin | True | True | False | 3 | 0.825 | 0.02 | 0.0 | 0.3925 | 5.043411877302531 | 6.0 | 0.001649539175473776 |
| 0 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.3075 | 7.388361375004492 | 7.925 | 0.0 |
| 0 | mute | virgin | True | False | True | 8 | 5.475 | 0.15 | 0.0 | 0.195 | 11.449196599571353 | 3.975 | 0.01600719611348165 |
| 0 | noscent | virgin | True | False | True | 108 | 55.425 | 3.395 | 0.0 | 0.305 | 14.847919483679625 | 11.925 | 0.013403056929611004 |
| 0 | mated | mated | True | False | True | 29 | 15.575 | 0.43 | 0.0 | 0.31 | 14.308783276312072 | 3.45 | 0.007122061692555155 |
| 1 | song | virgin | True | True | False | 25 | 9.8 | 0.395 | 0.0 | 0.3175 | 3.7891540149829193 | 2.575 | 0.009649961927261154 |
| 1 | jittered | virgin | True | True | False | 17 | 6.675 | 0.28 | 0.0 | 0.295 | 8.695738755147607 | 4.275 | 0.009649961927261154 |
| 1 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.2175 | 10.847606918735082 | 0.45 | 0.0 |
| 1 | mute | virgin | True | True | False | 10 | 4.875 | 0.085 | 0.0 | 0.3325 | 6.3312887327596465 | 0.725 | 0.00499210848573107 |
| 1 | noscent | virgin | True | False | True | 14 | 3.425 | 0.13 | 0.0 | 0.2725 | 22.102895838975073 | 0.9 | 0.006876032118618652 |
| 1 | mated | mated | True | False | True | 18 | 6.675 | 0.095 | 0.0 | 0.2725 | 21.687292679410717 | 1.4 | 0.00466198851384456 |
| 2 | song | virgin | True | False | True | 21 | 7.2 | 0.275 | 0.0 | 0.3175 | 21.4790825844085 | 15.55 | 0.00845721259654911 |
| 2 | jittered | virgin | True | True | False | 26 | 14.95 | 0.735 | 0.0 | 0.365 | 2.9058785809416268 | 0.125 | 0.008457212596549113 |
| 2 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.435 | 3.8014303074077582 | 1.625 | 0.0 |
| 2 | mute | virgin | True | False | True | 20 | 7.8 | 0.235 | 0.0 | 0.225 | 19.47192123902374 | 0.05 | 0.0005873278917014756 |
| 2 | noscent | virgin | True | False | True | 28 | 10.95 | 0.515 | 0.008333333333333333 | 0.3325 | 17.60506655319748 | 3.675 | 0.01365756657004834 |
| 2 | mated | mated | True | True | False | 47 | 17.625 | 0.56 | 0.0 | 0.2825 | 3.1092466148853206 | 3.15 | 0.011491496502705355 |
| 3 | song | virgin | False | False | True | 1 | 0.075 | 0.015 | 0.0 | 0.18 | 17.537085325173834 | 0.275 | 0.002589582694178652 |
| 3 | jittered | virgin | True | False | True | 118 | 34.15 | 1.535 | 0.0 | 0.425 | 14.245119280796066 | 0.575 | 0.0025895826941786525 |
| 3 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.2925 | 13.53560172218513 | 0.275 | 0.0 |
| 3 | mute | virgin | True | True | False | 8 | 1.7 | 0.075 | 0.0 | 0.1275 | 3.4376079320559683 | 0.15 | 0.002014093236279663 |
| 3 | noscent | virgin | True | False | True | 287 | 123.875 | 3.645 | 0.008333333333333333 | 0.215 | 14.294040527010637 | 167.475 | 0.04556582837610187 |
| 3 | mated | mated | False | True | False | 1 | 0.075 | 0.015 | 0.0 | 0.3075 | 10.235853613764483 | 0.275 | 0.002589582694178652 |
| 4 | song | virgin | True | False | True | 35 | 18.625 | 0.785 | 0.0 | 0.2975 | 19.555270233085547 | 1.45 | 0.0077394975094540385 |
| 4 | jittered | virgin | True | True | False | 60 | 30.05 | 0.73 | 0.0 | 0.3875 | 5.351206772848334 | 0.775 | 0.007739497509454039 |
| 4 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.31 | 11.519366799892063 | 1.175 | 0.0 |
| 4 | mute | virgin | False | True | False | 1 | 0.05 | 0.02 | 0.0 | 0.4125 | 2.8354984862608745 | 0.375 | 0.004142896028158046 |
| 4 | noscent | virgin | True | True | False | 112 | 42.9 | 0.885 | 0.0 | 0.3325 | 0.8927850532646637 | 65.275 | 0.07012048662627525 |
| 4 | mated | mated | True | False | True | 72 | 30.225 | 0.33 | 0.0 | 0.2975 | 13.97795387084141 | 29.225 | 0.006019603882083996 |
| 5 | song | virgin | True | False | True | 47 | 25.975 | 1.015 | 0.008333333333333333 | 0.36 | 18.428324711432072 | 3.15 | 0.018000056628155375 |
| 5 | jittered | virgin | True | False | True | 24 | 11.45 | 0.72 | 0.06666666666666667 | 0.375 | 7.641894629014657 | 2.625 | 0.01800005662815538 |
| 5 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.34 | 9.61653923299232 | 0.9 | 0.0 |
| 5 | mute | virgin | True | True | False | 50 | 13.75 | 0.47 | 0.016666666666666666 | 0.22 | 11.094074784794826 | 3.275 | 0.014803062062594428 |
| 5 | noscent | virgin | True | True | False | 51 | 19.6 | 0.84 | 0.0 | 0.3025 | 4.855473202108649 | 6.25 | 0.027323703997951004 |
| 5 | mated | mated | True | False | True | 34 | 11.35 | 0.14 | 0.008333333333333333 | 0.3025 | 4.319089379976139 | 4.275 | 0.02305138527961412 |
| 6 | song | virgin | True | True | False | 83 | 39.375 | 1.545 | 0.0 | 0.29 | 4.298284517807941 | 24.325 | 0.05529975118796243 |
| 6 | jittered | virgin | True | False | True | 139 | 56.2 | 1.66 | 0.0 | 0.4275 | 10.288357073485983 | 4.925 | 0.05529975118796242 |
| 6 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.3325 | 15.320653326332785 | 1.275 | 0.0 |
| 6 | mute | virgin | True | False | True | 26 | 15.7 | 0.445 | 0.0 | 0.3525 | 10.359297838948278 | 13.675 | 0.021788365220108942 |
| 6 | noscent | virgin | True | False | True | 28 | 12.0 | 0.27 | 0.0 | 0.5775 | 20.69411803575406 | 6.475 | 0.0193073341442676 |
| 6 | mated | mated | True | True | False | 70 | 31.75 | 0.42 | 0.0 | 0.505 | 10.94781324890709 | 1.075 | 0.005578034321041883 |
| 7 | song | virgin | True | False | True | 13 | 3.875 | 0.12 | 0.0 | 0.2425 | 13.460526162133021 | 0.975 | 0.007232539799581335 |
| 7 | jittered | virgin | True | False | True | 8 | 2.75 | 0.085 | 0.0 | 0.24 | 20.086328986198804 | 0.975 | 0.007232539799581335 |
| 7 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.2675 | 14.445020526387193 | 0.975 | 0.0 |
| 7 | mute | virgin | True | True | False | 31 | 10.4 | 0.315 | 0.0 | 0.36 | 3.8007382276840462 | 2.15 | 0.012616297494933246 |
| 7 | noscent | virgin | True | False | True | 22 | 10.0 | 0.19 | 0.0 | 0.31 | 7.979553870911352 | 2.9 | 0.015575331946488425 |
| 7 | mated | mated | True | False | True | 45 | 17.4 | 0.51 | 0.0 | 0.3275 | 9.045386867628759 | 3.6 | 0.02328582687292901 |
| 8 | song | virgin | True | False | True | 69 | 25.0 | 1.33 | 0.0 | 0.3025 | 9.092408402994081 | 1.175 | 0.005988541090895372 |
| 8 | jittered | virgin | True | True | False | 29 | 12.675 | 0.245 | 0.0 | 0.2175 | 8.344352897562635 | 4.875 | 0.005988541090895371 |
| 8 | silence | virgin | False | True | False | 0 | 0.0 | 0.0 | 0.0 | 0.23 | 5.732253611506273 | 1.2 | 0.0 |
| 8 | mute | virgin | True | False | True | 66 | 22.65 | 0.825 | 0.0 | 0.455 | 19.414152559445522 | 10.325 | 0.013675166526692577 |
| 8 | noscent | virgin | True | True | False | 102 | 51.475 | 1.005 | 0.0 | 0.285 | 6.420741262218981 | 13.775 | 0.03255318647910009 |
| 8 | mated | mated | True | False | True | 33 | 13.475 | 0.76 | 0.0 | 0.26 | 18.722484777444837 | 3.35 | 0.015220284066364756 |
| 9 | song | virgin | True | True | False | 7 | 2.5 | 0.45 | 0.0 | 0.3625 | 2.3700153853734447 | 0.15 | 0.0017364541565626974 |
| 9 | jittered | virgin | True | False | True | 36 | 15.625 | 0.43 | 0.0 | 0.2975 | 18.27068164489148 | 147.25 | 0.0017364541565626972 |
| 9 | silence | virgin | False | False | True | 0 | 0.0 | 0.0 | 0.0 | 0.29 | 7.676101272736284 | 0.15 | 0.0 |
| 9 | mute | virgin | True | False | True | 60 | 22.175 | 0.84 | 0.0 | 0.27 | 16.739278762385265 | 0.675 | 0.0025894242194116446 |
| 9 | noscent | virgin | True | False | True | 57 | 33.75 | 1.06 | 0.0 | 0.3925 | 13.227369645731748 | 5.05 | 0.014983762159596714 |
| 9 | mated | mated | True | False | True | 42 | 13.75 | 0.215 | 0.0 | 0.27 | 17.510282357244627 | 7.125 | 0.022744818428845882 |

## Seed spread
song: accept 9/10, approach 3/10, retreat 7/10. song: outcomes differ across seeds.
Accept seeds: [0, 1, 2, 4, 5, 6, 7, 8, 9]; no-accept seeds: [3].
jittered: accept 10/10, approach 5/10, retreat 5/10. jittered: outcomes differ across seeds.
Accept seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; no-accept seeds: [].
silence: accept 0/10, approach 1/10, retreat 9/10. silence: outcomes differ across seeds.
Accept seeds: []; no-accept seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
mute: accept 9/10, approach 5/10, retreat 5/10. mute: outcomes differ across seeds.
Accept seeds: [0, 1, 2, 3, 5, 6, 7, 8, 9]; no-accept seeds: [4].
noscent: accept 10/10, approach 3/10, retreat 7/10. noscent: outcomes differ across seeds.
Accept seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; no-accept seeds: [].
mated: accept 9/10, approach 3/10, retreat 7/10. mated: outcomes differ across seeds.
Accept seeds: [0, 1, 2, 4, 5, 6, 7, 8, 9]; no-accept seeds: [3].

## P9 she can say no
{'mean': -2.1449999999999996, 'se': 3.4233568645734582, 'n': 10, 'paired_differences': [-11.549999999999999, 3.125000000000001, -10.425, 0.0, -11.600000000000001, 14.625000000000002, 7.625, -13.524999999999999, 11.525, -11.25], 'metric': 'vpodn_hz', 'control': 'mated', 'direction': 'song - control', 'verdict': 'not supported'}

## P10 rejection (descriptive, no verdict)
Retreat fraction: adjacent pre-window distances that increase / all adjacent pairs.
{'seed': 0, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.05263157894736842}
{'seed': 0, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.10526315789473684}
{'seed': 1, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.18796992481203006}
{'seed': 1, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.2957393483709273}
{'seed': 2, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.19047619047619047}
{'seed': 2, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.23308270676691728}
{'seed': 3, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.17543859649122806}
{'seed': 3, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.14285714285714285}
{'seed': 4, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.3157894736842105}
{'seed': 4, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.3182957393483709}
{'seed': 5, 'condition': 'song', 'ovidn_hz': 0.008333333333333333, 'retreat_fraction': 0.2681704260651629}
{'seed': 5, 'condition': 'mated', 'ovidn_hz': 0.008333333333333333, 'retreat_fraction': 0.15037593984962405}
{'seed': 6, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.20802005012531327}
{'seed': 6, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.22055137844611528}
{'seed': 7, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.12531328320802004}
{'seed': 7, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.23558897243107768}
{'seed': 8, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.10526315789473684}
{'seed': 8, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.3283208020050125}
{'seed': 9, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.08020050125313283}
{'seed': 9, 'condition': 'mated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.2631578947368421}

## P11 he sees her
{'mean': 0.09909998498580877, 'se': 0.1739563528005651, 'n': 10, 'paired_differences': [-0.770600233100233, 0.02756089055301654, 1.4081645388732003, 0.061123429416112374, -0.2731356146470234, 0.06354166666666666, 0.05197580467128782, -0.08017791469868632, 0.13641291615165246, 0.3661343659720942], 'excluded_seeds': [], 'metric': 'lc10a_hz', 'control': 'no-sight windows within song trial', 'direction': 'sight - no sight', 'verdict': 'not supported'}
Paired within song trial; exclude trials missing either category; fewer than two pairs is undetermined.
{'seed': 0, 'sight_fraction': 0.48, 'lc10a_sight_hz': 0.9549242424242425, 'lc10a_no_sight_hz': 1.7255244755244754}
{'seed': 1, 'sight_fraction': 0.3175, 'lc10a_sight_hz': 0.1357193987115247, 'lc10a_no_sight_hz': 0.10815850815850815}
{'seed': 2, 'sight_fraction': 0.3175, 'lc10a_sight_hz': 1.748890479599141, 'lc10a_no_sight_hz': 0.3407259407259407}
{'seed': 3, 'sight_fraction': 0.18, 'lc10a_sight_hz': 0.36666666666666664, 'lc10a_no_sight_hz': 0.30554323725055427}
{'seed': 4, 'sight_fraction': 0.2975, 'lc10a_sight_hz': 0.08739495798319327, 'lc10a_no_sight_hz': 0.3605305726302167}
{'seed': 5, 'sight_fraction': 0.36, 'lc10a_sight_hz': 0.10303030303030303, 'lc10a_no_sight_hz': 0.03948863636363636}
{'seed': 6, 'sight_fraction': 0.29, 'lc10a_sight_hz': 0.3579937304075234, 'lc10a_no_sight_hz': 0.3060179257362356}
{'seed': 7, 'sight_fraction': 0.2425, 'lc10a_sight_hz': 0.16944704779756328, 'lc10a_no_sight_hz': 0.2496249624962496}
{'seed': 8, 'sight_fraction': 0.3025, 'lc10a_sight_hz': 0.3131480090157776, 'lc10a_no_sight_hz': 0.17673509286412514}
{'seed': 9, 'sight_fraction': 0.3625, 'lc10a_sight_hz': 0.5532288401253919, 'lc10a_no_sight_hz': 0.18709447415329766}

## P12 adaptation (descriptive, no verdict)
Spearman correlations: a[1:] and m[1:] versus previous distance/speed, and m[1:] versus LC10a[:-1]; None means constant data.
{'seed': 0, 'condition': 'song', 'a_distance_lagged_rho': -0.19813014981565816, 'a_speed_lagged_rho': 0.18883697996807328, 'm_distance_lagged_rho': 0.16613202480076625, 'm_speed_lagged_rho': -0.09884940696192941, 'm_lc10a_lagged_rho': 0.6369626411383309}
{'seed': 0, 'condition': 'jittered', 'a_distance_lagged_rho': -0.08236021872426065, 'a_speed_lagged_rho': 0.4025822484059302, 'm_distance_lagged_rho': 0.040215675217935135, 'm_speed_lagged_rho': -0.1442221741565146, 'm_lc10a_lagged_rho': 0.1212249280680827}
{'seed': 0, 'condition': 'silence', 'a_distance_lagged_rho': 0.060240730460614275, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.04108585275195242, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.05147407830711069}
{'seed': 0, 'condition': 'mute', 'a_distance_lagged_rho': -0.20213976443696535, 'a_speed_lagged_rho': 0.874221600907485, 'm_distance_lagged_rho': -0.2897643870552494, 'm_speed_lagged_rho': -0.022026938086061685, 'm_lc10a_lagged_rho': -0.16548169946557745}
{'seed': 0, 'condition': 'noscent', 'a_distance_lagged_rho': -0.14097994618824097, 'a_speed_lagged_rho': 0.2593617379262779, 'm_distance_lagged_rho': 0.023955828056755717, 'm_speed_lagged_rho': -0.023063169269842403, 'm_lc10a_lagged_rho': -0.038476092864650945}
{'seed': 0, 'condition': 'mated', 'a_distance_lagged_rho': 0.030488281379258433, 'a_speed_lagged_rho': 0.4239158509896897, 'm_distance_lagged_rho': -0.10136721502023482, 'm_speed_lagged_rho': -0.07143103485279963, 'm_lc10a_lagged_rho': 0.23181402249939095}
{'seed': 1, 'condition': 'song', 'a_distance_lagged_rho': 0.15523672239109884, 'a_speed_lagged_rho': 0.24458588459773978, 'm_distance_lagged_rho': 0.16482137115484233, 'm_speed_lagged_rho': 0.07262412860715027, 'm_lc10a_lagged_rho': -0.09057946442511075}
{'seed': 1, 'condition': 'jittered', 'a_distance_lagged_rho': 0.1035290336613353, 'a_speed_lagged_rho': 0.3727664832560627, 'm_distance_lagged_rho': 0.16071882879572857, 'm_speed_lagged_rho': -0.08044690152477946, 'm_lc10a_lagged_rho': -0.21969246570677667}
{'seed': 1, 'condition': 'silence', 'a_distance_lagged_rho': -0.05263091929135959, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': 0.05939429235859611, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.06418063545007456}
{'seed': 1, 'condition': 'mute', 'a_distance_lagged_rho': 0.01293068042096407, 'a_speed_lagged_rho': 0.4314965867598659, 'm_distance_lagged_rho': -0.2549364927581016, 'm_speed_lagged_rho': -0.18702415096687977, 'm_lc10a_lagged_rho': 0.0713570429428897}
{'seed': 1, 'condition': 'noscent', 'a_distance_lagged_rho': -0.16751832361780797, 'a_speed_lagged_rho': 0.21339650466325188, 'm_distance_lagged_rho': -0.19221030207971454, 'm_speed_lagged_rho': 0.13370344901629008, 'm_lc10a_lagged_rho': 0.26080383413847086}
{'seed': 1, 'condition': 'mated', 'a_distance_lagged_rho': -0.15143807929187975, 'a_speed_lagged_rho': 0.23677535112802514, 'm_distance_lagged_rho': -0.06701721287806246, 'm_speed_lagged_rho': -0.15026242744779422, 'm_lc10a_lagged_rho': -0.07344639384061698}
{'seed': 2, 'condition': 'song', 'a_distance_lagged_rho': -0.03941022179540993, 'a_speed_lagged_rho': 0.01894861678801472, 'm_distance_lagged_rho': 0.16267420584661704, 'm_speed_lagged_rho': 0.055846223729706164, 'm_lc10a_lagged_rho': 0.12924588378074067}
{'seed': 2, 'condition': 'jittered', 'a_distance_lagged_rho': 0.12193666448704038, 'a_speed_lagged_rho': 0.0777111418494373, 'm_distance_lagged_rho': -0.0774490073713747, 'm_speed_lagged_rho': 0.26530385356529557, 'm_lc10a_lagged_rho': 0.2582005070943901}
{'seed': 2, 'condition': 'silence', 'a_distance_lagged_rho': -0.11456104385368003, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': 0.5542448148142507, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.1659103573907532}
{'seed': 2, 'condition': 'mute', 'a_distance_lagged_rho': -0.07152557770650077, 'a_speed_lagged_rho': -0.015032685002731316, 'm_distance_lagged_rho': -0.4638132091298634, 'm_speed_lagged_rho': -0.23242688087203428, 'm_lc10a_lagged_rho': 0.5749355463094901}
{'seed': 2, 'condition': 'noscent', 'a_distance_lagged_rho': -0.06017446151887948, 'a_speed_lagged_rho': 0.4215407219412962, 'm_distance_lagged_rho': 0.35011109426732745, 'm_speed_lagged_rho': -0.33765110137496446, 'm_lc10a_lagged_rho': 0.07710671048046741}
{'seed': 2, 'condition': 'mated', 'a_distance_lagged_rho': 0.2026808758289694, 'a_speed_lagged_rho': 0.21889489853479044, 'm_distance_lagged_rho': 0.05485989936317491, 'm_speed_lagged_rho': -0.17690472484078876, 'm_lc10a_lagged_rho': -0.018400274551039316}
{'seed': 3, 'condition': 'song', 'a_distance_lagged_rho': -0.09172590002092003, 'a_speed_lagged_rho': -0.004362825406580458, 'm_distance_lagged_rho': -0.07513864199708735, 'm_speed_lagged_rho': 0.059642794366203206, 'm_lc10a_lagged_rho': 0.06361188220038337}
{'seed': 3, 'condition': 'jittered', 'a_distance_lagged_rho': -0.07425635200794944, 'a_speed_lagged_rho': -0.04036286622826018, 'm_distance_lagged_rho': -0.25209092484063034, 'm_speed_lagged_rho': 0.12766158041985198, 'm_lc10a_lagged_rho': -0.06900767436984291}
{'seed': 3, 'condition': 'silence', 'a_distance_lagged_rho': -0.08706237661697222, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.09354764359512255, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': 0.0806261498396597}
{'seed': 3, 'condition': 'mute', 'a_distance_lagged_rho': 0.05060379198156832, 'a_speed_lagged_rho': -0.009556530577301017, 'm_distance_lagged_rho': 0.2645467116052616, 'm_speed_lagged_rho': -0.13299622873369527, 'm_lc10a_lagged_rho': -0.2521556311518837}
{'seed': 3, 'condition': 'noscent', 'a_distance_lagged_rho': 0.23340042565972763, 'a_speed_lagged_rho': 0.289263636745556, 'm_distance_lagged_rho': -0.32792817632144183, 'm_speed_lagged_rho': 0.019635853833145006, 'm_lc10a_lagged_rho': 0.18659005062529785}
{'seed': 3, 'condition': 'mated', 'a_distance_lagged_rho': -0.09605786523422587, 'a_speed_lagged_rho': -0.002512562814070352, 'm_distance_lagged_rho': 0.01806250018023481, 'm_speed_lagged_rho': 0.08457165119493316, 'm_lc10a_lagged_rho': 0.22504444432767476}
{'seed': 4, 'condition': 'song', 'a_distance_lagged_rho': -0.08480559832430659, 'a_speed_lagged_rho': 0.20852493070375686, 'm_distance_lagged_rho': 0.3159761814825893, 'm_speed_lagged_rho': -0.08404693876958133, 'm_lc10a_lagged_rho': 0.21687613895599703}
{'seed': 4, 'condition': 'jittered', 'a_distance_lagged_rho': 0.0457053444594903, 'a_speed_lagged_rho': 0.07637948557654278, 'm_distance_lagged_rho': -0.0007911745890134282, 'm_speed_lagged_rho': 0.08724163770543267, 'm_lc10a_lagged_rho': -0.03230050102448195}
{'seed': 4, 'condition': 'silence', 'a_distance_lagged_rho': 0.01492083291034957, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.25508894586302144, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': 0.008567706132513613}
{'seed': 4, 'condition': 'mute', 'a_distance_lagged_rho': 0.056914036960358946, 'a_speed_lagged_rho': -0.0035577615527401193, 'm_distance_lagged_rho': -0.5085658727693052, 'm_speed_lagged_rho': -0.01491233306097975, 'm_lc10a_lagged_rho': 0.5623046823837448}
{'seed': 4, 'condition': 'noscent', 'a_distance_lagged_rho': 0.6149710924736882, 'a_speed_lagged_rho': 0.8831089110221221, 'm_distance_lagged_rho': -0.12343691305784416, 'm_speed_lagged_rho': -0.19707606535695876, 'm_lc10a_lagged_rho': -0.035349292412790674}
{'seed': 4, 'condition': 'mated', 'a_distance_lagged_rho': 0.5096167080307807, 'a_speed_lagged_rho': 0.6942306502016455, 'm_distance_lagged_rho': -0.21020990362555653, 'm_speed_lagged_rho': -0.22346371573895352, 'm_lc10a_lagged_rho': -0.07718234421509797}
{'seed': 5, 'condition': 'song', 'a_distance_lagged_rho': -0.229463672144858, 'a_speed_lagged_rho': 0.32277759550596785, 'm_distance_lagged_rho': 0.08203952967068888, 'm_speed_lagged_rho': -0.10778052190765333, 'm_lc10a_lagged_rho': -0.05748022201690985}
{'seed': 5, 'condition': 'jittered', 'a_distance_lagged_rho': -0.1399488399138526, 'a_speed_lagged_rho': 0.3470409608414915, 'm_distance_lagged_rho': 0.08573225415917872, 'm_speed_lagged_rho': -0.12962729838331238, 'm_lc10a_lagged_rho': -0.06811772569788596}
{'seed': 5, 'condition': 'silence', 'a_distance_lagged_rho': 0.026746612665025968, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.09934364041219752, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.0971273434278795}
{'seed': 5, 'condition': 'mute', 'a_distance_lagged_rho': -0.20219728978212187, 'a_speed_lagged_rho': 0.16988250158200227, 'm_distance_lagged_rho': 0.00829891788929568, 'm_speed_lagged_rho': -0.20414590480452094, 'm_lc10a_lagged_rho': 0.28850170988094365}
{'seed': 5, 'condition': 'noscent', 'a_distance_lagged_rho': -0.12771430100695413, 'a_speed_lagged_rho': 0.354960169477633, 'm_distance_lagged_rho': -0.16550696392719763, 'm_speed_lagged_rho': 7.13618881418098e-05, 'm_lc10a_lagged_rho': -0.16092451218175355}
{'seed': 5, 'condition': 'mated', 'a_distance_lagged_rho': -0.05882920335179394, 'a_speed_lagged_rho': 0.34588286698756127, 'm_distance_lagged_rho': 0.060158669403657876, 'm_speed_lagged_rho': 0.011805446894585007, 'm_lc10a_lagged_rho': 0.1259401835631937}
{'seed': 6, 'condition': 'song', 'a_distance_lagged_rho': -0.34097497672870414, 'a_speed_lagged_rho': 0.6335000457313416, 'm_distance_lagged_rho': 0.5086100377885409, 'm_speed_lagged_rho': -0.2154091781584542, 'm_lc10a_lagged_rho': 0.025540646780379775}
{'seed': 6, 'condition': 'jittered', 'a_distance_lagged_rho': -0.08879245115155497, 'a_speed_lagged_rho': 0.39006298370378933, 'm_distance_lagged_rho': 0.05974680767388399, 'm_speed_lagged_rho': -0.28336677439214003, 'm_lc10a_lagged_rho': 0.04997942485095058}
{'seed': 6, 'condition': 'silence', 'a_distance_lagged_rho': 0.017331866691309435, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': 0.3417231677806188, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.14916227485626726}
{'seed': 6, 'condition': 'mute', 'a_distance_lagged_rho': 0.2369547751960383, 'a_speed_lagged_rho': 0.9603132355889582, 'm_distance_lagged_rho': -0.0515283106946421, 'm_speed_lagged_rho': -0.4055491586403448, 'm_lc10a_lagged_rho': -0.09119970289268511}
{'seed': 6, 'condition': 'noscent', 'a_distance_lagged_rho': -0.251749991024927, 'a_speed_lagged_rho': 0.5759084392387581, 'm_distance_lagged_rho': -0.4742664363752712, 'm_speed_lagged_rho': -0.022453789173480457, 'm_lc10a_lagged_rho': 0.14075045989557758}
{'seed': 6, 'condition': 'mated', 'a_distance_lagged_rho': -0.020617102077601486, 'a_speed_lagged_rho': 0.0954373645967572, 'm_distance_lagged_rho': -0.5359517811240054, 'm_speed_lagged_rho': -0.24065947442240934, 'm_lc10a_lagged_rho': -0.19144264682954626}
{'seed': 7, 'condition': 'song', 'a_distance_lagged_rho': -0.15304773851253922, 'a_speed_lagged_rho': 0.24449197281312113, 'm_distance_lagged_rho': 0.4772317295473531, 'm_speed_lagged_rho': -0.09131296069520027, 'm_lc10a_lagged_rho': -0.28016491390285136}
{'seed': 7, 'condition': 'jittered', 'a_distance_lagged_rho': -0.13901577780885163, 'a_speed_lagged_rho': 0.4269947940375774, 'm_distance_lagged_rho': 0.28874694225445513, 'm_speed_lagged_rho': -0.08770007048016286, 'm_lc10a_lagged_rho': -0.06510447284245495}
{'seed': 7, 'condition': 'silence', 'a_distance_lagged_rho': 0.001787658195326752, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.34971434765032283, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': 0.030953856326378263}
{'seed': 7, 'condition': 'mute', 'a_distance_lagged_rho': 0.12588076977779636, 'a_speed_lagged_rho': 0.26059762071510334, 'm_distance_lagged_rho': -0.5527399729359375, 'm_speed_lagged_rho': -0.08178098785014268, 'm_lc10a_lagged_rho': 0.11845226824491266}
{'seed': 7, 'condition': 'noscent', 'a_distance_lagged_rho': -0.1288744717818429, 'a_speed_lagged_rho': 0.36710995135846014, 'm_distance_lagged_rho': 0.10994491471750123, 'm_speed_lagged_rho': -0.17903730131266357, 'm_lc10a_lagged_rho': 0.11564820588497537}
{'seed': 7, 'condition': 'mated', 'a_distance_lagged_rho': -0.12212939915283007, 'a_speed_lagged_rho': 0.3051013816658137, 'm_distance_lagged_rho': 0.16381072007481903, 'm_speed_lagged_rho': -0.004124313331104833, 'm_lc10a_lagged_rho': -0.021114686369368103}
{'seed': 8, 'condition': 'song', 'a_distance_lagged_rho': 0.054481562996292254, 'a_speed_lagged_rho': 0.035171344648113584, 'm_distance_lagged_rho': -0.03419143749521154, 'm_speed_lagged_rho': -0.3736767402663364, 'm_lc10a_lagged_rho': 0.020123707207331913}
{'seed': 8, 'condition': 'jittered', 'a_distance_lagged_rho': -0.11973650661080783, 'a_speed_lagged_rho': 0.06998282222101315, 'm_distance_lagged_rho': -0.1576889179575236, 'm_speed_lagged_rho': -0.06882239067529577, 'm_lc10a_lagged_rho': 0.12727619226783715}
{'seed': 8, 'condition': 'silence', 'a_distance_lagged_rho': 0.14774588255344645, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': 0.04852562391765097, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': -0.04290417053580833}
{'seed': 8, 'condition': 'mute', 'a_distance_lagged_rho': 0.09968570027989698, 'a_speed_lagged_rho': -0.04681100171917023, 'm_distance_lagged_rho': 0.25363212102151883, 'm_speed_lagged_rho': -0.1668423816597136, 'm_lc10a_lagged_rho': -0.0016157805836198135}
{'seed': 8, 'condition': 'noscent', 'a_distance_lagged_rho': 0.22275332970415546, 'a_speed_lagged_rho': 0.2254478007908208, 'm_distance_lagged_rho': 0.44031378581526315, 'm_speed_lagged_rho': 0.10069976849078549, 'm_lc10a_lagged_rho': 0.0002898439827685569}
{'seed': 8, 'condition': 'mated', 'a_distance_lagged_rho': -0.18088646630006208, 'a_speed_lagged_rho': 0.2517984693451762, 'm_distance_lagged_rho': 0.44452612979977785, 'm_speed_lagged_rho': -0.34566194281283785, 'm_lc10a_lagged_rho': 0.34015002350683315}
{'seed': 9, 'condition': 'song', 'a_distance_lagged_rho': 0.08896583798697534, 'a_speed_lagged_rho': -0.007169450177411942, 'm_distance_lagged_rho': -0.43675650156460083, 'm_speed_lagged_rho': -0.04678654759222776, 'm_lc10a_lagged_rho': 0.4749239958792193}
{'seed': 9, 'condition': 'jittered', 'a_distance_lagged_rho': -0.08087108826416055, 'a_speed_lagged_rho': -0.017157575371968277, 'm_distance_lagged_rho': -0.40294248438766694, 'm_speed_lagged_rho': -0.0526438033055675, 'm_lc10a_lagged_rho': 0.07528413561327416}
{'seed': 9, 'condition': 'silence', 'a_distance_lagged_rho': -0.10388214989561136, 'a_speed_lagged_rho': None, 'm_distance_lagged_rho': -0.08441716428384241, 'm_speed_lagged_rho': None, 'm_lc10a_lagged_rho': 0.024872072185310526}
{'seed': 9, 'condition': 'mute', 'a_distance_lagged_rho': -0.05483033998332316, 'a_speed_lagged_rho': 0.0438422010741176, 'm_distance_lagged_rho': -0.19680843724145486, 'm_speed_lagged_rho': 0.018180337915543382, 'm_lc10a_lagged_rho': 0.11856226814842376}
{'seed': 9, 'condition': 'noscent', 'a_distance_lagged_rho': -0.4358102772551261, 'a_speed_lagged_rho': 0.3774404834527464, 'm_distance_lagged_rho': 0.013851117621066441, 'm_speed_lagged_rho': -0.02020022714739304, 'm_lc10a_lagged_rho': -0.04864282158240723}
{'seed': 9, 'condition': 'mated', 'a_distance_lagged_rho': -0.27630425520310553, 'a_speed_lagged_rho': 0.31546095183918116, 'm_distance_lagged_rho': 0.47880473008498303, 'm_speed_lagged_rho': -0.034837082475492866, 'm_lc10a_lagged_rho': 0.0660943438904925}

## P7 his response (descriptive)
MEASURED: P1 active windows, LC10a mean rate per window, lagged amplitude/mode correlations with her previous distance/speed. None means constant data. No adaptation mechanism was added.
| Seed | Condition | P1 active windows | LC10a Hz | a-distance | a-speed | m-distance | m-speed |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | song | 117 | 1.3556363636363635 | -0.19813014981565816 | 0.18883697996807328 | 0.16613202480076625 | -0.09884940696192941 |
| 0 | jittered | 98 | 0.15072727272727274 | -0.08236021872426065 | 0.4025822484059302 | 0.040215675217935135 | -0.1442221741565146 |
| 0 | silence | 116 | 0.11927272727272728 | 0.060240730460614275 | None | -0.04108585275195242 | None |
| 0 | mute | 154 | 0.8919999999999999 | -0.20213976443696535 | 0.874221600907485 | -0.2897643870552494 | -0.022026938086061685 |
| 0 | noscent | 232 | 0.21054545454545454 | -0.14097994618824097 | 0.2593617379262779 | 0.023955828056755717 | -0.023063169269842403 |
| 0 | mated | 146 | 0.11327272727272727 | 0.030488281379258433 | 0.4239158509896897 | -0.10136721502023482 | -0.07143103485279963 |
| 1 | song | 234 | 0.11690909090909091 | 0.15523672239109884 | 0.24458588459773978 | 0.16482137115484233 | 0.07262412860715027 |
| 1 | jittered | 112 | 0.09654545454545455 | 0.1035290336613353 | 0.3727664832560627 | 0.16071882879572857 | -0.08044690152477946 |
| 1 | silence | 111 | 0.16636363636363638 | -0.05263091929135959 | None | 0.05939429235859611 | None |
| 1 | mute | 59 | 0.04872727272727273 | 0.01293068042096407 | 0.4314965867598659 | -0.2549364927581016 | -0.18702415096687977 |
| 1 | noscent | 171 | 0.14709090909090908 | -0.16751832361780797 | 0.21339650466325188 | -0.19221030207971454 | 0.13370344901629008 |
| 1 | mated | 103 | 0.09181818181818181 | -0.15143807929187975 | 0.23677535112802514 | -0.06701721287806246 | -0.15026242744779422 |
| 2 | song | 204 | 0.7878181818181819 | -0.03941022179540993 | 0.01894861678801472 | 0.16267420584661704 | 0.055846223729706164 |
| 2 | jittered | 212 | 0.664 | 0.12193666448704038 | 0.0777111418494373 | -0.0774490073713747 | 0.26530385356529557 |
| 2 | silence | 287 | 0.2107272727272727 | -0.11456104385368003 | None | 0.5542448148142507 | None |
| 2 | mute | 400 | 1.6047272727272723 | -0.07152557770650077 | -0.015032685002731316 | -0.4638132091298634 | -0.23242688087203428 |
| 2 | noscent | 400 | 0.45127272727272727 | -0.06017446151887948 | 0.4215407219412962 | 0.35011109426732745 | -0.33765110137496446 |
| 2 | mated | 110 | 0.12763636363636363 | 0.2026808758289694 | 0.21889489853479044 | 0.05485989936317491 | -0.17690472484078876 |
| 3 | song | 92 | 0.3165454545454546 | -0.09172590002092003 | -0.004362825406580458 | -0.07513864199708735 | 0.059642794366203206 |
| 3 | jittered | 55 | 0.15272727272727274 | -0.07425635200794944 | -0.04036286622826018 | -0.25209092484063034 | 0.12766158041985198 |
| 3 | silence | 101 | 0.15327272727272725 | -0.08706237661697222 | None | -0.09354764359512255 | None |
| 3 | mute | 184 | 1.5134545454545454 | 0.05060379198156832 | -0.009556530577301017 | 0.2645467116052616 | -0.13299622873369527 |
| 3 | noscent | 400 | 0.06309090909090909 | 0.23340042565972763 | 0.289263636745556 | -0.32792817632144183 | 0.019635853833145006 |
| 3 | mated | 122 | 0.08781818181818181 | -0.09605786523422587 | -0.002512562814070352 | 0.01806250018023481 | 0.08457165119493316 |
| 4 | song | 100 | 0.2792727272727273 | -0.08480559832430659 | 0.20852493070375686 | 0.3159761814825893 | -0.08404693876958133 |
| 4 | jittered | 86 | 0.09872727272727272 | 0.0457053444594903 | 0.07637948557654278 | -0.0007911745890134282 | 0.08724163770543267 |
| 4 | silence | 71 | 0.11781818181818181 | 0.01492083291034957 | None | -0.25508894586302144 | None |
| 4 | mute | 43 | 0.7094545454545456 | 0.056914036960358946 | -0.0035577615527401193 | -0.5085658727693052 | -0.01491233306097975 |
| 4 | noscent | 400 | 0.2198181818181818 | 0.6149710924736882 | 0.8831089110221221 | -0.12343691305784416 | -0.19707606535695876 |
| 4 | mated | 118 | 0.052000000000000005 | 0.5096167080307807 | 0.6942306502016455 | -0.21020990362555653 | -0.22346371573895352 |
| 5 | song | 74 | 0.062363636363636364 | -0.229463672144858 | 0.32277759550596785 | 0.08203952967068888 | -0.10778052190765333 |
| 5 | jittered | 136 | 0.10418181818181818 | -0.1399488399138526 | 0.3470409608414915 | 0.08573225415917872 | -0.12962729838331238 |
| 5 | silence | 18 | 0.12127272727272727 | 0.026746612665025968 | None | -0.09934364041219752 | None |
| 5 | mute | 170 | 0.7523636363636363 | -0.20219728978212187 | 0.16988250158200227 | 0.00829891788929568 | -0.20414590480452094 |
| 5 | noscent | 400 | 0.38345454545454544 | -0.12771430100695413 | 0.354960169477633 | -0.16550696392719763 | 7.13618881418098e-05 |
| 5 | mated | 91 | 0.07836363636363636 | -0.05882920335179394 | 0.34588286698756127 | 0.060158669403657876 | 0.011805446894585007 |
| 6 | song | 187 | 0.3210909090909091 | -0.34097497672870414 | 0.6335000457313416 | 0.5086100377885409 | -0.2154091781584542 |
| 6 | jittered | 72 | 0.06963636363636364 | -0.08879245115155497 | 0.39006298370378933 | 0.05974680767388399 | -0.28336677439214003 |
| 6 | silence | 57 | 0.17709090909090908 | 0.017331866691309435 | None | 0.3417231677806188 | None |
| 6 | mute | 400 | 0.14527272727272728 | 0.2369547751960383 | 0.9603132355889582 | -0.0515283106946421 | -0.4055491586403448 |
| 6 | noscent | 123 | 0.25781818181818184 | -0.251749991024927 | 0.5759084392387581 | -0.4742664363752712 | -0.022453789173480457 |
| 6 | mated | 103 | 0.10345454545454548 | -0.020617102077601486 | 0.0954373645967572 | -0.5359517811240054 | -0.24065947442240934 |
| 7 | song | 34 | 0.23018181818181815 | -0.15304773851253922 | 0.24449197281312113 | 0.4772317295473531 | -0.09131296069520027 |
| 7 | jittered | 100 | 0.022181818181818178 | -0.13901577780885163 | 0.4269947940375774 | 0.28874694225445513 | -0.08770007048016286 |
| 7 | silence | 59 | 0.27945454545454546 | 0.001787658195326752 | None | -0.34971434765032283 | None |
| 7 | mute | 121 | 0.33872727272727265 | 0.12588076977779636 | 0.26059762071510334 | -0.5527399729359375 | -0.08178098785014268 |
| 7 | noscent | 207 | 0.20563636363636362 | -0.1288744717818429 | 0.36710995135846014 | 0.10994491471750123 | -0.17903730131266357 |
| 7 | mated | 78 | 0.05509090909090908 | -0.12212939915283007 | 0.3051013816658137 | 0.16381072007481903 | -0.004124313331104833 |
| 8 | song | 108 | 0.21799999999999997 | 0.054481562996292254 | 0.035171344648113584 | -0.03419143749521154 | -0.3736767402663364 |
| 8 | jittered | 33 | 1.060909090909091 | -0.11973650661080783 | 0.06998282222101315 | -0.1576889179575236 | -0.06882239067529577 |
| 8 | silence | 59 | 0.12709090909090912 | 0.14774588255344645 | None | 0.04852562391765097 | None |
| 8 | mute | 113 | 0.22909090909090907 | 0.09968570027989698 | -0.04681100171917023 | 0.25363212102151883 | -0.1668423816597136 |
| 8 | noscent | 314 | 0.15309090909090908 | 0.22275332970415546 | 0.2254478007908208 | 0.44031378581526315 | 0.10069976849078549 |
| 8 | mated | 99 | 0.11418181818181818 | -0.18088646630006208 | 0.2517984693451762 | 0.44452612979977785 | -0.34566194281283785 |
| 9 | song | 41 | 0.3198181818181818 | 0.08896583798697534 | -0.007169450177411942 | -0.43675650156460083 | -0.04678654759222776 |
| 9 | jittered | 217 | 0.07781818181818181 | -0.08087108826416055 | -0.017157575371968277 | -0.40294248438766694 | -0.0526438033055675 |
| 9 | silence | 400 | 0.1449090909090909 | -0.10388214989561136 | None | -0.08441716428384241 | None |
| 9 | mute | 221 | 0.09745454545454546 | -0.05483033998332316 | 0.0438422010741176 | -0.19680843724145486 | 0.018180337915543382 |
| 9 | noscent | 400 | 0.02090909090909091 | -0.4358102772551261 | 0.3774404834527464 | 0.013851117621066441 | -0.02020022714739304 |
| 9 | mated | 190 | 0.032545454545454544 | -0.27630425520310553 | 0.31546095183918116 | 0.47880473008498303 | -0.034837082475492866 |

P5 joint verdict: not supported.
MEASURED jittered/song RMS ratios: [{'seed': 0, 'jittered_song_rms_ratio': 1.0}, {'seed': 1, 'jittered_song_rms_ratio': 1.0}, {'seed': 2, 'jittered_song_rms_ratio': 1.0000000000000002}, {'seed': 3, 'jittered_song_rms_ratio': 1.0000000000000002}, {'seed': 4, 'jittered_song_rms_ratio': 1.0000000000000002}, {'seed': 5, 'jittered_song_rms_ratio': 1.0000000000000002}, {'seed': 6, 'jittered_song_rms_ratio': 0.9999999999999998}, {'seed': 7, 'jittered_song_rms_ratio': 1.0}, {'seed': 8, 'jittered_song_rms_ratio': 0.9999999999999999}, {'seed': 9, 'jittered_song_rms_ratio': 0.9999999999999999}]
CHOSEN ear settings: {'bands_hz': [[100, 500], [500, 2500]], 'filter_order': 4, 'filter': 'sosfiltfilt', 'padlen': 27, 'subwindows': 10, 'lif_steps_per_subwindow': 25, 'rms_full': 0.1253444659031983, 'reference': 'CHOSEN: JO-A RMS of one second of full pure pulse train', 'note': 'CHOSEN: her brain runs in real time so pulse timing can reach it'}
Replay uses song command, mode and distance with a disclosed trial-wide energy gain; P3 cannot isolate timing when energy differs substantially. Mute tests the command chain; its name does not guarantee silence.
MEASURED: mean world step 0.270482 s; female brain step 0.138315 s. Estimated ten-seed cost (6 x 400 windows each): 1.8 hours, excluding setup and rendering.

## What it means
Under this protocol, P1 was supported; P2 was supported; P3 was not supported; P4 was not supported; P5_command was not supported; P5_rms was not supported; P6 was not supported; P8 was not supported; P8b was descriptive; no verdict; P9 was not supported; P11 was not supported. These comparisons concern this simulator and input encoding.
She hears him, in real time and through a real waveform: on every seed his song put her vpoDN above zero, on nine of ten for at least three windows, and her pC1 rose with it; on every seed the same brain went silent when the sound was cut. That is the answer this map gives, and it comes from the wiring, not from a rule. The rest did not hold in this run. The rhythm of the song made no measurable difference against a jittered copy with the same energy, so what she hears here is energy, not timing. Taking her scent away did not lower his P1, silencing his P1 outputs did not lower his pIP10 or his song, and her LC10a-facing cells did not fire more when her silhouette was on his eye; each of those was measured and did not clear the bar, which is not the same as untested. She could not be made to say no through her own sex-peptide pathway, because in this map those axons end without synapses. The addendum below shows what the command group can do when it is driven from above.

## Limitations
- her state is a chosen tonic drive on SpsP, an identity taken from the FlyWire name and not verified; oviDN is a readout of a descending command, she has no body to extrude an ovipositor with; the start geometry is chosen so that she is visible to him
- His wing motor neurons run near ceiling from background activity; the song is taken from the command neuron pIP10, not from the motor sum. Mode still uses the motor means.
- Her ear now hears a waveform in 5 ms sub-windows; the brain runs in real time for her. His brain also runs in real time.
- Pulse rhythm and carriers are chosen synthesis, not measured spike timing or a biomechanical wing model.
- The fork's zero-phase filter uses the whole current 50 ms block; boundary effects and within-block lookahead remain. Sample bins alternate lengths at 22050 Hz.
- Jitter changes pulse density as well as timing (mean IPI 37.5 ms versus 35 ms). Replay holds source amplitude, mode and distance fixed; one trial-wide gain matches energy. Local envelopes and spectral energy can still differ.
- The contact chemosensory cells are a putative receptor label (putative_ppk23), not verified ppk23 expression.
- His female-scent input is a chosen drive at the smell ceiling with distance falloff, not a measured pheromone plume; the cVA channel is held at zero because there is no other male.
- He is inside the loop: his trajectory and song can change when she moves differently. Mute additionally changes his P1 outgoing gains.
- vpoDN identified as DNp37 by alias, 2 cells; pC1a-e 10 cells. No pheromone channel to her.
- In closed-loop 50 ms windows, silencing P1 outputs did not reduce pIP10 in the v4 smoke run; pIP10 is driven mainly by AVLP717m and aIPg7 in this graph, so the song dependence on P1 is not established here.
- Male P1 membership is the dictionary's uncertain 86-cell group. Outgoing-gain lesion need not silence P1's own spikes.
- No adaptation mechanism was added on his side; correlation does not establish causation.
- CHOSEN: female raw weights for parity; female eye blind. Earlier luminance silence measured vpoDN 54-206 Hz; contrast/motion vision remains a later step.
- Dark and silence coincide with the blind female eye.
- Male eye and soma-side availability are recorded per trial; gains remain uncalibrated.
- Distance changes include both bodies; approach is not an isolated female command.
- Retired v3 limitations: rate-only ear, motor-sum song, 12 ms brain windows, separate motor-reference clipping and shuffled-rate multiset no longer describe this protocol.

# Courtship experiment

Run: 10 seeds x 4 conditions x 400 steps; quick=False.
Quick runs are smoke tests; their two-seed verdicts are not the full ten-seed experiment.

## Question
Does his song change her graph's answer and their distance?

## Measured versus chosen
MEASURED: positions and headings, realised female speed, delivered sound, song, pC1 and vpoDN window rates.
IMPLEMENTATION: brain class flysim_gpu.FlyBrainGPU; torch devices {'male': 'cuda', 'female': 'cuda'}. Device is not a scientific choice. Required equivalence: same numbers on either device, verified by test (COURTSHIP_REAL_BRAIN=1); a failing test invalidates this claim.
CHOSEN: accept means vpoDN > 0 Hz in at least 3 windows; approach/retreat compare last and first quarter mean distance. Equal distance is neither. Geometry is sampled before each window; speed is displacement during it.
CHOSEN: paired seeds, annotation-backed male eye when available, uncalibrated gains, 12 ms brain windows, and the rate-to-motion mapping. No outcomes were tuned to differ across seeds.

CHOSEN: FEMALE_EXC_SCALE = 1.0; FEMALE_EYE = blind.
Conditions, same seed and same arena start for all four (paired):
- `song`: the room as built; his pulse and sine motor sums reach her JO-A and JO-B separately through `sound_hz` with distance attenuation.
- `silence`: her incoming `sound_hz` is forced to (0.0, 0.0) every step (he still sings; she does not hear). The Room-level hook overrides only her incoming sound; it does not modify his incoming sound channel.
- `shuffled`: her incoming `sound_hz` sequence is the `song` trial's sequence for the same seed, permuted as paired rows in time with a seed-derived permutation (run `song` first for that seed, keep its per-step sound series, then feed the permutation). This matches total sound energy; only the timing/coupling is broken.

- `dark`: her eye is blind and incoming sound is forced to (0.0, 0.0); this is the baseline. With FEMALE_EYE blind, dark and silence coincide; both are retained because they differ if FEMALE_EYE changes.

CHOSEN before data: female raw weights (FEMALE_EXC_SCALE = 1.0) give parity with the male. Her eye is blind (FEMALE_EYE = "blind"): uniform grey carries no information and floods her brain (measured silence vpoDN 54-206 Hz with eye, 0 Hz with blind). Contrast/motion vision is a later step.

CHOSEN: pulse-motor sum drives JO-A and sine-motor sum drives JO-B through sound_hz(rate, distance), each normalised by song_full_hz with its actual group size (8 and 2 respectively in these data) and the brain's refractory period; separate references preserve each population's fraction of its ceiling.
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
- Seed spread (maintainer's requirement, descriptive): number of seeds with accept / approach / retreat per condition; if every seed gives the same outcome in a condition, the report says so in one plain sentence and does not soften it.

## Predictions
| Prediction | Paired difference | SE | n | Verdict |
|---|---:|---:|---:|---|
| P1: vpodn_hz, song - control (silence) | 132.729 | 12.571542949440596 | 10 | supported |
| P2: pc1_hz, song - control (silence) | 3.18333 | 0.4932589564804151 | 10 | supported |
| P3: vpodn_hz, song - control (shuffled) | -11.8125 | 16.88711413600893 | 10 | not supported |
| P4: last_distance_mm, control - song (silence) | -3.46519 | 2.8368622961359695 | 10 | not supported |

## P0: baseline (descriptive, no verdict)
Expected active windows: 0.
Seed 0, silence: 0 active windows.
Seed 0, dark: 0 active windows.
Seed 1, silence: 0 active windows.
Seed 1, dark: 0 active windows.
Seed 2, silence: 0 active windows.
Seed 2, dark: 0 active windows.
Seed 3, silence: 0 active windows.
Seed 3, dark: 0 active windows.
Seed 4, silence: 0 active windows.
Seed 4, dark: 0 active windows.
Seed 5, silence: 0 active windows.
Seed 5, dark: 0 active windows.
Seed 6, silence: 0 active windows.
Seed 6, dark: 0 active windows.
Seed 7, silence: 0 active windows.
Seed 7, dark: 0 active windows.
Seed 8, silence: 0 active windows.
Seed 8, dark: 0 active windows.
Seed 9, silence: 0 active windows.
Seed 9, dark: 0 active windows.

## Per-seed outcomes
MEASURED: her_sound_a_clipped and her_sound_b_clipped count delayed song windows exceeding their references before the condition override; silence/dark counts describe attempted input and shuffled counts describe its live source, not replay clipping. None means unavailable in an older trace.
| Seed | Condition | Accept | Approach | Retreat | Active windows | her_sound_a_clipped | her_sound_b_clipped |
|---:|---|---|---|---|---:|---:|---:|
| 0 | song | True | False | True | 328 | 0 | 123 |
| 0 | silence | False | False | True | 0 | 0 | 78 |
| 0 | shuffled | True | False | True | 266 | 0 | 59 |
| 0 | dark | False | False | True | 0 | 0 | 78 |
| 1 | song | True | False | True | 304 | 0 | 84 |
| 1 | silence | False | False | True | 0 | 0 | 20 |
| 1 | shuffled | True | True | False | 334 | 0 | 50 |
| 1 | dark | False | False | True | 0 | 0 | 20 |
| 2 | song | True | True | False | 295 | 0 | 152 |
| 2 | silence | False | True | False | 0 | 0 | 67 |
| 2 | shuffled | True | False | True | 302 | 0 | 5 |
| 2 | dark | False | True | False | 0 | 0 | 67 |
| 3 | song | True | True | False | 124 | 0 | 180 |
| 3 | silence | False | True | False | 0 | 0 | 49 |
| 3 | shuffled | True | False | True | 281 | 0 | 56 |
| 3 | dark | False | True | False | 0 | 0 | 49 |
| 4 | song | True | False | True | 331 | 0 | 34 |
| 4 | silence | False | False | True | 0 | 22 | 106 |
| 4 | shuffled | True | True | False | 313 | 0 | 87 |
| 4 | dark | False | False | True | 0 | 22 | 106 |
| 5 | song | True | True | False | 339 | 0 | 71 |
| 5 | silence | False | False | True | 0 | 0 | 37 |
| 5 | shuffled | True | True | False | 268 | 0 | 73 |
| 5 | dark | False | False | True | 0 | 0 | 37 |
| 6 | song | True | False | True | 255 | 0 | 132 |
| 6 | silence | False | False | True | 0 | 0 | 29 |
| 6 | shuffled | True | False | True | 332 | 0 | 42 |
| 6 | dark | False | False | True | 0 | 0 | 29 |
| 7 | song | True | True | False | 333 | 0 | 81 |
| 7 | silence | False | False | True | 0 | 0 | 80 |
| 7 | shuffled | True | True | False | 307 | 0 | 72 |
| 7 | dark | False | False | True | 0 | 0 | 80 |
| 8 | song | True | False | True | 218 | 4 | 61 |
| 8 | silence | False | False | True | 0 | 0 | 50 |
| 8 | shuffled | True | False | True | 295 | 0 | 130 |
| 8 | dark | False | False | True | 0 | 0 | 50 |
| 9 | song | True | False | True | 161 | 0 | 206 |
| 9 | silence | False | False | True | 0 | 0 | 34 |
| 9 | shuffled | True | True | False | 288 | 0 | 43 |
| 9 | dark | False | False | True | 0 | 0 | 34 |

JO-B clipped-window totals over all seeds are 1124 for song, 550 for silence, 617 for shuffled, and 550 for dark.

## Seed spread
song: accept 10/10, approach 4/10, retreat 6/10. song: outcomes differ across seeds.
silence: accept 0/10, approach 2/10, retreat 8/10. silence: outcomes differ across seeds.
shuffled: accept 10/10, approach 5/10, retreat 5/10. shuffled: outcomes differ across seeds.
dark: accept 0/10, approach 2/10, retreat 8/10. dark: outcomes differ across seeds.

## P5: his song
This measures whether his song already depends on her; no adaptation mechanism was added. Correlation does not establish causation. None means a constant series or fewer than two samples.
| Seed | Condition | Song-distance Spearman | Song-speed Spearman |
|---:|---|---:|---:|
| 0 | song | 0.15941359020544604 | -0.09284795309094314 |
| 0 | silence | -0.6506655580170417 | None |
| 0 | shuffled | 0.3211857914147358 | -0.09719876313647577 |
| 0 | dark | -0.6506655580170417 | None |
| 1 | song | 0.453664977085091 | -0.2893589846002932 |
| 1 | silence | -0.6404778472294056 | None |
| 1 | shuffled | 0.45536758047111053 | -0.0571506427156956 |
| 1 | dark | -0.6404778472294056 | None |
| 2 | song | 0.32016721988874 | -0.2500129237727937 |
| 2 | silence | -0.11810440486086823 | None |
| 2 | shuffled | 0.0018131590911873922 | 0.043701051421584765 |
| 2 | dark | -0.11810440486086823 | None |
| 3 | song | 0.7247446796626783 | -0.7165130859570162 |
| 3 | silence | -0.08433479225599438 | None |
| 3 | shuffled | 0.4196910008702564 | -0.12349385777186966 |
| 3 | dark | -0.08433479225599438 | None |
| 4 | song | 0.37447952339230817 | -0.05899076813465773 |
| 4 | silence | 0.0551141261202332 | None |
| 4 | shuffled | -0.47155200045934037 | 0.10032904778187225 |
| 4 | dark | 0.0551141261202332 | None |
| 5 | song | 0.11438823207205172 | 0.1193552186807508 |
| 5 | silence | -0.010713437033262517 | None |
| 5 | shuffled | -0.028617730465832082 | -0.11602482261581372 |
| 5 | dark | -0.010713437033262517 | None |
| 6 | song | 0.12197238673542396 | -0.11728832853760313 |
| 6 | silence | -0.05055641886347585 | None |
| 6 | shuffled | 0.36548324358243883 | 0.15849536419391275 |
| 6 | dark | -0.05055641886347585 | None |
| 7 | song | -0.4479301860708252 | 0.003354470038891549 |
| 7 | silence | -0.033073749520211804 | None |
| 7 | shuffled | 0.15311766947211453 | -0.17723881843047795 |
| 7 | dark | -0.033073749520211804 | None |
| 8 | song | 0.11625206214760378 | 0.18403890715640311 |
| 8 | silence | 0.29145380181133823 | None |
| 8 | shuffled | -0.2625282835148971 | 0.030277954359591228 |
| 8 | dark | 0.29145380181133823 | None |
| 9 | song | 0.031593712068485665 | -0.4319644036761996 |
| 9 | silence | -0.12023797646610143 | None |
| 9 | shuffled | 0.10468704823228829 | -0.20911824696448844 |
| 9 | dark | -0.12023797646610143 | None |

## P6 his adaptation (descriptive)
His next pulse/sine window against her previous distance and speed; no verdict or adaptation mechanism. None means unavailable or constant data.
| Seed | Condition | P1 active windows | Pulse-distance lagged rho | Pulse-speed lagged rho | Sine-distance lagged rho | Sine-speed lagged rho |
|---:|---|---:|---:|---:|---:|---:|
| 0 | song | 64 | 0.1605439023290089 | -0.08896453701430614 | -0.12940942684894735 | 0.06346969975822975 |
| 0 | silence | 178 | -0.6467652774685838 | None | -0.5389454478693867 | None |
| 0 | shuffled | 110 | 0.3250170056785077 | -0.04597561426844078 | -0.04547277034890184 | 0.1119968759246195 |
| 0 | dark | 178 | -0.6467652774685838 | None | -0.5389454478693867 | None |
| 1 | song | 124 | 0.4563263319727492 | -0.28332848076340106 | 0.23903004911976192 | 0.021139675765106655 |
| 1 | silence | 400 | -0.6359645550048025 | None | -0.14268624431780405 | None |
| 1 | shuffled | 80 | 0.46949768194823877 | -0.0028532303066197175 | -0.519235665975638 | 0.11251849236498505 |
| 1 | dark | 400 | -0.6359645550048025 | None | -0.14268624431780405 | None |
| 2 | song | 199 | 0.32789090145086114 | -0.23516255961024335 | 0.5496170813508665 | -0.2503349284507096 |
| 2 | silence | 64 | -0.11813486776127531 | None | 0.12883302066376753 | None |
| 2 | shuffled | 59 | 0.0031569142646665988 | 0.08303289002412856 | -0.0034290772394626984 | 0.0861870221847697 |
| 2 | dark | 64 | -0.11813486776127531 | None | 0.12883302066376753 | None |
| 3 | song | 400 | 0.7314426999219241 | -0.7088146847002189 | 0.34065328317863575 | -0.24317766145004444 |
| 3 | silence | 18 | -0.08705599804500819 | None | -0.007268293471841826 | None |
| 3 | shuffled | 131 | 0.41224415650590795 | -0.17318552162466477 | -0.052952596579327806 | -0.15647425295213235 |
| 3 | dark | 18 | -0.08705599804500819 | None | -0.007268293471841826 | None |
| 4 | song | 102 | 0.3908158393098666 | -0.047810293302697156 | -0.21788951967046497 | 0.062276068268729286 |
| 4 | silence | 155 | 0.06341091408670614 | None | -0.20481661963698666 | None |
| 4 | shuffled | 92 | -0.45915299419540656 | 0.12196376741508426 | 0.3190060766156849 | 0.024385090504238014 |
| 4 | dark | 155 | 0.06341091408670614 | None | -0.20481661963698666 | None |
| 5 | song | 107 | 0.1282531373429601 | 0.11988091786986088 | -0.05007623242459639 | -0.014329409491700421 |
| 5 | silence | 128 | -0.01558337128685529 | None | -0.12158487487671746 | None |
| 5 | shuffled | 63 | -0.03236492735498877 | -0.11077158525180938 | -0.1570971849297997 | -0.02945656795004348 |
| 5 | dark | 128 | -0.01558337128685529 | None | -0.12158487487671746 | None |
| 6 | song | 396 | 0.13657466177709343 | -0.12012510403210648 | 0.3544645031643244 | -0.16478031883771405 |
| 6 | silence | 396 | -0.05742600995651179 | None | -0.035010159398890045 | None |
| 6 | shuffled | 80 | 0.3816191148552097 | 0.16985991984674026 | -0.3153603946012762 | -0.010555908113760235 |
| 6 | dark | 396 | -0.05742600995651179 | None | -0.035010159398890045 | None |
| 7 | song | 400 | -0.45093745741195135 | 0.001740578208736972 | 0.03978765291080845 | 0.1327125981738672 |
| 7 | silence | 80 | -0.02394588403321657 | None | 0.09723836850435685 | None |
| 7 | shuffled | 400 | 0.15445934604875294 | -0.20993904545181535 | 0.6795501977105317 | -0.4808278510807672 |
| 7 | dark | 80 | -0.02394588403321657 | None | 0.09723836850435685 | None |
| 8 | song | 75 | 0.12697476581816278 | 0.1477943722519972 | -0.035070289061916886 | -0.1043604720991337 |
| 8 | silence | 123 | 0.3136549640050101 | None | -0.30058710039222736 | None |
| 8 | shuffled | 22 | -0.2574441786942138 | 0.04274900914207322 | 0.046055897907431226 | -0.03942812505867161 |
| 8 | dark | 123 | 0.3136549640050101 | None | -0.30058710039222736 | None |
| 9 | song | 338 | 0.0323226991505737 | -0.46137236303234774 | 0.1915401074076417 | -0.136365644510033 |
| 9 | silence | 363 | -0.12568343670896145 | None | -0.24355531037965747 | None |
| 9 | shuffled | 66 | 0.10411487439702302 | -0.21243847415826778 | -0.022046842804488473 | 0.14541930918822193 |
| 9 | dark | 363 | -0.12568343670896145 | None | -0.24355531037965747 | None |

## What it means
Under this protocol, P1 was supported; P2 was supported; P3 was not supported; P4 was not supported. These comparisons concern this simulator and input encoding.
She heard him. On every seed, his song put her vpoDN above zero for a large part of the trial (124 to 339 of 400 windows) and left it at zero when the sound was taken away, and her pC1 rose with it; that is the answer this graph gives, and it comes from the wiring, not from a rule. Two things did not happen. The timing of the song did not matter: a shuffled copy of the same sound did as well as the song itself, which is what a rate-based ear predicts and what a waveform ear should be built to test. And she did not walk towards him: she walks when sung to and stands when not, but nothing in this readout turns hearing into a heading. On his side, the P1 group was active in as few as 18 windows on one seed and in all 400 on another, and his next window of song did not follow her previous distance or speed in any consistent way; no adaptation was built in and none appeared. The sentence the numbers support is "in this simulator, his song reaches her pC1 and vpoDN and changes what she does, but not where she goes", not "she chose him".

## Limitations
- The two-cell sine reference (song_full_hz for 2 cells) lies below the sum a 12 ms window can reach, so her JO-B drive is clipped at its ceiling in some song windows; the count is reported per trial. Pulse did not clip on the quick seeds.
- The contact chemosensory cells are a putative receptor label (putative_ppk23), not verified ppk23 expression.
- His female-scent input is a chosen drive at the smell ceiling with distance falloff, not a measured pheromone plume; the cVA channel is held at zero because there is no other male.
- He is inside the loop: his trajectory and song differ between conditions because she moves differently, not because his input path changed; only her incoming sound is overridden.
- Pulse and sine are separate motor population sums, not acoustic waveforms.
- Her ear receives a rate, not a waveform.
- 12 ms brain per 50 ms world.
- vpoDN identified as DNp37 by alias, 2 cells.
- pC1a-e 10 cells.
- No pheromone channel to her.
- Male P1 membership is the dictionary's uncertain 86-cell group.
- No adaptation mechanism was added on his side.
- CHOSEN before data: female loaded raw (FEMALE_EXC_SCALE = 1.0) for parity with the male.
- CHOSEN before data: female eye blind; uniform grey carries no information and floods the brain (measured silence vpoDN 54-206 Hz with eye, 0 Hz with blind).
- Dark is blind and silent; with FEMALE_EYE blind it coincides with silence. Contrast/motion vision is a later step.
- Male eye and soma-side availability are recorded per trial; gains remain uncalibrated.
- Distance changes include both bodies; approach is not an isolated female command.
- Shuffling matches the input rate multiset, not the downstream neural response.

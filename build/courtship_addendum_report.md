# Courtship v6 addendum

Baseline: {'path': 'build/courtship_v5_experiment.json', 'sha256': 'ebb98df47ce9f00e576fb158d858a3c150121e9ffdb38697117d1caaf9b52ccd', 'date': 'not recorded'}
Quick runs are smoke tests, not the full experiment.
Protocol v6 ADDENDUM. Only gated and p1drive are run; controls are the v5 song records paired by seed. Seed RNGs, start geometry, steps, brains and all other v5 song settings are retained.
CHOSEN gated: identical to v5 song (virgin), with outgoing gains 0.0 on exactly pC1a, pC1b, pC1c, pC1d, pC1e; other gains one. a mated female cannot be encoded through her own sex-peptide pathway in this map, because the SPSN and SAG axons carry no synapses here; the receptivity gate is closed by hand instead, as a lesion, the way the tests lesion.
CHOSEN p1drive: identical to v5 song, plus P1_DRIVE_HZ = 100 Hz to his P1 cells every window. asks whether the courtship command group can drive the song and her answer from above; it is an intervention, not a claim that P1 fires like this on its own.
Predictions, fixed before data:
- P9' she can be made to say no: her vpoDN mean rate, v5 song > gated (paired by seed), > 2 SE.
- P10' rejection (descriptive): oviDN mean and retreat fraction, gated vs v5 song.
- P13 the command can sing: his pIP10 mean rate and the delivered song RMS, p1drive > v5 song, both > 2 SE (joint verdict as P5).
- P14 the command reaches her: her vpoDN mean rate, p1drive > v5 song, > 2 SE.
- P15 (descriptive): his P1 active windows and LC10a in p1drive vs v5 song.
- Seed spread: per condition, seeds with accept / no-accept and approach / retreat; say plainly when every seed gives the same outcome.

| Seed | Condition | Accept | Approach | Retreat | vpoDN Hz | pIP10 Hz | Delivered RMS |
|---:|---|---|---|---|---:|---:|---:|
| 0 | gated | True | False | True | 26.225 | 1.475 | 0.0016402505762727882 |
| 0 | p1drive | True | False | True | 127.425 | 91.975 | 0.0649890754528579 |
| 1 | gated | True | True | False | 22.85 | 4.625 | 0.026957619444142374 |
| 1 | p1drive | True | False | True | 106.8 | 90.85 | 0.06371270055152374 |
| 2 | gated | False | True | False | 0.15 | 0.475 | 0.005555731646766057 |
| 2 | p1drive | True | False | True | 149.3 | 142.825 | 0.09143474535443892 |
| 3 | gated | True | False | True | 14.6 | 0.375 | 0.002947262489187087 |
| 3 | p1drive | True | False | True | 115.0 | 100.8 | 0.05312635227676349 |
| 4 | gated | True | False | True | 13.925 | 2.125 | 0.011250861057319284 |
| 4 | p1drive | True | False | True | 115.8 | 78.375 | 0.06733654384846395 |
| 5 | gated | True | False | True | 6.625 | 2.475 | 0.013711797168044855 |
| 5 | p1drive | True | False | True | 89.975 | 58.1 | 0.08237156300934706 |
| 6 | gated | True | True | False | 18.5 | 1.175 | 0.0063984947427947044 |
| 6 | p1drive | True | True | False | 124.925 | 77.075 | 0.09375175935458235 |
| 7 | gated | True | False | True | 1.15 | 2.525 | 0.01468544534159138 |
| 7 | p1drive | True | True | False | 102.725 | 68.625 | 0.08011854893798392 |
| 8 | gated | True | False | True | 11.1 | 3.95 | 0.01174044542870274 |
| 8 | p1drive | True | True | False | 113.625 | 124.425 | 0.09711233585790678 |
| 9 | gated | True | False | True | 0.675 | 3.075 | 0.015917832744288166 |
| 9 | p1drive | True | False | True | 75.725 | 98.8 | 0.07818213159140282 |
P9': {'mean': 2.0650000000000004, 'se': 4.615701162577818, 'n': 10, 'paired_differences': [-22.200000000000003, -13.05, 7.05, -14.525, 4.699999999999999, 19.35, 20.875, 2.725, 13.9, 1.825], 'metric': 'vpodn_hz', 'control': 'v5 song', 'direction': 'v5 song - gated', 'verdict': 'not supported'}
P13_command: {'mean': 88.2025, 'se': 8.124684993893608, 'n': 10, 'paired_differences': [91.77499999999999, 88.27499999999999, 127.27499999999999, 100.52499999999999, 76.925, 54.95, 52.75, 67.65, 123.25, 98.64999999999999], 'metric': 'pip10_hz', 'control': 'v5 song', 'direction': 'p1drive - v5 song', 'verdict': 'supported'}
P13_rms: {'mean': 0.0653792619469197, 'se': 0.005014808355792059, 'n': 10, 'paired_differences': [0.06333953627738412, 0.054062738624262585, 0.08297753275788981, 0.05053676958258484, 0.05959704633900991, 0.06437150638119168, 0.03845200816661992, 0.0728860091384026, 0.0911237947670114, 0.07644567743484013], 'metric': 'delivered_rms', 'control': 'v5 song', 'direction': 'p1drive - v5 song', 'verdict': 'supported'}
P14: {'mean': 98.485, 'se': 7.370489392766867, 'n': 10, 'paired_differences': [123.39999999999999, 97.0, 142.10000000000002, 114.925, 97.175, 63.99999999999999, 85.55, 98.85, 88.625, 73.225], 'metric': 'vpodn_hz', 'control': 'v5 song', 'direction': 'p1drive - v5 song', 'verdict': 'supported'}
P13 joint verdict: supported
P10' (descriptive):
{'seed': 0, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.2857142857142857}
{'seed': 0, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.05263157894736842}
{'seed': 1, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.21303258145363407}
{'seed': 1, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.18796992481203006}
{'seed': 2, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.15037593984962405}
{'seed': 2, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.19047619047619047}
{'seed': 3, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.18295739348370926}
{'seed': 3, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.17543859649122806}
{'seed': 4, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.11528822055137844}
{'seed': 4, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.3157894736842105}
{'seed': 5, 'condition': 'gated', 'ovidn_hz': 0.008333333333333333, 'retreat_fraction': 0.2807017543859649}
{'seed': 5, 'condition': 'song', 'ovidn_hz': 0.008333333333333333, 'retreat_fraction': 0.2681704260651629}
{'seed': 6, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.3508771929824561}
{'seed': 6, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.20802005012531327}
{'seed': 7, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.22556390977443608}
{'seed': 7, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.12531328320802004}
{'seed': 8, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.13533834586466165}
{'seed': 8, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.10526315789473684}
{'seed': 9, 'condition': 'gated', 'ovidn_hz': 0.0, 'retreat_fraction': 0.2706766917293233}
{'seed': 9, 'condition': 'song', 'ovidn_hz': 0.0, 'retreat_fraction': 0.08020050125313283}
P15 (descriptive):
{'seed': 0, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.12636363636363637}
{'seed': 0, 'condition': 'song', 'p1_active_windows': 117, 'lc10a_hz': 1.3556363636363635}
{'seed': 1, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.07836363636363636}
{'seed': 1, 'condition': 'song', 'p1_active_windows': 234, 'lc10a_hz': 0.11690909090909091}
{'seed': 2, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.21472727272727274}
{'seed': 2, 'condition': 'song', 'p1_active_windows': 204, 'lc10a_hz': 0.7878181818181819}
{'seed': 3, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.078}
{'seed': 3, 'condition': 'song', 'p1_active_windows': 92, 'lc10a_hz': 0.3165454545454546}
{'seed': 4, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.144}
{'seed': 4, 'condition': 'song', 'p1_active_windows': 100, 'lc10a_hz': 0.2792727272727273}
{'seed': 5, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.11800000000000001}
{'seed': 5, 'condition': 'song', 'p1_active_windows': 74, 'lc10a_hz': 0.062363636363636364}
{'seed': 6, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.13727272727272727}
{'seed': 6, 'condition': 'song', 'p1_active_windows': 187, 'lc10a_hz': 0.3210909090909091}
{'seed': 7, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.062}
{'seed': 7, 'condition': 'song', 'p1_active_windows': 34, 'lc10a_hz': 0.23018181818181815}
{'seed': 8, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.08181818181818182}
{'seed': 8, 'condition': 'song', 'p1_active_windows': 108, 'lc10a_hz': 0.21799999999999997}
{'seed': 9, 'condition': 'p1drive', 'p1_active_windows': 400, 'lc10a_hz': 0.23399999999999999}
{'seed': 9, 'condition': 'song', 'p1_active_windows': 41, 'lc10a_hz': 0.3198181818181818}
{'n': 10, 'accept': 9, 'approach': 3, 'retreat': 7, 'no_accept': 1, 'accept_seeds': [0, 1, 3, 4, 5, 6, 7, 8, 9], 'no_accept_seeds': [2], 'sentence': 'gated: outcomes differ across seeds.'}
{'n': 10, 'accept': 10, 'approach': 3, 'retreat': 7, 'no_accept': 0, 'accept_seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'no_accept_seeds': [], 'sentence': 'p1drive: outcomes differ across seeds.'}

## What it means

Driving his P1 raised his pIP10 by 88.2 Hz and increased the song energy reaching her, supporting both parts of P13. Her vpoDN rose by 98.5 Hz, supporting P14: the command reached her. Gated did not produce a no: her vpoDN stayed above zero on every seed, and its mean fell by 2.1 Hz without supporting P9'. Accept occurred on ten of ten p1drive seeds and nine of ten gated seeds; each condition had approach on three seeds and retreat on seven.

## Limitations
- Measured on the real graphs in GPU 250-step probes: the female brain-only FlyWire FAFB v783 map has no outgoing synapses from any of seven SpsP cells or either of two AN_SMP_2 (SAG) cells. Driving SpsP at 25-400 Hz or SAG at 25-200 Hz left pC1, vpoDN and oviDN unchanged within noise. v5 mated drives a dead switch: P9 cannot be supported for this anatomical reason. Gated is a manual lesion, not a mating-state encoding.
- Measured male probes: silence gave P1 0 Hz; Or47b scent gave P1 24-50 Hz, putative_ppk23 contact 24-46 Hz, and LC10a vision 26 Hz. LC10a drive gave pIP10 111 Hz; direct P1 drive at 100 Hz gave pIP10 76 Hz. P1 can drive pIP10, but vision also drives it, explaining why mute did not lower pIP10 in the loop. These are motivating probes, not v6 outcomes.
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
- The baseline is a separate v5 run; its path and byte SHA256 identify the controls. P1 drive is an imposed intervention, not spontaneous firing. The retained v5 state-drive limitation describes the disconnected tonic input; v6 closes pC1 by hand.

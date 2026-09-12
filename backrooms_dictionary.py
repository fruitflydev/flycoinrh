"""
The backrooms dictionary: which cells of the male CNS connectome may be named
on the page, by what name, and on whose authority.

Why this file exists. The backrooms page prints lines like
"A  P1 fires 41 Hz (courtship command neurons)". Every such line names a group
of neurons. Rule 7 of the build says a group may be named only if the type
strings in this dataset match a published identity, with the paper and figure
cited, and the number of cells measured. This module is the single place where
those matches are made and their evidence written down, so the captioner and
the page cannot name anything that is not here.

MEASURED (from build/graph.npz `types` and data/body-annotations.feather)
  * Which type strings exist among the 165,122 traced neurons, how many cells
    each has per side, and the annotators' fruDsx / dimorphism labels.
  * For groups the annotators tied to a paper themselves (their `synonyms`
    column, e.g. "Ruta 2010: DC1", "Asahina 2014: TK-FruM"), the list of type
    names carrying that tag. Those lists are copied here as literal names so
    that resolving a group at run time needs nothing but `fb.types`.

CHOSEN (and said so in each entry)
  * Which published population each group stands for, and the words used for
    its role. Each entry cites the paper and, where a figure could be read in
    this session, the figure. Where the figure number could not be checked the
    entry says "figure not verified".
  * `confidence`: "measured" when the dataset's own labels carry the published
    name (type, mancType, hemibrainType, flywireType or synonyms) and the paper
    describes that population in a male; "uncertain" when the name sits on two
    distinct types, the count disagrees with the paper, the annotators' own
    labels disagree with each other, or the role was shown only in females.
    Uncertain groups are shown on the page with that word.
  * Groups reported ABSENT (`present` False, empty `types`) are the names the
    build asked for that no type string carries. They resolve to nothing and
    the page cannot print them.

A type entry that starts with "^" is a regular expression matched against the
whole type string (re.fullmatch); any other entry is an exact type name.
Nothing here loads the brain. `groups(fb)` needs only `fb.types` (a str array,
one per neuron), which is what flysim.FlyBrain and any drop-in replacement
expose, so swapping the brain class later touches nothing in this file.

  py backrooms_dictionary.py --write     rebuild build/backrooms_dictionary.json
  py backrooms_dictionary.py --check     confirm the JSON on disk is current
  py -m pytest -q test_backrooms_dictionary.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
BUILD = ROOT / "build"
GRAPH = BUILD / "graph.npz"
ANNOTATIONS = ROOT / "data" / "body-annotations.feather"
JSON_PATH = BUILD / "backrooms_dictionary.json"

MEASURED = "measured"
UNCERTAIN = "uncertain"

# The roamer's motor readout (pumpui.FlyPilot.motor, reused by plume_fly):
# these four types are read as the fly's walking command. They are listed in
# the dictionary so the recorder can log them, flagged `motor` so tests can
# check that no other group overlaps them.
MOTOR_READOUT = {
    "DNa02": "steering (left/right asymmetry)",
    "DNa01": "forward (the roamer's reading)",
    "MDN": "backward walking",
    "DNp09": "stop (the roamer's reading; Bidaye 2020 shows forward walking, "
             "freezing only at strong activation)",
}

# pC1 subtypes the annotators tag "Cachero 2010: pMP-e; Yu 2010: pMP4"
# (MEASURED from the synonyms column; 25 types, 86 cells, 43 per side).
_P1_TYPES = [
    "pC1_12a", "pC1_13a", "pC1_14a", "pC1_14b", "pC1_15a", "pC1_15b",
    "pC1_15c", "pC1_16a", "pC1_16b", "pC1_17a", "pC1_17b", "pC1_1a",
    "pC1_1b", "pC1_2a", "pC1_2a/2b", "pC1_2b", "pC1_2c", "pC1_3a",
    "pC1_3b", "pC1_3c", "pC1_5a", "pC1_5b", "pC1_6a", "pC1_7a", "pC1_7b",
]
# annotators' tag "Ruta 2010: DC1" (8 types, 66 cells) and "Ruta 2010: LC1"
# (3 types, 25 cells)
_DC1_TYPES = ["AVLP753m", "LH001m", "LH002m", "LH003m", "LH006m", "LH008m",
              "PVLP205m", "SLP160"]
_LC1_TYPES = ["AVLP743m", "LH004m", "LH007m"]
# annotators' tag "Zhou 2014: pCd": 66 cells on 16 types, but the tag is
# per cell and three types carry it on only some of their cells
# (CB1508,CB2021 2 of 16; CB1671,CB2138,CB2610 1 of 5; CB2349,CB2520 1 of 7).
# Only the 13 types tagged on every cell are listed (62 cells), because the
# group resolves by type name and must not sweep in untagged cells.
_PCD_TYPES = ["CB0405", "CB0975", "CB1379", "DNpe034", "SCL002m", "SMP285",
              "SMP286", "SMP700m", "SMP710m", "SMP720m", "SMP721m", "SMP726m",
              "SMP727m"]
# annotators' tag "Nojima 2021: pC2l": 37 cells on 16 types, of which only 9
# types carry it on every cell (20 cells). The mixed types are left out for
# the same reason (AVLP715m 2 of 4, AVLP718m 4 of 5, PVLP204m 4 of 6,
# PVLP209m 2 of 13, PVLP210m 3 of 6, SIP108m 1 of 4, SIP109m 1 of 4).
_PC2L_TYPES = ["AVLP567", "AVLP713m", "AVLP716m", "AVLP735m", "AVLP736m",
               "AVLP737m", "AVLP738m", "SIP137m_a", "SIP137m_b"]
# every vnc_motor type whose subclass is "wm" (wing motor); one wm cell has an
# empty type string and is left out because it cannot be named
_WING_MN_TYPES = [
    "DLMn a, b", "DLMn c-f", "DVMn 1a-c", "DVMn 2a, b", "DVMn 3a, b",
    "MNwm35", "MNwm36", "STTMm", "TTMn", "b1 MN", "b2 MN", "b3 MN",
    "hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN", "i1 MN", "i2 MN", "iii1 MN",
    "iii3 MN", "ps1 MN", "ps2 MN", "tp1 MN", "tp2 MN", "tpn MN",
]

_FIG_NOT_VERIFIED = "figure not verified in this session"


def _entry(name, types, role, citation, confidence, *, present=True,
           motor=False, contains=(), note="", channel=""):
    return {
        "name": name,
        "types": list(types),
        "role": role,
        "citation": citation,
        "confidence": confidence,
        "present": present,
        "motor": motor,
        "contains": list(contains),
        "note": note,
        "channel": channel,
    }


DICTIONARY = {
    # ---- smell: the cVA pathway -------------------------------------------
    "ORN_DA1": _entry(
        "ORN_DA1", ["ORN_DA1"],
        "cVA pheromone receptor neurons (Or67d, glomerulus DA1)",
        "Kurtovic, Widmer & Dickson 2007 Nature 446:542 (Or67d neurons "
        "respond to cVA); Couto, Alenius & Dickson 2005 Curr Biol 15:1535 "
        "(Or67d projects to DA1); " + _FIG_NOT_VERIFIED + ". Identity: the "
        "type name ORN_DA1.",
        MEASURED, channel="smell",
        note="Cell bodies are in the antenna, so there is no somaSide; the "
             "annotators' rootSide gives R 105, L 51, unknown 48. This is the "
             "group the room drives with the other fly's presence."),
    "DA1_PN": _entry(
        "DA1 PN", ["DA1_lPN", "DA1_vPN"],
        "cVA projection neurons (DA1 glomerulus to lateral horn)",
        "Datta et al. 2008 Nature 452:473 (DA1 projection neurons, "
        "dimorphic targets); Ruta et al. 2010 Nature 468:686; "
        + _FIG_NOT_VERIFIED + ". Identity: type names DA1_lPN, DA1_vPN.",
        MEASURED, channel="smell"),
    "DC1": _entry(
        "DC1", _DC1_TYPES,
        "third-order cVA neurons, fru+ (Ruta 2010 DC1)",
        "Ruta et al. 2010 Nature 468:686 (DC1 receive DA1 PN input and are "
        "tuned to cVA); " + _FIG_NOT_VERIFIED + ". Identity: the annotators' "
        "synonyms column tags these 8 types 'Ruta 2010: DC1'.",
        MEASURED, channel="smell"),
    "LC1": _entry(
        "LC1", _LC1_TYPES,
        "third-order cVA neurons, fru+ (Ruta 2010 LC1)",
        "Ruta et al. 2010 Nature 468:686; " + _FIG_NOT_VERIFIED + ". "
        "Identity: the annotators' synonyms column tags these 3 types "
        "'Ruta 2010: LC1'.",
        MEASURED, channel="smell"),
    "DNp13": _entry(
        "DNp13 (Ruta 2010 DN1)", ["DNp13"],
        "dsx+ descending neuron tagged Ruta 2010 DN1 (cVA pathway); pMN1, "
        "the ovipositor-extrusion command in females (Mezzera 2020)",
        "Ruta et al. 2010 Nature 468:686 (DN1, described there as a "
        "male-specific descending neuron excited by DC1 and responding to "
        "cVA); Kimura et al. 2015 PLoS ONE 10:e0126445 (pMN1, dsx+, "
        "ovipositor extension in females); Mezzera et al. 2020 Curr Biol "
        "30:3736 (DNp13 = pMN1 drives full ovipositor extrusion in females); "
        + _FIG_NOT_VERIFIED + ". Identity: type DNp13, whose synonyms "
        "column reads 'Kimura 2015: pMN1; Ruta 2010: DN1'; the annotators' "
        "dimorphism label is 'sexually dimorphic' (present in both sexes), "
        "not male-specific.",
        UNCERTAIN, channel="smell",
        note="The tag says a male-specific cVA descending neuron (Ruta "
             "2010), the annotators' dimorphism label says sexually "
             "dimorphic, and the best-documented role of pMN1 / DNp13 is a "
             "female motor command (Mezzera 2020): the paper and the labels "
             "disagree, so the group is labelled uncertain."),

    # ---- hearing ------------------------------------------------------------
    "JO_A": _entry(
        "JO-A", ["^JO-A.*"],
        "sound-sensitive Johnston's organ neurons, subgroup A",
        "Kamikouchi et al. 2009 Nature 458:165 (subgroups A and B respond to "
        "vibration, C and E to static deflection); Yorozu et al. 2009 Nature "
        "458:201; " + _FIG_NOT_VERIFIED + ". Identity: type names JO-A1..A4 "
        "and JO-A-unclear.",
        MEASURED, channel="sound",
        note="Antennal cells: no somaSide; rootSide L 18, R 32. The "
             "annotators' subclass calls 49 of the 50 'auditory' and 1 "
             "'wind_gravity'. This is the group the room drives with the "
             "other fly's song."),
    "JO_B": _entry(
        "JO-B", ["^JO-B.*"],
        "sound-sensitive Johnston's organ neurons, subgroup B",
        "Kamikouchi et al. 2009 Nature 458:165; Yorozu et al. 2009 Nature "
        "458:201; " + _FIG_NOT_VERIFIED + ". Identity: type names "
        "JO-B1_a..B4_b and JO-B-unclear.",
        UNCERTAIN, channel="sound",
        note="Antennal cells: no somaSide; rootSide L 60, R 29. The "
             "annotators' subclass calls 54 of the 89 'auditory', 34 "
             "'wind_gravity' and leaves 1 blank; the group follows the type "
             "name, as the FACTS do. The annotators' labels disagree with "
             "each other on 35 of 89 cells, so the group is labelled "
             "uncertain."),

    # ---- sight ----------------------------------------------------------------
    "LC10a": _entry(
        "LC10a", ["LC10a"],
        "small-object tracking visual projection neurons used in courtship "
        "pursuit, fru+",
        "Ribeiro et al. 2018 Cell 174:607 (LC10 silencing abolishes "
        "orientation toward and pursuit of the female); Hindmarsh Sten et "
        "al. 2021 Nature 595:549 (LC10a, gated by P1); "
        + _FIG_NOT_VERIFIED + ". Identity: type name LC10a.",
        MEASURED, channel="sight",
        note="LC10b, LC10c-1/2, LC10d, LC10e and LoVP76 (LC10f) exist too "
             "(689 more cells) and are not named, since the courtship "
             "evidence is for LC10a."),

    # ---- courtship command ------------------------------------------------
    "pC1": _entry(
        "pC1", ["^pC1_.*", "^pC1x_.*"],
        "dsx+ male courtship and aggression command cluster (contains P1)",
        "Rideout et al. 2010 Nat Neurosci 13:458 (pC1 dsx cluster); Nojima "
        "et al. 2021 Curr Biol 31:2276; Sten et al. 2025 (pC1x); "
        + _FIG_NOT_VERIFIED + ". Identity: type names pC1_* and pC1x_* "
        "(not LPC1/LLPC1, which are optic-lobe types).",
        MEASURED, contains=["P1"], channel="courtship"),
    "P1": _entry(
        "P1", _P1_TYPES,
        "courtship command neurons (P1 = pMP4 = pMP-e), fru+ dsx+",
        "Kimura et al. 2008 Neuron 59:759 (P1); von Philipsborn et al. 2011 "
        "Neuron 69:509 (P1 activation drives song; "
        + _FIG_NOT_VERIFIED + "); Hoopfer et al. 2015 eLife 4:e11346 Fig. 1 "
        "(P1, 'also known as pMP4/-e', activation promotes wing extension "
        "and aggression); Inagaki et al. 2014 Nat Methods 11:325. Identity: "
        "the 25 pC1 subtypes the annotators tag 'Cachero 2010: pMP-e; Yu "
        "2010: pMP4'.",
        UNCERTAIN, channel="courtship",
        note="86 cells here (43 per side) against about 20 per hemisphere "
             "in Kimura 2008, so the tag covers more cells than the "
             "published P1 cluster; named with the label 'uncertain'."),
    "pCd": _entry(
        "pCd", _PCD_TYPES,
        "dsx+ cluster; in males cVA-responsive and required for P1-evoked "
        "persistent courtship and aggression",
        "Jung et al. 2020 Neuron 105:322 (male pCd neurons respond to cVA "
        "and are required for P1-evoked persistent courtship and "
        "aggression; " + _FIG_NOT_VERIFIED + "); Zhou et al. 2014 Neuron "
        "83:149 (the name pCd: a dsx+ cluster studied there for female "
        "receptivity; the source of the annotators' tag, not of the male "
        "role). Identity: the annotators' synonyms column tags 66 cells on "
        "16 types 'Zhou 2014: pCd' (Rideout 2010 pC3; Nojima 2021 pCd-1, "
        "pCd-2); the 13 types tagged on every cell are listed (62 cells).",
        UNCERTAIN, channel="courtship",
        note="The tag is spread over many types including one descending "
             "neuron (DNpe034, mancType oviDN), is per cell rather than per "
             "type on three of them (those are left out, 4 tagged cells "
             "lost), and the annotators split it into pCd-1 and pCd-2. "
             "Labelled uncertain for that spread. The name comes from a "
             "female study (Zhou 2014); the male role is Jung 2020."),
    "pC2l": _entry(
        "pC2l", _PC2L_TYPES,
        "dsx+ song-responsive neurons",
        "Deutsch et al. 2019 eLife 8:e49890 (pC2l respond to courtship "
        "song); " + _FIG_NOT_VERIFIED + ". Identity: the annotators' "
        "synonyms column tags 37 cells on 16 types 'Nojima 2021: pC2l'; "
        "the 9 types tagged on every cell are listed (20 cells).",
        UNCERTAIN, channel="sound",
        note="No type is named pC2; the name exists only as a per-cell "
             "synonym on 16 AVLP/PVLP/SIP types, 7 of them tagged on only "
             "some cells (left out, 17 tagged cells lost). Labelled "
             "uncertain for that spread."),
    "mAL": _entry(
        "mAL", ["^mAL_m.*"],
        "fru+ GABAergic neurons inhibiting P1 (contact pheromone pathway)",
        "Kimura et al. 2005 Nature 438:229 (mAL, fru+, male-enlarged); "
        "Clowney et al. 2015 Neuron 87:1036 Fig. 3-4 (mAL inhibition "
        "converges on P1); Kallman, Kim & Scott 2015 eLife 4:e11188. "
        "Identity: type names mAL_m1..m11 (fru_high, male-specific).",
        MEASURED, channel="courtship",
        note="Types mAL4A..mAL6, mALB*, mALD* (hemibrain lineage names, no "
             "fru label, synonym 'mAL') are left out."),
    "vAB3": _entry(
        "vAB3", ["AN09B017e", "AN09B017f", "AN09B017g"],
        "ascending pheromone neurons exciting P1 (ppk25 foreleg input)",
        "Clowney et al. 2015 Neuron 87:1036 Fig. 2-3 (vAB3 excitation onto "
        "P1). Identity: the annotators' synonyms column tags these 3 types "
        "'Yu 2010: vAB3' (one also 'von Philipsborn 2011: vAB3').",
        MEASURED, channel="courtship"),
    "PPN1": _entry(
        "PPN1", ["AN05B102a"],
        "pheromone processing neuron of the leg-to-brain pathway",
        "Kallman, Kim & Scott 2015 eLife 4:e11188; " + _FIG_NOT_VERIFIED
        + ". Identity: the annotators' synonyms column tags AN05B102a "
        "'Kallman 2015: PPN1'.",
        MEASURED, channel="courtship"),

    # ---- aggression -------------------------------------------------------
    "Tk_FruM": _entry(
        "Tk-FruM", ["AVLP727m"],
        "male-specific tachykinin neurons promoting aggression",
        "Asahina et al. 2014 Cell 156:221 Fig. 2 (about 4 male-specific "
        "Tk-GAL4^FruM cells per hemisphere in the lateral protocerebrum; "
        "activation promotes aggression) and Fig. 3 (no effect on "
        "courtship). Identity: the annotators' synonyms column tags "
        "AVLP727m 'Asahina 2014: TK-FruM'.",
        UNCERTAIN, channel="aggression",
        note="5 cells here, 3 left and 2 right, against 4.2 +/- 0.8 per "
             "hemisphere (about 8 in all) in the paper: the count disagrees "
             "with the paper, so the group is labelled uncertain."),
    "aIPg": _entry(
        "aIPg", ["^aIPg.*"],
        "aggression neurons (shown in females; fru+ in this male, male role "
        "not shown)",
        "Schretter et al. 2020 eLife 9:e58942 Fig. 1 (aIPg activation "
        "raises female aggression) and Fig. 5 (aIPg in the EM volume); "
        "Cachero et al. 2010 Curr Biol 20:1589 (aIP-g, fru+, dimorphic). "
        "Identity: type names aIPg1..aIPg10 and aIPg_m1..m4.",
        UNCERTAIN, channel="aggression",
        note="Schretter 2020 saw no expression of two aIPg lines in males "
             "and no male effect, so the role is female evidence only."),

    # ---- song: brain and nerve-cord premotor -----------------------------
    "pIP10": _entry(
        "pIP10", ["pIP10"],
        "song descending neuron (drives both pulse and sine song)",
        "von Philipsborn et al. 2011 Neuron 69:509 (pIP10; "
        + _FIG_NOT_VERIFIED + "); Lillvis et al. 2024 Curr Biol 34:808 "
        "Fig. 4; Shiozaki et al. 2024 Nat Neurosci 27:1954 Fig. 6. "
        "Identity: type and mancType pIP10 (synonyms: Kimura 2008 P2b, "
        "Cachero 2010 pIP-a, Yu 2010 pIP1).",
        MEASURED, channel="song"),
    "pMP2": _entry(
        "pMP2", ["pMP2"],
        "pulse-song descending neuron",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 3 (pMP2 required for "
        "pulse song); Shiozaki et al. 2024 Nat Neurosci 27:1954 Fig. 6 "
        "(pulse-selective). Identity: type and mancType pMP2.",
        MEASURED, channel="song"),
    "dPR1": _entry(
        "dPR1", ["dPR1", "IN08B001"],
        "pulse-song premotor neuron",
        "von Philipsborn et al. 2011 Neuron 69:509 (dPR1; "
        + _FIG_NOT_VERIFIED + "); Lillvis et al. 2024 Curr Biol 34:808 "
        "Fig. 3; Shiozaki et al. 2024 Nat Neurosci 27:1954 Fig. 1 "
        "(pulse-selective). Identity: mancType dPR1 on one type and the "
        "synonym 'Yu 2010, Lillvis 2024: dPr1' on IN08B001.",
        UNCERTAIN, channel="song",
        note="The dataset carries the name on two distinct types (2 cells "
             "each); both are included and the group is labelled uncertain."),
    "TN1A": _entry(
        "TN1A", ["^TN1a_.*"],
        "sine-song premotor neurons (also active during pulse)",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 2 (TN1A necessary and "
        "sufficient for sine song); Shiozaki et al. 2024 Nat Neurosci "
        "27:1954 Fig. 1 and 3. Identity: mancType TN1a (types TN1a_a..i).",
        MEASURED, channel="song",
        note="TN1c_a..d (13 cells, mancType TN1c) are not in the cited "
             "figures and are left out."),
    "dMS2": _entry(
        "dMS2", ["dMS2"],
        "sine-song premotor neurons",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 2 (dMS2 necessary and "
        "sufficient for sine song). Identity: type and mancType dMS2 "
        "(synonym 'Yu 2010, Lillvis 2024: dMS2').",
        MEASURED, channel="song"),
    "dMS9": _entry(
        "dMS9", ["dMS9"],
        "pulse-song premotor neurons",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 3 (dMS9 required for "
        "pulse song). Identity: type and mancType dMS9 (synonym 'Lillvis "
        "2024: dMS9').",
        MEASURED, channel="song"),
    "vMS12": _entry(
        "vMS12", ["^vMS12_.*"],
        "pulse-song premotor neurons that drive the ps1 wing motor neuron",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 3 (required for pulse "
        "song) and Fig. 8D (ps1 receives its input mainly from vMS12). "
        "Identity: mancType vMS12 (types vMS12_a..e; synonym 'Cachero 2010: "
        "vMs-m').",
        MEASURED, channel="song"),
    "vPR6": _entry(
        "vPR6", ["vPR6"],
        "song premotor neurons (pulse timing in von Philipsborn 2011; not "
        "part of the core circuit in Lillvis 2024)",
        "von Philipsborn et al. 2011 Neuron 69:509 (vPR6; "
        + _FIG_NOT_VERIFIED + "); Lillvis et al. 2024 Curr Biol 34:808 "
        "supplementary figures. Identity: type and mancType vPR6.",
        MEASURED, channel="song"),
    "vMS11": _entry(
        "vMS11", ["vMS11"],
        "song premotor neurons (von Philipsborn 2011)",
        "von Philipsborn et al. 2011 Neuron 69:509 (vMS11; "
        + _FIG_NOT_VERIFIED + "). Identity: type and mancType vMS11.",
        UNCERTAIN, channel="song",
        note="The published vMS11 is fru+; the 14 cells named vMS11 here "
             "carry no fruDsx label, so the match is labelled uncertain."),
    "vPR9": _entry(
        "vPR9", ["^vPR9_.*", "IN00A017"],
        "GABAergic song neurons acting on both song modes",
        "Lillvis et al. 2024 Curr Biol 34:808 Fig. 5-6; Shiozaki et al. "
        "2024 Nat Neurosci 27:1954 Extended Data Fig. 8. Identity: mancType "
        "vPR9 (types vPR9_a..c) and the synonym 'Yu 2010, Lillvis 2024: "
        "vPR9' on IN00A017.",
        UNCERTAIN, channel="song",
        note="Two labels on distinct types, and the vPR9_* cells carry no "
             "somaSide (midline cells)."),

    # ---- song: wing motor neurons ------------------------------------------
    "song_pulse_mn": _entry(
        "pulse-song wing motor neurons", ["ps1 MN", "i1 MN", "iii1 MN", "b3 MN"],
        "wing motor neurons wired from pulse-song premotor neurons",
        "Shirangi, Stern & Truman 2013 Cell Rep 5:678 Fig. 4 (inhibiting the "
        "ps1 motoneuron lowers pulse carrier frequency and amplitude, sine "
        "song intact); Lillvis et al. 2024 Curr Biol 34:808 Fig. 8D (ps1, "
        "i1, iii1 and b3 are innervated by pulse-specific neurons). "
        "Identity: MANC motor-neuron type names 'ps1 MN', 'i1 MN', "
        "'iii1 MN', 'b3 MN' (subclass wm).",
        MEASURED, contains=["song_ps1"], channel="song",
        note="This is the group the room reads as the fly's song. The "
             "evidence for i1, iii1 and b3 is wiring (Fig. 8D); for ps1 it "
             "is silencing (Shirangi 2013 Fig. 4)."),
    "song_ps1": _entry(
        "ps1 MN", ["ps1 MN"],
        "pulse-song wing motor neuron (pleurosternal muscle ps1)",
        "Shirangi, Stern & Truman 2013 Cell Rep 5:678 Fig. 4; Lillvis et "
        "al. 2024 Curr Biol 34:808 Fig. 8D. Identity: type 'ps1 MN'.",
        MEASURED, channel="song"),
    "song_sine_hg1": _entry(
        "hg1 MN", ["hg1 MN"],
        "sine-song wing motor neuron (male-enlarged muscle hg1)",
        "Shirangi, Stern & Truman 2013 Cell Rep 5:678 Fig. 1 (hg1 is "
        "male-enlarged) and Fig. 2 (silencing the hg1 motoneuron removes "
        "sine song, pulse song intact). Identity: type 'hg1 MN'.",
        MEASURED, channel="song",
        note="The build's FACTS listed hg1 with ps1 as pulse-song motor "
             "neurons; Shirangi 2013 assigns hg1 to sine song, so hg1 is "
             "kept out of the pulse group and named here as sine."),
    "wing_mn_all": _entry(
        "wing motor neurons", _WING_MN_TYPES,
        "all named wing motor neurons (steering and power muscles)",
        "MANC nomenclature: Takemura et al. 2024 eLife 13:RP97769 and "
        "Cheong et al. 2024 eLife 13:RP96084 (male adult nerve cord); "
        "Azevedo et al. 2024 Nature 631:360 (wing motor neurons of the "
        "female VNC). Identity: vnc_motor types with subclass 'wm'.",
        MEASURED, contains=["song_pulse_mn", "song_ps1", "song_sine_hg1"],
        channel="song",
        note="One wm cell has an empty type string and is left out. The "
             "room falls back to this group for the song only if the "
             "pulse group were absent; it is not."),

    # ---- motor readout (the roamer's mapping; recorded, not captioned as
    #      circuits, and never overlapped by any group above) ---------------
    "DNa02": _entry(
        "DNa02", ["DNa02"], "steering descending neurons (left/right asymmetry)",
        "Rayshubskiy et al. 2020 bioRxiv 2020.04.04.024703 (DNa02 steering); "
        "the roamer's mapping, pumpui.FlyPilot. Identity: type DNa02.",
        MEASURED, motor=True, channel="motor",
        note="DNae001 carries hemibrainType DNa01 and DNp71 carries "
             "hemibrainType DNp09; both are excluded, as in the roamer, which "
             "matches the type name exactly."),
    "DNa01": _entry(
        "DNa01", ["DNa01"], "forward walking descending neurons (the roamer's reading)",
        "Chen et al. 2018 Nat Commun 9:4390 and Rayshubskiy et al. 2020 "
        "describe DNa01 as a steering neuron; the roamer reads it as "
        "forward. Identity: type DNa01.",
        MEASURED, motor=True, channel="motor"),
    "MDN": _entry(
        "MDN", ["MDN"], "backward walking descending neurons (moonwalker)",
        "Bidaye et al. 2014 Science 344:97 (MDN). Identity: type MDN "
        "(synonym 'Carreira-Rosario 2018: DNp50').",
        MEASURED, motor=True, channel="motor"),
    "DNp09": _entry(
        "DNp09", ["DNp09"],
        "forward-walking descending neurons that freeze the fly at strong "
        "activation; read as stop by the roamer",
        "Bidaye et al. 2020 Neuron 108:469 (DNp09 drives forward walking, "
        "freezing at strong activation); the roamer reads it as stop. "
        "Identity: type DNp09.",
        MEASURED, motor=True, channel="motor"),

    # ---- absent: asked for by name, no type string carries it ------------
    "P1_by_name": _entry(
        "P1 (by name)", [],
        "no type is named P1", "The only 'P1' string is hemibrainType "
        "'P1-9' on OA-AL2i1, an octopaminergic optic-lobe type, unrelated. "
        "P1 is reachable only through the pC1 subtypes tagged pMP-e/pMP4 "
        "(group 'P1').",
        UNCERTAIN, present=False, channel="courtship"),
    "aDN": _entry(
        "aDN (courtship descending)", [],
        "no courtship type is named aDN", "Types 'ADNM1 MN' and 'ADNM2 MN' "
        "are abdominal motor neurons (mancType ADNM1/2), not a courtship "
        "descending neuron.",
        UNCERTAIN, present=False, channel="courtship"),
    "pC2_by_name": _entry(
        "pC2 (by name)", [],
        "no type is named pC2", "pC2l exists only as the annotators' "
        "synonym on 16 types (group 'pC2l').",
        UNCERTAIN, present=False, channel="sound"),
    "vpoDN": _entry(
        "vpoDN", [],
        "female vaginal-plate-opening descending neuron; not in a male",
        "Wang et al. 2021 Nature 589:577 (vpoDN). vpoEN and vpoIN types "
        "exist here (fru+), vpoDN does not.",
        UNCERTAIN, present=False, channel="courtship"),
}

MOTOR_KEYS = tuple(k for k, v in DICTIONARY.items() if v["motor"])
ABSENT_KEYS = tuple(k for k, v in DICTIONARY.items() if not v["present"])


# ---- resolution ------------------------------------------------------------

def match_mask(names, spec):
    """
    Boolean mask over `names` (unique type strings) for one dictionary entry's
    `types` list: entries starting with '^' are regular expressions matched
    against the whole string, the rest are exact names. fullmatch is chosen
    so that '^pC1_.*' cannot pick up LPC1 or LLPC1 and 'DNa01' cannot pick
    up DNae001.
    """
    names = np.asarray(names, dtype=str)
    m = np.zeros(len(names), dtype=bool)
    for t in spec:
        if t.startswith("^"):
            rx = re.compile(t)
            m |= np.fromiter((rx.fullmatch(n) is not None for n in names),
                             dtype=bool, count=len(names))
        else:
            m |= names == t
    return m


def resolve(types, spec):
    """Index array into `types` (one str per neuron) for a `types` spec."""
    types = np.asarray(types, dtype=str)
    uniq, inv = np.unique(types, return_inverse=True)
    m = match_mask(uniq, spec)
    return np.flatnonzero(m[inv]).astype(np.int64)


def groups(fb, keys=None):
    """
    {key: index array} for every dictionary group, resolved against fb.types.
    Absent groups resolve to empty arrays. Only fb.types is touched, so any
    object with a per-neuron `types` str array serves, brain or not.
    """
    types = np.asarray(fb.types, dtype=str)
    uniq, inv = np.unique(types, return_inverse=True)
    out = {}
    for k in (keys or DICTIONARY):
        m = match_mask(uniq, DICTIONARY[k]["types"])
        out[k] = np.flatnonzero(m[inv]).astype(np.int64)
    return out


def present_groups(fb):
    """The non-empty groups only: what the recorder records and the page shows."""
    return {k: v for k, v in groups(fb).items() if len(v)}


# ---- counts and the JSON ----------------------------------------------------

def load_types(graph_path=GRAPH):
    """The per-neuron type strings and body ids, without touching the weights."""
    z = np.load(graph_path, allow_pickle=False)
    return z["types"].astype(str), z["bodies"]


def _annotations(bodies, path=ANNOTATIONS):
    import pandas as pd
    ann = pd.read_feather(path)
    ann = ann.drop_duplicates("bodyId").set_index("bodyId").reindex(bodies)
    cols = ["somaSide", "rootSide", "fruDsx", "dimorphism", "subclass"]
    return {c: ann[c].fillna("").astype(str).to_numpy() for c in cols}


def _counter(values):
    u, c = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return {str(k): int(n) for k, n in zip(u, c)}


def counts(types, ann, idx):
    """
    Per-group numbers: cells, cells per side (somaSide, else rootSide for
    antennal sensory cells), fruDsx and dimorphism labels, cells per type.
    """
    side = ann["somaSide"][idx].copy()
    src = "somaSide"
    if len(idx) and not (side != "").any():
        side = ann["rootSide"][idx]
        src = "rootSide"
    return {
        "n": int(len(idx)),
        "L": int((side == "L").sum()),
        "R": int((side == "R").sum()),
        "side_source": src if len(idx) else "",
        "side_other": int(((side != "L") & (side != "R")).sum()),
        "fruDsx": _counter(ann["fruDsx"][idx]),
        "dimorphism": _counter(ann["dimorphism"][idx]),
        "subclass": _counter(ann["subclass"][idx]),
        "per_type": _counter(types[idx]),
    }


def build(graph_path=GRAPH, ann_path=ANNOTATIONS):
    """The dictionary with measured counts attached, as a plain dict."""
    types, bodies = load_types(graph_path)
    ann = _annotations(bodies, ann_path)
    resolved = groups(type("T", (), {"types": types})())
    out = {
        "what": "Which neurons the backrooms page may name, on whose authority, "
                "and how many of them this dataset holds. Counts are measured "
                "from build/graph.npz and data/body-annotations.feather; "
                "identities and role words are chosen and cited per entry.",
        "n_neurons": int(len(types)),
        "confidence_meaning": {
            MEASURED: "the dataset's own labels carry the published name and "
                      "the cited paper describes that population in a male",
            UNCERTAIN: "the name sits on two distinct types, the count "
                       "disagrees with the paper, the annotators' labels "
                       "disagree with each other, or the role was shown only "
                       "in females; shown on the page with this word",
        },
        "types_syntax": "an entry starting with '^' is a regular expression "
                        "matched against the whole type string; any other "
                        "entry is an exact type name",
        "motor_readout": MOTOR_READOUT,
        "groups": {},
    }
    for k in sorted(DICTIONARY):
        e = dict(DICTIONARY[k])
        e["counts"] = counts(types, ann, resolved[k])
        e["resolved"] = bool(len(resolved[k]))
        out["groups"][k] = e
    return out


def dumps(d):
    """Deterministic text: sorted keys, fixed indent, ASCII only, newline at end."""
    return json.dumps(d, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def write_json(path=JSON_PATH, **kw):
    text = dumps(build(**kw))
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return text


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--write" in argv:
        text = write_json()
        d = json.loads(text)
        for k, e in d["groups"].items():
            c = e["counts"]
            flag = "" if e["present"] else "  ABSENT"
            print(f"{k:16s} n={c['n']:4d} L={c['L']:3d} R={c['R']:3d} "
                  f"{e['confidence']:9s}{flag}")
        print(f"wrote {JSON_PATH}")
        return 0
    if "--check" in argv:
        fresh = dumps(build())
        disk = Path(JSON_PATH).read_text(encoding="utf-8")
        print("current" if fresh == disk else "STALE: run --write")
        return 0 if fresh == disk else 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())

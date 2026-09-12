"""
The backrooms dictionary, checked against the real type strings without
loading the brain: build/graph.npz is opened for its `types` and `bodies`
arrays only (np.load reads keys lazily, so the weights stay on disk).

What matters and why:
  * every group resolves to cells, or says it is absent - a name that
    resolves to nothing must never reach a caption;
  * no captioned group shares a cell with the roamer's motor readout, so a
    "circuit fires" line can never be a restatement of the walking command;
  * groups overlap only where the dictionary declares containment (P1 in
    pC1, ps1 in the pulse group, the song groups in all wing motor neurons);
  * the JSON is a pure function of the data: two builds are byte-identical
    and the file on disk is current;
  * the counts the build's FACTS rely on are the counts here.

  py -m pytest -q test_backrooms_dictionary.py
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np

import backrooms_dictionary as bd

ROOT = Path(__file__).parent
HAVE_DATA = bd.GRAPH.exists() and bd.ANNOTATIONS.exists()


class FakeBrain:
    """Anything with a per-neuron `types` str array is enough for groups()."""

    def __init__(self, types):
        self.types = np.asarray(types, dtype=str)


class DictionaryShape(unittest.TestCase):
    def test_every_entry_has_the_fields_the_page_needs(self):
        for k, e in bd.DICTIONARY.items():
            for f in ("name", "types", "role", "citation", "confidence",
                      "present", "motor", "contains", "note", "channel"):
                self.assertIn(f, e, f"{k} lacks {f}")
            self.assertIn(e["confidence"], (bd.MEASURED, bd.UNCERTAIN), k)
            self.assertTrue(e["citation"].strip(), f"{k} has no citation")
            self.assertTrue(e["role"].strip(), f"{k} has no role")
            self.assertTrue(e["channel"].strip(), f"{k} has no channel")

    def test_absent_groups_have_no_types_and_present_groups_do(self):
        for k, e in bd.DICTIONARY.items():
            if e["present"]:
                self.assertTrue(e["types"], f"{k} present but lists no types")
            else:
                self.assertEqual(e["types"], [], f"{k} absent but lists types")

    def test_uncertain_groups_say_why(self):
        for k, e in bd.DICTIONARY.items():
            if e["confidence"] == bd.UNCERTAIN and e["present"]:
                self.assertTrue(e["note"].strip(), f"{k} is uncertain with no note")

    def test_contains_refers_to_real_keys(self):
        for k, e in bd.DICTIONARY.items():
            for c in e["contains"]:
                self.assertIn(c, bd.DICTIONARY, f"{k} contains unknown {c}")

    def test_motor_keys_are_exactly_the_roamers_four(self):
        self.assertEqual(set(bd.MOTOR_KEYS), set(bd.MOTOR_READOUT))
        self.assertEqual(set(bd.MOTOR_READOUT), {"DNa02", "DNa01", "MDN", "DNp09"})

    def test_module_calls_no_model_and_no_network(self):
        src = (ROOT / "backrooms_dictionary.py").read_text(encoding="utf-8")
        for bad in ("openai", "anthropic", "transformers", "llama_cpp",
                    "requests", "httpx", "urllib", "socket"):
            self.assertIsNone(re.search(rf"^\s*(import|from)\s+{bad}\b", src, re.M),
                              f"backrooms_dictionary imports {bad}")


class Confidence(unittest.TestCase):
    """
    The module's own rule for 'uncertain' (two types, count mismatch, the
    annotators' labels disagreeing, a role shown only in females), applied
    to the entries the review found marked 'measured' against it.
    """

    def test_groups_whose_labels_counts_or_sex_disagree_are_uncertain(self):
        for k in ("JO_B", "Tk_FruM", "DNp13", "pCd", "P1", "aIPg", "dPR1", "vMS11", "vPR9", "pC2l"):
            e = bd.DICTIONARY[k]
            self.assertEqual(e["confidence"], bd.UNCERTAIN, k)
            self.assertTrue(e["note"].strip(), k)

    def test_the_notes_name_the_reason(self):
        self.assertIn("disagree", bd.DICTIONARY["JO_B"]["note"])
        self.assertIn("35 of 89", bd.DICTIONARY["JO_B"]["note"])
        self.assertIn("count disagrees", bd.DICTIONARY["Tk_FruM"]["note"])
        self.assertIn("female", bd.DICTIONARY["DNp13"]["note"])
        self.assertIn("Jung 2020", bd.DICTIONARY["pCd"]["note"])

    def test_the_male_pcd_role_is_cited_to_the_male_paper(self):
        e = bd.DICTIONARY["pCd"]
        self.assertTrue(e["citation"].startswith("Jung et al. 2020 Neuron 105:322"))
        self.assertIn("Zhou et al. 2014 Neuron 83:149", e["citation"])
        self.assertIn("female receptivity", e["citation"])       # what Zhou 2014 is about
        self.assertIn("in males", e["role"])

    def test_dnp13_role_names_both_documented_roles(self):
        e = bd.DICTIONARY["DNp13"]
        self.assertIn("Mezzera et al. 2020 Curr Biol 30:3736", e["citation"])
        self.assertIn("Kimura et al. 2015 PLoS ONE 10:e0126445", e["citation"])
        self.assertIn("ovipositor", e["role"])
        self.assertIn("sexually dimorphic", e["citation"])

    def test_dnp09_role_says_what_the_paper_found(self):
        e = bd.DICTIONARY["DNp09"]
        self.assertTrue(e["role"].startswith("forward-walking"))
        self.assertIn("read as stop by the roamer", e["role"])
        self.assertIn("forward walking", bd.MOTOR_READOUT["DNp09"])

    def test_citations_checked_this_session(self):
        self.assertIn("Nat Commun 9:4390", bd.DICTIONARY["DNa01"]["citation"])
        self.assertNotIn("Cell Rep 23:1231", bd.DICTIONARY["DNa01"]["citation"])
        self.assertIn("eLife 13:RP97769", bd.DICTIONARY["wing_mn_all"]["citation"])
        self.assertNotIn("eLife 12:RP97769", bd.DICTIONARY["wing_mn_all"]["citation"])
        self.assertIn("2021 Nature 589:577", bd.DICTIONARY["vpoDN"]["citation"])


class Matching(unittest.TestCase):
    def test_regex_entries_match_the_whole_string(self):
        names = ["pC1_1a", "pC1x_a", "LPC1", "LLPC1", "pC1", "xpC1_1a"]
        m = bd.match_mask(names, ["^pC1_.*", "^pC1x_.*"])
        self.assertEqual(list(m), [True, True, False, False, False, False])

    def test_exact_entries_do_not_match_lookalikes(self):
        names = ["DNa01", "DNae001", "DNp09", "DNp71", "MDN", "MDN2"]
        self.assertEqual(list(bd.match_mask(names, ["DNa01"])),
                         [True, False, False, False, False, False])
        self.assertEqual(list(bd.match_mask(names, ["MDN"])),
                         [False, False, False, False, True, False])

    def test_groups_needs_only_a_types_array(self):
        fb = FakeBrain(["ORN_DA1", "JO-A1", "JO-B2", "ps1 MN", "DNa02",
                        "pC1_1a", "LPC1", "ORN_DA1"])
        g = bd.groups(fb)
        self.assertEqual(list(g["ORN_DA1"]), [0, 7])
        self.assertEqual(list(g["JO_A"]), [1])
        self.assertEqual(list(g["JO_B"]), [2])
        self.assertEqual(list(g["song_ps1"]), [3])
        self.assertEqual(list(g["song_pulse_mn"]), [3])
        self.assertEqual(list(g["DNa02"]), [4])
        self.assertEqual(list(g["pC1"]), [5])
        self.assertEqual(list(g["P1"]), [5])
        self.assertEqual(list(g["aDN"]), [])
        self.assertEqual(g["ORN_DA1"].dtype, np.int64)

    def test_resolve_is_the_same_as_groups(self):
        fb = FakeBrain(["a", "ps1 MN", "b", "ps1 MN"])
        self.assertEqual(list(bd.resolve(fb.types, ["ps1 MN"])), [1, 3])


@unittest.skipUnless(HAVE_DATA, "graph.npz / annotations not present")
class RealTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types, cls.bodies = bd.load_types()
        cls.groups = bd.groups(FakeBrain(cls.types))

    def test_the_dataset_is_the_male_cns(self):
        self.assertEqual(len(self.types), 165122)

    def test_every_group_resolves_or_is_marked_absent(self):
        for k, e in bd.DICTIONARY.items():
            n = len(self.groups[k])
            if e["present"]:
                self.assertGreater(n, 0, f"{k} is marked present but resolves to nothing")
            else:
                self.assertEqual(n, 0, f"{k} is marked absent but resolves to {n} cells")

    def test_no_circuit_group_overlaps_the_motor_readout(self):
        motor = np.concatenate([self.groups[k] for k in bd.MOTOR_KEYS])
        self.assertEqual(len(motor), 2 + 2 + 4 + 2)
        for k, e in bd.DICTIONARY.items():
            if e["motor"]:
                continue
            clash = np.intersect1d(self.groups[k], motor)
            self.assertEqual(len(clash), 0, f"{k} shares cells with the motor readout")

    def test_groups_overlap_only_where_declared(self):
        keys = [k for k in bd.DICTIONARY if bd.DICTIONARY[k]["present"]]
        allowed = set()
        for k in keys:
            for c in bd.DICTIONARY[k]["contains"]:
                allowed.add((k, c))
                allowed.add((c, k))
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                clash = np.intersect1d(self.groups[a], self.groups[b])
                if (a, b) in allowed:
                    self.assertGreater(len(clash), 0, f"{a} declares {b} but they share nothing")
                else:
                    self.assertEqual(len(clash), 0, f"{a} and {b} share {len(clash)} cells undeclared")

    def test_declared_containment_is_containment(self):
        for k, e in bd.DICTIONARY.items():
            for c in e["contains"]:
                inner, outer = self.groups[c], self.groups[k]
                self.assertTrue(np.isin(inner, outer).all(), f"{c} is not inside {k}")

    def test_counts_the_facts_rely_on(self):
        want = {"JO_A": 50, "JO_B": 89, "ORN_DA1": 204, "DNa02": 2, "DNa01": 2,
                "MDN": 4, "DNp09": 2, "pIP10": 2, "LC10a": 275, "song_ps1": 2,
                "song_sine_hg1": 2, "song_pulse_mn": 8, "P1": 86, "pC1": 156,
                "Tk_FruM": 5, "vPR6": 8, "vMS11": 14, "DNp13": 2}
        for k, n in want.items():
            self.assertEqual(len(self.groups[k]), n, k)

    def test_lookalike_descending_neurons_are_excluded(self):
        t = self.types
        self.assertIn("DNae001", set(t))
        self.assertIn("DNp71", set(t))
        self.assertFalse(np.isin(np.flatnonzero(t == "DNae001"), self.groups["DNa01"]).any())
        self.assertFalse(np.isin(np.flatnonzero(t == "DNp71"), self.groups["DNp09"]).any())

    def test_optic_lobe_pc1_lookalikes_are_excluded(self):
        t = self.types
        for bad in ("LPC1", "LLPC1"):
            self.assertGreater((t == bad).sum(), 0)
            self.assertFalse(np.isin(np.flatnonzero(t == bad), self.groups["pC1"]).any())


@unittest.skipUnless(HAVE_DATA, "graph.npz / annotations not present")
class JsonFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text1 = bd.dumps(bd.build())
        cls.text2 = bd.dumps(bd.build())

    def test_build_is_deterministic(self):
        self.assertEqual(self.text1, self.text2)

    def test_file_on_disk_is_current(self):
        self.assertTrue(bd.JSON_PATH.exists(), "run: py backrooms_dictionary.py --write")
        self.assertEqual(bd.JSON_PATH.read_text(encoding="utf-8"), self.text1)

    def test_json_counts_match_resolution(self):
        d = json.loads(self.text1)
        g = bd.groups(FakeBrain(bd.load_types()[0]))
        self.assertEqual(d["n_neurons"], 165122)
        for k, e in d["groups"].items():
            self.assertEqual(e["counts"]["n"], len(g[k]), k)
            self.assertEqual(e["resolved"], bool(len(g[k])), k)
            self.assertEqual(e["counts"]["L"] + e["counts"]["R"] + e["counts"]["side_other"],
                             e["counts"]["n"], k)

    def test_json_is_ascii_and_has_no_timestamp(self):
        self.assertTrue(self.text1.isascii())
        self.assertNotRegex(self.text1, r"20\d\d-\d\d-\d\dT")

    def test_measured_sensory_groups_have_agreeing_annotator_labels(self):
        # the rule: a group is 'measured' only if the annotators' own labels
        # agree. For the antennal sensory groups the subclass column is a
        # functional label (auditory / wind_gravity), so the majority label
        # must cover at least 90 % of the labelled cells (JO-B: 54 of 88
        # 'auditory' -> uncertain; JO-A: 49 of 50 -> measured). VNC groups
        # carry MANC morphology codes there (BI, II, IR), which say nothing
        # about identity and are not held to this.
        d = json.loads(self.text1)["groups"]
        for k in ("JO_A", "JO_B", "ORN_DA1"):
            e = d[k]
            labels = {s: n for s, n in e["counts"]["subclass"].items() if s}
            if not labels:
                continue
            agree = max(labels.values()) / sum(labels.values())
            if e["confidence"] == bd.MEASURED:
                self.assertGreaterEqual(agree, 0.9, f"{k} is measured but its labels disagree: {labels}")
            else:
                self.assertLess(agree, 0.9, f"{k} is uncertain but its labels agree: {labels}")
        self.assertEqual(d["JO_B"]["counts"]["subclass"], {"": 1, "auditory": 54, "wind_gravity": 34})
        self.assertEqual(d["JO_A"]["counts"]["subclass"], {"auditory": 49, "wind_gravity": 1})

    def test_antennal_groups_count_sides_by_root(self):
        d = json.loads(self.text1)["groups"]
        for k in ("ORN_DA1", "JO_A", "JO_B"):
            self.assertEqual(d[k]["counts"]["side_source"], "rootSide", k)
        for k in ("pC1", "DNa02", "song_ps1"):
            self.assertEqual(d[k]["counts"]["side_source"], "somaSide", k)


if __name__ == "__main__":
    unittest.main()

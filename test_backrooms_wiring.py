"""
The three backrooms parts joined: backrooms_world.World (on a fake room, so
no brain is loaded) stepped by backrooms.Loop with section 1's real
Captioner, the way backrooms.build_objects joins them on the real brain.

What is held to: the loop accepts World's record as it is (no key renamed
between the two files), the captioner reads that record without a KeyError
and stays silent on the fake brain's flat traces, the frame and the state
file carry both flies' numbers and every present dictionary group, and
build_objects is the only place the parts meet.

  py -m pytest -q test_backrooms_wiring.py
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import backrooms
import backrooms_dictionary as bd
import backrooms_world as bw
from test_backrooms_world import make_parts

ROOT = Path(__file__).parent


def fake_world(seed=5, spont=None):
    fb, eye, groups, motor = make_parts(spont)
    return bw.World(seed=seed, room=bw.Room(fb, eye, groups, motor, seed=seed))


class Wiring(unittest.TestCase):
    def test_build_objects_names_world_and_sizes_from_its_groups(self):
        src = inspect.getsource(backrooms.build_objects)
        self.assertIn("backrooms_world.World(seed=seed, ram_check=ram_check)", src)
        self.assertIn("sizes_from_groups", src)
        self.assertTrue(hasattr(bw, "World"))

    def test_loop_ticks_a_world_through_the_real_captioner(self):
        world = fake_world(spont={26: 120.0, 27: 60.0, 28: 200.0, 29: 200.0})   # a steady turn and walk
        sizes = backrooms.sizes_from_groups(world.groups)
        self.assertEqual(sizes, {k: len(v) for k, v in world.groups.items()})
        captioner = backrooms.Captioner(sizes, dt=backrooms.WORLD_DT_S)
        with tempfile.TemporaryDirectory() as tmp:
            loop = backrooms.Loop(world, captioner, out=Path(tmp) / "build",
                                  dictionary=bd.JSON_PATH)
            frames = [loop.tick() for _ in range(8)]
            self.assertEqual([f["step"] for f in frames], list(range(1, 9)))
            self.assertAlmostEqual(frames[-1]["t"], 8 * backrooms.WORLD_DT_S)
            for f in frames:
                json.dumps(f)
                self.assertEqual(sorted(f["flies"]), ["A", "B"])
                for fly in f["flies"].values():
                    for k in backrooms.FLY_FRAME_KEYS:
                        self.assertIn(k, fly, k)
                    self.assertEqual(sorted(fly["rates"]), sorted(world.groups))
                    self.assertEqual(sorted(fly["drive"]), ["JO_A", "JO_B", "ORN_DA1"])
                    self.assertIsInstance(fly["x"], float)
            state = json.loads((Path(tmp) / "build" / "backrooms_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["step"], 8)
            self.assertEqual(state["world"]["world_class"], "backrooms_world.World")
            self.assertEqual(sorted(state["flies"]), ["A", "B"])
            self.assertIn("record", state["world"])
            steps = (Path(tmp) / "build" / "backrooms_steps.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(steps), 8)
            row = json.loads(steps[-1])
            self.assertEqual(row["flies"]["A"]["window"], 8)
            self.assertTrue(row["flies"]["A"]["state_carried"])
            # the fake brain echoes a constant drive: the captioner may write
            # geometry lines from the walk but never a "fires" line, since no
            # rate ever rises above twice its own baseline
            for ln in loop.lines:
                self.assertNotIn(ln["kind"], ("onset", "offset"), ln["text"])
                self.assertIn(ln["fly"], ("A", "B"))

    def test_world_keys_cover_what_the_captioner_reads(self):
        """Captioner.update reads x, y, heading, rates, drive, song per fly: all present, none renamed."""
        world = fake_world()
        r = world.step()
        for fly in r["flies"].values():
            for k in ("x", "y", "heading", "rates", "drive", "song"):
                self.assertIn(k, fly)
            for k in backrooms.SENSORY_TEMPLATES:
                self.assertIn(k, fly["drive"], k)


if __name__ == "__main__":
    unittest.main()

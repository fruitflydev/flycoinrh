"""
backrooms.py's server and loop, driven with fakes: no brain is loaded here.

What is held to: the loop writes the state file whole-or-not-at-all, appends
the transcript one complete line at a time and continues its numbering after
a restart, logs every step, hands each frame to every websocket, and the
routes serve what the page reads. The world and the captioner are stand-ins
that obey the interface written in section 2 of backrooms.py.

  py -m pytest -q test_backrooms_server.py
"""
import asyncio
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import backrooms

HERE = Path(__file__).parent

FAKE_DICT = {
    "n_neurons": 165122,
    "groups": {
        "P1": {"name": "P1", "role": "courtship command neurons",
               "citation": "von Philipsborn 2011", "confidence": "uncertain",
               "channel": "courtship", "present": True, "motor": False,
               "contains": [], "counts": {"n": 86, "L": 43, "R": 43}},
        "JO_A": {"name": "JO-A", "role": "sound-sensitive JO neurons",
                 "citation": "Kamikouchi 2009", "confidence": "measured",
                 "channel": "sound", "present": True, "motor": False,
                 "contains": [], "counts": {"n": 50, "L": 26, "R": 24}},
        "DNa02": {"name": "DNa02", "role": "steering", "citation": "roamer",
                  "confidence": "measured", "channel": "motor",
                  "present": True, "motor": True, "contains": [],
                  "counts": {"n": 2, "L": 1, "R": 1}},
        "P1_by_name": {"name": "P1_by_name", "role": "no type is named P1",
                       "citation": "", "confidence": "uncertain",
                       "channel": "courtship", "present": False,
                       "motor": False, "contains": [],
                       "counts": {"n": 0, "L": 0, "R": 0}},
    },
}


class FakeWorld:
    """Two flies on circles; rates are a deterministic function of the step."""

    def __init__(self):
        self.n = 0

    def meta(self):
        return {"fake": True, "arena_mm": 20.0}

    def step(self):
        self.n += 1
        k = self.n
        flies = {}
        for i, name in enumerate(("A", "B")):
            ang = 0.1 * k + i * np.pi
            x = 10 + 6 * np.cos(ang)
            y = 10 + 6 * np.sin(ang)
            flies[name] = {
                "x": np.float64(x), "y": np.float64(y),
                "heading": np.float32(np.degrees(ang) % 360),
                "speed": 3.0, "turn": 20.0,
                "distance": np.float64(12.0), "bearing": np.float64(0.0),
                "rates": {"P1": np.float64(10 + 5 * i + k % 3),
                          "JO_A": np.float64(2.0), "DNa02": np.float64(1.0)},
                "counts": {"P1": np.int64(3), "JO_A": np.int64(1),
                           "DNa02": np.int64(0)},
                "motor": {"steer_L": 1.0, "steer_R": 2.0, "fwd_L": 3.0,
                          "fwd_R": 4.0, "back": 0.0, "stop": 0.0},
                "drive": {"ORN_DA1": 80.0, "JO_A": 10.0, "JO_B": 10.0},
                "song": np.float64(0.0),
                "spikes": np.array([1, 2, 3]),
                "nan_here": float("nan"),
            }
        return {"step": k, "t": k * 0.05, "flies": flies}


class FakeCaptioner:
    """
    Section 1's shape: update(t, flies) -> lines, a `seq` the loop may set.
    One line every third step (t a multiple of 0.15 s), none otherwise.
    """

    def __init__(self):
        self.seq = 0
        self.calls = []

    def update(self, t, flies):
        self.calls.append((t, sorted(flies)))
        step = int(round(t / 0.05))
        if step % 3:
            return []
        a = flies["A"]["rates"]["P1"]
        # a real template filled from the real dictionary, so the reload filter keeps it
        e = backrooms.bd.DICTIONARY["P1"]
        text = backrooms.TEMPLATES["onset"].format(
            fly="A", name=e["name"], value=float(a), role=e["role"],
            cite=backrooms.cite_key(e["citation"]), unc=backrooms.UNCERTAIN_SUFFIX)
        line = {"seq": self.seq, "t": t, "fly": "A", "kind": "onset",
                "group": "P1", "value": a, "text": text}
        self.seq += 1
        return [line]


def lines_text(a):
    """The text FakeCaptioner writes for a P1 rate `a`: the real onset template on the real P1 entry."""
    e = backrooms.bd.DICTIONARY["P1"]
    return backrooms.TEMPLATES["onset"].format(
        fly="A", name=e["name"], value=float(a), role=e["role"],
        cite=backrooms.cite_key(e["citation"]), unc=backrooms.UNCERTAIN_SUFFIX)


def fresh(tmp):
    tmp = Path(tmp)
    dpath = tmp / "dict.json"
    dpath.write_text(json.dumps(FAKE_DICT), encoding="utf-8")
    loop = backrooms.Loop(FakeWorld(), FakeCaptioner(), out=tmp / "build",
                          dictionary=dpath)
    return loop, dpath


class Plain(unittest.TestCase):
    def test_numpy_and_nan_become_json(self):
        p = backrooms.plain({"a": np.float32(1.5), "b": np.array([1, 2]),
                             "c": float("nan"), "d": (np.int64(3),),
                             "e": float("inf")})
        self.assertEqual(p, {"a": 1.5, "b": [1, 2], "c": None, "d": [3],
                             "e": None})
        json.loads(backrooms.dumps(p))   # parses without a default hook

    def test_dumps_never_emits_bare_nan(self):
        self.assertNotIn("NaN", backrooms.dumps({"x": float("nan")}))


class AtomicWrite(unittest.TestCase):
    def test_leaves_the_whole_file_and_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            backrooms.atomic_write(p, b'{"a":1}')
            self.assertEqual(p.read_bytes(), b'{"a":1}')
            self.assertEqual([q.name for q in Path(tmp).iterdir()], ["s.json"])

    def test_a_failed_rename_leaves_the_old_file_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            backrooms.atomic_write(p, b"old")
            real = backrooms.os.replace

            def boom(a, b):
                raise OSError("disk went away")

            backrooms.os.replace = boom
            try:
                with self.assertRaises(OSError):
                    backrooms.atomic_write(p, b"new")
            finally:
                backrooms.os.replace = real
            self.assertEqual(p.read_bytes(), b"old")

    def test_a_rename_blocked_by_a_reader_is_retried_then_lands(self):
        """Windows: PermissionError while a reader holds the target; two refusals, then the rename goes through."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            backrooms.atomic_write(p, b"old")
            real = backrooms.os.replace
            calls = []

            def held_twice(a, b):
                calls.append(1)
                if len(calls) <= 2:
                    raise PermissionError(13, "Access is denied")
                real(a, b)

            backrooms.os.replace = held_twice
            try:
                backrooms.atomic_write(p, b"new")
            finally:
                backrooms.os.replace = real
            self.assertEqual(len(calls), 3)
            self.assertEqual(p.read_bytes(), b"new")
            self.assertEqual([q.name for q in Path(tmp).iterdir()], ["s.json"])

    def test_a_target_held_for_good_raises_after_the_tries_with_the_old_file_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            backrooms.atomic_write(p, b"old")
            real = backrooms.os.replace
            calls = []

            def held(a, b):
                calls.append(1)
                raise PermissionError(13, "Access is denied")

            backrooms.os.replace = held
            try:
                with self.assertRaises(PermissionError):
                    backrooms.atomic_write(p, b"new")
            finally:
                backrooms.os.replace = real
            self.assertEqual(len(calls), backrooms.RENAME_TRIES)
            self.assertEqual(p.read_bytes(), b"old")


class LoopFiles(unittest.TestCase):
    def test_tick_returns_a_frame_for_both_flies_without_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            f = loop.tick()
            self.assertEqual(f["type"], "frame")
            self.assertEqual(sorted(f["flies"]), ["A", "B"])
            for fly in f["flies"].values():
                self.assertEqual(sorted(fly), sorted(backrooms.FLY_FRAME_KEYS))
                self.assertNotIn("spikes", fly)
                self.assertEqual(fly["drive"]["ORN_DA1"], 80.0)
            json.dumps(f)   # plain python only
            self.assertIsInstance(f["flies"]["A"]["x"], float)
            self.assertGreater(f["slowdown"], 0)

    def test_the_captioner_is_fed_t_and_both_flies_once_per_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            loop.tick()
            loop.tick()
            self.assertEqual(loop.captioner.calls,
                             [(0.05, ["A", "B"]), (0.1, ["A", "B"])])

    def test_state_file_is_whole_json_with_the_banner_and_present_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            for _ in range(3):
                loop.tick()
            s = json.loads(loop.state_path.read_text(encoding="utf-8"))
            self.assertEqual(s["step"], 3)
            self.assertEqual(s["how"], list(backrooms.BANNER_PARAGRAPHS))
            self.assertEqual(sorted(s["groups"]), ["DNa02", "JO_A", "P1"])
            self.assertNotIn("P1_by_name", s["groups"])
            self.assertEqual(s["groups"]["P1"]["confidence"], "uncertain")
            self.assertEqual(s["groups"]["P1"]["n"], 86)
            self.assertEqual(s["world"], {"fake": True, "arena_mm": 20.0})
            self.assertEqual(s["captioner"]["confirm_steps"], backrooms.CONFIRM_STEPS)
            self.assertEqual(s["captioner"]["templates"], backrooms.TEMPLATES)
            self.assertEqual(s["n_neurons"], 165122)
            self.assertEqual(s["brain_ms_per_world_step"], 12.0)
            self.assertIsNone(s["flies"]["A"]["nan_here"])
            self.assertFalse(loop.state_path.with_name("backrooms_state.json.tmp").exists())
            self.assertEqual(json.loads(loop.state_text()), s)

    def test_transcript_is_appended_one_complete_line_per_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            for _ in range(9):
                loop.tick()
            rows = [json.loads(l) for l in
                    loop.transcript_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
            self.assertEqual([r["step"] for r in rows], [3, 6, 9])
            self.assertEqual(rows[0]["fly"], "A")
            self.assertEqual(rows[0]["group"], "P1")
            self.assertTrue(rows[0]["text"].startswith("A  P1 fires"))
            self.assertAlmostEqual(rows[1]["t"], 0.3)
            self.assertEqual(loop.lines_after(0), rows)
            self.assertEqual(loop.lines_after(2), rows[2:])
            self.assertEqual(loop.lines_after(3), [])
            # the captioner's own numbering was set to agree with the store's
            self.assertEqual(loop.captioner.seq, 4)

    def test_every_step_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            for _ in range(4):
                loop.tick()
            rows = [json.loads(l) for l in
                    loop.steps_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["step"] for r in rows], [1, 2, 3, 4])
            self.assertEqual(rows[0]["flies"]["A"]["spikes"], [1, 2, 3])
            self.assertEqual(rows[0]["flies"]["A"]["counts"]["P1"], 3)

    def test_the_step_log_rotates_at_its_cap(self):
        cap = backrooms.STEPS_CAP_BYTES
        backrooms.STEPS_CAP_BYTES = 2000
        try:
            with tempfile.TemporaryDirectory() as tmp:
                loop, _ = fresh(tmp)
                for _ in range(8):
                    loop.tick()
                older = loop.steps_path.with_name("backrooms_steps.jsonl.1")
                self.assertTrue(older.exists())
                self.assertTrue(loop.steps_path.exists())
                self.assertLess(loop.steps_path.stat().st_size, 4000)
        finally:
            backrooms.STEPS_CAP_BYTES = cap

    def test_numbering_continues_after_a_restart_and_a_torn_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, dpath = fresh(tmp)
            for _ in range(6):
                loop.tick()
            with open(loop.transcript_path, "ab") as f:
                f.write(b'{"seq": 99, "text": "torn')   # no newline, no close
            cap = FakeCaptioner()
            again = backrooms.Loop(FakeWorld(), cap, out=Path(tmp) / "build",
                                   dictionary=dpath)
            self.assertEqual(again.seq, 2)
            self.assertEqual(cap.seq, 3)
            self.assertEqual(len(again.lines), 2)
            for _ in range(3):
                again.tick()
            self.assertEqual(again.seq, 3)
            self.assertEqual(again.lines_after(2)[0]["seq"], 3)

    def test_rows_not_written_by_the_code_are_dropped_on_reload(self):
        """
        The one path by which a person could put words on the page: a row
        appended to build/backrooms_transcript.jsonl by hand, served back
        after a restart. Such rows are dropped, counted in the state, and
        their sequence numbers are never reused.
        """
        with tempfile.TemporaryDirectory() as tmp:
            loop, dpath = fresh(tmp)
            for _ in range(6):
                loop.tick()                                      # seq 1, 2
            with open(loop.transcript_path, "ab") as f:
                f.write(json.dumps({"seq": 9999, "step": 6, "t": 0.3, "at": 0.0, "fly": "A",
                                    "kind": "onset", "group": "", "value": 0.0,
                                    "text": "A  feels lonely"}).encode() + b"\n")
                f.write(json.dumps({"seq": 9999, "step": 6, "t": 0.3, "at": 0.0, "fly": "A",
                                    "kind": "onset", "group": "P1", "value": 5.0,
                                    "text": "A  P1 fires 5 Hz"}).encode() + b"\n")     # allowed words, no template
                f.write(json.dumps({"seq": 9999, "step": 6, "t": 0.3, "at": 0.0, "fly": "A",
                                    "kind": "onset", "group": "pCd", "value": 87.0,
                                    "text": "A  pCd fires 87 Hz (dsx+ cluster required for cVA-evoked "
                                            "courtship and aggression, Zhou 2014; uncertain match)"}
                                   ).encode() + b"\n")                                    # an older dictionary's line
                f.write(json.dumps({"seq": 10000, "step": 6, "t": 0.3, "at": 0.0, "fly": "A",
                                    "kind": "poem", "group": "", "value": 0.0,
                                    "text": lines_text(5.0)}).encode() + b"\n")          # a real line, wrong kind
            cap = FakeCaptioner()
            again = backrooms.Loop(FakeWorld(), cap, out=Path(tmp) / "build", dictionary=dpath)
            self.assertEqual(len(again.lines), 2)
            self.assertEqual(again.dropped_on_reload, 4)
            self.assertEqual(again.seq, 10000)                   # past every stored row
            self.assertEqual(cap.seq, 10001)
            s = again.snapshot()
            self.assertEqual(s["lines_dropped_on_reload"], 4)
            self.assertNotIn("lonely", json.dumps(s))
            self.assertNotIn("Zhou", json.dumps(s))
            self.assertNotIn("poem", json.dumps(s["last_lines"]))
            for _ in range(3):
                again.tick()
            self.assertEqual(again.lines_after(10000)[0]["seq"], 10001)
            self.assertNotIn("lonely", json.dumps(again.lines_after(0)))

    def test_group_meta_without_a_dictionary_is_empty(self):
        self.assertEqual(backrooms.load_group_meta(Path("nowhere") / "x.json"), {})

    def test_group_meta_from_the_real_dictionary_names_only_present_groups(self):
        if not backrooms.DICT_PATH.exists():
            self.skipTest("no build/backrooms_dictionary.json")
        meta = backrooms.load_group_meta(backrooms.DICT_PATH)
        self.assertEqual(len(meta), 35)
        for absent in ("P1_by_name", "aDN", "pC2_by_name", "vpoDN"):
            self.assertNotIn(absent, meta)
        self.assertEqual(meta["ORN_DA1"]["n"], 204)
        self.assertEqual(meta["JO_A"]["n"], 50)
        self.assertEqual(meta["JO_B"]["n"], 89)
        self.assertEqual(meta["P1"]["confidence"], "uncertain")
        self.assertTrue(meta["DNa02"]["motor"])


class HeldFiles(unittest.TestCase):
    """Windows: a reader holding a file blocks its rename. Neither the cap nor the live feed may suffer for it."""

    def test_a_rotation_refused_by_a_held_file_is_retried_at_the_next_step(self):
        cap = backrooms.STEPS_CAP_BYTES
        backrooms.STEPS_CAP_BYTES = 2000
        real = backrooms.os.replace
        held = {"on": True, "refused": 0}

        def replace(a, b):
            if held["on"] and str(a).endswith("backrooms_steps.jsonl"):
                held["refused"] += 1
                raise PermissionError(13, "Access is denied")
            real(a, b)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                loop, _ = fresh(tmp)
                backrooms.os.replace = replace
                for _ in range(8):
                    loop.tick()
                older = loop.steps_path.with_name("backrooms_steps.jsonl.1")
                self.assertFalse(older.exists())
                self.assertGreaterEqual(held["refused"], 2)           # retried every step, not 64 MB later
                self.assertEqual(loop.rotation_failures, held["refused"])
                self.assertGreater(loop._steps_bytes, 2000)            # the count was not reset
                self.assertEqual(loop.snapshot()["rotation_failures"], held["refused"])
                held["on"] = False
                loop.tick()                                            # the reader let go: rotated now
                self.assertTrue(older.exists())
                self.assertLess(loop._steps_bytes, 2000)
        finally:
            backrooms.os.replace = real
            backrooms.STEPS_CAP_BYTES = cap

    def test_a_held_state_file_does_not_abort_the_step_or_the_feed(self):
        real = backrooms.atomic_write

        def held(path, data):
            raise PermissionError(13, "Access is denied")

        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            loop.tick()
            offered = []
            loop._offer = offered.append
            backrooms.atomic_write = held
            try:
                for _ in range(3):
                    frame = loop.tick()                                # steps 2, 3, 4; a line at step 3
            finally:
                backrooms.atomic_write = real
            self.assertEqual([f["step"] for f in offered], [2, 3, 4])   # every frame reached the watchers
            self.assertEqual(frame["step"], 4)
            self.assertEqual(loop.state_write_failures, 3)
            self.assertEqual(loop.snapshot()["step"], 4)               # /state is current in memory
            self.assertEqual(loop.snapshot()["state_write_failures"], 3)
            on_disk = json.loads(loop.state_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["step"], 1)                       # the old file is whole
            rows = [json.loads(l) for l in loop.transcript_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["step"] for r in rows], [3])           # the transcript was appended
            steps = loop.steps_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(steps), 4)                            # and every step logged
            loop.tick()                                                # the reader let go
            self.assertEqual(json.loads(loop.state_path.read_text(encoding="utf-8"))["step"], 5)


@unittest.skipUnless(backrooms.DICT_PATH.exists() and backrooms.bd.GRAPH.exists()
                     and backrooms.bd.ANNOTATIONS.exists(), "dictionary or data not present")
class DictionaryCurrency(unittest.TestCase):
    def test_the_server_starts_only_on_a_current_dictionary_file(self):
        self.assertTrue(backrooms.dictionary_is_current())
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "d.json"
            stale.write_text(backrooms.DICT_PATH.read_text(encoding="utf-8").replace("Jung", "Someone"),
                             encoding="utf-8")
            self.assertFalse(backrooms.dictionary_is_current(stale))
            self.assertFalse(backrooms.dictionary_is_current(Path(tmp) / "none.json"))


class Frames(unittest.TestCase):
    def test_a_slow_watcher_keeps_the_latest_frames(self):
        async def go():
            loop = backrooms.Loop.__new__(backrooms.Loop)
            loop.queues = {}
            q = loop.subscribe()
            for i in range(backrooms.FRAME_QUEUE + 5):
                loop._offer({"i": i})
            await asyncio.sleep(0.05)
            got = []
            while not q.empty():
                got.append(q.get_nowait()["i"])
            return got
        got = asyncio.run(go())
        self.assertEqual(len(got), backrooms.FRAME_QUEUE)
        self.assertEqual(got[-1], backrooms.FRAME_QUEUE + 4)


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.tmp = tempfile.TemporaryDirectory()
        cls.loop, cls.dpath = fresh(cls.tmp.name)
        cls.app = backrooms.make_app(cls.loop, dictionary=cls.dpath)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.tmp.cleanup()

    def test_index_is_the_page_with_the_banner(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        for para in backrooms.BANNER_PARAGRAPHS:
            self.assertIn(para, r.text)

    def test_state_and_transcript(self):
        for _ in range(6):
            self.loop.tick()
        s = self.client.get("/state").json()
        self.assertGreaterEqual(s["step"], 6)
        self.assertEqual(s["how"], list(backrooms.BANNER_PARAGRAPHS))
        t = self.client.get("/transcript", params={"after": 0}).json()
        self.assertGreaterEqual(len(t["lines"]), 2)
        self.assertEqual(t["lines"][0]["seq"], 1)
        self.assertEqual(t["seq"], self.loop.seq)
        t2 = self.client.get("/transcript", params={"after": t["seq"]}).json()
        self.assertEqual(t2["lines"], [])
        self.assertEqual(self.client.get("/transcript?after=-1").status_code, 422)

    def test_dictionary_json_is_the_file(self):
        r = self.client.get("/dictionary.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), FAKE_DICT)

    def test_missing_dictionary_is_a_404(self):
        from fastapi.testclient import TestClient
        app = backrooms.make_app(self.loop, dictionary=Path(self.tmp.name) / "none.json")
        with TestClient(app) as c:
            self.assertEqual(c.get("/dictionary.json").status_code, 404)

    def test_websocket_gets_hello_then_every_frame(self):
        with self.client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            self.assertEqual(hello["type"], "hello")
            self.assertEqual(hello["state"]["how"], list(backrooms.BANNER_PARAGRAPHS))
            before = self.loop.step
            f1 = self.loop.tick()
            f2 = self.loop.tick()
            got1 = ws.receive_json()
            got2 = ws.receive_json()
            self.assertEqual(got1["type"], "frame")
            self.assertEqual(got1["step"], before + 1)
            self.assertEqual(got2["step"], before + 2)
            self.assertEqual(got1["flies"]["A"]["x"], f1["flies"]["A"]["x"])
            self.assertEqual(got2["seq"], f2["seq"])
        time.sleep(0.05)
        self.assertEqual(self.loop.queues, {})


class Autostart(unittest.TestCase):
    def test_the_loop_runs_from_startup_and_stops_at_shutdown(self):
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            loop, dpath = fresh(tmp)
            app = backrooms.make_app(loop, dictionary=dpath, autostart=True,
                                     steps=5)
            with TestClient(app) as c:
                for _ in range(50):
                    if loop.step >= 5:
                        break
                    time.sleep(0.05)
                self.assertGreaterEqual(loop.step, 5)
                self.assertEqual(c.get("/state").json()["step"], loop.step)
            self.assertFalse(loop.running)


class Tunnel(unittest.TestCase):
    def test_the_address_is_read_from_cloudflared_output(self):
        line = "2026-09-13 INF |  https://tiny-words-are-here.trycloudflare.com  |"
        self.assertEqual(backrooms.tunnel_url(line),
                         "https://tiny-words-are-here.trycloudflare.com")
        self.assertIsNone(backrooms.tunnel_url("Registered tunnel connection"))
        self.assertIsNone(backrooms.tunnel_url(None))

    def test_set_tunnel_lands_in_the_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = fresh(tmp)
            loop.set_tunnel("https://x-y.trycloudflare.com")
            s = json.loads(loop.state_path.read_text(encoding="utf-8"))
            self.assertEqual(s["tunnel"], "https://x-y.trycloudflare.com")
            loop.tick()
            s = json.loads(loop.state_path.read_text(encoding="utf-8"))
            self.assertEqual(s["tunnel"], "https://x-y.trycloudflare.com")

    def test_fly_tunnel_0_opens_nothing(self):
        old = os.environ.get("FLY_TUNNEL")
        os.environ["FLY_TUNNEL"] = "0"
        try:
            self.assertIsNone(backrooms.start_tunnel(4720, lambda u: None))
        finally:
            if old is None:
                del os.environ["FLY_TUNNEL"]
            else:
                os.environ["FLY_TUNNEL"] = old


class Honesty(unittest.TestCase):
    def test_the_module_imports_no_model_and_no_http_client(self):
        src = (HERE / "backrooms.py").read_text(encoding="utf-8")
        for name in ("openai", "anthropic", "transformers", "llama_cpp",
                     "requests", "httpx", "urllib.request", "aiohttp"):
            self.assertIsNone(re.search(rf"^\s*(import|from)\s+{re.escape(name)}\b",
                                        src, re.M), name)

    def test_free_ram_is_a_number_or_unknown(self):
        v = backrooms.free_ram_gb()
        self.assertTrue(v is None or v > 0)


if __name__ == "__main__":
    unittest.main()

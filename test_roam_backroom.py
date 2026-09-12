"""
roam.py's half of the backroom: the door, the fence and the wiring.

No browser, no connectome, no socket and no page. What these check is the part
of roam that decides where the fly may go, which step runs while it is in the
room, and what a watcher is told - plus three properties that hold only as long
as nobody edits them away: that no dopamine is handed out in this file any
more, that nothing in the room branch can reach a click, and that the coin side
is not imported at all unless the room is switched on.

The source checks read roam.py with ast rather than running it, because running
the loop needs a browser and the properties are about code that must never be
reachable in the first place.

  py -m pytest -q test_roam_backroom.py
"""
import ast
import contextlib
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import calibration
import roam

HERE = Path(__file__).parent
SRC = (HERE / "roam.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC, filename="roam.py")
PARENT = {c: p for p in ast.walk(TREE) for c in ast.iter_child_nodes(p)}
PORT = 4660


@contextlib.contextmanager
def env(**kw):
    """
    Set FLY_* variables for the body of a test; None unsets one.

    roam reads its settings through launch.load_env, which lays every FLY_*
    variable of the process over .env - so this is also the check that the
    settings work from the environment run_all.py hands its children, and not
    only from a file.
    """
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def guards(node):
    """Every `if` between a node and the top of the file, as source text."""
    out = []
    cur = node
    while cur in PARENT:
        up = PARENT[cur]
        if isinstance(up, ast.If):
            if any(cur is s for s in up.body):
                out.append(ast.unparse(up.test))
            elif any(cur is s for s in up.orelse):
                out.append(f"not ({ast.unparse(up.test)})")
        cur = up
    return out


def calls(text):
    """Every call in roam.py whose own source contains this text."""
    return [n for n in ast.walk(TREE)
            if isinstance(n, ast.Call) and text in ast.unparse(n)]


def out_of_the_room(node):
    """True when some `if` above this node says the fly is not in the room."""
    return any("not in_room" in g or g == "not (in_room)" for g in guards(node))


class Port(unittest.TestCase):
    """Every test here talks about one process on one port."""

    def setUp(self):
        self._port = roam.STATE.get("port")
        roam.STATE["port"] = PORT

    def tearDown(self):
        if self._port is None:
            roam.STATE.pop("port", None)
        else:
            roam.STATE["port"] = self._port


class Door(Port):
    """allowed_host: the room is the one address on this machine, and only that."""

    def test_the_room_is_open_when_the_room_is_on(self):
        with env(FLY_BACKROOM="1"):
            self.assertEqual(roam.backroom_url(), f"http://127.0.0.1:{PORT}/backroom")
            self.assertTrue(roam.allowed_host(roam.backroom_url()))
            self.assertTrue(roam.allowed_host(roam.backroom_url() + "/"))

    def test_nothing_else_on_this_machine(self):
        others = [
            f"http://127.0.0.1:{PORT}/",
            f"http://127.0.0.1:{PORT}/state",
            f"http://127.0.0.1:{PORT}/frame.jpg",
            f"http://127.0.0.1:{PORT}/backroom/board.json",
            f"http://127.0.0.1:{PORT}/backroom/state",
            f"http://127.0.0.1:{PORT}/backroom?token=1",
            f"http://127.0.0.1:{PORT}/backroomer",
            f"http://localhost:{PORT}/backroom",
            f"http://0.0.0.0:{PORT}/backroom",
            f"http://[::1]:{PORT}/backroom",
            f"https://127.0.0.1:{PORT}/backroom",
            "http://127.0.0.1:4671/backroom",       # the executor's own port
            "http://127.0.0.1/backroom",
            # every other spelling of this machine, and the networks it sits on.
            # 127.0.0.0/8 is all loopback, so 127.0.0.2 is the executor's own
            # port by another name, and 169.254.169.254 is the metadata service
            # a container would answer from.
            f"http://127.0.0.2:{PORT}/backroom",
            "http://127.0.0.2:4671/health",
            f"http://[::ffff:127.0.0.1]:{PORT}/backroom",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.9/admin",
            "http://[fe80::1]/",
        ]
        with env(FLY_BACKROOM="1"):
            for open_mode in (False, True):
                with mock.patch.object(roam, "OPEN", open_mode):
                    for url in others:
                        with self.subTest(url=url, open=open_mode):
                            self.assertFalse(roam.allowed_host(url))

    def test_what_counts_as_this_machine(self):
        for host in ("127.0.0.1", "127.0.0.2", "localhost", "dev.localhost", "::1",
                     "[::1]", "0.0.0.0", "::ffff:127.0.0.1", "169.254.169.254",
                     "10.0.0.5", "192.168.1.1", "172.16.0.9", "fe80::1", "224.0.0.1"):
            self.assertTrue(roam.this_machine(host), host)
        for host in ("en.wikipedia.org", "8.8.8.8", "", None, "2606:4700::1111"):
            self.assertFalse(roam.this_machine(host), host)

    def test_the_door_is_shut_when_the_room_is_off(self):
        with env(FLY_BACKROOM=None):
            self.assertFalse(roam.allowed_host(roam.backroom_url()))
        with env(FLY_BACKROOM="0"):
            self.assertFalse(roam.allowed_host(roam.backroom_url()))

    def test_open_mode_does_not_open_this_machine(self):
        with env(FLY_BACKROOM=None), mock.patch.object(roam, "OPEN", True):
            self.assertTrue(roam.allowed_host("https://example.com/anything"))
            self.assertFalse(roam.allowed_host(roam.backroom_url()))
            self.assertFalse(roam.allowed_host(f"http://127.0.0.1:{PORT}/state"))

    def test_the_web_fence_is_what_it_was(self):
        with env(FLY_BACKROOM="1"), mock.patch.object(roam, "OPEN", False):
            self.assertTrue(roam.allowed_host("https://en.wikipedia.org/wiki/Fly"))
            self.assertTrue(roam.allowed_host("https://www.ponsfamily.com/launchpad/explore"))
            self.assertFalse(roam.allowed_host("https://example.com/"))
            self.assertFalse(roam.allowed_host("not a url at all"))

    def test_is_backroom_is_not_a_prefix_test(self):
        self.assertTrue(roam.is_backroom(f"http://127.0.0.1:{PORT}/backroom"))
        self.assertFalse(roam.is_backroom(f"http://127.0.0.1:{PORT}/backroom/board.json"))
        self.assertFalse(roam.is_backroom("https://en.wikipedia.org/wiki/Backroom"))
        self.assertFalse(roam.is_backroom(None))


class Restart(Port):
    """next_place: where a life begins again."""

    def test_a_seed_and_nothing_else_when_the_room_is_off(self):
        with env(FLY_BACKROOM=None):
            rng = random.Random(11)
            got = [roam.next_place(rng) for _ in range(50)]
        # the same stream a plain rng.choice would have produced: with the room
        # off nothing else is drawn, so today's roam is bit for bit today's roam
        same = random.Random(11)
        self.assertEqual([u for u, _ in got],
                         [same.choice(roam.SEEDS) for _ in range(50)])
        self.assertEqual({w for _, w in got}, {"a seed"})

    def test_the_share_is_the_share(self):
        n = 4000
        with env(FLY_BACKROOM="1", FLY_BACKROOM_SHARE="0.35"):
            rng = random.Random(4)
            room = sum(1 for _ in range(n) if roam.next_place(rng)[1] == "the backroom")
        self.assertAlmostEqual(room / n, 0.35, delta=0.03)

    def test_never_and_always(self):
        with env(FLY_BACKROOM="1", FLY_BACKROOM_SHARE="0"):
            rng = random.Random(2)
            self.assertEqual({roam.next_place(rng)[1] for _ in range(200)}, {"a seed"})
        with env(FLY_BACKROOM="1", FLY_BACKROOM_SHARE="1"):
            rng = random.Random(2)
            urls = {roam.next_place(rng)[0] for _ in range(200)}
            self.assertEqual(urls, {roam.backroom_url()})

    def test_a_room_that_could_not_be_built_is_not_a_destination(self):
        # FLY_BACKROOM=1 with no mushroom body: the page would be a room with
        # nothing behind it, so the fly is not sent there
        with env(FLY_BACKROOM="1", FLY_BACKROOM_SHARE="1"):
            rng = random.Random(3)
            self.assertEqual({roam.next_place(rng, False)[1] for _ in range(50)}, {"a seed"})

    def test_a_settings_reader_that_cannot_be_broken_by_a_typo(self):
        with env(FLY_BACKROOM_SHARE="banana", FLY_BACKROOM_STEPS="soon"):
            self.assertEqual(roam.backroom_share(), roam.BACKROOM_SHARE)
            self.assertEqual(roam.backroom_steps(), roam.BACKROOM_STEPS)
        with env(FLY_BACKROOM_SHARE="5", FLY_BACKROOM_STEPS="7"):
            self.assertEqual(roam.backroom_share(), 1.0)
            self.assertEqual(roam.backroom_steps(), 7)
        with env(FLY_BACKROOM_SHARE=None, FLY_BACKROOM_STEPS=None):
            self.assertEqual(roam.backroom_share(), 0.35)
            self.assertEqual(roam.backroom_steps(), 120)


class Source(unittest.TestCase):
    """Properties of the file itself, read with ast."""

    def test_no_dopamine_is_handed_out_here(self):
        hit = [ast.unparse(n) for n in ast.walk(TREE)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "dopamine"]
        self.assertEqual(hit, [], "only profit and loss teach, and the room delivers those")

    def test_the_click_cannot_be_reached_from_the_room(self):
        clicks = calls("page.mouse.click")
        self.assertEqual(len(clicks), 1, "there is one place a click happens")
        for node in clicks:
            self.assertTrue(out_of_the_room(node), ast.unparse(node))

    def test_the_room_is_never_asked_what_is_under_the_cursor(self):
        asked = calls("page.evaluate(UNDER_JS")
        self.assertTrue(asked)
        for node in asked:
            self.assertTrue(out_of_the_room(node), ast.unparse(node))

    def test_the_room_runs_the_step_when_the_fly_is_in_it(self):
        stepped = calls("room.step(")
        self.assertTrue(stepped)
        for node in stepped:
            self.assertIn("in_room", guards(node))

    def test_the_mushroom_body_is_not_observed_twice(self):
        seen = [n for n in ast.walk(TREE)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("observe", "forget")
                and ast.unparse(n).startswith("mb.")]
        self.assertTrue(seen)
        for node in seen:
            self.assertTrue(out_of_the_room(node), ast.unparse(node))

    def test_every_step_of_the_pilot_is_calibrated(self):
        steps = [n for n in ast.walk(TREE)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "step" and ast.unparse(n.func) == "pilot.step"]
        self.assertTrue(steps)
        for node in steps:
            self.assertIn("gains", [k.arg for k in node.keywords], ast.unparse(node))

    def test_the_coin_side_is_not_imported_unless_it_is_asked_for(self):
        top = set()
        for node in TREE.body:
            if isinstance(node, ast.Import):
                top.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertEqual(top & {"backroom", "pons", "olfaction", "eth_abi", "requests"},
                         set(), "roam must import without the coin side installed")


class Room(Port):
    """The routes a watcher and the room's own page read."""

    class FakeRoom:
        in_room = True

        def board(self):
            return {"cards": [{"token": "0x" + "1" * 40}], "holdings": [], "updated": 7}

        def state(self):
            return {"in_room": True, "visits": 1}

    def setUp(self):
        super().setUp()
        self._room = roam.STATE.get("room")
        roam.STATE["room"] = None
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        roam.STATE["room"] = self._room
        super().tearDown()

    def book(self, obj):
        p = self.tmp / "backroom" / "public" / "public.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")

    def test_it_says_off_when_there_is_nothing(self):
        with env(FLY_BACKROOM=None, FLY_STATE_DIR=str(self.tmp)):
            s = roam.backroom_view()
        self.assertFalse(s["enabled"])
        self.assertIsNone(s["book"])
        self.assertIsNone(s["room"])
        self.assertEqual(s["mode"], "paper")
        self.assertIsInstance(s["updated"], int)

    def test_it_reads_the_book_the_executor_wrote(self):
        self.book({"mode": "paper", "eth_balance": 0.5, "positions": []})
        roam.STATE["room"] = self.FakeRoom()
        with env(FLY_BACKROOM="1", FLY_STATE_DIR=str(self.tmp)):
            s = roam.backroom_view()
        self.assertTrue(s["enabled"])
        self.assertEqual(s["book"]["eth_balance"], 0.5)
        self.assertEqual(s["room"], {"in_room": True, "visits": 1})

    def test_a_book_it_cannot_read_is_no_book(self):
        self.book("{ this is not json")
        with env(FLY_BACKROOM="1", FLY_STATE_DIR=str(self.tmp)):
            s = roam.backroom_view()
        self.assertIsNone(s["book"])

    def test_the_board_is_empty_until_the_room_exists(self):
        self.assertEqual(roam.backroom_board(),
                         {"cards": [], "holdings": [], "updated": 0})

    def test_the_board_is_the_room_s_own(self):
        roam.STATE["room"] = self.FakeRoom()
        self.assertEqual(roam.backroom_board()["updated"], 7)

    def test_the_page_is_served_from_this_repo(self):
        self.assertTrue((HERE / "web" / "backroom.html").exists())
        self.assertTrue(str(roam.backroom_page().path).endswith("backroom.html"))


class Published(Port):
    """What lands in roam_state.json."""

    def stats(self):
        return {"steps": 1, "clicks": 0, "vetoes": 0, "hops": 1, "blocked": 0,
                "scrolled": 0, "started": 0.0, "visited": [], "events": []}

    def test_the_room_shows_up_only_when_it_is_on(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(roam, "OUT", tmp):
            roam.publish(self.stats(), b"", "http://x/", {"stop": 1.0}, None, None)
            off = json.loads((tmp / "roam_state.json").read_text())
            roam.publish(self.stats(), b"", "http://x/", {"stop": 1.0}, None,
                         {"enabled": True, "mode": "paper"})
            on = json.loads((tmp / "roam_state.json").read_text())
        self.assertNotIn("backroom", off)
        self.assertEqual(on["backroom"]["mode"], "paper")


class Tunnel(unittest.TestCase):
    """FLY_TUNNEL=0: a run nobody outside this machine can watch."""

    def tearDown(self):
        roam.TUNNEL["proc"] = None
        roam.TUNNEL["url"] = None

    def test_zero_launches_nothing(self):
        with env(FLY_TUNNEL="0"), mock.patch.object(roam, "CFD", Path(__file__)), \
                mock.patch("subprocess.Popen") as popen:
            roam.start_tunnel(PORT)
        popen.assert_not_called()
        self.assertIsNone(roam.TUNNEL["proc"])

    def test_and_that_it_is_the_setting_doing_it(self):
        with env(FLY_TUNNEL=None), mock.patch.object(roam, "CFD", Path(__file__)), \
                mock.patch("subprocess.Popen") as popen:
            roam.start_tunnel(PORT)
        self.assertEqual(popen.call_count, 1)


class Wiring(unittest.TestCase):
    """load_brain and load_room, with everything heavy faked."""

    class FakeBrain:
        n = 3
        bodies = [1, 2, 3]

    class FakeMB:
        def __init__(self, fb, calibration=None, **kw):
            self.calibration = calibration

        def stats(self):
            return {"synapses": 1, "reward_side": 1, "punish_side": 1, "depressed": 0}

    class FakeNose:
        def __init__(self, fb, equal_sniff=0.0, **kw):
            self.max_hz = 0.0
            self.equal_sniff = float(equal_sniff)
            self.built_with = dict(kw, equal_sniff=equal_sniff)
            self.cells = 2635

    def setUp(self):
        self.saved = {k: roam.STATE.get(k) for k in
                      ("brain", "pilot", "mb", "gains", "xy", "room")}
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        for k, v in self.saved.items():
            roam.STATE[k] = v

    def test_the_brain_is_built_with_the_measured_calibration(self):
        roam.STATE["brain"] = None
        with mock.patch.object(roam, "FlyBrain", self.FakeBrain), \
                mock.patch.object(roam, "FlyPilot", lambda fb, sim_steps=60: "PILOT"), \
                mock.patch.object(roam, "MushroomBody", self.FakeMB), \
                mock.patch.object(roam, "soma_xy", lambda fb: None), \
                mock.patch.object(roam.calibration, "gains_for",
                                  lambda fb, setting: ("GAINS", setting)):
            fb, pilot = roam.load_brain()
        self.assertEqual(pilot, "PILOT")
        self.assertEqual(roam.STATE["gains"], ("GAINS", calibration.CHOSEN))
        self.assertEqual(roam.STATE["mb"].calibration, calibration.CHOSEN)

    def test_a_missing_side_table_does_not_take_the_roamer_down(self):
        """build/mb_sides.json is derived and gitignored, so an image can lack it."""
        def boom(fb, calibration=None, **kw):
            raise FileNotFoundError("build/mb_sides.json is missing. It is written by mb_sides.py")

        roam.STATE["brain"] = None
        with mock.patch.object(roam, "FlyBrain", self.FakeBrain), \
                mock.patch.object(roam, "FlyPilot", lambda fb, sim_steps=60: "PILOT"), \
                mock.patch.object(roam, "MushroomBody", boom), \
                mock.patch.object(roam, "soma_xy", lambda fb: None), \
                mock.patch.object(roam.calibration, "gains_for", lambda fb, setting: None):
            fb, pilot = roam.load_brain()
        self.assertIsNotNone(fb)                 # the fly still roams
        self.assertIsNone(roam.STATE["mb"])      # with nothing to learn with

    def test_the_room_will_not_open_without_a_mushroom_body(self):
        roam.STATE.update({"brain": "BRAIN", "pilot": "PILOT", "mb": None,
                           "gains": "GAINS", "room": None})
        modules = {"backroom": types.SimpleNamespace(Room=lambda *a, **k: "ROOM"),
                   "olfaction": types.SimpleNamespace(Nose=self.FakeNose)}
        with env(FLY_BACKROOM="1", FLY_STATE_DIR=str(self.tmp)), \
                mock.patch.dict(sys.modules, modules):
            self.assertIsNone(roam.load_room())
        self.assertIsNone(roam.STATE["room"])

    def test_a_room_that_cannot_be_built_does_not_take_the_roamer_down(self):
        """
        load_room is called from inside roam(), after Chromium and the page are
        up and before the fly has opened anything. An image without the DoOR
        tables raises FileNotFoundError there; unguarded it escaped roam(), and
        begin()'s loop then restarted the whole browser every six seconds for
        ever - a blank stream, and a service answering /status while doing
        nothing at all.
        """
        said = []

        def boom():
            raise FileNotFoundError("DoOR data not found in ./door or ./data/door")

        with mock.patch.object(roam, "load_room", boom), \
                mock.patch.object(roam, "say", lambda *p: said.append(" ".join(map(str, p)))):
            self.assertIsNone(roam.open_room())
        self.assertTrue(said and "backroom stays shut" in said[0], said)
        self.assertIn("DoOR", said[0])

    def test_a_room_that_can_be_built_is_returned_untouched(self):
        with mock.patch.object(roam, "load_room", lambda: "ROOM"):
            self.assertEqual(roam.open_room(), "ROOM")

    def test_the_mushroom_body_keeps_its_gains_in_the_state_dir(self):
        """
        mushroom.py resolves FLY_STATE_DIR from the process environment alone,
        and roam resolves it through .env as well. Started by hand with
        FLY_STATE_DIR in .env only, the ledger, the room and the looks went to
        the configured directory and everything the fly had learned went to the
        repo's build/ - on a volume-backed host, the one file the volume is for
        would be the one file outside it.
        """
        made = {}

        class RecordingMB(self.FakeMB):
            def __init__(self, fb, calibration=None, **kw):
                super().__init__(fb, calibration=calibration, **kw)
                made.update(kw)

        roam.STATE["brain"] = None
        with env(FLY_STATE_DIR=str(self.tmp)), \
                mock.patch.object(roam, "FlyBrain", self.FakeBrain), \
                mock.patch.object(roam, "FlyPilot", lambda fb, sim_steps=60: "PILOT"), \
                mock.patch.object(roam, "MushroomBody", RecordingMB), \
                mock.patch.object(roam, "soma_xy", lambda fb: None), \
                mock.patch.object(roam.calibration, "gains_for", lambda fb, setting: None):
            roam.load_brain()
        self.assertEqual(Path(made["store"]).parent, self.tmp)
        self.assertEqual(Path(made["store"]).name, "mb_gains.v2.npz")

    def test_no_room_when_the_room_is_off(self):
        roam.STATE["room"] = None
        with env(FLY_BACKROOM=None):
            self.assertIsNone(roam.load_room())
        self.assertIsNone(roam.STATE["room"])

    def test_the_room_is_handed_the_executor_and_the_token(self):
        made = {}

        class FakeRoom:
            def __init__(self, fb, pilot, mb, nose, gains, state_dir, executor_url,
                         intent_token, **kw):
                made.update(fb=fb, pilot=pilot, mb=mb, nose=nose, gains=gains,
                            state_dir=state_dir, executor_url=executor_url,
                            intent_token=intent_token)

        roam.STATE.update({"brain": "BRAIN", "pilot": "PILOT", "mb": "MB",
                           "gains": "GAINS", "room": None})
        modules = {"backroom": types.SimpleNamespace(Room=FakeRoom),
                   "olfaction": types.SimpleNamespace(Nose=self.FakeNose)}
        with env(FLY_BACKROOM="1", FLY_EXECUTOR_PORT="4999", FLY_INTENT_TOKEN="abc",
                 FLY_STATE_DIR=str(self.tmp)), mock.patch.dict(sys.modules, modules):
            room = roam.load_room()
        self.assertIsNotNone(room)
        self.assertEqual(made["executor_url"], "http://127.0.0.1:4999")
        self.assertEqual(made["intent_token"], "abc")
        self.assertEqual(Path(made["state_dir"]), self.tmp)
        self.assertEqual(made["gains"], "GAINS")
        self.assertEqual(made["nose"].max_hz,
                         calibration.SETTINGS[calibration.CHOSEN]["odour_max_hz"])
        # every coin's smell scaled toward the same total, and the nose is
        # built that way rather than adjusted afterwards: without it geosmin fires
        # 42% of the Kenyon cells against isopentyl acetate's 4.9% and the loud
        # coins swamp the quiet ones (measured 2026-09-12)
        self.assertEqual(made["nose"].built_with["equal_sniff"], roam.EQUAL_SNIFF)
        self.assertEqual(made["nose"].equal_sniff, roam.EQUAL_SNIFF)
        self.assertEqual(roam.EQUAL_SNIFF, 2.0)
        # the room holds no key, no port of its own and no way to reach the
        # executor except the loopback address it was handed here
        self.assertIs(roam.STATE["room"], room)


if __name__ == "__main__":
    unittest.main()

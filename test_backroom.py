"""
backroom.py without the connectome, the page or the executor.

The brain, the pilot, the mushroom body, the nose, the page and the executor
are all faked, so these tests run in milliseconds, open no socket and load no
graph. What they check is the room's rules: how a look is read and what it is
judged against, when a dwell counts, what a stop on a card does, what reaches
the executor, and when a weight is allowed to move.

The decision rule these were rewritten for (measured 2026-09-12 by
backroom_screen.py, evidence in build/backroom_screen.json): a look is read off
the mushroom body's output synapses, a card's leaning is (A-V)/(A+V), and the
drive is that leaning against the mean leaning of the other cards this visit
has looked at. There is no blank control run and no second run of any kind -
one run of the brain a step.

  py -m pytest -q test_backroom.py
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import calibration
from backroom import Room, amount_for, sign_for

TOKEN_A = "0x1111111111111111111111111111111111111111"
TOKEN_B = "0x2222222222222222222222222222222222222222"
IMG = np.full((800, 1280), 0.05, dtype=np.float32)

# card A top left, card B beside it; anything past x=700 is bare ground
RECTS = [
    {"token": TOKEN_A, "held": False, "x": 0.0, "y": 0.0, "w": 280.0, "h": 200.0},
    {"token": TOKEN_B, "held": True, "x": 300.0, "y": 0.0, "w": 280.0, "h": 200.0},
]
IN_A = (100.0, 100.0)
IN_B = (400.0, 100.0)
OUTSIDE = (900.0, 600.0)


class FakeBrain:
    """
    Answers the one run the room makes outside a step: the re-presentation of a
    stored look when a sell settles.

    Eight neurons, of which 1-4 are Kenyon cells with one output synapse each
    (see FakeMB). A room step runs the brain through the pilot and nowhere
    else, so during stepping nothing may arrive here at all.
    """

    def __init__(self):
        self.n = 8
        self.runs = []
        self.fired = np.array([1, 3], dtype=np.int64)

    def run(self, drive, steps, gains=None, record=None, seed=0):
        self.runs.append({"drive": drive, "steps": steps, "seed": seed})
        return {"_fired": self.fired}


class FakeEye:
    def look(self, img, cx, cy):
        return {(0, 1): float(np.mean(img)) * 180.0}


def card_at(cx, cy):
    for r in RECTS:
        if r["x"] <= cx < r["x"] + r["w"] and r["y"] <= cy < r["y"] + r["h"]:
            return r["token"]
    return None


class FakePilot:
    """
    The one run of a step: the Kenyon cells a look at a card fires.

    A look at card A fires cells 1 and 3 and a look at card B fires 2 and 4, so
    the two cards' readings can be set independently by weight (see FakeMB).
    Off every card nothing fires; no reading is taken there.
    """

    def __init__(self):
        self.eye = FakeEye()
        self.sim_steps = 60
        self.click = False
        self.calls = []
        self.fired = {TOKEN_A: np.array([1, 3], dtype=np.int64),
                      TOKEN_B: np.array([2, 4], dtype=np.int64)}

    def step(self, img, cx, cy, gains=None, seed=0, detail=False,
             extra_drive=None, extra_record=None):
        self.calls.append({"cx": cx, "cy": cy, "seed": seed, "extra_drive": extra_drive})
        fired = self.fired.get(card_at(cx, cy), np.array([], dtype=np.int64))
        return (0.0, 0.0, self.click, {"stop": 400.0},
                {"fired": fired, "firing": int(len(fired))})


class FakeMB:
    """
    Four KC->MBON synapses, one per Kenyon cell, two on each side.

    The room reads a look as calibration.syn_drive does - the cells that fired
    times the weights dopamine changes - so the weights are what a test sets:
    base is [A of card A, A of card B, V of card A, V of card B], since a look
    at card A fires cells 1 and 3 and a look at card B fires cells 2 and 4.
    """

    reward_side = np.array([10, 11])
    punish_side = np.array([20, 21])
    calibration = "pn05_apl10_kc03"
    sides_sha = "0f0f"

    def __init__(self, fb):
        self.fb = fb
        self.log = []
        # cells 1-4 are the Kenyon cells, as FakeBrain's docstring says, and a
        # look record counts how many of them fired the way the gate does
        self.kc = np.array([1, 2, 3, 4], dtype=np.int64)
        self.pre = np.array([1, 2, 3, 4], dtype=np.int64)
        self.side = np.array([-1, -1, 1, 1], dtype=np.int8)   # approach, approach, avoid, avoid
        self.base = np.array([300.0, 100.0, 100.0, 100.0], dtype=np.float64)
        self.gain = np.ones(4, dtype=np.float64)

    def observe(self, fired):
        self.log.append(("observe", 0 if fired is None else int(len(fired))))

    def forget(self):
        self.log.append(("forget", None))

    def forget_trace(self):
        self.log.append(("forget_trace", None))

    def dopamine(self, valence, amount=1.0):
        self.log.append(("dopamine", (int(valence), round(float(amount), 6))))
        return 7

    def apply(self):
        self.log.append(("apply", None))

    def save(self):
        self.log.append(("save", None))
        return True


class FakeNose:
    def smell(self, name, symbol="", description=""):
        return {"odorants": [{"name": "geosmin", "weight": 1.0, "why": "test"}],
                "profile": {"DM1": 0.5}}

    def drive(self, smell):
        return {(5, 6): 100.0}


class FakePage:
    def __init__(self, rects=None):
        self.rects = RECTS if rects is None else rects

    async def evaluate(self, js, arg=None):
        return [dict(r) for r in self.rects]


class FakeHttp:
    """The executor, over no network at all."""

    def __init__(self):
        self.posts = []
        self.gets = []
        self.intent_reply = (200, {"status": "booked", "event": {}})
        self.intent_raises = None
        self.marks = {}
        self.events = {"events": [], "last": 0}

    def post_json(self, url, body, headers=None, timeout=None):
        self.posts.append({"url": url, "body": body, "headers": dict(headers or {}),
                           "timeout": timeout})
        if url.endswith("/intent"):
            if self.intent_raises is not None:
                raise self.intent_raises
            return self.intent_reply
        if url.endswith("/marks"):
            return 200, {"marks": {t: self.marks.get(t, {"value_eth": None, "venue": None,
                                                         "block": None, "reason": "unknown"})
                                   for t in body["tokens"]}}
        return 404, {}

    def get_json(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return 200, json.loads(json.dumps(self.events))

    def intents(self):
        return [p for p in self.posts if p["url"].endswith("/intent")]


class Jobs:
    """The worker threads, run when the test says so rather than whenever."""

    def __init__(self):
        self.queue = []

    def __call__(self, fn, *args):
        self.queue.append((fn, args))

    def flush(self):
        while self.queue:
            fn, args = self.queue.pop(0)
            fn(*args)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.now = [1_700_000_000.0]
        self.fb = FakeBrain()
        self.pilot = FakePilot()
        self.mb = FakeMB(self.fb)
        self.http = FakeHttp()
        self.jobs = Jobs()
        self.page = FakePage()
        self.board = []
        # The book, not the page, decides what is held - so the fixture's book
        # has to agree with the card the fixture draws as held.
        self.write_book([{"token": TOKEN_B, "symbol": "MUD", "name": "Mud"}])
        self.room = Room(self.fb, self.pilot, self.mb, FakeNose(), None, self.tmp,
                         "http://127.0.0.1:4671", "secret",
                         fetch_board=lambda: list(self.board), http=self.http,
                         clock=lambda: self.now[0], spawn=self.jobs)
        # a room that already knows where it is in the executor's event stream;
        # a room that does not is what JoiningTheEventStream is about
        self.room.seq_known = True

    # -- driving the fly --------------------------------------------------
    def step(self, at, seed=1, click=None):
        if click is not None:
            self.pilot.click = click
        return asyncio.run(self.room.step(self.page, IMG, at[0], at[1], seed))

    def readings(self, A=(300.0, 100.0), B=(100.0, 100.0)):
        """What a look at card A and a look at card B are worth: (approach, avoid) each."""
        self.mb.base[:] = [A[0], B[0], A[1], B[1]]

    def likes(self):
        """Card A leans toward approach, card B leans nowhere."""
        self.readings(A=(300.0, 100.0), B=(100.0, 100.0))

    def dislikes(self):
        """Card A leans toward avoidance, card B leans nowhere."""
        self.readings(A=(100.0, 300.0), B=(100.0, 100.0))

    def look_around(self, at):
        """
        One look at some other card first.

        A card is judged against the other cards this visit has looked at, so a
        fly that has seen nothing else has nothing to commit on. Every test
        about what a commit does needs the room to have been walked first; the
        one about the rule itself is TheFirstCardOfAVisit.
        """
        for r in self.page.rects:
            if not (r["x"] <= at[0] < r["x"] + r["w"] and r["y"] <= at[1] < r["y"] + r["h"]):
                self.step((r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0), seed=0)
                return True
        return False

    def commit(self, at, steps=2, flush=True, look_around=True):
        """
        Look around the room, look at a card long enough, then stop on it.

        An intent leaves on a worker, so nothing reaches the executor until the
        worker runs; flush=False leaves it in flight, which is what the busy
        rule is about.
        """
        if look_around:
            self.look_around(at)
        for i in range(steps):
            self.step(at, seed=i + 1, click=(i == steps - 1))
        self.pilot.click = False
        if flush:
            self.jobs.flush()

    def looks(self):
        return sorted(p.stem for p in (self.room.dir / "looks").glob("*.json"))

    def write_book(self, positions, mode="paper"):
        p = self.tmp / "backroom" / "public" / "public.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"mode": mode, "positions": positions}), encoding="utf-8")

    def pump_events(self, events, last=None):
        """One executor poll: ask on this call, apply on the next, as roam does."""
        self.http.events = {"events": events,
                            "last": len(events) if last is None else last}
        self.room.poll_events()
        self.jobs.flush()
        self.room.poll_events()


# --------------------------------------------------------------------------
class Dwell(Base):
    def test_leaving_the_card_ends_the_dwell(self):
        self.step(IN_A)
        self.assertEqual(self.room._dwell["steps"], 1)
        self.step(OUTSIDE)
        self.assertIsNone(self.room._dwell)

    def test_another_card_starts_a_new_dwell(self):
        self.step(IN_A)
        self.step(IN_A)
        self.assertEqual(self.room._dwell["steps"], 2)
        self.step(IN_B)
        self.assertEqual(self.room._dwell["token"], TOKEN_B)
        self.assertEqual(self.room._dwell["steps"], 1)

    def test_two_looks_of_evidence_are_needed(self):
        self.commit(IN_A, steps=1)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.looks(), [])
        self.assertEqual(self.room.counters["commits"], 0)

    def test_a_look_at_another_card_is_not_evidence_about_this_one(self):
        """The dwell is consecutive steps on one card; the room behind it is not."""
        self.step(IN_B)
        self.commit(IN_A, steps=1, look_around=False)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.looks(), [])
        self.assertEqual(self.room.counters["commits"], 0)

    def test_a_stop_off_every_card_does_nothing(self):
        self.step(OUTSIDE, click=True)
        self.step(OUTSIDE, click=True)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.looks(), [])
        self.assertEqual(self.room.counters["commits"], 0)

    def test_a_commit_spends_the_evidence(self):
        self.commit(IN_A)
        self.assertIsNone(self.room._dwell)

    def test_one_run_of_the_brain_per_step(self):
        """
        The blank control run is gone and so is the second card's run: what the
        fly sees and smells, the stop and the Kenyon cells that fired all come
        out of the one run a walking fly can afford.
        """
        for _ in range(4):
            self.step(IN_A)
            self.step(IN_B)
        self.assertEqual(len(self.pilot.calls), 8)
        self.assertEqual(self.fb.runs, [])

    def test_the_reference_is_the_other_cards_this_visit_has_looked_at(self):
        self.step(IN_A)
        self.assertEqual(self.room._dwell["reference"], [])   # nothing seen before it
        self.step(IN_B)
        self.assertEqual(self.room._dwell["reference"], [TOKEN_A])
        self.step(IN_A)
        self.assertEqual(self.room._dwell["reference"], [TOKEN_B])

    def test_a_card_is_never_its_own_reference(self):
        self.step(IN_A)
        self.step(IN_A)
        self.assertEqual(self.room._dwell["reference"], [])
        self.assertEqual(self.room._dwell["pairs"], 0)

    def test_walking_out_empties_the_room_behind_the_fly(self):
        asyncio.run(self.room.enter(self.page))
        self.step(IN_A)
        self.step(IN_B)
        self.assertEqual(self.room._dwell["reference"], [TOKEN_A])
        self.room.leave()
        asyncio.run(self.room.enter(self.page))
        self.step(IN_B)
        self.assertEqual(self.room._dwell["reference"], [])

    def test_one_card_in_the_room_is_not_a_choice(self):
        self.page.rects = [RECTS[0]]
        self.likes()
        self.commit(IN_A)
        self.assertEqual(self.http.intents(), [])
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        self.assertIsNone(look["drive"])
        self.assertEqual(look["pairs"], 0)
        self.assertEqual(look["reference"], [])

    def test_off_a_card_there_is_no_smell_and_no_reading(self):
        self.step(IN_A)
        self.step(OUTSIDE)
        self.assertIsNone(self.pilot.calls[-1]["extra_drive"])
        self.assertEqual(self.fb.runs, [])
        self.assertEqual(sorted(self.room._seen), [TOKEN_A])


class HowALookIsRead(Base):
    """
    A look is worth what its Kenyon cells carry into the mushroom body's output
    synapses, split by which dopamine cluster reaches them - not what the MBONs
    fired at. MEASURED 2026-09-12 (build/backroom_screen.json): with the rate
    readout both sugar and shock lowered the score and a rewarded coin came out
    backwards, because in a whole-brain simulation those MBONs also carry
    recurrent input from everything else.
    """

    def test_only_the_synapses_whose_kenyon_cell_fired_are_counted(self):
        # card B's two synapses are enormous; the fly is looking at card A
        self.readings(A=(300.0, 100.0), B=(9e6, 9e6))
        self.step(IN_A)
        self.assertEqual((self.room._dwell["A"], self.room._dwell["V"]), (300.0, 100.0))
        self.assertEqual(self.room._seen[TOKEN_A], (300.0, 100.0))

    def test_the_sides_are_split_by_which_dopamine_reaches_them(self):
        # cell 1's synapse is on the approach side (PPL1 input), cell 3's on the
        # avoidance side, and a look at card A fires both
        self.readings(A=(317.0, 101.0))
        self.step(IN_A)
        self.assertEqual(self.room._dwell["A"], 317.0)
        self.assertEqual(self.room._dwell["V"], 101.0)

    def test_the_reading_is_what_calibration_says_it_is(self):
        self.readings(A=(300.0, 100.0))
        self.step(IN_A)
        self.assertEqual(self.room._seen[TOKEN_A],
                         calibration.syn_drive(self.mb, np.array([1, 3])))

    def test_depressing_an_avoidance_synapse_lifts_the_card(self):
        """What reward dopamine does, done by hand: the same look leans further toward approach."""
        self.likes()
        self.step(IN_B)
        self.step(IN_A)
        before = self.room._drive_of(self.room._dwell)
        self.mb.gain[2] = 0.5                     # card A's avoidance synapse, depressed
        self.step(IN_B)
        self.step(IN_A)
        self.assertGreater(self.room._drive_of(self.room._dwell), before)

    def test_the_dwell_averages_its_looks(self):
        self.likes()
        self.step(IN_B)
        self.step(IN_A)
        one = self.room._drive_of(self.room._dwell)
        self.step(IN_A)
        self.assertEqual(self.room._dwell["steps"], 2)
        self.assertEqual(self.room._dwell["pairs"], 2)
        self.assertAlmostEqual(self.room._drive_of(self.room._dwell), one)


class ALookThatReadNothing(Base):
    """
    A run in which no Kenyon cell fired measured nothing about the card.

    calibration.leaning scores an empty reading 0.0, and 0.0 is not neutral
    here: every leaning the first paper run measured was between -0.18 and
    -0.31, so an empty reading is about a quarter of the range above every real
    card. It reached the reference of look 1789229112324-0009 and moved that
    sell's drive from -0.068 to -0.171 - and the drive is the order size.
    """

    def blind(self, card="A"):
        """Make one card's synapses carry nothing, so its look reads (0, 0)."""
        if card == "A":
            self.readings(A=(0.0, 0.0), B=(100.0, 300.0))
        else:
            self.readings(A=(300.0, 100.0), B=(0.0, 0.0))

    def test_an_empty_reading_is_not_the_room_behind_the_fly(self):
        self.blind("B")
        self.step(IN_B)
        self.step(IN_A)
        self.assertEqual(self.room._dwell["reference"], [])
        self.assertEqual(self.room._dwell["pairs"], 0)
        self.assertIsNone(self.room._drive_of(self.room._dwell))

    def test_an_empty_reading_is_not_a_look_at_all(self):
        self.blind("A")
        self.step(IN_A)
        self.assertEqual(self.room._dwell["steps"], 0)
        self.assertEqual(self.room._dwell["blind"], 1)
        self.assertEqual(self.room._seen, {})
        self.assertEqual(self.room.counters["looks"], 0)

    def test_a_blind_step_is_not_evidence_and_commits_nothing(self):
        self.step(IN_B)                                   # a real card behind it
        self.blind("A")
        self.commit(IN_A, look_around=False)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.looks(), [])

    def test_a_real_reading_next_to_an_empty_one_still_counts(self):
        self.readings(A=(300.0, 100.0), B=(0.0, 0.0))
        self.step(IN_B)                                   # nothing measured
        self.step(IN_A)
        self.assertEqual(sorted(self.room._seen), [TOKEN_A])
        self.assertEqual(self.room._dwell["steps"], 1)


class DepressingMB(FakeMB):
    """
    A mushroom body whose dopamine really moves weights, the way mushroom.py's
    does: a punishment depresses the approach side of every synapse that was
    eligible, and this fixture depresses the whole side, because the measured
    shift is almost entirely common to every card (build/backroom_screen.json:
    12 pairings moved the trained coin's leaning -0.0254 and an untouched
    coin's -0.0267, so the untouched coin moved further).
    """

    def dopamine(self, valence, amount=1.0):
        FakeMB.dopamine(self, valence, amount)
        self.gain[self.side == -1] *= 0.9
        return 4


class AfterALesson(Base):
    """
    A delivery moves weights, and a reading taken before it is priced at the
    old ones. syn_drive is base * gain, so comparing a card read after a lesson
    against cards read before it hands the whole common shift to the drive -
    which is the one thing the relative rule exists to cancel, and the drive is
    the order size.
    """

    def setUp(self):
        super().setUp()
        self.mb = DepressingMB(self.fb)
        self.room.mb = self.mb
        self.room.readout = calibration.readout(self.mb)
        self.readings(A=(200.0, 100.0), B=(200.0, 100.0))   # two cards read alike
        self.room.mark_ref[TOKEN_B] = 1.0
        self.http.marks[TOKEN_B] = {"value_eth": 0.5}       # a fall: punishment

    def punish(self):
        self.step(IN_A)                                     # read at the old weights
        self.step(IN_B)                                     # asks for the mark
        self.jobs.flush()
        self.step(IN_B)                                     # the answer lands: a lesson

    def test_the_lesson_is_delivered(self):
        self.punish()
        self.assertIn(("dopamine", (-1, 0.5)), self.mb.log)

    def test_the_room_behind_the_fly_is_emptied(self):
        self.punish()
        self.assertEqual(self.room._seen, {})
        self.assertEqual(self.room._dwell["steps"], 0)
        self.assertEqual(self.room._dwell["pairs"], 0)
        self.assertEqual(self.room._dwell["reference"], [])

    def test_the_next_card_has_nothing_to_be_judged_against_yet(self):
        self.punish()
        self.step(IN_A)
        self.assertIsNone(self.room._drive_of(self.room._dwell))

    def test_and_once_it_has_walked_two_cards_again_the_lesson_is_not_the_drive(self):
        self.punish()
        self.step(IN_B)
        self.step(IN_A)
        # both cards now read under the depressed weights, and the brain reads
        # them alike, so the drive is 0. Against the stale reference it was
        # -0.0476, which is larger than the whole lesson the gate measured
        self.assertEqual(self.room._drive_of(self.room._dwell), 0.0)

    def test_a_settled_sell_empties_it_too(self):
        self.step(IN_A)
        self.step(IN_B)
        look = self.looks()[-1] if self.looks() else None
        self.commit(IN_A, look_around=False)
        look = self.looks()[-1]
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([{"seq": 1, "kind": "booked", "side": "sell", "token": TOKEN_A,
                           "eth": 0.5, "gas_eth": 0.0, "fraction": 1.0, "look_id": look}])
        self.assertIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room._seen, {})


class Commit(Base):
    def test_liking_a_coin_is_a_buy(self):
        self.likes()
        self.commit(IN_A)
        sent = self.http.intents()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["body"]["side"], "buy")
        self.assertEqual(sent[0]["body"]["token"], TOKEN_A)
        self.assertGreater(sent[0]["body"]["drive"], 0)

    def test_liking_a_coin_it_holds_is_still_a_buy(self):
        self.readings(A=(100.0, 100.0), B=(300.0, 100.0))
        self.commit(IN_B)
        self.assertEqual(self.http.intents()[0]["body"]["side"], "buy")

    def test_disliking_a_coin_it_holds_is_a_sell(self):
        self.readings(A=(100.0, 100.0), B=(100.0, 300.0))
        self.commit(IN_B)
        sent = self.http.intents()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["body"]["side"], "sell")
        self.assertLess(sent[0]["body"]["drive"], 0)

    def test_disliking_a_coin_it_does_not_hold_sends_nothing(self):
        self.dislikes()
        self.commit(IN_A)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.room.counters["dislikes"], 1)
        self.assertEqual(self.room.last_intents[-1]["status"], "dislike")
        self.assertEqual(len(self.looks()), 1)          # the dislike is still on the record

    def test_an_indifferent_commit_sends_nothing(self):
        self.readings(A=(100.0, 100.0), B=(100.0, 100.0))
        self.commit(IN_A)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.room.counters["commits"], 1)
        self.assertEqual(len(self.looks()), 1)          # the look is kept either way

    def test_a_card_that_leans_exactly_like_the_room_is_indifferent(self):
        """Not "quiet" and not "loud": the same leaning, at five times the size."""
        self.readings(A=(300.0, 100.0), B=(60.0, 20.0))
        self.commit(IN_A)
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        self.assertEqual(look["drive"], 0.0)
        self.assertEqual(self.http.intents(), [])

    def test_above_the_room_is_positive_and_below_it_is_negative(self):
        self.likes()
        self.commit(IN_A)
        self.assertGreater(self.http.intents()[0]["body"]["drive"], 0)
        self.dislikes()
        self.commit(IN_A)
        last = json.loads((self.room.dir / "looks" / f"{self.looks()[-1]}.json").read_text())
        self.assertLess(last["drive"], 0)

    def test_doubling_both_sides_of_a_card_changes_nothing(self):
        """The leaning is a ratio, so a coin cannot buy a drive by firing harder."""
        self.readings(A=(300.0, 100.0), B=(100.0, 100.0))
        self.commit(IN_A)
        self.readings(A=(600.0, 200.0), B=(200.0, 200.0))
        self.commit(IN_A)
        quiet, loud = (p["body"]["drive"] for p in self.http.intents())
        self.assertAlmostEqual(loud, quiet)

    def test_the_intent_body_is_exactly_the_five_fields(self):
        self.likes()
        self.commit(IN_A)
        body = self.http.intents()[0]["body"]
        self.assertEqual(set(body), {"token", "side", "drive", "seen_at", "look_id"})
        self.assertEqual(body["seen_at"], self.now[0])
        self.assertTrue((self.room.dir / "looks" / f"{body['look_id']}.json").exists())

    def test_the_intent_token_is_the_only_credential(self):
        self.likes()
        self.commit(IN_A)
        self.assertEqual(self.http.intents()[0]["headers"]["X-Fly-Intent"], "secret")

    def test_the_drive_reaches_the_executor_unclipped(self):
        self.readings(A=(317.0, 101.0), B=(97.0, 103.0))
        self.commit(IN_A)
        want = calibration.relative((317.0, 101.0), [(97.0, 103.0)])
        self.assertEqual(self.http.intents()[0]["body"]["drive"], want)
        self.assertNotEqual(want, round(want, 3))       # not a rounded number

    def test_nothing_clips_it_but_the_ends_of_the_range(self):
        self.readings(A=(400.0, 0.0), B=(0.0, 400.0))   # +1 against -1
        self.commit(IN_A)
        self.assertEqual(self.http.intents()[0]["body"]["drive"], 1.0)

    def test_a_commit_while_an_intent_is_in_flight_is_not_sent(self):
        self.likes()
        self.commit(IN_A, flush=False)                  # the worker is queued, not run
        self.commit(IN_B, flush=False)
        self.jobs.flush()
        self.assertEqual(len(self.http.intents()), 1)
        self.assertEqual(self.room.counters["busy"], 1)
        self.assertEqual(self.room.last_intents[-1]["status"], "busy")
        self.assertEqual(len(self.looks()), 2)          # both stops are still recorded

    def test_the_look_record_holds_what_a_replay_needs(self):
        self.likes()
        self.commit(IN_A)
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        for key in ("look_id", "at", "token", "symbol", "name", "smell", "card_rect", "cursor",
                    "seed", "drive", "A", "V", "leaning", "dwell_steps", "pairs", "reference",
                    "reference_leaning", "calibration", "sides_sha", "board_item", "crop",
                    "frame", "mode"):
            self.assertIn(key, look)
        for gone in ("A0", "V0", "ground"):
            self.assertNotIn(gone, look, "the blank control is gone and so are its fields")
        self.assertEqual([r["token"] for r in look["reference"]], [TOKEN_B])
        self.assertTrue(look["reference"][0]["smell"]["profile"])
        # A and V are the dwell's sums, the leaning is its mean, and the
        # reference is what the other card leaned when the fly looked at it
        self.assertEqual((look["A"], look["V"]), (600.0, 200.0))
        self.assertAlmostEqual(look["leaning"], 0.5)
        self.assertEqual(look["reference"][0]["leaning"], 0.0)
        self.assertEqual(look["reference_leaning"], 0.0)
        self.assertAlmostEqual(look["drive"], 0.5)
        self.assertEqual(look["cursor"], list(IN_A))
        self.assertEqual(look["dwell_steps"], 2)
        self.assertEqual(look["calibration"], "pn05_apl10_kc03")
        self.assertEqual(look["mode"], "paper")
        self.assertEqual(look["crop"], [0, 0, 250, 205])   # the window, clipped at the corner

    def test_the_record_cannot_be_read_two_ways(self):
        """
        A reader who recomputes (A-V)/(A+V) from the stored sums gets a
        different number from the stored leaning, which is the mean of the
        per-step ratios: up to 0.042 apart in the first paper run, which is the
        size of the whole lesson the offline gate measured. Both are written
        down, named for what they are.
        """
        self.likes()
        self.step(IN_B)
        self.readings(A=(300.0, 100.0))
        self.step(IN_A)
        self.readings(A=(100.0, 100.0))                   # the same card, read differently
        self.step(IN_A, click=True)
        self.pilot.click = False
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        self.assertEqual((look["A"], look["V"]), (400.0, 200.0))
        self.assertEqual((look["A_mean"], look["V_mean"]), (200.0, 100.0))
        self.assertAlmostEqual(look["leaning"], 0.25)                    # (0.5 + 0.0) / 2
        self.assertAlmostEqual(look["leaning_of_sums"], 1.0 / 3.0)       # 200 / 600
        self.assertNotAlmostEqual(look["leaning"], look["leaning_of_sums"])

    def test_the_record_says_how_much_of_the_window_was_this_card(self):
        """
        The offline gate held picture, cursor and seed identical so that only
        smell differed. The room cannot: the eye window is 300x210 and the
        cards are 280x200 fourteen pixels apart, so a stop away from a card's
        centre reads its neighbours too - 29.5% to 85.4% on card over the first
        paper run's nine commits. A reader of one look can now see which.
        """
        self.likes()
        self.commit(IN_A)
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        # the window is clipped to [0,0,250,205] at the corner and the card is
        # 280x200, so 250x200 of that window is the card being judged
        self.assertAlmostEqual(look["on_card"], 50000.0 / 51250.0, places=4)

    def test_the_record_says_how_much_of_the_brain_was_in_it(self):
        """
        Two coins in the first paper run recorded byte-identical approach sums
        (678.150021) over three steps each: at the floor the same lowest
        threshold cells fire whatever the coin is, and the reading stops being
        about the coin. The gate's coins fired 5.0% to 23.5% of the Kenyon
        cells; a look far below that is worth knowing about.
        """
        self.likes()
        self.commit(IN_A)
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        self.assertEqual(look["kc_mean"], 2.0)             # cells 1 and 3, every step
        self.assertAlmostEqual(look["kc_frac"], 0.5)       # two of the four Kenyon cells
        self.assertEqual(look["blind_steps"], 0)

    def test_a_refused_reply_is_reported_on_a_later_step(self):
        self.likes()
        self.http.intent_reply = (200, {"status": "refused", "reason": "nothing to spend"})
        self.commit(IN_A)
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertEqual(self.room.last_intents[-1]["status"], "refused")
        self.assertEqual(self.room.last_intents[-1]["reason"], "nothing to spend")


class TheFirstCardOfAVisit(Base):
    """
    A card is judged against the other cards this visit has looked at, so the
    first card of a visit is judged against nothing.

    MEASURED 2026-09-12 (build/backroom_screen.json): an absolute reading -
    against a blank frame or against nothing at all - carries almost none of
    what the mushroom body learned, because learning here is partly global. So
    a stop with no other card behind it is recorded and sends nothing, rather
    than being read as a preference.
    """

    def test_nothing_commits_before_a_second_card_has_been_seen(self):
        self.likes()
        self.commit(IN_A, look_around=False)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.room.counters["commits"], 1)
        self.assertEqual(self.room.counters["intents"], 0)
        look = json.loads((self.room.dir / "looks" / f"{self.looks()[0]}.json").read_text())
        self.assertIsNone(look["drive"])
        self.assertEqual(look["pairs"], 0)

    def test_the_stop_is_logged_with_a_reason(self):
        self.likes()
        self.commit(IN_A, look_around=False)
        last = self.room.last_intents[-1]
        self.assertEqual(last["status"], "no reference")
        self.assertEqual(last["side"], "none")
        self.assertIsNone(last["drive"])
        self.assertIn("no other card", last["reason"])

    def test_and_then_the_next_card_can_commit(self):
        self.likes()
        self.commit(IN_A, look_around=False)               # sends nothing
        self.commit(IN_B, look_around=False)               # A is behind it now
        self.assertEqual(len(self.http.intents()), 1)
        self.assertEqual(self.http.intents()[0]["body"]["token"], TOKEN_B)

    def test_a_fresh_visit_starts_with_nothing_behind_the_fly(self):
        self.likes()
        asyncio.run(self.room.enter(self.page))
        self.jobs.flush()
        self.commit(IN_B, look_around=False)               # first card of this visit
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.room.last_intents[-1]["status"], "no reference")


class Amount(unittest.TestCase):
    def test_a_doubling_and_a_halving_weigh_the_same(self):
        self.assertAlmostEqual(amount_for(2.0), 0.5)
        self.assertAlmostEqual(amount_for(0.5), 0.5)

    def test_no_move_is_no_lesson(self):
        self.assertEqual(amount_for(1.0), 0.0)
        self.assertEqual(sign_for(1.0), 0)

    def test_total_loss_is_the_whole_lesson(self):
        self.assertEqual(amount_for(0.0), 1.0)
        self.assertEqual(amount_for(-3.0), 1.0)

    def test_small_moves_are_small(self):
        self.assertAlmostEqual(amount_for(1.01), 0.00990099, places=6)

    def test_direction(self):
        self.assertEqual(sign_for(2.0), 1)
        self.assertEqual(sign_for(0.5), -1)


class LookingAtWhatItHolds(Base):
    def setUp(self):
        super().setUp()
        self.room.mark_ref[TOKEN_B] = 1.0
        self.http.marks[TOKEN_B] = {"value_eth": 2.0, "venue": "curve", "block": 9}

    def test_the_first_dwell_step_only_asks(self):
        self.step(IN_B)
        self.assertEqual([p["url"] for p in self.http.posts], [])   # still queued
        self.jobs.flush()
        self.assertTrue(self.http.posts[0]["url"].endswith("/marks"))
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])

    def test_the_next_step_pairs_the_coin_with_what_it_did(self):
        self.step(IN_B)
        self.jobs.flush()
        mark = len(self.mb.log)
        self.step(IN_B)
        self.assertEqual(self.mb.log[mark:],
                         [("observe", 2), ("forget", None), ("dopamine", (1, 0.5)),
                          ("apply", None), ("save", None)])
        self.assertEqual(self.room.mark_ref[TOKEN_B], 2.0)
        self.assertEqual(self.room.counters["sugar"], 1)

    def test_a_fall_is_punishment(self):
        self.http.marks[TOKEN_B] = {"value_eth": 0.5}
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        self.assertIn(("dopamine", (-1, 0.5)), self.mb.log)
        self.assertEqual(self.room.counters["shock"], 1)

    def test_one_delivery_per_dwell(self):
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        self.step(IN_B)
        self.step(IN_B)
        self.assertEqual([k for k, _ in self.mb.log].count("dopamine"), 1)

    def test_leaving_before_the_answer_delivers_nothing(self):
        self.step(IN_B)
        self.step(OUTSIDE)
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.mark_ref[TOKEN_B], 1.0)

    def test_an_unquotable_mark_teaches_nothing(self):
        self.http.marks[TOKEN_B] = {"value_eth": None}
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.mark_ref[TOKEN_B], 1.0)

    def test_a_coin_with_no_reference_only_sets_one(self):
        self.room.mark_ref.pop(TOKEN_B)
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.mark_ref[TOKEN_B], 2.0)

    def test_a_coin_it_does_not_hold_is_never_marked(self):
        self.step(IN_A)
        self.jobs.flush()
        self.step(IN_A)
        self.assertEqual(self.http.posts, [])

    def test_the_record_says_it_is_paper(self):
        self.write_book([{"token": TOKEN_B, "symbol": "MUD"}])
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        line = json.loads((self.room.dir / "dopamine.jsonl").read_text().splitlines()[0])
        self.assertEqual(line["mode"], "paper")
        self.assertEqual(line["kind"], "look")
        self.assertEqual(line["q"], 2.0)
        self.assertEqual(line["synapses_hit"], 7)
        self.assertEqual(line["sign"], 1)


class Settlements(Base):
    def a_look(self, at=IN_A):
        """
        A stored look with its picture, which is what a real sell carries: the
        commit that sold the coin wrote one. A settled sell with no stored
        picture is Settlements' own test below.
        """
        self.commit(at, look_around=False)
        self.mb.log.clear()
        return self.looks()[-1]

    def booked(self, side, **kw):
        ev = {"seq": 1, "at": self.now[0], "kind": "booked", "side": side, "token": TOKEN_A,
              "symbol": "AAA", "name": "A", "drive": 0.5, "look_id": None, "venue": "curve",
              "block": 9, "eth": 0.0, "tokens": 0.0, "gas_eth": 0.0}
        ev.update(kw)
        return ev

    def test_a_booked_buy_sets_the_reference_and_teaches_nothing(self):
        self.pump_events([self.booked("buy", eth=0.30, gas_eth=0.02)])
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_A], 0.32)
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.counters["booked"], 1)
        self.assertEqual(self.room.last_seq, 1)

    def test_buys_add_up(self):
        self.pump_events([self.booked("buy", eth=0.30, gas_eth=0.02)])
        self.now[0] += 10
        self.pump_events([self.booked("buy", seq=2, eth=0.10, gas_eth=0.01)], last=2)
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_A], 0.43)

    def test_a_booked_sell_shows_the_coin_again_and_teaches(self):
        look = self.a_look()
        self.room.mark_ref[TOKEN_A] = 1.0
        before = len(self.fb.runs)
        self.pump_events([self.booked("sell", eth=1.1, gas_eth=0.1, fraction=0.5,
                                      look_id=look)])
        self.assertEqual(len(self.fb.runs), before + 1)        # the re-presentation run
        # the wipe first: nothing the fly looked at while the sell was settling
        # may take the lesson - and the wipe after, so this coin cannot take
        # the lesson of whatever card the fly is dwelling on next
        self.assertEqual(self.mb.log, [("forget_trace", None), ("observe", 2),
                                       ("dopamine", (1, 0.5)), ("apply", None), ("save", None),
                                       ("forget_trace", None)])
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_A], 0.5)

    def test_the_coin_just_taught_is_no_longer_eligible(self):
        """
        The trace decays but never clears, so without the wipe on the way out
        the re-presented coin is still eligible at 55% on the next step and
        takes that share of the next card's mark lesson.
        """
        look = self.a_look()
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([self.booked("sell", eth=1.1, gas_eth=0.1, fraction=0.5,
                                      look_id=look)])
        after = [k for k, _ in self.mb.log]
        self.assertEqual(after[-1], "forget_trace")
        self.assertGreater(after.index("dopamine"), after.index("forget_trace"))

    def test_a_losing_sell_is_punishment(self):
        look = self.a_look()
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([self.booked("sell", eth=0.3, gas_eth=0.05, fraction=1.0,
                                      look_id=look)])
        self.assertIn(("dopamine", (-1, 0.75)), self.mb.log)
        self.assertNotIn(TOKEN_A, self.room.mark_ref)          # the position closed
        line = json.loads((self.room.dir / "dopamine.jsonl").read_text().splitlines()[0])
        self.assertEqual(line["kind"], "sell")
        self.assertAlmostEqual(line["q"], 0.25)

    def test_a_sell_whose_picture_is_gone_teaches_nothing(self):
        """
        With no stored picture the coin can only be shown to the fly on a
        uniform ground-grey frame, which this project measured as a stronger
        stimulus than a card (13,075 approach MBONs against 6,796) and which
        every card shares - so the lesson would land on the whole board. The
        record can be missing: a failed PNG write, a prune, an event carrying a
        look_id this room never wrote.
        """
        look = self.a_look()
        (self.room.dir / "looks" / f"{look}.png").unlink()
        self.room.mark_ref[TOKEN_A] = 1.0
        before = len(self.fb.runs)
        self.pump_events([self.booked("sell", eth=0.3, gas_eth=0.05, fraction=1.0,
                                      look_id=look)])
        self.assertEqual(len(self.fb.runs), before)            # nothing was shown at all
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertFalse((self.room.dir / "dopamine.jsonl").exists())

    def test_a_sell_with_a_look_id_this_room_never_wrote_teaches_nothing(self):
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([self.booked("sell", eth=0.3, gas_eth=0.05, fraction=1.0,
                                      look_id="1700000000000-9999")])
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])

    def test_the_re_presentation_carries_the_coins_own_smell(self):
        look = self.a_look()
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([self.booked("sell", eth=1.1, gas_eth=0.1, fraction=0.5,
                                      look_id=look)])
        self.assertIn((5, 6), self.fb.runs[-1]["drive"])        # the nose's own key
        self.assertEqual(self.fb.runs[-1]["steps"], self.pilot.sim_steps)

    def test_a_sell_with_no_reference_teaches_nothing(self):
        self.pump_events([self.booked("sell", eth=1.1, gas_eth=0.1, fraction=0.5)])
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])

    def test_a_refusal_teaches_nothing(self):
        self.pump_events([{"seq": 1, "kind": "refused", "side": "buy", "token": TOKEN_A,
                           "reason": "nothing to spend", "look_id": "x"}])
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.counters["refused"], 1)
        self.assertEqual(self.room.counters["booked"], 0)

    def test_events_are_asked_for_after_the_last_one_seen(self):
        self.pump_events([self.booked("buy", eth=0.1)])
        self.now[0] += 10
        self.room.poll_events()
        self.jobs.flush()
        self.assertTrue(self.http.gets[-1].endswith("/events?after=1"))


class Looks(Base):
    def test_pruning_keeps_what_an_open_position_points_at(self):
        self.room.max_looks = 2
        self.likes()
        self.commit(IN_A)
        kept = self.looks()[0]
        self.jobs.flush()                                   # the intent comes back booked
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.pump_events([{"seq": 1, "kind": "booked", "side": "buy", "token": TOKEN_A,
                           "eth": 0.2, "gas_eth": 0.01, "look_id": kept}])
        self.assertIn(kept, self.room.refs[TOKEN_A])

        for i in range(4):                                  # fill the room with newer looks
            self.now[0] += 1
            self.commit(IN_A, flush=False)
        self.assertIn(kept, self.looks())
        self.assertLessEqual(len(self.looks()), 3)
        self.assertTrue((self.room.dir / "looks" / f"{kept}.png").exists())

    def test_a_closed_position_lets_its_looks_age_out(self):
        self.likes()
        self.commit(IN_A)
        kept = self.looks()[0]
        self.jobs.flush()
        self.step(OUTSIDE)
        self.room.mark_ref[TOKEN_A] = 1.0
        self.pump_events([{"seq": 1, "kind": "booked", "side": "sell", "token": TOKEN_A,
                           "eth": 1.0, "gas_eth": 0.0, "fraction": 1.0, "look_id": kept}])
        self.assertNotIn(TOKEN_A, self.room.refs)

    def test_a_refused_intent_does_not_pin_its_look(self):
        self.likes()
        self.http.intent_reply = (200, {"status": "refused", "reason": "stale"})
        self.commit(IN_A)
        look = self.looks()[0]
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertEqual(self.room.protected_looks(), set())
        self.assertNotIn(TOKEN_A, self.room.refs)
        self.assertTrue((self.room.dir / "looks" / f"{look}.json").exists())


class Board(Base):
    def coin(self, token, native=True, name="Banana"):
        return {"token": token, "name": name, "symbol": "BNNA", "description": "a yellow fruit",
                "logo_url": "https://example.invalid/logo.png", "market_cap_usd": 41000.0,
                "progress_pct": 12.5, "quote_is_native": native}

    def test_only_native_quote_coins_reach_the_board(self):
        self.board = [self.coin(TOKEN_A), self.coin(TOKEN_B, native=False)]
        self.room.refresh_board(force=True)
        self.jobs.flush()
        self.assertEqual([c["token"] for c in self.room.board()["cards"]], [TOKEN_A])

    def test_a_failed_fetch_keeps_the_last_good_board(self):
        self.board = [self.coin(TOKEN_A)]
        self.room.refresh_board(force=True)
        self.jobs.flush()

        def boom():
            raise RuntimeError("pons listing unavailable")
        self.room.fetch_board = boom
        self.now[0] += 60
        self.room.refresh_board(force=True)
        self.jobs.flush()
        self.assertEqual([c["token"] for c in self.room.board()["cards"]], [TOKEN_A])

    def test_a_failed_fetch_is_said_once_and_shown(self):
        """
        An empty grid is the room's one total failure and it used to be its
        quietest: no rects, no dwell, no commit ever, while every published
        number agreed that nothing was wrong. The last good board still
        survives; the reason it is not being replaced does not have to be a
        secret.
        """
        said = []
        self.room._say = said.append

        def boom():
            raise ModuleNotFoundError("No module named 'eth_abi'")
        self.room.fetch_board = boom
        for _ in range(3):
            self.now[0] += 60
            self.room.refresh_board(force=True)
            self.jobs.flush()
        self.assertEqual(len(said), 1, said)
        self.assertIn("eth_abi", said[0])
        self.assertIn("eth_abi", self.room.state()["board_error"])
        self.assertEqual(self.room.state()["board_size"], 0)

    def test_a_board_that_arrives_clears_the_complaint(self):
        def boom():
            raise RuntimeError("pons listing unavailable")
        self.room.fetch_board = boom
        self.room.refresh_board(force=True)
        self.jobs.flush()
        self.assertIsNotNone(self.room.state()["board_error"])
        self.room.fetch_board = lambda: [self.coin(TOKEN_A)]
        self.now[0] += 60
        self.room.refresh_board(force=True)
        self.jobs.flush()
        self.assertIsNone(self.room.state()["board_error"])

    def test_a_holding_is_drawn_even_when_it_is_off_the_board(self):
        self.board = [self.coin(TOKEN_A)]
        self.room.refresh_board(force=True)
        self.jobs.flush()
        self.write_book([{"token": TOKEN_A, "symbol": "BNNA", "name": "Banana"},
                         {"token": TOKEN_B, "symbol": "MUD", "name": "Mud"}])
        board = self.room.board()
        self.assertTrue(board["cards"][0]["held"])
        self.assertEqual([h["token"] for h in board["holdings"]], [TOKEN_A, TOKEN_B])
        self.assertEqual(board["holdings"][0]["logo"], "https://example.invalid/logo.png")

    def test_the_room_remembers_a_coins_words_across_a_restart(self):
        self.board = [self.coin(TOKEN_A)]
        self.room.refresh_board(force=True)
        self.jobs.flush()
        again = Room(self.fb, self.pilot, self.mb, FakeNose(), None, self.tmp,
                     "http://127.0.0.1:4671", "secret", fetch_board=lambda: [],
                     http=self.http, clock=lambda: self.now[0], spawn=self.jobs)
        self.assertEqual(again.meta[TOKEN_A]["description"], "a yellow fruit")
        self.assertEqual(again.board()["updated"], 0)        # stale by definition


class State(Base):
    def test_what_the_page_and_the_stream_are_told(self):
        self.likes()
        asyncio.run(self.room.enter(self.page))
        self.jobs.flush()
        self.step(IN_B)                                  # the room behind the fly
        self.step(IN_A)
        s = self.room.state()
        self.assertTrue(s["in_room"])
        self.assertEqual(s["visits"], 1)
        self.assertEqual(s["seen"], 2)
        self.assertEqual(s["now"]["token"], TOKEN_A)
        self.assertEqual(s["now"]["dwell_steps"], 1)
        self.assertGreater(s["now"]["drive"], 0)
        self.assertEqual(s["learning"], {"sugar": 0, "shock": 0, "last": []})
        for key in ("looks", "commits", "intents", "booked", "refused", "dislikes", "busy",
                    "last_intents", "board_size", "board_error", "board_updated"):
            self.assertIn(key, s)

    def test_looks_and_commits_are_not_the_same_number(self):
        """
        counters["looks"] was incremented only inside _write_look, which only
        _commit calls, so the two published figures could never differ: the
        first paper run's panel would have read looks 9, commits 9 for a fly
        that had looked at a card on most of 1,592 room steps. A look is a step
        that read a card; a commit is a stop that became a decision.
        """
        self.likes()
        asyncio.run(self.room.enter(self.page))
        self.jobs.flush()
        for _ in range(3):
            self.step(IN_B)
            self.step(IN_A)
        self.step(IN_A, click=True)
        self.pilot.click = False
        s = self.room.state()
        self.assertEqual(s["commits"], 1)
        self.assertEqual(s["looks"], 7)
        self.assertEqual(self.room.counters["looks"], 7)

    def test_the_first_card_of_a_visit_has_no_drive_to_report(self):
        self.likes()
        asyncio.run(self.room.enter(self.page))
        self.jobs.flush()
        self.step(IN_A)
        self.assertEqual(self.room.state()["seen"], 1)
        self.assertIsNone(self.room.state()["now"]["drive"])

    def test_leaving_ends_the_visit(self):
        asyncio.run(self.room.enter(self.page))
        self.step(IN_A)
        self.room.leave()
        self.assertFalse(self.room.state()["in_room"])
        self.assertIsNone(self.room.state()["now"])


class Isolation(unittest.TestCase):
    def test_the_room_imports_nothing_that_can_sign(self):
        import ast
        src = (Path(__file__).parent / "backroom.py").read_text(encoding="utf-8")
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        for banned in ("eth_account", "executor", "tradebook", "rhwallet", "rhprovider", "web3"):
            self.assertNotIn(banned, names)

    def test_nothing_here_signs_or_sends_a_transaction(self):
        src = (Path(__file__).parent / "backroom.py").read_text(encoding="utf-8")
        for banned in ("sendRawTransaction", "sign_transaction", "PrivateKey", "Account.create"):
            self.assertNotIn(banned, src)


class SlowExecutor(Base):
    """
    The executor answers late, or not at all.

    Quoting the chain takes the executor longer than the room's ordinary wait -
    14 to 15 s against the public RPC in the first paper run - so neither of
    these is an edge case. That run booked a buy several seconds after the room
    had given up waiting for the reply.
    """

    def test_a_timed_out_intent_keeps_its_look_pinned(self):
        self.likes()
        self.http.intent_raises = TimeoutError("timed out")
        self.commit(IN_A)
        look = self.looks()[0]
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertEqual(self.room.last_intents[-1]["status"], "unreachable")
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.assertIn(look, self.room.refs.get(TOKEN_A, []))
        self.assertIn(look, self.room.protected_looks())

    def test_a_booking_that_lands_after_the_timeout_still_has_its_look(self):
        self.likes()
        self.http.intent_raises = TimeoutError("timed out")
        self.commit(IN_A)
        look = self.looks()[0]
        self.jobs.flush()
        self.step(OUTSIDE)                                  # the room gives up on the reply
        self.http.intent_raises = None
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.pump_events([{"seq": 1, "kind": "booked", "side": "buy", "token": TOKEN_A,
                           "eth": 0.2, "gas_eth": 0.01, "look_id": look}])
        self.room.max_looks = 1
        for _ in range(3):
            self.now[0] += 1
            self.commit(IN_A, flush=False)
        self.assertIn(look, self.looks())
        self.assertTrue((self.room.dir / "looks" / f"{look}.png").exists())

    def test_a_busy_executor_does_not_pin_the_look(self):
        self.likes()
        self.http.intent_reply = (409, {"status": "busy"})
        self.commit(IN_A)
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertNotIn(TOKEN_A, self.room.refs)

    def test_the_two_chain_calls_wait_longer_than_the_rest(self):
        self.likes()
        self.commit(IN_A)
        self.jobs.flush()
        intent = [p for p in self.http.posts if p["url"].endswith("/intent")][-1]
        self.assertGreaterEqual(intent["timeout"], 20.0)
        self.room._mark_worker(TOKEN_A, {"done": False})
        mark = [p for p in self.http.posts if p["url"].endswith("/marks")][-1]
        self.assertGreaterEqual(mark["timeout"], 20.0)

    def test_what_counts_as_an_answer(self):
        """
        The executor's own replies, not hand-written status strings.

        Its 400 says {"status": "error"} and its 403 says {"status":
        "forbidden"}, so a rule that read the body never recognised either and
        those looks stayed pinned for ever.
        """
        import backroom
        for status, http in (("refused", 200), ("busy", 409), ("error", 400), ("forbidden", 403)):
            self.assertTrue(backroom.answered_unbooked(status, http), (status, http))
        for status, http in (("booked", 200), ("unreachable", None),
                             ("error", 500), ("error", 502)):
            self.assertFalse(backroom.answered_unbooked(status, http), (status, http))

    def test_a_four_hundred_does_not_pin_its_look(self):
        self.likes()
        self.http.intent_reply = (400, {"status": "error", "reason": "the body must be exactly"})
        self.commit(IN_A)
        self.jobs.flush()
        self.step(OUTSIDE)
        self.assertNotIn(TOKEN_A, self.room.refs)
        self.assertEqual(self.room.last_intents[-1]["status"], "error")

    def test_an_unauthenticated_room_does_not_pin_its_looks(self):
        # roam started outside run_all has no intent token, so every post is 403
        self.likes()
        self.http.intent_reply = (403, {"status": "forbidden"})
        for _ in range(3):
            self.now[0] += 1
            self.commit(IN_A)
            self.step(OUTSIDE)
        self.assertEqual(self.room.refs, {})

    def test_a_five_hundred_still_pins_it(self):
        self.likes()
        self.http.intent_reply = (500, {"status": "error", "reason": "boom"})
        self.commit(IN_A)
        self.jobs.flush()
        self.step(OUTSIDE)
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.assertIn(TOKEN_A, self.room.refs)


class HeldIsTheBookNotThePage(Base):
    """
    web/backroom.html repaints data-held on its own twenty-second poll, and a
    dwell is one to a dozen steps, so the card in front of the fly can be a
    whole dwell out of date. The executor's book is the authority.
    """

    def test_a_coin_bought_seconds_ago_is_sold_rather_than_disliked(self):
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"},
                         {"token": TOKEN_B, "symbol": "MUD", "name": "Mud"}])
        self.dislikes()
        self.commit(IN_A)                          # the card still says data-held="0"
        sent = self.http.intents()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["body"]["side"], "sell")
        self.assertEqual(self.room.counters["dislikes"], 0)

    def test_a_position_that_closed_is_not_sold_again(self):
        self.write_book([])                        # the card still says data-held="1"
        self.readings(A=(100.0, 100.0), B=(100.0, 300.0))
        self.commit(IN_B)
        self.assertEqual(self.http.intents(), [])
        self.assertEqual(self.room.counters["dislikes"], 1)

    def test_a_mark_follows_the_book_too(self):
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.step(IN_A)
        self.jobs.flush()
        self.assertTrue(self.http.posts[-1]["url"].endswith("/marks"))
        self.assertEqual(self.http.posts[-1]["body"], {"tokens": [TOKEN_A]})

    def test_the_page_is_still_what_the_fly_looks_at(self):
        # the attribute is not read for the decision, but the room must not
        # start ignoring the rectangles either
        self.write_book([])
        self.step(IN_B)
        self.assertEqual(self.room._dwell["token"], TOKEN_B)
        self.assertFalse(self.room._dwell["held"])


class TheFirstLookAtAFreshPosition(Base):
    """
    A buy's reference is what it cost; a mark is what selling would return.
    Both ends pay the venue's fee and the creator tax, so the first look at a
    coin the fly has just bought is below its cost whatever the coin did.
    """

    def buy(self, eth=0.30, gas=0.02):
        self.write_book([{"token": TOKEN_A, "symbol": "AAA", "name": "A"}])
        self.pump_events([{"seq": 1, "kind": "booked", "side": "buy", "token": TOKEN_A,
                           "eth": eth, "gas_eth": gas, "look_id": None}])

    def look(self):
        self.step(IN_A)
        self.jobs.flush()
        self.step(IN_A)
        self.step(OUTSIDE)

    def test_it_sets_the_reference_and_teaches_nothing(self):
        self.buy()
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_A], 0.32)
        self.http.marks[TOKEN_A] = {"value_eth": 0.30}     # the round trip, not a loss
        self.look()
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertEqual(self.room.mark_ref[TOKEN_A], 0.30)
        self.assertNotIn(TOKEN_A, self.room.from_cost)

    def test_the_look_after_that_is_measured_against_the_mark(self):
        self.buy()
        self.http.marks[TOKEN_A] = {"value_eth": 0.30}
        self.look()
        self.http.marks[TOKEN_A] = {"value_eth": 0.60}
        self.look()
        self.assertIn(("dopamine", (1, 0.5)), self.mb.log)
        self.assertEqual(self.room.mark_ref[TOKEN_A], 0.60)

    def test_a_sell_before_any_look_teaches_nothing_either(self):
        """
        The same rule on the other path. A sell whose position was never marked
        can only be measured against what it cost, which is the comparison
        _learn_from_mark refuses: both ends pay the fee, so a coin that did
        nothing still reads as a loss. Measured on this fixture before the fix:
        the identical flat round trip taught shock 0.060 when the fly had not
        looked at the coin in between and 0.0105 when it had - gaze history
        deciding the size of a lesson about a trade.
        """
        self.buy()
        self.now[0] += 10                                      # the next poll is due
        self.pump_events([{"seq": 2, "kind": "booked", "side": "sell", "token": TOKEN_A,
                           "eth": 0.18, "gas_eth": 0.02, "fraction": 1.0, "look_id": None}], last=2)
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertNotIn(TOKEN_A, self.room.from_cost)
        self.assertNotIn(TOKEN_A, self.room.mark_ref)          # the position closed

    def test_a_sell_after_a_mark_is_measured_against_the_mark(self):
        self.buy()
        self.http.marks[TOKEN_A] = {"value_eth": 0.30}
        self.look()                                            # the first mark sets the reference
        self.assertNotIn(TOKEN_A, self.room.from_cost)
        self.commit(IN_A, look_around=False)                   # the stop that wrote the sell's look
        self.mb.log.clear()
        self.now[0] += 10
        self.pump_events([{"seq": 2, "kind": "booked", "side": "sell", "token": TOKEN_A,
                           "eth": 0.17, "gas_eth": 0.02, "fraction": 1.0,
                           "look_id": self.looks()[-1]}], last=2)
        self.assertIn(("dopamine", (-1, 0.5)), self.mb.log)    # 0.15 against 0.30


class MarksThatCrossASettlement(Base):
    """A mark takes the executor 14-15 s to quote; the book can move inside it."""

    def test_a_mark_quoted_across_a_booking_teaches_nothing(self):
        self.room.mark_ref[TOKEN_B] = 1.0
        self.http.marks[TOKEN_B] = {"value_eth": 2.0}
        self.step(IN_B)                                        # asks
        self.jobs.flush()                                      # the answer is in hand
        self.pump_events([{"seq": 1, "kind": "booked", "side": "buy", "token": TOKEN_B,
                           "eth": 0.5, "gas_eth": 0.01, "look_id": None}])
        self.step(IN_B)                                        # would have delivered
        self.assertNotIn("dopamine", [k for k, _ in self.mb.log])
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_B], 1.51)

    def test_an_undisturbed_mark_still_teaches(self):
        self.room.mark_ref[TOKEN_B] = 1.0
        self.http.marks[TOKEN_B] = {"value_eth": 2.0}
        self.step(IN_B)
        self.jobs.flush()
        self.step(IN_B)
        self.assertIn(("dopamine", (1, 0.5)), self.mb.log)

    def test_the_reference_is_the_one_the_request_was_made_against(self):
        self.room.mark_ref[TOKEN_B] = 1.0
        self.http.marks[TOKEN_B] = {"value_eth": 2.0}
        self.step(IN_B)
        self.jobs.flush()
        self.room.mark_ref[TOKEN_B] = 4.0                      # moved by something else
        self.step(IN_B)
        self.assertIn(("dopamine", (1, 0.5)), self.mb.log)     # 2.0 against 1.0, not 4.0


class JoiningTheEventStream(Base):
    """
    Where a room starts reading from. room.json is best effort and the mushroom
    body is stored separately, so a room that lost its place and asked for
    everything would deliver dopamine a second time for every sell on file.
    """

    def setUp(self):
        super().setUp()
        self.room.seq_known = False                # as if room.json were missing
        self.room.last_seq = 0

    def booked(self, side, **kw):
        ev = {"seq": 1, "at": self.now[0], "kind": "booked", "side": side, "token": TOKEN_A,
              "symbol": "AAA", "name": "A", "drive": 0.5, "look_id": None, "venue": "curve",
              "block": 9, "eth": 0.0, "tokens": 0.0, "gas_eth": 0.0}
        ev.update(kw)
        return ev

    def history(self):
        return [self.booked("buy", seq=1, eth=0.2, gas_eth=0.01),
                self.booked("sell", seq=2, eth=0.5, gas_eth=0.01, fraction=1.0),
                self.booked("buy", seq=3, eth=0.2, gas_eth=0.01)]

    def test_a_room_with_no_place_on_disk_learns_nothing_from_the_backlog(self):
        self.pump_events(self.history(), last=3)
        self.assertEqual(self.room.last_seq, 3)
        self.assertEqual(self.mb.log, [])
        self.assertEqual(self.room.counters["booked"], 0)
        self.assertEqual(self.room.mark_ref, {})

    def test_and_then_it_applies_what_happens_after_it_arrived(self):
        self.pump_events(self.history(), last=3)
        self.now[0] += 10
        self.pump_events([self.booked("buy", seq=4, eth=0.3, gas_eth=0.02)], last=4)
        self.assertAlmostEqual(self.room.mark_ref[TOKEN_A], 0.32)
        self.assertEqual(self.room.counters["booked"], 1)

    def test_the_place_it_took_is_written_down(self):
        self.pump_events(self.history(), last=3)
        again = Room(self.fb, self.pilot, self.mb, FakeNose(), None, self.tmp,
                     "http://127.0.0.1:4671", "secret", fetch_board=lambda: [],
                     http=self.http, clock=lambda: self.now[0], spawn=self.jobs)
        self.assertTrue(again.seq_known)
        self.assertEqual(again.last_seq, 3)

    def test_a_ledger_that_restarted_is_followed_back_down(self):
        self.room.seq_known = True
        self.room.last_seq = 40
        self.pump_events([], last=3)
        self.assertEqual(self.room.last_seq, 3)

    def test_events_the_executor_will_not_vouch_for_are_not_applied(self):
        self.room.seq_known = True
        self.http.events = {"events": [self.booked("buy", eth=0.3, gas_eth=0.02)], "last": 1,
                            "error": "line 3 is not JSON"}
        self.room.poll_events()
        self.jobs.flush()
        self.room.poll_events()
        self.assertEqual(self.room.counters["booked"], 0)
        self.assertEqual(self.room.last_seq, 0)
        self.assertEqual(self.room.mark_ref, {})


class Eligibility(Base):
    """Which synapses a lesson is allowed to reach."""

    def test_arriving_at_a_card_drops_what_came_before_it(self):
        self.step(IN_A)
        self.assertIn(("forget_trace", None), self.mb.log)
        n = len(self.mb.log)
        self.step(IN_A)                                  # the same card, still one look
        self.assertNotIn(("forget_trace", None), self.mb.log[n:])
        self.step(IN_B)                                  # another coin
        self.assertIn(("forget_trace", None), self.mb.log[n:])

    def test_the_look_itself_is_observed_after_the_wipe(self):
        self.step(IN_A)
        self.assertEqual([k for k, _ in self.mb.log][:3], ["forget_trace", "observe", "forget"])

    def test_off_the_cards_nothing_is_wiped(self):
        self.step(OUTSIDE)
        self.assertNotIn(("forget_trace", None), self.mb.log)


if __name__ == "__main__":
    unittest.main()

"""
backrooms_talk on fakes: a fake model, a fake world (and the real TalkRoom on
backrooms_world's fake brain), a local HTTP relay. No model or connectome is
loaded and nothing leaves this machine.

  engine venv:  python -m unittest backrooms_talk.tests.test_talk -v
  repo root:    py -m pytest -q backrooms_talk     (skipped without torch / transformers)

What is held to:
  * turn order: round-robin A, B, C, D; the world steps between turns; a
    dropped turn passes to the next fly after at most two new samples;
  * the prompt holds only the fixed instruction, the fixed frame text, the
    topic, the hints and the last six accepted lines;
  * topics: the opening topic from a seed theme, a switch on a brain jump or
    after TOPIC_MAX lines, topic drops by reason, the payload's topic fields;
  * normalisation is exactly the disclosed list and changes no word;
  * filter drops are counted per reason, archived with the text, never
    posted and never edited; readout strings that fail are left out;
  * the line checks drop names, brands and the listed subjects, long and
    unfinished lines, numbers with units not in the hints, and repeats;
    uncounted never, edited never;
  * every payload matches the JSON Schema in TURN_SCHEMA.md;
  * the relay client retries 5xx and network errors with backoff, not 4xx;
  * the supervisor backs off, resets after a long run and stops on low RAM.
"""
try:
    import pytest
except ImportError:                 # the engine's venv runs this file with unittest
    pytest = None
if pytest is not None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

import http.server
import json
import math
import re
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import backrooms as br
import backrooms_world as bw
from test_backrooms_world import FakeBrain, FakeEye, fake_sides
import backrooms_dictionary as bd

import backrooms_talk
from backrooms_talk import engine as E
from backrooms_talk import grounding as G
from backrooms_talk import prompt as P
from backrooms_talk import readout as R
from backrooms_talk import relay_client as RC
from backrooms_talk import supervise as S
from backrooms_talk import topics as T
from backrooms_talk import world as W

PKG = Path(backrooms_talk.__file__).resolve().parent
NAMES = W.FLY_NAMES
SAFETY = E.load_safety()


# =============================================================================
# fakes
# =============================================================================

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeWorld:
    """
    Four flies in a row 4 mm apart along x, still. Every rate is a constant
    except fly A's P1, which jumps to 40 Hz from step 30 on, so the readout's
    ranking and the captioner both have something to report. Each step
    advances the shared clock by one second.
    """

    SIZES = {"ORN_DA1": 204, "JO_A": 50, "JO_B": 89, "LC10a": 275, "P1": 86, "pC1": 156,
             "song_pulse_mn": 8, "DNa02": 2, "DNa01": 2, "MDN": 4, "DNp09": 2}

    def __init__(self, clock=None, step_s=1.0):
        self.sizes = dict(self.SIZES)
        self.dt = bw.WORLD_DT_S
        self.clock, self.step_s = clock, step_s
        self.n = 0

    def step(self):
        self.n += 1
        if self.clock is not None:
            self.clock.now += self.step_s
        flies = {}
        for i, name in enumerate(NAMES):
            rates = {k: 0.0 for k in self.sizes}
            rates["LC10a"] = 3.0
            rates["ORN_DA1"] = 20.0
            if name == "A" and self.n >= 30:
                rates["P1"] = 40.0
            flies[name] = {
                "x": 2.0 + 4.0 * i, "y": 10.0, "heading": 0.0, "speed": 2.0, "turn": 0.0,
                "others": {o: {"distance": 4.0 * abs(i - j), "bearing": 0.0 if j > i else 180.0}
                           for j, o in enumerate(NAMES) if o != name},
                "rates": rates, "drive": {"ORN_DA1": 150.0, "JO_A": 12.0, "JO_B": 12.0},
                "drive_from": {}, "song": 900.0, "fired": 16512, "active_fraction": 0.1, "total_hz": 2.5,
            }
        return {"step": self.n, "t": self.n * self.dt, "flies": flies, "step_s": 0.0}


FAKE_TOPIC_WORDS = ("amber", "basil", "cedar", "delta", "ember", "fjord", "grove", "heron", "ivory")


class FakeModel:
    """Scripted replies; records every call. script(name, call_no) -> (text, tokens, finished);
    topic(call_no) -> (topic, tokens, finished) for topic requests (recorded in topic_calls)."""

    def __init__(self, script=None, topic=None):
        self.calls, self.topic_calls = [], []
        self.script = script or (lambda name, k: (f"I am at {k} mm from the wall, fly {name} here.", 12, True))
        self.topic = topic or (lambda k: (f"the {FAKE_TOPIC_WORDS[k % len(FAKE_TOPIC_WORDS)]} question", 4, True))

    def generate(self, messages, seed, **kw):
        if messages[0]["content"] == T.TOPIC_SYSTEM:
            self.topic_calls.append({"messages": messages, "seed": seed, "kw": kw})
            text, n, fin = self.topic(len(self.topic_calls))
            return text, n, fin, 0.01
        name = re.search(r"Write fly (\w)'s line in the first person",
                         messages[1]["content"]).group(1)
        self.calls.append({"messages": messages, "seed": seed, "kw": kw, "name": name})
        text, n, fin = self.script(name, len(self.calls))
        return text, n, fin, 0.01


class FakeRelay:
    def __init__(self, fail=()):
        self.payloads, self.fail = [], set(fail)

    def post_turn(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        k = len(self.payloads)
        ok = k not in self.fail
        return {"ok": ok, "status": 200 if ok else 422, "attempts": 1, "body": {"seq": k}, "error": None if ok else "HTTP 422"}


def make_engine(tmp, model=None, dry=True, relay=None, clock=None, **kw):
    clock = clock or Clock()
    world = FakeWorld(clock)
    kw.setdefault("turn_seconds", 3.0)
    kw.setdefault("warmup_steps", 2)
    kw.setdefault("ground", None)          # the scripted lines are not about the readout; Grounding tests use the real check
    eng = E.Engine(world, model or FakeModel(), safety=SAFETY, relay=relay, dry=dry,
                   archive_path=Path(tmp) / "archive.jsonl", clock=clock, sleep=lambda s: None,
                   say=lambda *a: None, **kw)
    return eng, world, clock


def archive_rows(tmp):
    p = Path(tmp) / "archive.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


# ---- a small JSON Schema checker for the subset TURN_SCHEMA.md uses ----------

def schema_from_md(path=E.TURN_SCHEMA_PATH):
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    return json.loads(m.group(1))


def validate(value, schema, where="$"):
    """Raise AssertionError on the first violation; supports the keywords the schema uses."""
    t = schema.get("type")
    if t == "object":
        assert isinstance(value, dict), f"{where}: not an object"
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            assert k in value, f"{where}: missing {k}"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            assert not extra, f"{where}: extra keys {sorted(extra)}"
        for k, v in value.items():
            if k in props:
                validate(v, props[k], f"{where}.{k}")
    elif t == "array":
        assert isinstance(value, list), f"{where}: not an array"
        assert len(value) >= schema.get("minItems", 0), f"{where}: too few items"
        assert len(value) <= schema.get("maxItems", 10 ** 9), f"{where}: too many items"
        for i, v in enumerate(value):
            validate(v, schema["items"], f"{where}[{i}]")
    elif t == "string":
        assert isinstance(value, str), f"{where}: not a string"
        assert len(value) >= schema.get("minLength", 0), f"{where}: too short"
        assert len(value) <= schema.get("maxLength", 10 ** 9), f"{where}: too long"
    elif t == "boolean":
        assert isinstance(value, bool), f"{where}: not a boolean"
    elif t == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), f"{where}: not an integer"
    elif t == "number":
        assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: not a number"
        assert math.isfinite(value), f"{where}: not finite"
    if "enum" in schema:
        assert value in schema["enum"], f"{where}: {value!r} not in enum"
    if "minimum" in schema:
        assert value >= schema["minimum"], f"{where}: below minimum"
    if "maximum" in schema:
        assert value <= schema["maximum"], f"{where}: above maximum"


# =============================================================================

class TurnOrder(unittest.TestCase):
    def test_round_robin_with_the_world_running_between_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, world, clock = make_engine(tmp)
            res = eng.run(9)
            self.assertEqual([r["speaker"] for r in res], list("ABCDABCDA"))
            self.assertTrue(all(r["accepted"] for r in res))
            self.assertEqual(world.n, 2 + 9 * 3)                       # warm-up, then 3 s = 3 steps per turn
            self.assertEqual([r["world_steps_before"] for r in res], [3] * 9)
            self.assertEqual([r["numbers"]["window_steps"] for r in res[4:8]], [12] * 4)   # 4 slots x 3 steps
            self.assertEqual([c["name"] for c in eng.model.calls], list("ABCDABCDA"))

    def test_a_dropped_turn_passes_to_the_next_fly_after_two_new_samples(self):
        def script(name, k):
            if name == "B":
                return ("B would buy it.", 6, True)                   # financial, every time
            return (f"{name} reads 3 Hz.", 5, True)
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(script)
            eng, _, _ = make_engine(tmp, model=model)
            res = eng.run(4)
            self.assertEqual([r["speaker"] for r in res], list("ABCD"))
            self.assertEqual([r["accepted"] for r in res], [True, False, True, True])
            self.assertEqual([c["name"] for c in model.calls], ["A", "B", "B", "B", "C", "D"])
            seeds = [c["seed"] for c in model.calls if c["name"] == "B"]
            self.assertEqual(len(set(seeds)), 3)                        # a new seed per sample
            self.assertEqual(seeds, [E.turn_seed(eng.seed, 2, a) for a in range(3)])
            self.assertEqual(eng.drops, {"financial": 3})
            # C's prompt carries A's line only: B had none
            self.assertIn("A: A reads 3 Hz.", model.calls[4]["messages"][1]["content"])
            self.assertNotIn("buy", model.calls[4]["messages"][1]["content"])
            self.assertEqual(res[2]["payload"]["to"], "A")

    def test_the_retry_cap_is_two(self):
        with self.assertRaises(ValueError):
            E.Engine(FakeWorld(), FakeModel(), safety=SAFETY, retries_on_drop=3)


class Prompt(unittest.TestCase):
    def test_prompt_is_system_frame_topic_hints_and_history_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, _, _ = make_engine(tmp)
            res = eng.run(11)
            calls = eng.model.calls
            for k, (call, r) in enumerate(zip(calls, res)):
                sys_msg, user = call["messages"]
                self.assertEqual(sys_msg, {"role": "system", "content": P.SYSTEM_PROMPT})
                self.assertEqual(len(call["messages"]), 2)
                name, pl = r["speaker"], r["payload"]
                lines = user["content"].split("\n")
                hist = [(x["speaker"], x["text"]) for x in res[max(0, k - E.HISTORY):k]]
                to = hist[-1][0] if hist and hist[-1][0] != name else None
                switched_lines = pl["topic_id"] > 1 and pl["topic_turn"] <= 2
                n_shown = (min(P.PROMPT_LINES, max(P.SWITCH_LINES, pl["topic_turn"] - 1)) if pl["topic_id"] > 1
                           else P.PROMPT_LINES)
                shown = hist[-n_shown:]
                note = P.topic_note(pl["topic_id"], pl["topic_turn"] == 1)
                answer = P.ANSWER.format(to=to) if to and hist[-1][1].endswith("?") else ""
                focus = P.FOCUS.format(topic=pl["topic"]) if switched_lines else ""
                expect = [P.TOPIC_LINE.format(topic=pl["topic"]), note, "", P.HINTS_HEADER.format(name=name)] + \
                         ([f"- {h}" for h in r["hints"]] or [P.NO_HINTS]) + \
                         ["", P.TRANSCRIPT_HEADER] + ([f"{s}: {t}" for s, t in shown] or [P.NO_LINES]) + \
                         ["", P.CLOSING.format(name=name, reply=P.REPLY.format(to=to) if to else "", answer=answer,
                                               move=r["move"], focus=focus,
                                               opener=r["opener"].format(to=to) if r["opener"] else "")]
                self.assertEqual(lines, expect)
                self.assertEqual(r["move"], P.pick_move(E.turn_seed(eng.seed, r["turn"], E.MOVE_ATTEMPT), to))
                earlier = [x["opener"] for x in res[max(0, k - P.OPENER_MEMORY):k]]
                self.assertEqual(r["opener"], P.pick_opener(E.turn_seed(eng.seed, r["turn"], E.OPENER_ATTEMPT), to,
                                                            earlier))
                self.assertTrue(r["opener"] == "" or r["opener"] not in earlier)
                self.assertTrue(R.HINTS_MIN <= len(r["hints"]) <= R.HINTS_MAX, r["hints"])
                self.assertEqual(pl["basis"], r["basis"])                    # the numeric readout is posted
                self.assertEqual(pl.get("to"), to)
                for b in r["basis"]:
                    if b not in r["hints"]:
                        self.assertNotIn(f"- {b}", lines)                    # the model is shown hints, not the basis
            self.assertEqual(res[0]["payload"]["topic"], "the basil question")
            self.assertIn(P.TOPIC_OPEN, calls[0]["messages"][1]["content"])
            self.assertIn(P.TOPIC_KEEP, calls[1]["messages"][1]["content"])
            self.assertEqual(P.PROMPT_LINES, 6)
            self.assertEqual(len([l for l in calls[10]["messages"][1]["content"].split("\n")
                                  if re.match(r"^[ABCD]: ", l)]), 4)   # topic 2's own four lines, not six
            self.assertEqual(res[T.TOPIC_MAX]["payload"]["topic_turn"], 1)          # a switched topic's first line
            new_topic_user = calls[T.TOPIC_MAX]["messages"][1]["content"]
            self.assertEqual(len([l for l in new_topic_user.split("\n") if re.match(r"^[ABCD]: ", l)]), P.SWITCH_LINES)
            self.assertTrue(new_topic_user.endswith(P.FOCUS.format(topic=res[T.TOPIC_MAX]["payload"]["topic"])))
            self.assertEqual(len(eng.history), E.HISTORY)                # eight kept for the repeat check

    def test_readout_strings_that_fail_the_filter_are_left_out_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, _, _ = make_engine(tmp)
            real_take = eng.readout.take

            def take(name):
                basis, hints, numbers = real_take(name)
                return (basis + ["MDN 83 Hz (backward walking descending neurons (moonwalker))", "x" * 301],
                        hints + ["the price of sugar is high"], numbers)
            eng.readout.take = take
            r = eng.run(1)[0]
            self.assertNotIn("moonwalker", eng.model.calls[0]["messages"][1]["content"])
            self.assertNotIn("price", eng.model.calls[0]["messages"][1]["content"])
            self.assertFalse(any("moonwalker" in b for b in r["payload"]["basis"]))
            self.assertEqual(eng.basis_drops, {"financial": 2, "basis_length": 1})
            rows = [x for x in archive_rows(tmp) if x["kind"] == "dropped_basis"]
            self.assertEqual(sorted(x["reason"] for x in rows), ["basis_length", "financial"])
            self.assertEqual([x["text"] for x in archive_rows(tmp) if x["kind"] == "dropped_hint"],
                             ["the price of sugar is high"])

    def test_system_prompt_rules(self):
        s = P.SYSTEM_PROMPT
        for must in ("four curious flies", "endless yellow room", "one or two short casual sentences",
                     "Stay on the current topic unless you are told the topic has changed",
                     "Reply to the last fly who spoke", "ask a question or disagree", "use them or ignore them",
                     "Do not start every line with I", "unless you copy its number exactly from your hints",
                     "humans", "Use plain words", "real named people, brands, politics, elections, religion",
                     "money or trading, medical claims, hate, sexual content or how to harm anyone"):
            self.assertIn(must, s)
        self.assertNotIn("Do not mention people", s)                    # humans are fine now
        # the instruction is published by the page, not posted as a line; its filter
        # hits are the safety list's own words
        self.assertIsNotNone(SAFETY.check(s, max_chars=10_000))
        clean = s.replace("money or trading, ", "").replace("sexual content ", "")
        self.assertIsNone(SAFETY.check(clean, max_chars=10_000))


class Normalisation(unittest.TestCase):
    def test_each_rule(self):
        n = P.normalise
        self.assertEqual(n("‘a’ ‚b‛", "A"), "'a' 'b'")
        self.assertEqual(n("“c” „d‟", "A"), '"c" "d"')
        for d in "‐‑‒–":
            self.assertEqual(n(f"left{d}right", "A"), "left-right")
        for d in "—―":
            self.assertEqual(n(f"dream{d}maybe", "A"), "dream - maybe")
            self.assertEqual(n(f"dream {d} maybe", "A"), "dream - maybe")
        self.assertEqual(n("so… on", "A"), "so... on")
        self.assertEqual(n("  \n\tspaced out \n ", "A"), "spaced out")
        self.assertEqual(n("much.  \nWhat do you think?", "A"), "much. What do you think?")
        self.assertEqual(n("A: my line", "A"), "my line")
        self.assertEqual(n("Fly A : my line", "A"), "my line")
        self.assertEqual(n("fly A:my line", "A"), "my line")
        self.assertEqual(n("A: A: twice", "A"), "A: twice")            # one label only
        self.assertEqual(n("B: not mine", "A"), "B: not mine")         # another fly's label stays
        self.assertEqual(n("A is near: yes", "A"), "A is near: yes")   # not a label
        self.assertEqual(n("—A: x", "A"), "- A: x")               # the label must lead

    def test_no_word_changes(self):
        rng = np.random.default_rng(3)
        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;!?'\"()-")
        for _ in range(500):
            s = "".join(rng.choice(alphabet, size=int(rng.integers(1, 80))))
            out = P.normalise(s, "Q")
            self.assertEqual(out, re.sub(r"\s+", " ", s).strip())

    def test_the_list_is_disclosed_in_the_engine_docstring(self):
        doc = E.__doc__
        for phrase in ("curly quotes", "straight", "dashes", "ellipsis", "trimmed", "<fly name>:",
                       "Content words are never changed"):
            self.assertIn(phrase, doc)
        self.assertEqual(len(P.NORMALISATION), 5)


class Filter(unittest.TestCase):
    def test_drops_are_counted_by_reason_archived_and_never_posted(self):
        replies = iter([
            ("I see B near me.", 6, True),                      # A ok
            ("Check www.example.com now.", 6, True),            # B link
            ("I would kill for that song.", 6, True),           # B threat
            ("“Fine” — B is 4 mm away…", 6, True),   # B ok after normalisation
            ("Half a sentence that never", 60, False),          # C unfinished
            ("It keeps getting cut off and", 60, False),        # C unfinished
            ("still cut", 60, False),                           # C unfinished -> C has no line
            ("D here, P1 is quiet.", 6, True),                  # D ok
        ])
        with tempfile.TemporaryDirectory() as tmp:
            relay = FakeRelay()
            eng, _, _ = make_engine(tmp, model=FakeModel(lambda name, k: next(replies)), dry=False, relay=relay)
            res = eng.run(4)
            self.assertEqual([r["accepted"] for r in res], [True, True, False, True])
            self.assertEqual(eng.drops, {"link": 1, "threat": 1, "unfinished": 3})
            self.assertEqual([p["text"] for p in relay.payloads],
                             ["I see B near me.", '"Fine" - B is 4 mm away...', "D here, P1 is quiet."])
            dropped = [x for x in archive_rows(tmp) if x["kind"] == "dropped"]
            self.assertEqual([(x["speaker"], x["reason"], x["text"]) for x in dropped], [
                ("B", "link", "Check www.example.com now."),
                ("B", "threat", "I would kill for that song."),
                ("C", "unfinished", "Half a sentence that never"),
                ("C", "unfinished", "It keeps getting cut off and"),
                ("C", "unfinished", "still cut")])
            accepted = [x for x in archive_rows(tmp) if x["kind"] == "accepted"]
            self.assertEqual(len(accepted), 3)
            self.assertEqual(accepted[1]["raw"], "“Fine” — B is 4 mm away…")
            self.assertTrue(all(x["post"]["ok"] for x in accepted))
            self.assertEqual(eng.counts()["posted"], 3)

    def test_a_line_the_relay_refuses_is_not_added_to_the_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            relay = FakeRelay(fail={2})
            eng, _, _ = make_engine(tmp, dry=False, relay=relay)
            eng.run(3)
            self.assertEqual([s for s, _ in eng.history], ["A", "C"])
            self.assertEqual(eng.post_failures, 1)
            self.assertEqual([p["engine_seq"] for p in relay.payloads], [1, 2, 3])

    def test_posting_needs_a_relay(self):
        with self.assertRaises(ValueError):
            E.Engine(FakeWorld(), FakeModel(), safety=SAFETY, dry=False)


class Schema(unittest.TestCase):
    def test_engine_payloads_match_turn_schema_md(self):
        schema = schema_from_md()
        with tempfile.TemporaryDirectory() as tmp:
            relay = FakeRelay()
            eng, _, _ = make_engine(tmp, dry=False, relay=relay)
            eng.run(6)
            self.assertEqual(len(relay.payloads), 6)
            for p in relay.payloads:
                validate(p, schema)
                self.assertEqual(list(p["meta"]), list(E.META_KEYS))
                self.assertTrue(set(E.TOPIC_KEYS) <= set(p))
                self.assertEqual(p["fly_index"], NAMES.index(p["speaker"]))
            self.assertNotIn("to", relay.payloads[0])
            self.assertEqual(relay.payloads[1]["to"], "A")
            self.assertEqual(set(schema["properties"]["meta"]["required"]), set(E.META_KEYS))
            self.assertLessEqual(len(E.META_KEYS), 12)                 # the relay's meta cap
            # A readout string the relay would refuse is left out by the engine, not sent.
            self.assertEqual(schema["properties"]["basis"]["items"]["maxLength"], E.BASIS_ITEM_MAX_CHARS)
            self.assertEqual(schema["properties"]["basis"]["maxItems"], R.BASIS_MAX)

    def test_the_checker_rejects_what_the_schema_forbids(self):
        schema = schema_from_md()
        good = E.make_payload("A", "B", "a line", 1, 0, ["x"], {k: (1 if k in ("window_steps", "seed", "new_tokens") else 0.5)
                                                                for k in E.META_KEYS},
                              {"topic": "the light", "topic_id": 2, "topic_turn": 1, "switched": True})
        validate(good, schema)
        bad = [dict(good, extra=1), dict(good, basis=["x"] * 7), dict(good, speaker="E"),
               dict(good, topic=""), dict(good, topic="x" * 121), dict(good, topic_id=0), dict(good, topic_turn=0),
               dict(good, switched=1), {k: v for k, v in good.items() if k != "topic"},
               dict(good, fly_index=4), dict(good, text=""), dict(good, meta=dict(good["meta"], seed=True)),
               dict(good, meta=dict(good["meta"], song_hz=float("nan"))),
               dict(good, meta={k: v for k, v in good["meta"].items() if k != "seed"})]
        for b in bad:
            with self.assertRaises(AssertionError):
                validate(b, schema)


class RelayRetry(unittest.TestCase):
    def serve(self, statuses):
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append({"auth": self.headers.get("Authorization"), "ctype": self.headers.get("Content-Type"),
                             "body": json.loads(body), "path": self.path})
                code = statuses[min(len(seen) - 1, len(statuses) - 1)]
                out = json.dumps({"ok": code == 200, "seq": len(seen)}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}", seen

    def test_5xx_is_retried_with_backoff(self):
        url, seen = self.serve([503, 502, 200])
        waits = []
        c = RC.RelayClient(url, "t" * 40, sleep=waits.append, say=lambda *a: None)
        r = c.post_turn({"speaker": "A", "text": "hi"})
        self.assertTrue(r["ok"])
        self.assertEqual((r["attempts"], r["status"]), (3, 200))
        self.assertEqual(waits, [RC.BACKOFF_S, RC.BACKOFF_S * 2])
        self.assertEqual([s["path"] for s in seen], ["/turn"] * 3)
        self.assertEqual(seen[0]["auth"], "Bearer " + "t" * 40)
        self.assertEqual(seen[0]["ctype"], "application/json")
        self.assertEqual(seen[2]["body"], {"speaker": "A", "text": "hi"})
        self.assertNotIn("t" * 40, repr(c))

    def test_4xx_is_not_retried(self):
        url, seen = self.serve([422])
        c = RC.RelayClient(url, "t" * 40, sleep=lambda s: None, say=lambda *a: None)
        r = c.post_turn({"speaker": "A", "text": "hi"})
        self.assertFalse(r["ok"])
        self.assertEqual((r["attempts"], r["status"], len(seen)), (1, 422, 1))

    def test_429_is_retried_and_gives_up_after_the_cap(self):
        url, seen = self.serve([429])
        waits = []
        c = RC.RelayClient(url, "t" * 40, retries=3, sleep=waits.append, say=lambda *a: None)
        r = c.post_turn({"speaker": "A", "text": "hi"})
        self.assertFalse(r["ok"])
        self.assertEqual((r["attempts"], len(seen)), (4, 4))
        self.assertEqual(waits, [2.0, 4.0, 8.0])

    def test_a_refused_connection_is_retried(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        port = srv.server_address[1]
        srv.server_close()                                   # nothing listens there now
        waits = []
        c = RC.RelayClient(f"http://127.0.0.1:{port}", "t" * 40, retries=2, timeout_s=2,
                           sleep=waits.append, say=lambda *a: None)
        r = c.post_turn({"speaker": "A", "text": "hi"})
        self.assertFalse(r["ok"])
        self.assertEqual(r["attempts"], 3)
        self.assertEqual(len(waits), 2)
        self.assertIsNone(r["status"])

    def test_env_helpers_append_only_a_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".env"
            p.write_bytes(b"RELAY_TOKEN=abc")
            self.assertTrue(RC.ensure_env_key(p, "RELAY_URL", "https://x.example"))
            self.assertFalse(RC.ensure_env_key(p, "RELAY_URL", "https://y.example"))
            self.assertEqual(RC.read_env(p), {"RELAY_TOKEN": "abc", "RELAY_URL": "https://x.example"})
            self.assertEqual(p.read_text(), "RELAY_TOKEN=abc\nRELAY_URL=https://x.example\n")


# =============================================================================
# the four-fly room and the readout, on backrooms_world's fake brain
# =============================================================================

class FakeBatchBrain(FakeBrain):
    """FakeBrain plus run_batch: each column is FakeBrain.run with its own drive, seed and state."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.batches = []

    def run_batch(self, drives, steps, gains=None, record=None, seeds=None, spike_log=False,
                  states=None, random_sources=None):
        self.batches.append(len(drives))
        return [FakeBrain.run(self, d, steps, gains=gains, record=record, seed=s, state=st)
                for d, s, st in zip(drives, seeds, states)]


def fake_room(cls=FakeBatchBrain, seed=3, start=None, spont=None):
    fb = cls(spont=spont)
    eye = FakeEye(fb)
    groups = bd.present_groups(fb)
    motor = bw.motor_groups(fb, fake_sides(fb))
    arena = bw.Arena(seed=seed, names=NAMES, start=start) if start else None
    return W.TalkRoom(fb, eye, groups, motor, seed=seed, arena=arena)


def strip(rec):
    rec = json.loads(json.dumps(rec))
    rec.pop("step_s")
    return rec


class FourFlyRoom(unittest.TestCase):
    CORNERS = [(0.5, 0.5, 0.0), (5.0, 0.5, 0.0), (19.5, 19.5, 0.0), (19.5, 15.0, 0.0)]

    def test_one_batched_call_per_step_equals_stepping_in_turn(self):
        ps1 = FakeBrain().where(type_re="^ps1 MN$")
        spont = {26: 120.0, 27: 60.0, 28: 200.0, int(ps1[0]): 150.0}
        a, b = fake_room(FakeBatchBrain, spont=spont), fake_room(FakeBrain, spont=spont)
        ra = [strip(a.step()) for _ in range(5)]
        rb = [strip(b.step()) for _ in range(5)]
        self.assertEqual(ra, rb)
        self.assertEqual(a.fb.batches, [4] * 5)
        self.assertEqual({n: body.seed for n, body in a.bodies.items()}, {"A": 13, "B": 14, "C": 15, "D": 16})
        self.assertEqual(sorted(ra[0]["flies"]), list(NAMES))
        self.assertTrue(all(ra[4]["flies"][n]["state_carried"] for n in NAMES))
        self.assertNotEqual(ra[0]["flies"]["A"]["x"], ra[4]["flies"]["A"]["x"])     # the DN drive walks them

    def test_each_fly_hears_and_smells_the_other_three(self):
        ps1 = FakeBrain().where(type_re="^ps1 MN$")
        room = fake_room(start=[(8.0, 10.0, 0.0), (10.0, 10.0, 0.0), (12.0, 10.0, 0.0), (10.0, 13.0, 0.0)],
                         spont={int(ps1[0]): 120.0})
        r1 = room.step()
        self.assertEqual({n: r1["flies"][n]["drive"]["JO_A"] for n in NAMES}, {n: 0.0 for n in NAMES})
        fl = room.arena.by_name
        ch = room.channels
        r2 = room.step()                                       # nobody moves: no DN drive
        for n in NAMES:
            me = fl[n]
            d = [bw.distance_mm(me, fl[o]) for o in NAMES if o != n]
            self.assertAlmostEqual(r2["flies"][n]["drive"]["JO_A"], ch.sound_hz_many([(120.0, x) for x in d]))
            self.assertAlmostEqual(r2["flies"][n]["drive"]["ORN_DA1"], ch.smell_hz_many(d))
            self.assertEqual(r2["flies"][n]["drive"]["ORN_DA1"], 200.0)       # three near flies: clipped
            self.assertEqual(r2["flies"][n]["drive_from"], {})                # several sources: none named
            self.assertEqual(sorted(r2["flies"][n]["others"]), sorted(o for o in NAMES if o != n))
            self.assertAlmostEqual(r2["flies"][n]["active_fraction"], r2["flies"][n]["fired"] / room.fb.n)
        # the eye sees more than one fly: the frame holds every other fly
        me = fl["B"]
        img = ch.sight_frame_many(me, [fl["A"], fl["C"], fl["D"]])
        self.assertGreater((img < bw.GROUND_GREY).sum(), (ch.sight_frame(me, fl["A"]) < bw.GROUND_GREY).sum())

    def test_a_single_source_is_named_for_the_captioner(self):
        room = fake_room(start=self.CORNERS)
        r = room.step()
        self.assertEqual(r["flies"]["A"]["drive_from"], {"ORN_DA1": "B"})
        self.assertEqual(r["flies"]["C"]["drive_from"], {"ORN_DA1": "D"})
        self.assertEqual(r["flies"]["A"]["drive"]["ORN_DA1"], room.channels.smell_hz(4.5))

    def test_readout_on_the_room(self):
        room = fake_room(spont={26: 120.0, 27: 60.0})
        ro = R.Readout(room.sizes, NAMES, dt=room.dt)
        for _ in range(20):
            ro.update(room.step())
        ro.settle()
        for _ in range(10):
            ro.update(room.step())
        basis, hints, numbers = ro.take("A")
        self.assertLessEqual(len(basis), R.BASIS_MAX)
        self.assertTrue(R.HINTS_MIN <= len(hints) <= R.HINTS_MAX)
        for h in hints:
            self.assertIsNone(SAFETY.check(h), h)
        self.assertEqual(sum(b.startswith("walked ") for b in basis), 1)
        self.assertEqual(len([b for b in basis if re.match(r"^[BCD] is \d+\.\d mm away, ", b)]), 3)
        self.assertEqual(numbers["window_steps"], 10)
        for s in basis:
            self.assertIsNone(SAFETY.check(s), s)
        with self.assertRaises(ValueError):
            ro.take("A")                                       # its window is closed and empty


class ReadoutTemplatesPassTheFilter(unittest.TestCase):
    """
    A readout string the filter drops is lost to the model, so every template
    is checked over the range of its numbers. The first dry run lost one:
    "D 13.6 mm away" folds to a filtered word, hence "D is 13.6 mm away".
    """

    def test_every_template_over_its_numbers(self):
        sizes = {k: 5 for k, e in bd.DICTIONARY.items() if e["present"]}
        ro = R.Readout(sizes, NAMES)
        bad = []
        for d10 in range(0, 290):
            t = ro.walked_line(d10 / 7.0, d10 / 13.0)
            if SAFETY.check(t):
                bad.append(t)
            for b in (0.1, 7.0, -13.0, 45.0, 135.0, -179.0):
                for o in NAMES:
                    t = ro.other_line(o, {"distance": d10 / 10.0, "bearing": b})
                    if SAFETY.check(t):
                        bad.append(t)
        for k in ro.keys:
            for hz in range(0, 520):
                for base in (None, float(hz // 3)):
                    t = ro.group_line(k, float(hz), base)
                    if SAFETY.check(t):
                        bad.append(t)
        for song in range(0, 4000):
            t = ro.song_line(float(song), float(song % 201), float((song * 7) % 201))
            if SAFETY.check(t):
                bad.append(t)
        hints = set()
        for k in ro.keys:
            for score in (-2.0, -0.1, 0.3, 1.5, 9.0):
                hints.add(ro.group_hint(k, score))
        for change in (-0.5, -0.05, 0.0, 0.05, 0.5):
            hints.add(ro._first(R.BRAIN_WORDS, change))
        hints.update(R.SENSE_WORDS.values())
        for d10 in range(0, 290):
            hints.add(ro.walk_hint(d10 / 7.0, 2.0))
            for b in (0.1, 45.0, -45.0, 170.0):
                for o in NAMES:
                    hints.add(ro.nearest_hint(o, {"distance": d10 / 10.0, "bearing": b}))
        bad += [h for h in sorted(hints) if SAFETY.check(h) or G.check(h + ".", [h])]
        self.assertEqual(bad[:5], [])
        self.assertTrue(set(R.ROLE_WORDS) >= set(ro.keys))                 # every captioned group has a role word


class ReadoutRanking(unittest.TestCase):
    def test_the_group_that_rose_most_above_its_baseline_leads(self):
        world = FakeWorld()
        ro = R.Readout(world.sizes, NAMES, dt=world.dt)
        for _ in range(20):
            ro.update(world.step())
        ro.settle()
        for _ in range(15):                                    # P1 jumps to 40 Hz at step 30
            ro.update(world.step())
        basis, hints, numbers = ro.take("A")
        top_b = ro.top_groups("B")
        # steps 21-35 in A's window, P1 at 40 Hz on steps 30-35: 16 Hz against a 0 Hz baseline
        self.assertEqual(basis[0], "P1 neurons: 16 Hz now, baseline 0 Hz")
        self.assertAlmostEqual(numbers["top_group_rate_hz"], 40.0 * 6 / 15)
        self.assertEqual(numbers["top_group_baseline_hz"], 0.0)
        self.assertEqual(numbers["nearest_fly_mm"], 4.0)
        self.assertEqual(numbers["window_steps"], 15)
        self.assertEqual(basis[2:], ["walked 1.5 mm in the last 0.8 s", "B is 4.0 mm away, straight ahead",
                                     "C is 8.0 mm away, straight ahead", "D is 12.0 mm away, straight ahead"])
        self.assertEqual(len(basis), R.BASIS_MAX)
        self.assertFalse(any("song level" in b for b in basis))        # the song level goes to meta only
        self.assertAlmostEqual(numbers["song_hz"], 900.0)
        # B's rates never changed: every score is 0 and the tie goes to dictionary order
        self.assertEqual([k for k, *_ in top_b], ["ORN_DA1", "LC10a"])
        self.assertEqual([s for *_, s in top_b], [0.0, 0.0])
        # hints: P1 rose far above its 0 Hz baseline (score 16 / floor); no previous window yet
        self.assertEqual(hints, ["your courtship neurons are firing far more than usual",
                                 "B is 4.0 mm away, straight ahead, the closest fly",
                                 "you walked 1.5 mm since your last turn"])
        self.assertEqual(ro.sense_hint((150.0, 12.0), 150.0, 12.0), None)
        self.assertEqual(ro.sense_hint((200.0, 150.0), 190.0, 60.0), "the wing song you hear got quieter")
        self.assertEqual(ro.sense_hint((100.0, 150.0), 190.0, 160.0), "the smell of the other flies got stronger")
        self.assertIsNone(numbers["brain_change"])
        self.assertEqual(R.bearing_words(35), "35 deg to the left")
        self.assertEqual(R.bearing_words(-120.4), "120 deg to the right")

    def test_group_names_ending_in_neurons_get_no_second_neurons(self):
        ro = R.Readout({"wing_mn_all": 5, "song_sine_hg1": 1}, NAMES)
        self.assertEqual(ro.group_line("wing_mn_all", 12.0, 3.0), "wing motor neurons: 12 Hz now, baseline 3 Hz")
        self.assertEqual(ro.group_line("song_sine_hg1", 12.0, None), "hg1 MN neurons: 12 Hz now, no baseline yet")

    def test_captioner_lines_are_not_shown(self):
        world = FakeWorld()
        ro = R.Readout(world.sizes, NAMES, dt=world.dt)
        for _ in range(29):
            ro.update(world.step())
        ro.settle()
        new = []
        for _ in range(10):
            new += ro.update(world.step())
        basis, _, _ = ro.take("A")
        a_lines = [ln["text"] for ln in new if ln["fly"] == "A"]
        self.assertTrue(a_lines)                               # the captioner still runs
        self.assertEqual(R.EVENT_LINES, 0)
        self.assertFalse(set(a_lines) & set(basis))
        self.assertFalse(any(s.startswith("A  ") for s in basis))


class Grounding(unittest.TestCase):
    HINTS = ["your courtship neurons are firing much more than usual", "your whole brain got a little busier",
             "B is 4.0 mm away, to your left, the closest fly", "you hear faint wing song",
             "you walked 14.6 mm since your last turn"]

    def test_free_talk_is_kept(self):
        for t in ("Do you think the humans ever notice us?",
                  "I walked 14.6 mm and the walls still look the same. Why is everything yellow?",
                  "No, B, the light is not warmer here.",
                  "B is 4.0 mm away from me, and I can hear a little wing song.",
                  "Being small means every crumb is a feast!",
                  "Fly B, what do you smell right now?",
                  "I disagree... the hum is coming from the ceiling.",
                  "I'm not sure I agree with D-that helping might matter more.",       # a dash normalised to "-"
                  "I can't deny it, A-there definitely feels different lately."):
            self.assertIsNone(G.check(t, self.HINTS), t)

    def test_each_reason(self):
        cases = {
            "I heard Elon talking about the room.": "name",
            "My friend Gerald lives by the window.": "name",
            "The NASA people would love this room.": "name",
            "Maybe the humans are drinking pepsi again.": "name",
            "Do flies vote in elections?": "politics",
            "Maybe the light is heaven.": "religion",
            "Our wings could cure a disease.": "medical",
            "Those racist humans again.": "hate",
            "Here is how to make poison from sugar.": "harm_instructions",
            "One. Two. Three. Four sentences is too many.": "too_long",
            "a" * 250 + ".": "too_long",
            "I think the light is": "unfinished",
            "I walked 12.0 mm today.": "number",
            "My brain runs at 40 Hz.": "number",
            "C is 4.0 mm away.": "number",                                 # B's distance given to C
            "It took 3 seconds to cross.": "number",
            "The walls beat faster when I walk closer, about half a centimeter.": "number",
            "You are five millimeters from me.": "number",
            "I walked 14.6 millimeters and a few inches more.": "number",
        }
        for t, why in cases.items():
            self.assertEqual(G.check(t, self.HINTS), why, t)

    def test_repeats_over_the_last_lines(self):
        recent = ["Do you think the humans ever notice us?"]
        self.assertEqual(G.check("Do you think the humans ever notice us?", self.HINTS, recent), "repeat")
        self.assertEqual(G.check("Do you think humans ever notice us?", self.HINTS, recent), "repeat")
        self.assertIsNone(G.check("The humans never notice anything, I think.", self.HINTS, recent))

    def test_short_third_sentence_is_kept(self):
        self.assertIsNone(G.check("I disagree. It is not a language. What about you, B?", self.HINTS))

    def test_echoes_same_openings_and_off_topic(self):
        recent = ["I feel my neurons buzzing, just like the room's pulse.",
                  "Maybe the walls are connected to something bigger, beyond our senses.",
                  "The walls are talking in a language only we can hear, our thoughts sync up."]
        line = "Really sensing that pulse now, like the walls whisper. Our feelings sync up."
        self.assertEqual(G.echoes(line, recent, "the rhythm of the walls breathing"), {"puls", "sens", "sync"})
        self.assertEqual(G.check(line, self.HINTS, recent, topic="the rhythm of the walls breathing", topic_slot=5),
                         "repeat")
        self.assertIsNone(G.check("Humans build rooms like this one, do they not?", self.HINTS, recent,
                                  topic="the rhythm of the walls breathing", topic_slot=5))
        self.assertEqual(G.check("I think it hums.", self.HINTS, ["I think the light is warm."]), "same_opening")
        loop = ["Why does the distance clue matter?", "Well, the distance feels important.", "The distance is odd."]
        self.assertEqual(G.motifs("Hmm, that distance again.", loop), {"distanc"})
        self.assertEqual(G.check("Hmm, that distance again.", self.HINTS, loop), "repeat")
        self.assertIsNone(G.check("Hmm, that distance again.", self.HINTS, loop[1:]))
        self.assertEqual(G.check("Fly C, I agree. Fly D, that is a good guess.", self.HINTS, speaker="D"), "self_name")
        self.assertIsNone(G.check("Fly C, I agree with you.", self.HINTS, speaker="D"))
        self.assertIsNone(G.check("I wonder if it hums.", self.HINTS, ["I think the light is warm."]))
        topic = "the color of shadows shifting on the floor"
        self.assertEqual(G.check("Listening to the walls' pulse, it gets louder.", self.HINTS, [], topic=topic,
                                 topic_slot=1), "off_topic")
        self.assertIsNone(G.check("My shadow looks bigger on this floor.", self.HINTS, [], topic=topic, topic_slot=2))
        self.assertIsNone(G.check("Listening to the walls' pulse, it gets louder.", self.HINTS, [], topic=topic,
                                  topic_slot=3))

    def test_numbers(self):
        self.assertEqual(G.quantities("LC10a at 3 Hz, 125-degree turn, 2.2 seconds, 8 cells, 50%"),
                         {(3.0, "hz"), (125.0, "deg"), (2.2, "s"), (8.0, ""), (50.0, "%")})
        self.assertIsNone(G.check("There are 4 of us and 6 legs each.", self.HINTS))   # bare numbers are fine
        self.assertIsNone(G.check("I walked 14.6 mm.", self.HINTS))
        self.assertEqual(G.check("I walked 14.6 cm.", self.HINTS), "number")

    def test_proper_names(self):
        self.assertEqual(G.proper_names("Hello. Why is it yellow? I'm not sure, B."), [])
        self.assertEqual(G.proper_names("I think Paris is far."), ["Paris"])
        self.assertEqual(G.content_reason("the walls of the room"), None)

    def test_the_engine_drops_by_reason_and_never_edits_or_keeps_them(self):
        def script(name, k):
            if name == "A":
                return ("Let us talk about politics.", 9, True)
            return ("I walked 99.9 mm in the last 1.0 s.", 12, True) if k % 2 else (
                "Fly A, I wonder what the humans are doing today.", 8, True)
        with tempfile.TemporaryDirectory() as tmp:
            relay = FakeRelay()
            eng, _, _ = make_engine(tmp, model=FakeModel(script, topic=lambda k: ("what humans do", 3, True)),
                                    dry=False, relay=relay, ground=G.check)
            res = eng.run(4)
            self.assertFalse(res[0]["accepted"])
            self.assertEqual(eng.drops["politics"], 3)
            self.assertTrue(set(eng.drops) <= set(G.REASONS))
            self.assertTrue(relay.payloads)
            for p in relay.payloads:
                self.assertIsNone(G.check(p["text"], []))
            self.assertNotIn("politics", " ".join(t for _, t in eng.history))
            dropped = [x for x in archive_rows(tmp) if x["kind"] == "dropped"]
            self.assertTrue(all(x["reason"] in G.REASONS for x in dropped))
            self.assertEqual(dropped[0]["text"], "Let us talk about politics.")


class Topics(unittest.TestCase):
    def test_switch_rule(self):
        tp = T.Topics(seed=1)
        self.assertEqual(tp.switch_reason(None), "opening")
        tp.set("the hum", "opening")
        self.assertEqual(tp.fields(), {"topic": "the hum", "topic_id": 1, "topic_turn": 1, "switched": False})
        self.assertIsNone(tp.switch_reason(0.5))                       # a jump before TOPIC_MIN lines
        for _ in range(T.TOPIC_MIN):
            tp.accepted()
        self.assertIsNone(tp.switch_reason(T.SWITCH_JUMP * 0.99))
        self.assertIsNone(tp.switch_reason(None))
        self.assertEqual(tp.switch_reason(T.SWITCH_JUMP), "brain_jump")
        self.assertEqual(tp.switch_reason(-T.SWITCH_JUMP), "brain_jump")   # calmer counts too
        for _ in range(T.TOPIC_MAX - T.TOPIC_MIN):
            tp.accepted()
        self.assertEqual(tp.switch_reason(0.0), "topic_max")
        tp.set("sleep", "topic_max")
        self.assertEqual(tp.slots, 0)
        for k in range(T.TOPIC_MAX_SLOTS - 1):
            self.assertEqual(tp.slot(), k + 1)
            self.assertIsNone(tp.switch_reason(0.0))                   # no line yet: the slot cap decides
        tp.slot()
        self.assertEqual(tp.switch_reason(0.0), "topic_max")
        tp.slots = 0
        self.assertEqual(tp.fields(), {"topic": "sleep", "topic_id": 2, "topic_turn": 1, "switched": True})
        tp.accepted()
        self.assertEqual(tp.fields()["switched"], False)
        self.assertTrue(6 <= T.TOPIC_MAX <= 10)
        self.assertIn(tp.theme, T.SEED_THEMES)
        self.assertEqual(T.opening_theme(1), T.opening_theme(1))

    def test_topic_cleaning_and_checks(self):
        self.assertEqual(T.clean_topic(' Topic: "why the walls hum." '), "why the walls hum")
        self.assertEqual(T.clean_topic("the smell of rain"), "the smell of rain")
        self.assertEqual(T.clean_topic("1. the texture of walls  \nthe taste of dew\nthe"), "the texture of walls")
        self.assertEqual(T.first_line("- a\n- b"), ("- a", True))
        openers = {P.pick_opener(k, "B", [P.OPENERS[2], ""]) for k in range(300)}
        self.assertEqual(openers, set(P.OPENERS) - {P.OPENERS[2]})
        self.assertFalse(any("{to}" in P.pick_opener(k, None) for k in range(100)))
        moves = {P.pick_move(k, "B") for k in range(200)}
        self.assertEqual(moves, {m.format(to="B") for m in P.MOVES})
        self.assertTrue(all("fly" not in P.pick_move(k, None) or "yourself" in P.pick_move(k, None) for k in range(50)))
        recent = ["the smell of rain"]
        cases = {"the smell of rain": "topic_repeat", "The Smell of Rain": "name",
                 "buying sugar cheap": "financial", "a very long topic with far too many words in it": "topic_too_long",
                 "flies and religion": "religion", "": "empty"}
        for t, why in cases.items():
            self.assertEqual(T.check_topic(t, recent, SAFETY), why, t)
        self.assertEqual(T.check_topic("why the light", recent, SAFETY, finished=False), "unfinished")
        self.assertEqual(T.check_topic("walls that breathe in rhythm", ["the rhythm of the walls breathing"], SAFETY),
                         "topic_repeat")
        used = ["the room", "humans", "ideas"]
        cats = {T.next_category(k, used) for k in range(100)}
        self.assertEqual(cats, set(T.CATEGORIES) - set(used))
        self.assertTrue(set(T.THEME_CATEGORY.values()) <= set(T.CATEGORIES))
        self.assertIsNone(T.check_topic("how humans see us", recent, SAFETY))

    def test_the_engine_switches_and_posts_topic_fields(self):
        topics = iter([("the endless yellow walls", 4, True), ("Topic: Elon", 3, True), ("sugar and crumbs", 4, True)])
        with tempfile.TemporaryDirectory() as tmp:
            relay = FakeRelay()
            model = FakeModel(topic=lambda k: next(topics))
            eng, _, _ = make_engine(tmp, model=model, dry=False, relay=relay)
            res = eng.run(T.TOPIC_MAX + 2)
            ps = relay.payloads
            self.assertEqual([(p["topic_id"], p["topic_turn"], p["switched"]) for p in ps],
                             [(1, i + 1, False) for i in range(T.TOPIC_MAX)] + [(2, 1, True), (2, 2, False)])
            self.assertEqual(ps[0]["topic"], "the endless yellow walls")
            self.assertEqual(ps[T.TOPIC_MAX]["topic"], "sugar and crumbs")
            self.assertEqual(eng.topic_drops, {"name": 1})
            self.assertEqual(res[T.TOPIC_MAX]["topic_switch"], {"topic": "sugar and crumbs", "kept": True, "tries": 2,
                                                                "reason": "topic_max"})
            self.assertIn(P.TOPIC_NEW, model.calls[T.TOPIC_MAX]["messages"][1]["content"])
            self.assertIn("the endless yellow walls", model.topic_calls[1]["messages"][1]["content"])   # recent topics
            cat = [x for x in archive_rows(tmp) if x["kind"] == "topic"][1]["category"]
            self.assertIn(T.CATEGORIES[cat], model.topic_calls[1]["messages"][1]["content"])
            self.assertNotEqual(cat, T.THEME_CATEGORY[eng.topics.theme])
            self.assertEqual(model.topic_calls[0]["kw"]["max_new_tokens"], T.TOPIC_MAX_NEW_TOKENS)
            self.assertIn(f"Seed theme: {eng.topics.theme}.", model.topic_calls[0]["messages"][1]["content"])
            rows = archive_rows(tmp)
            self.assertEqual([(x["kind"], x.get("reason")) for x in rows if "topic" in x["kind"]],
                             [("topic", "opening"), ("dropped_topic", "name"), ("topic", "topic_max")])
            self.assertEqual(E.resume_topics(Path(tmp) / "archive.jsonl", eng.mode),
                             {"topic": "sugar and crumbs", "topic_id": 2, "turns": 2,
                              "recent": ["the endless yellow walls", "sugar and crumbs"]})

    def test_a_brain_jump_switches_and_the_opening_falls_back_to_its_theme(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(topic=lambda k: ("buy low", 3, True) if k <= 3 else (f"new topic {k}", 3, True))
            eng, _, _ = make_engine(tmp, model=model)
            real_take = eng.readout.take
            jumps = iter([None, None, None, None, 0.2, 0.0])

            def take(name):
                b, h, n = real_take(name)
                return b, h, dict(n, brain_change=next(jumps))
            eng.readout.take = take
            res = eng.run(6)
            self.assertEqual(res[0]["payload"]["topic"], eng.topics.recent[0])
            self.assertEqual(res[0]["payload"]["topic"], eng.topics.theme)       # three dropped: the seed theme
            self.assertEqual(eng.topic_drops, {"financial": 3})
            self.assertEqual(res[4]["topic_switch"]["reason"], "brain_jump")
            self.assertEqual(res[4]["payload"]["topic"], "new topic 4")
            self.assertTrue(res[4]["payload"]["switched"])
            self.assertIsNone(res[5]["topic_switch"])


class Supervisor(unittest.TestCase):
    def make(self, codes, ups, ram=16.0):
        clock = Clock()
        codes, ups = list(codes), list(ups)
        logs = []

        def spawn():
            clock.now += ups.pop(0)
            return codes.pop(0)
        sup = S.Supervisor("unused", spawn=spawn, sleep=lambda s: None, clock=clock,
                           free_ram=(ram if callable(ram) else (lambda: ram)), min_free_gb=6.0, log=logs.append)
        return sup, logs

    def test_backoff_doubles_resets_after_a_long_run_and_stops_when_finished(self):
        sup, logs = self.make([1, 1, 1, 1, 1, 0], [5, 5, 5, 900, 5, 5])
        self.assertEqual(sup.run(), "finished")
        self.assertEqual(sup.waits, [10.0, 20.0, 40.0, 10.0, 20.0])
        self.assertEqual(sup.starts, 6)

    def test_backoff_is_capped(self):
        sup, _ = self.make([1] * 9, [1] * 9)
        self.assertEqual(sup.run(max_starts=9), "max_starts")
        self.assertEqual(sup.waits[-1], S.BACKOFF_MAX_S)

    def test_low_ram_waits_and_starts_again_when_it_recovers(self):
        rams = iter([16.0, 4.0, 4.0, 16.0, 16.0])
        sup, logs = self.make([1, 0], [1, 1], ram=lambda: next(rams))
        self.assertEqual(sup.run(), "finished")
        self.assertEqual(sup.starts, 2)
        self.assertEqual(sup.ram_waits, 2)
        self.assertTrue(any("free RAM 4.0 GB" in l and "waiting 120 s" in l for l in logs))

    def test_low_ram_stops_only_past_a_wait_cap(self):
        sup, logs = self.make([1], [1], ram=4.0)
        self.assertEqual(sup.run(max_ram_waits=3), "ram")
        self.assertEqual((sup.starts, sup.ram_waits), (0, 3))

    def test_generator_ram_refusal_waits_then_restarts(self):
        sup, logs = self.make([S.EXIT_RAM, 0], [1, 1])
        self.assertEqual(sup.run(), "finished")
        self.assertEqual((sup.starts, sup.ram_waits, sup.waits), (2, 1, []))

    def test_config_error_stops(self):
        sup, _ = self.make([S.EXIT_CONFIG], [1])
        self.assertEqual(sup.run(), "config")

    def test_pid_files_are_written_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def spawn():
                seen["supervise"] = S.read_pid(Path(tmp) / "logs" / S.SUPERVISE_PID)
                return 0
            sup = S.Supervisor(tmp, spawn=spawn, sleep=lambda s: None, free_ram=lambda: 16.0, min_free_gb=6.0,
                               log=lambda l: None, write_pids=True)
            self.assertEqual(sup.run(), "finished")
            import os
            self.assertEqual(seen["supervise"], os.getpid())
            self.assertFalse((Path(tmp) / "logs" / S.SUPERVISE_PID).exists())

    def test_stop_ends_the_supervisor_first_and_skips_a_stale_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            S.write_pid(logs / S.SUPERVISE_PID, 111)
            S.write_pid(logs / S.GENERATOR_PID, 222)
            killed = []
            out = S.stop(tmp, say=lambda *a: None, is_python=lambda pid: pid == 111, kill=killed.append)
            self.assertEqual(killed, [111])
            self.assertEqual(out, [(S.SUPERVISE_PID, 111)])
            self.assertFalse((logs / S.SUPERVISE_PID).exists())
            self.assertFalse((logs / S.GENERATOR_PID).exists())      # stale: removed, nothing killed


class Cli(unittest.TestCase):
    def test_flags(self):
        from backrooms_talk.__main__ import parse_args, main, EXIT_CONFIG
        a = parse_args(["--dry", "--turns", "12", "--relay", "http://127.0.0.1:9", "--home", "h"])
        self.assertEqual((a.dry, a.turns, a.relay, a.home), (True, 12, "http://127.0.0.1:9", "h"))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--dry", "--home", tmp]), EXIT_CONFIG)        # no model folder, no LICENSE


class Honesty(unittest.TestCase):
    def test_account_says_what_the_lines_are(self):
        h = backrooms_talk.HOW_LINES_ARE_MADE
        for must in ("LFM2.5-1.2B-Instruct", "not a fly's thoughts", "flies do not use language", "labels",
                     "MaleCNS v1.0", "CC BY 4.0", "leaky integrate-and-fire", "simplification",
                     "3-5 plain hints", "use or ignore", "jumps past a chosen threshold", "the model picks the new topic",
                     "can still say things the hints do not support", "never edited", "measured", "chosen"):
            self.assertIn(must, h)
        doc = backrooms_talk.__doc__
        self.assertIn("MEASURED", doc)
        self.assertIn("CHOSEN", doc)
        self.assertIsNone(SAFETY.check(h[:600]))

    def test_the_readme_publishes_the_instruction_word_for_word_and_no_retired_method(self):
        # the page shows only the room and the lines; the instruction and credits are published here
        text = (PKG / "README.md").read_text(encoding="utf-8")
        self.assertIn(P.SYSTEM_PROMPT, text)
        self.assertIn(P.CLOSING.format(name="A", reply=P.REPLY.format(to="D"), answer="<answer>", move="<move>",
                                       focus="<focus>", opener="<opener>"), text)
        self.assertIn(P.CLOSING.format(name="A", reply="", answer="", move="", focus="", opener=""), text)
        for opener in P.OPENERS:
            self.assertIn(opener.strip(), text)
        self.assertIn(P.ANSWER.format(to="{to}").strip(), text)
        self.assertIn(P.FOCUS.format(topic="{topic}").strip(), text)
        for frame in (P.TOPIC_OPEN, P.TOPIC_NEW, P.TOPIC_KEEP, P.HINTS_HEADER.format(name="A"), P.NO_HINTS,
                      P.TRANSCRIPT_HEADER, P.NO_LINES, P.TOPIC_LINE.format(topic="<topic>")):
            self.assertIn(frame, text)
        for move in P.MOVES:
            self.assertIn(move.strip(), text)
        for gone in ("abstract recurrence", "next-word scores", "Nothing is generating", "graph statistics",
                     "on the page"):
            self.assertNotIn(gone, text)
        for must in ("leaky integrate-and-fire", "MaleCNS v1.0", "CC BY 4.0", "not a fly's thoughts",
                     "LFM Open License", "basis"):
            self.assertIn(must, text)

    def test_no_unnamed_upstream_project_is_named(self):
        banned = [("f" + "lm"), ("nft" + "echie"), ("Worm" + "uth")]
        for p in list(PKG.rglob("*.py")) + list(PKG.rglob("*.md")):
            text = p.read_text(encoding="utf-8").lower()
            for word in banned:
                self.assertIsNone(re.search(r"(?<![a-z])" + word.lower() + r"(?![a-z])", text), f"{word} in {p.name}")

    def test_the_engine_loads_the_relays_filter_file(self):
        self.assertEqual(Path(SAFETY.__file__).resolve(), (W.REPO / "backrooms_relay" / "safety.py").resolve())
        self.assertEqual(SAFETY.check("buy now"), "financial")

    def test_no_network_library_but_urllib_in_the_client(self):
        for p in PKG.glob("*.py"):
            src = p.read_text(encoding="utf-8")
            for bad in ("requests", "httpx", "aiohttp", "openai", "anthropic", "socket"):
                self.assertIsNone(re.search(rf"^\s*(import|from)\s+{bad}\b", src, re.M), f"{p.name} imports {bad}")
            if p.name != "relay_client.py":
                self.assertIsNone(re.search(r"^\s*(import|from)\s+urllib", src, re.M), p.name)


if __name__ == "__main__":
    unittest.main()

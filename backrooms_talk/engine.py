"""
The generator: four simulated flies keep running; every TURN_SECONDS of wall
time the next fly (round-robin A, B, C, D) gets one line from the language
model, on the current topic, from plain hints of its own simulated brain and
body and the last accepted lines.

HOW LINES ARE MADE: see HOW_LINES_ARE_MADE in backrooms_talk/__init__.py
(the canonical plain-words account, with what is measured and what is
chosen). In short: a small language model (LFM2.5-1.2B-Instruct) writes each
line from a readout of that fly's simulated brain activity on the Janelia
male CNS connectome (MaleCNS v1.0, CC BY 4.0) in a simulated room. The
simulator is a uniform leaky integrate-and-fire model, a simplification. The
lines are not a fly's thoughts; flies do not use language; the names are
labels.

One turn, in order
  1. Readout.take(fly): at most BASIS_MAX numeric readout strings, 3-5 plain
     hints made from the same measured values, and the numbers.
  2. Each readout string and each hint passes backrooms_relay/safety.py
     (loaded from that file) or is left out and counted by reason; a string
     longer than BASIS_ITEM_MAX_CHARS is left out as "basis_length". The
     `basis` (posted) is the numeric readout the hints were made from; the
     hints are what the model is shown.
  3. The topic (topics.py): on the first turn, or when the speaker's
     whole-brain rate jumped by at least topics.SWITCH_JUMP from its previous
     window (after topics.TOPIC_MIN lines on the topic), or when the topic has
     had topics.TOPIC_MAX lines (or topics.TOPIC_MAX_SLOTS turn slots), the
     model proposes a new topic (seeds
     turn_seed(turn, 10 + attempt), at most topics.TOPIC_TRIES samples, each
     checked and dropped by reason, never edited), in a category drawn with
     turn_seed(turn, 98) from those the last topics did not use.
  4. The prompt: SYSTEM_PROMPT (constant) + the topic and whether it just
     changed + the hints + the last accepted lines (prompt.PROMPT_LINES = 6,
     or on a switched-to topic only its own lines, at least
     prompt.SWITCH_LINES = 2), one
     conversational move (prompt.MOVES, drawn with turn_seed(turn, 99)) and
     one opener (prompt.OPENERS, drawn with turn_seed(turn, 97), not one of
     the last three turns' openers), in fixed frame text
     (prompt.build_messages; nothing else). The model
     samples with seed = turn_seed(turn, attempt).
  5. FORMATTING-ONLY NORMALISATION before the filter, and nothing else:
       - curly quotes become straight quotes (U+2018 U+2019 U+201A U+201B -> '
         and U+201C U+201D U+201E U+201F -> ");
       - dashes (U+2010 to U+2013) become "-", long dashes (U+2014, U+2015)
         become " - ";
       - an ellipsis character (U+2026) becomes "...";
       - every run of whitespace (newlines included) becomes one space, and
         the ends are trimmed;
       - one leading "<fly name>:" label of the speaking fly ("A:" or
         "Fly A:") is stripped.
     Content words are never changed.
  6. The line passes safety.check or is DROPPED and counted by reason; a reply
     the model did not finish itself (cut at max_new_tokens) is dropped as
     "unfinished". Then the line checks (backrooms_talk/grounding.py) drop,
     by reason, a line that names a real person or brand, or touches
     politics, religion, medical claims, hate or harm instructions; is longer
     than three sentences or 240 characters; does not end a sentence; quotes
     a number with a unit that its hints do not hold, or an amount in words
     with a unit ("number"); repeats one of the last HISTORY accepted lines
     or reuses three content words of the last four, or one word three of
     them used ("repeat"); opens with
     the same two words as one of the last three ("same_opening"); names the
     speaking fly itself ("self_name"); or, in a switched-to
     topic's first two turn slots, shares no content word with the topic
     ("off_topic"). A dropped line is never edited. The
     same prompt is sampled again with a new seed at most RETRIES_ON_DROP (2)
     more times; then the turn ends with no line and the next fly speaks.
  7. An accepted line is posted to the relay (retries with backoff) with its
     topic fields or, with --dry, printed. Every accepted and dropped line,
     topic, and dropped readout string or hint is appended to the local
     archive.

CHOSEN here: TURN_SECONDS, HISTORY, RETRIES_ON_DROP, WARMUP_STEPS,
BASIS_ITEM_MAX_CHARS, the seed rule, OOM_WAIT_S; the sampling settings are in
model.py, the readout's and hints' shape in readout.py, the prompt's shape in
prompt.py, the topic rule in topics.py and the checks' thresholds in
grounding.py.
"""
import ctypes
import importlib.util
import itertools
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

from backrooms_talk import grounding
from backrooms_talk import model as lm
from backrooms_talk import topics as T
from backrooms_talk.prompt import OPENER_MEMORY, SYSTEM_PROMPT, build_messages, normalise, pick_move, pick_opener, reply_to
from backrooms_talk.readout import BASIS_MAX, Readout
from backrooms_talk.world import FLY_NAMES, REPO

TURN_SECONDS = 9.0           # CHOSEN: wall seconds between turns (the world runs between them)
HISTORY = 8                  # CHOSEN: accepted lines kept for the repeat check (the prompt shows the last 6)
RETRIES_ON_DROP = 2          # CHOSEN: new samples of the same prompt after a drop (cap: 2)
WARMUP_STEPS = 40            # CHOSEN: world steps (2 s of room time) before the first turn
MIN_STEPS_PER_TURN = 1       # every turn's window holds at least one world step
BASIS_ITEM_MAX_CHARS = 200   # the relay's cap (TURN_SCHEMA.md): a longer readout string is left out, never cut
OOM_WAIT_S = 120.0           # CUDA out of memory: wait, then retry; never kill another process
OOM_TRIES = 30
SEED_MUL, TURN_MUL = 1_000_003, 1_009   # CHOSEN: turn_seed = (seed x SEED_MUL + turn x TURN_MUL + attempt) mod 2^31
MOVE_ATTEMPT = 99            # the turn's move (prompt.MOVES) is drawn with turn_seed(turn, 99)
CATEGORY_ATTEMPT = 98        # a switch's topic category is drawn with turn_seed(turn, 98)
OPENER_ATTEMPT = 97          # the turn's opener (prompt.OPENERS) is drawn with turn_seed(turn, 97)

SAFETY_PATH = REPO / "backrooms_relay" / "safety.py"
TURN_SCHEMA_PATH = Path(__file__).resolve().parent / "TURN_SCHEMA.md"
META_KEYS = ("active_fraction", "brain_rate_hz", "song_hz", "sound_drive_hz", "smell_drive_hz",
             "top_group_rate_hz", "top_group_baseline_hz", "nearest_fly_mm", "world_s",
             "window_steps", "seed", "new_tokens")
TOPIC_KEYS = ("topic", "topic_id", "topic_turn", "switched")


def load_safety(path=SAFETY_PATH):
    """The relay's own filter module, loaded from its file so the engine and the relay run the same bytes."""
    spec = importlib.util.spec_from_file_location("backrooms_relay_safety", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def turn_seed(seed, turn, attempt):
    return int((int(seed) * SEED_MUL + int(turn) * TURN_MUL + int(attempt)) % (2 ** 31))


def make_payload(speaker, to, text, engine_seq, fly_index, basis, meta, topic):
    """The POST /turn body, exactly as TURN_SCHEMA.md defines it (to is omitted when None).
    topic: {topic, topic_id, topic_turn, switched} (Topics.fields())."""
    p = {"speaker": speaker}
    if to is not None:
        p["to"] = to
    p.update({"text": text, "engine_seq": int(engine_seq), "fly_index": int(fly_index),
              "topic": str(topic["topic"]), "topic_id": int(topic["topic_id"]),
              "topic_turn": int(topic["topic_turn"]), "switched": bool(topic["switched"]),
              "basis": list(basis), "meta": {k: meta[k] for k in META_KEYS}})
    return p


def say_safe(*parts):
    """Print without raising on a console that cannot show a character."""
    try:
        print(*parts, flush=True)
    except Exception:
        try:
            print(*[str(p).encode("ascii", "backslashreplace").decode() for p in parts], flush=True)
        except Exception:
            pass


# ---- resources ------------------------------------------------------------------

def process_rss_gb():
    """(working set GB, peak working set GB) of this process; (nan, nan) if unreadable."""
    try:
        if os.name == "nt":
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.GetCurrentProcess.restype = wt.HANDLE
            k.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
            k.K32GetProcessMemoryInfo.restype = wt.BOOL
            if not k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb):
                raise OSError("GetProcessMemoryInfo failed")
            return c.WorkingSetSize / 2 ** 30, c.PeakWorkingSetSize / 2 ** 30
        rss = peak = float("nan")
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = float(line.split()[1]) / 2 ** 20
            elif line.startswith("VmHWM:"):
                peak = float(line.split()[1]) / 2 ** 20
        return rss, peak
    except Exception:
        return float("nan"), float("nan")


def vram_gb():
    """{allocated, peak_allocated, device_free, device_total} in GiB, or {} without CUDA."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"allocated": torch.cuda.memory_allocated() / 2 ** 30,
            "peak_allocated": torch.cuda.max_memory_allocated() / 2 ** 30,
            "device_free": free / 2 ** 30, "device_total": total / 2 ** 30}


# ---- the engine -------------------------------------------------------------------

class Engine:
    """
    `world` has step() -> record (TalkRoom's shape), .sizes, .dt; `model` has
    generate(messages, seed, **sampling) -> (text, new_tokens, finished, secs).
    With dry=True nothing is posted; otherwise `relay` has post_turn(payload).
    """

    def __init__(self, world, model, *, names=FLY_NAMES, safety=None, relay=None, dry=True,
                 archive_path=None, mode=None, seed=0, turn_seconds=TURN_SECONDS, history=HISTORY,
                 retries_on_drop=RETRIES_ON_DROP, warmup_steps=WARMUP_STEPS, sampling=None,
                 clock=time.monotonic, sleep=time.sleep, say=say_safe, oom_wait_s=OOM_WAIT_S,
                 dictionary=None, engine_seq0=0, history0=(), next_fly=0, ground=grounding.check,
                 topics0=None, recent_topics_path=None):
        if not dry and relay is None:
            raise ValueError("posting needs a relay client (or dry=True)")
        if retries_on_drop > 2:
            raise ValueError("no prompt is sampled again more than twice")
        self.world, self.model = world, model
        self.names = tuple(names)
        self.safety = safety or load_safety()
        self.relay, self.dry = relay, bool(dry)
        self.archive_path = Path(archive_path) if archive_path else None
        self.mode = mode or ("dry" if dry else "relay")
        self.seed = int(seed)
        self.turn_seconds = float(turn_seconds)
        self.history = deque(history0, maxlen=int(history))
        self.retries_on_drop = int(retries_on_drop)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.sampling = dict(sampling or {})
        self.clock, self.sleep, self.say = clock, sleep, say
        self.oom_wait_s = float(oom_wait_s)
        self.ground = ground
        self.readout = Readout(world.sizes, self.names, dictionary=dictionary, dt=world.dt)
        self.topics = T.Topics(seed=self.seed)
        self.recent_topics_path = Path(recent_topics_path) if recent_topics_path else None
        self.topics.recent.extend(load_recent_topics(self.recent_topics_path))
        if topics0:
            self.topics.current, self.topics.topic_id = topics0["topic"], int(topics0["topic_id"])
            self.topics.turns = self.topics.slots = int(topics0["turns"])
            self.topics.recent.extend(t for t in topics0.get("recent", ()) if t not in self.topics.recent)
        self.topic_drops = Counter()     # proposed topics, by reason
        self.openers = deque(maxlen=max(OPENER_MEMORY, 1))
        self.engine_seq = int(engine_seq0)
        self.turn_index = 0
        self.next_fly = int(next_fly) % len(self.names)
        self.drops = Counter()           # generated lines, by reason
        self.basis_drops = Counter()     # readout strings, by reason
        self.accepted = self.posted = self.post_failures = 0
        self.world_steps = 0
        self.warmed = False
        self.results = []

    # -- plumbing ---------------------------------------------------------------

    def _archive(self, row):
        if self.archive_path is None:
            return
        row = dict(row, at=time.time(), mode=self.mode)
        with open(self.archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _oom_retry(self, fn, what):
        for k in range(OOM_TRIES):
            try:
                return fn()
            except Exception as exc:
                if not lm.is_cuda_oom(exc) or k == OOM_TRIES - 1:
                    raise
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.say(f"CUDA out of memory during {what}; waiting {self.oom_wait_s:.0f} s and retrying "
                         "(no other process is touched)")
                self.sleep(self.oom_wait_s)

    def step_world(self):
        rec = self._oom_retry(self.world.step, "a world step")
        self.readout.update(rec)
        self.world_steps += 1
        return rec

    def warmup(self):
        t0 = self.clock()
        for _ in range(self.warmup_steps):
            self.step_world()
        self.readout.settle()
        self.warmed = True
        self.say(f"warm-up: {self.warmup_steps} world steps in {self.clock() - t0:.1f} s")

    # -- turns ------------------------------------------------------------------

    def run(self, turns=0):
        """`turns` turn slots (0 = forever). Returns this call's turn results."""
        if not self.warmed:
            self.warmup()
        out = []
        next_at = self.clock() + self.turn_seconds
        for _ in (range(int(turns)) if turns else itertools.count()):
            t_slot = self.clock()
            steps = 0
            while steps < MIN_STEPS_PER_TURN or self.clock() < next_at:
                self.step_world()
                steps += 1
            name = self.names[self.next_fly]
            self.next_fly = (self.next_fly + 1) % len(self.names)
            res = self.turn(name)
            res["world_steps_before"] = steps
            res["slot_wall_s"] = self.clock() - t_slot
            out.append(res)
            self.results.append(res)
            next_at = max(next_at + self.turn_seconds, self.clock())
        return out

    def turn(self, name):
        t0 = self.clock()
        self.turn_index += 1
        raw_basis, raw_hints, numbers = self.readout.take(name)
        basis = self._screen(name, raw_basis, "dropped_basis")[:BASIS_MAX]
        hints = self._screen(name, raw_hints, "dropped_hint")
        history = list(self.history)
        res = {"turn": self.turn_index, "speaker": name, "basis": basis, "hints": hints, "accepted": False,
               "attempts": [], "numbers": numbers, "topic_switch": None}
        why = self.topics.switch_reason(numbers.get("brain_change"))
        if why is not None:
            res["topic_switch"] = dict(self._new_topic(why, history), reason=why)
        tp = self.topics
        topic_slot = tp.slot()
        move = pick_move(turn_seed(self.seed, self.turn_index, MOVE_ATTEMPT), reply_to(name, history))
        opener = pick_opener(turn_seed(self.seed, self.turn_index, OPENER_ATTEMPT), reply_to(name, history),
                             self.openers if OPENER_MEMORY > 0 else ())
        self.openers.append(opener)
        res["move"], res["opener"] = move, opener
        topic_turn = tp.turns + 1
        messages = build_messages(name, history, hints, tp.current, tp.topic_id, tp.fresh, move, topic_turn,
                                  opener)

        for attempt in range(1 + self.retries_on_drop):
            seed = turn_seed(self.seed, self.turn_index, attempt)
            raw, n_new, finished, gen_s = self._oom_retry(
                lambda: self.model.generate(messages, seed=seed, **self.sampling), "generation")
            text = normalise(raw, name)
            reason = self.safety.check(text)
            if reason is None and not finished:
                reason = "unfinished"
            if reason is None and self.ground is not None:
                reason = self.ground(text, hints, [t for _, t in history], topic=tp.current,
                                     topic_slot=topic_slot if tp.topic_id > 1 else None, speaker=name)
            res["attempts"].append({"seed": seed, "gen_s": gen_s, "new_tokens": n_new, "reason": reason})
            if reason is not None:
                self.drops[reason] += 1
                self._archive({"kind": "dropped", "turn": self.turn_index, "speaker": name,
                               "attempt": attempt, "seed": seed, "reason": reason, "text": text})
                self.say(f"turn {self.turn_index} {name}: dropped ({reason}), attempt {attempt + 1}")
                continue

            to = reply_to(name, history)
            meta = dict(numbers, seed=seed, new_tokens=int(n_new))
            payload = make_payload(name, to, text, self.engine_seq + 1, self.names.index(name), basis, meta,
                                   self.topics.fields())
            post = None
            if self.dry:
                ok = True
            else:
                post = self.relay.post_turn(payload)
                ok = bool(post.get("ok"))
            self.accepted += 1
            self.engine_seq += 1
            if ok:
                self.history.append((name, text))
                self.topics.accepted()
                self.posted += 0 if self.dry else 1
            else:
                self.post_failures += 1
            self._archive({"kind": "accepted", "turn": self.turn_index, "attempt": attempt, "seed": seed,
                           "raw": raw, "payload": payload, "gen_s": gen_s,
                           "post": None if post is None else {k: post.get(k) for k in ("ok", "status", "attempts", "error", "body")}})
            res.update(accepted=True, text=text, raw=raw, payload=payload, post=post, posted_ok=ok)
            break
        res["turn_wall_s"] = self.clock() - t0
        self._print_turn(res)
        return res

    def _screen(self, name, strings, kind):
        """Readout strings or hints that pass the filter (and the relay's length cap); the rest counted."""
        kept = []
        for s in strings:
            reason = self.safety.check(s)
            if reason is None and len(s) > BASIS_ITEM_MAX_CHARS:
                reason = "basis_length"
            if reason is not None:
                self.basis_drops[reason] += 1
                self._archive({"kind": kind, "turn": self.turn_index, "speaker": name, "reason": reason, "text": s})
                continue
            kept.append(s)
        return kept

    def _new_topic(self, why, history):
        """Ask the model for a topic (opening: from the seed theme). Returns {topic, kept, tries}."""
        tp = self.topics
        theme = tp.theme if why == "opening" else None
        category = (T.THEME_CATEGORY.get(tp.theme) if theme is not None
                    else T.next_category(turn_seed(self.seed, self.turn_index, CATEGORY_ATTEMPT), tp.categories))
        messages = T.topic_messages(list(tp.recent), history, theme=theme, category=category)
        sampling = dict(self.sampling, max_new_tokens=T.TOPIC_MAX_NEW_TOKENS)
        for attempt in range(T.TOPIC_TRIES):
            seed = turn_seed(self.seed, self.turn_index, T.TOPIC_SEED_OFFSET + attempt)
            raw, n_new, finished, gen_s = self._oom_retry(
                lambda: self.model.generate(messages, seed=seed, **sampling), "topic generation")
            topic = T.clean_topic(raw)
            reason = T.check_topic(topic, list(tp.recent), self.safety, finished or T.first_line(raw)[1])
            if reason is None:
                tp.set(topic, why, category)
                save_recent_topics(self.recent_topics_path, tp.recent)
                self._archive({"kind": "topic", "turn": self.turn_index, "attempt": attempt, "seed": seed,
                               "reason": why, "category": category, "raw": raw, "topic": topic,
                               "topic_id": tp.topic_id})
                self.say(f"turn {self.turn_index}: new topic ({why}, {category}): {topic}")
                return {"topic": topic, "kept": True, "tries": attempt + 1}
            self.topic_drops[reason] += 1
            self._archive({"kind": "dropped_topic", "turn": self.turn_index, "attempt": attempt, "seed": seed,
                           "reason": reason, "text": topic})
            self.say(f"turn {self.turn_index}: topic dropped ({reason}), attempt {attempt + 1}")
        if why == "opening":
            tp.set(tp.theme, "opening_theme", category)
            self._archive({"kind": "topic", "turn": self.turn_index, "reason": "opening_theme",
                           "topic": tp.theme, "topic_id": tp.topic_id})
            return {"topic": tp.theme, "kept": False, "tries": T.TOPIC_TRIES}
        return {"topic": tp.current, "kept": False, "tries": T.TOPIC_TRIES}

    def _print_turn(self, r):
        head = f"turn {r['turn']} {r['speaker']}"
        if r["accepted"]:
            p = r["payload"]
            to = p.get("to")
            status = "dry" if self.dry else ("posted" if r["posted_ok"] else f"post failed: {r['post'].get('error')}")
            mark = " [SWITCH]" if p["switched"] else ""
            self.say(f"{head}{' -> ' + to if to else ''} [{status}] topic {p['topic_id']}.{p['topic_turn']}"
                     f" '{p['topic']}'{mark}: {r['text']}")
        else:
            self.say(f"{head}: no line (all {len(r['attempts'])} samples dropped)")
        for h in r["hints"]:
            self.say(f"    hint: {h}")

    def counts(self):
        return {"turns": self.turn_index, "accepted": self.accepted, "posted": self.posted,
                "post_failures": self.post_failures, "dropped_by_reason": dict(self.drops),
                "basis_dropped_by_reason": dict(self.basis_drops), "topic_dropped_by_reason": dict(self.topic_drops),
                "topics": [{"topic_id": i, "reason": why} for i, why in self.topics.switches],
                "world_steps": self.world_steps}


# ---- archive helpers for a restart --------------------------------------------------

def load_recent_topics(path):
    """The recent topics kept across runs (a JSON list), or [] when there is no readable file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8")) if path else []
    except (OSError, ValueError):
        return []
    return [str(t) for t in data if isinstance(t, str)][-T.RECENT_TOPICS:] if isinstance(data, list) else []


def save_recent_topics(path, recent):
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(list(recent)[-T.RECENT_TOPICS:], ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


def resume_from_archive(path, mode, names=FLY_NAMES, history=HISTORY):
    """(engine_seq0, history0, next_fly) from the accepted, posted lines of this mode in the archive."""
    path = Path(path)
    seq, hist, last = 0, deque(maxlen=history), None
    if not path.exists():
        return seq, [], 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("mode") != mode or row.get("kind") != "accepted":
                continue
            p = row.get("payload") or {}
            seq = max(seq, int(p.get("engine_seq", 0)))
            if (row.get("post") or {}).get("ok"):
                hist.append((p.get("speaker"), p.get("text")))
                last = p.get("speaker")
    nxt = (names.index(last) + 1) % len(names) if last in names else 0
    return seq, list(hist), nxt


def resume_topics(path, mode):
    """{topic, topic_id, turns, recent} from the last posted line of this mode that carries a topic, or None."""
    path = Path(path)
    if not path.exists():
        return None
    out, recent = None, deque(maxlen=T.RECENT_TOPICS)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            p = row.get("payload") or {}
            if row.get("mode") != mode or row.get("kind") != "accepted" or "topic" not in p:
                continue
            if not (row.get("post") or {}).get("ok"):
                continue
            if not recent or recent[-1] != p["topic"]:
                recent.append(p["topic"])
            out = {"topic": p["topic"], "topic_id": int(p["topic_id"]), "turns": int(p["topic_turn"])}
    if out:
        out["recent"] = list(recent)
    return out

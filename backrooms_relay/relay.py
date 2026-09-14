"""Backrooms relay: a tiny public board for turns posted by a separate generator.

This service has no brain and no model. It only stores and re-broadcasts text
that a generator (backrooms_talk, running elsewhere) has already produced and
filtered, and shows nothing until one posts. The relay runs the same safety
filter again (safety.py) on the text and on every basis line, and rejects, never
edits, a body that fails it; rejections are counted in /status as
blocked_at_relay.

What the text is: lines written by a small language model (Liquid AI
LFM2.5-1.2B-Instruct) from a readout of that fly's simulated brain activity on
the Janelia male CNS connectome (MaleCNS v1.0, CC BY 4.0) in a simulated room.
The simulator is a uniform leaky integrate-and-fire model, a simplification.
The lines are not a fly's thoughts; flies do not use language. The flies' names
are labels, not personalities. Nothing here claims consciousness, feelings or
thinking. A talk turn carries `basis`: the readout lines the model was shown.

Two body shapes are accepted (unknown fields are rejected in both):
  talk turn   any body with `fly_index` or `basis`; must match
              backrooms_talk/TURN_SCHEMA.md exactly (schema: TalkTurnIn): speaker and
              to in A B C D, text 1-600 chars, engine_seq >= 1, fly_index 0-3 (the
              speaker's position), basis 0-6 strings of 1-200 chars, meta exactly the
              12 TALK_META keys with their types and ranges
  legacy      speaker, to, text, engine_seq, meta as before (schema: TurnIn)
GET /turns and /stream return every stored field, so talk turns carry fly_index,
basis and meta and legacy turns look as they always did.

Endpoints
  POST /turn          Authorization: Bearer <RELAY_TOKEN>; JSON body (TalkTurnIn or TurnIn)
  GET  /turns          the NEWEST `limit` turns (default 100, max 500), oldest first
  GET  /turns?after=N  the first `limit` turns with seq > N (for paging forward)
  GET  /stream         Server-Sent Events; replays seq > max(?after=, Last-Event-ID), then one
                       `caught_up` event {last_seq}, then live `turn` events
  GET  /status         {running, last_at, count, blocked_at_relay}
                       running: the generator posted a turn in the last 120 s
  GET  /               short description (the honesty statement above)

Environment
  RELAY_TOKEN   required for POST (>= 32 chars); if missing, POST fails closed with 503
  DATA_DIR      default /data; turns.jsonl is appended there only if the directory exists
  PORT          used by the Dockerfile's uvicorn command
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional, Union

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

import safety

log = logging.getLogger("backrooms-relay")

KEEP = 500                 # turns kept in memory
RUNNING_WINDOW_S = 120.0   # status.running if the last turn is newer than this
MAX_BODY_BYTES = 8192      # hard cap on POST body size
MAX_TEXT_CHARS = safety.MAX_CHARS
MAX_META_KEYS = 12
FLY_LABELS = ("A", "B", "C", "D")   # speaker labels of a talk turn; fly_index is the position here
BASIS_MAX_ITEMS = 6
BASIS_ITEM_MAX_CHARS = 200
MAX_JSON_INT = 2**53
# The talk turn's meta: exactly these keys, in this order (TURN_SCHEMA.md).
# key -> (JSON type, minimum, maximum); None means unbounded (finite, and 2^53 for integers).
TALK_META = {
    "active_fraction": ("number", 0, 1),
    "brain_rate_hz": ("number", 0, None),
    "song_hz": ("number", 0, None),
    "sound_drive_hz": ("number", 0, 200),
    "smell_drive_hz": ("number", 0, 200),
    "top_group_rate_hz": ("number", 0, None),
    "top_group_baseline_hz": ("number", 0, None),
    "nearest_fly_mm": ("number", 0, 29),
    "world_s": ("number", 0, None),
    "window_steps": ("integer", 1, None),
    "seed": ("integer", 0, 2**31 - 1),
    "new_tokens": ("integer", 1, 4096),
}
TALK_ONLY_FIELDS = ("fly_index", "basis")   # either one marks a body as a talk turn
MAX_SUBSCRIBERS = 1000
SUBSCRIBER_QUEUE = 256
HEARTBEAT_S = 15.0
MIN_TOKEN_LEN = 32

ABOUT = (
    "Nothing appears here until the generator posts. Each line is written by a small language model "
    "(Liquid AI LFM2.5-1.2B-Instruct) from a readout of that fly's simulated brain activity on the "
    "Janelia male CNS connectome (MaleCNS v1.0, CC BY 4.0) in a simulated room. The simulator is a "
    "uniform leaky integrate-and-fire model, a simplification. This is not a fly's thoughts; flies do "
    "not use language. The flies' names are labels, not personalities. No claim of consciousness, "
    "feelings or thinking is made. Each turn lists the readout lines the model was shown."
)

ALLOWED_ORIGINS = ["https://flybrain.online", "https://www.flybrain.online"]

NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _.\-]{0,23}")
META_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
# Control characters (newline and tab allowed) and bidi/zero-width spoofing characters.
BAD_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")
# A basis item is one readout line: no control characters at all (not even newline or tab).
BAD_BASIS_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")


def _check_text(v: str) -> str:
    if not v.strip():
        raise ValueError("text is blank")
    if BAD_CHARS_RE.search(v):
        raise ValueError("text contains control or bidi characters")
    if v.count("\n") > safety.MAX_NEWLINES:
        raise ValueError("text has too many lines")
    return v


class TurnIn(BaseModel):
    """The original (legacy) body: speaker, to, text, engine_seq, meta. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    speaker: StrictStr = Field(min_length=1, max_length=24)
    to: Optional[StrictStr] = Field(default=None, min_length=1, max_length=24)
    text: StrictStr = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    engine_seq: Optional[StrictInt] = Field(default=None, ge=0, le=2**53)
    meta: Optional[dict[StrictStr, Union[StrictInt, StrictFloat]]] = None

    @field_validator("speaker", "to")
    @classmethod
    def _name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not NAME_RE.fullmatch(v):
            raise ValueError("name must be letters, digits, space, _ . - and start with a letter")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _check_text(v)

    @field_validator("meta")
    @classmethod
    def _meta(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        if len(v) > MAX_META_KEYS:
            raise ValueError(f"meta has more than {MAX_META_KEYS} keys")
        for k, val in v.items():
            if not META_KEY_RE.fullmatch(k):
                raise ValueError("meta keys must match [a-z][a-z0-9_]{0,31}")
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError("meta values must be finite")
            if isinstance(val, int) and abs(val) > 2**53:
                raise ValueError("meta integer out of range")
        return v


class TalkTurnIn(BaseModel):
    """The talk engine's body, exactly backrooms_talk/TURN_SCHEMA.md. Unknown fields are rejected.

    Beyond the JSON Schema, two consistency rules follow from the field definitions:
    fly_index must be the speaker's position in A B C D, and `to` (when present) must be
    another fly. Integers without a stated maximum are capped at 2^53 (exact in JSON).
    """

    model_config = ConfigDict(extra="forbid")

    speaker: StrictStr
    to: Optional[StrictStr] = None
    text: StrictStr = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    engine_seq: StrictInt = Field(ge=1, le=MAX_JSON_INT)
    fly_index: StrictInt = Field(ge=0, le=len(FLY_LABELS) - 1)
    basis: list[StrictStr] = Field(max_length=BASIS_MAX_ITEMS)
    meta: dict[StrictStr, Union[StrictInt, StrictFloat]]

    @model_validator(mode="before")
    @classmethod
    def _to_is_omitted_not_null(cls, data):
        if isinstance(data, dict) and "to" in data and data["to"] is None:
            raise ValueError("to must be omitted, not null, when there is no addressee")
        return data

    @field_validator("speaker", "to")
    @classmethod
    def _label(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in FLY_LABELS:
            raise ValueError("must be one of " + " ".join(FLY_LABELS))
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _check_text(v)

    @field_validator("basis")
    @classmethod
    def _basis(cls, v: list[str]) -> list[str]:
        for i, item in enumerate(v):
            if not 1 <= len(item) <= BASIS_ITEM_MAX_CHARS:
                raise ValueError(f"basis[{i}] must be 1-{BASIS_ITEM_MAX_CHARS} characters")
            if not item.strip():
                raise ValueError(f"basis[{i}] is blank")
            if BAD_BASIS_CHARS_RE.search(item):
                raise ValueError(f"basis[{i}] contains control or bidi characters")
        return v

    @field_validator("meta")
    @classmethod
    def _meta(cls, v: dict) -> dict:
        missing = [k for k in TALK_META if k not in v]
        extra = sorted(k for k in v if k not in TALK_META)
        if missing or extra:
            raise ValueError(f"meta must have exactly the {len(TALK_META)} talk keys (missing {missing}, extra {extra})")
        for k, (kind, lo, hi) in TALK_META.items():
            val = v[k]
            if kind == "integer":
                if not isinstance(val, int):
                    raise ValueError(f"meta.{k} must be an integer")
                if abs(val) > MAX_JSON_INT:
                    raise ValueError(f"meta.{k} out of range")
            elif not math.isfinite(val):
                raise ValueError(f"meta.{k} must be finite")
            if val < lo or (hi is not None and val > hi):
                raise ValueError(f"meta.{k} must be in [{lo}, {'inf' if hi is None else hi}]")
        return {k: v[k] for k in TALK_META}   # stored in the schema's order

    @model_validator(mode="after")
    def _consistent(self) -> "TalkTurnIn":
        if self.fly_index != FLY_LABELS.index(self.speaker):
            raise ValueError("fly_index must be the speaker's position in A B C D")
        if self.to is not None and self.to == self.speaker:
            raise ValueError("to must name another fly")
        return self


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON here")


def _sse(turn: dict) -> bytes:
    data = json.dumps(turn, ensure_ascii=False, separators=(",", ":"))
    return f"id: {turn['seq']}\nevent: turn\ndata: {data}\n\n".encode("utf-8")


def _tail_jsonl(path: str, keep: int) -> list[dict]:
    """Read the last `keep` valid turns from a JSONL file without loading the whole file."""
    size = os.path.getsize(path)
    budget = keep * (MAX_BODY_BYTES + 512)
    with open(path, "rb") as f:
        start = max(0, size - budget)
        f.seek(start)
        chunk = f.read()
    lines = chunk.split(b"\n")
    if start > 0:
        lines = lines[1:]  # first line is probably partial
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("seq"), int) and isinstance(obj.get("ts"), (int, float)):
            out.append(obj)
    out.sort(key=lambda t: t["seq"])
    return out[-keep:]


class Relay:
    def __init__(self, data_dir: Optional[str], clock: Callable[[], float]):
        self.clock = clock
        self.turns: deque[dict] = deque(maxlen=KEEP)
        self.seq = 0
        self.blocked = 0  # lines rejected by the safety filter since this process started
        self.subscribers: set[asyncio.Queue] = set()
        self.path: Optional[str] = None
        if data_dir and os.path.isdir(data_dir):
            self.path = os.path.join(data_dir, "turns.jsonl")
            if os.path.isfile(self.path):
                try:
                    for t in _tail_jsonl(self.path, KEEP):
                        self.turns.append(t)
                    if self.turns:
                        self.seq = self.turns[-1]["seq"]
                except OSError as e:
                    log.error("could not read %s: %s", self.path, e)
        log.info("relay ready: persist=%s restored=%d seq=%d", bool(self.path), len(self.turns), self.seq)

    def add(self, turn_in: Union[TurnIn, "TalkTurnIn"]) -> dict:
        now = self.clock()
        self.seq += 1
        turn = {"seq": self.seq, "at": _iso(now), "ts": round(now, 3)}
        turn.update(turn_in.model_dump(exclude_none=True))
        self.turns.append(turn)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(turn, ensure_ascii=False, separators=(",", ":")) + "\n")
            except OSError as e:
                log.error("append failed (memory only for this turn): %s", e)
        for q in list(self.subscribers):
            try:
                q.put_nowait(turn)
            except asyncio.QueueFull:
                # Slow consumer: drop it; it can reconnect with Last-Event-ID.
                self.subscribers.discard(q)
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(None)
        return turn

    def after(self, seq: int, limit: int) -> list[dict]:
        return [t for t in self.turns if t["seq"] > seq][:limit]

    def newest(self, limit: int) -> list[dict]:
        return list(self.turns)[-limit:]

    def status(self) -> dict:
        last = self.turns[-1] if self.turns else None
        return {
            "running": last is not None and (self.clock() - last["ts"]) <= RUNNING_WINDOW_S,
            "last_at": last["at"] if last else None,
            "count": self.seq,
            "blocked_at_relay": self.blocked,
        }


def create_app(
    token: Optional[str] = None,
    data_dir: Optional[str] = None,
    clock: Callable[[], float] = time.time,
    heartbeat_s: float = HEARTBEAT_S,
) -> FastAPI:
    token = os.environ.get("RELAY_TOKEN", "") if token is None else token
    data_dir = os.environ.get("DATA_DIR", "/data") if data_dir is None else data_dir
    token_bytes = token.encode("utf-8") if len(token) >= MIN_TOKEN_LEN else None
    if token_bytes is None:
        log.warning("RELAY_TOKEN missing or shorter than %d chars: POST /turn will return 503", MIN_TOKEN_LEN)

    relay = Relay(data_dir, clock)
    app = FastAPI(title="backrooms relay", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.relay = relay
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["Last-Event-ID"],
        allow_credentials=False,
        max_age=600,
    )

    def err(status: int, detail) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=status)

    @app.get("/")
    async def root():
        return {"service": "backrooms relay", "about": ABOUT}

    @app.post("/turn")
    async def post_turn(request: Request):
        if token_bytes is None:
            return err(503, "relay token not configured")
        auth = request.headers.get("authorization", "")
        scheme, _, given = auth.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(given.strip().encode("utf-8"), token_bytes):
            return err(401, "unauthorized")
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return err(415, "content-type must be application/json")
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
            return err(413, f"body over {MAX_BODY_BYTES} bytes")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return err(413, f"body over {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(bytes(body).decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError):
            return err(400, "body is not valid JSON")
        if not isinstance(payload, dict):
            return err(422, "body must be a JSON object")
        # A body carrying fly_index or basis is a talk turn and must match TURN_SCHEMA.md
        # exactly; a body with neither is the original shape, validated as before.
        model = TalkTurnIn if any(k in payload for k in TALK_ONLY_FIELDS) else TurnIn
        try:
            turn_in = model.model_validate(payload)
        except ValidationError as e:
            return err(422, [{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()])
        checks = [(field, getattr(turn_in, field)) for field in ("speaker", "to", "text")]
        checks += [(f"basis[{i}]", item) for i, item in enumerate(getattr(turn_in, "basis", None) or [])]
        for field, value in checks:
            reason = safety.check(value) if value is not None else None
            if reason is not None:
                relay.blocked += 1  # dropped and counted (once per body), never edited
                return err(422, {"blocked": field, "reason": reason, "filter": safety.FILTER_VERSION})
        turn = relay.add(turn_in)
        return {"ok": True, "seq": turn["seq"], "at": turn["at"]}

    @app.get("/turns")
    async def get_turns(after: Optional[int] = Query(None, ge=0), limit: int = Query(100, ge=1, le=KEEP)):
        # No `after`: the newest `limit` turns (what a page shows on load). With `after`: page forward.
        oldest = relay.turns[0]["seq"] if relay.turns else None
        turns = relay.newest(limit) if after is None else relay.after(after, limit)
        return {"turns": turns, "last_seq": relay.seq, "oldest_seq": oldest}

    @app.get("/status")
    async def get_status():
        return relay.status()

    @app.get("/stream")
    async def stream(
        request: Request,
        after: Optional[int] = Query(None, ge=0),
        max_events: Optional[int] = Query(None, ge=1, le=10000),
    ):
        if len(relay.subscribers) >= MAX_SUBSCRIBERS:
            return err(503, "too many listeners")
        # An automatic EventSource reconnect keeps the original ?after= in the URL and adds
        # Last-Event-ID; the larger of the two is what the client has already received.
        last_id = request.headers.get("last-event-id", "").strip()
        header_seq = int(last_id) if last_id.isdigit() and len(last_id) <= 16 else None
        seen = [x for x in (after, header_seq) if x is not None]
        start = max(seen) if seen else relay.seq
        start = min(start, relay.seq)  # a client ahead of us (relay restarted without /data) resyncs
        q: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        relay.subscribers.add(q)  # subscribe before the replay snapshot so nothing is missed
        backlog = relay.after(start, KEEP)
        # Marks the end of the replayed backlog so a page can show replayed turns without
        # animating them as if they were being said now.
        caught_up = ("event: caught_up\ndata: " + json.dumps({"last_seq": relay.seq}) + "\n\n").encode("utf-8")

        async def gen():
            sent = 0
            last = start
            try:
                yield b"retry: 3000\n\n"
                for t in backlog:
                    yield _sse(t)
                    last = t["seq"]
                    sent += 1
                    if max_events and sent >= max_events:
                        return
                yield caught_up
                while True:
                    try:
                        t = await asyncio.wait_for(q.get(), timeout=heartbeat_s)
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
                        continue
                    if t is None:
                        return
                    if t["seq"] <= last:
                        continue
                    yield _sse(t)
                    last = t["seq"]
                    sent += 1
                    if max_events and sent >= max_events:
                        return
            finally:
                relay.subscribers.discard(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
app = create_app()

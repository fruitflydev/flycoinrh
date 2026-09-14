# backrooms_relay

The relay runs on Railway with no volume, so turns are kept in memory only.
The generator is `backrooms_talk` (four simulated flies; a small language
model writes one line per turn from a readout of that fly's simulated brain
activity). The page that reads it, `site/web/backrooms.html`, says "generator
not running" while no turns arrive and "relay unreachable" when it cannot reach
the relay. The deployed relay (redeployed 2026-09-14) accepts the talk-turn
schema below.

## What it does

A small FastAPI service with no model in it. A separate generator posts short
lines of text; the relay stores the newest 500 and re-broadcasts them.

Two body shapes are accepted, unknown fields rejected in both:

- **talk turn** (any body with `fly_index` or `basis`): exactly
  `backrooms_talk/TURN_SCHEMA.md`: `speaker` and optional `to` in `A B C D`
  (`to` another fly, omitted rather than null), `text` 1-600 chars,
  `engine_seq` >= 1, `fly_index` 0-3 and equal to the speaker's position,
  `basis` 0-6 strings of 1-200 chars with no control characters, and `meta`
  with exactly the 12 numeric keys, each with its JSON type (integer keys
  `window_steps`, `seed`, `new_tokens` take integers only; booleans are never
  numbers) and range. `test_turn_schema_md_agrees_with_the_relay` fails if the
  relay and the schema file drift.
- **legacy**: `speaker`, `to`, `text`, `engine_seq`, `meta` as before.

`GET /turns` and `GET /stream` return every stored field, so talk turns carry
`fly_index`, `basis` and `meta` (in the schema's key order).

| endpoint | what it returns |
|---|---|
| `POST /turn` | accepts one line; needs `Authorization: Bearer <RELAY_TOKEN>` |
| `GET /turns` | the newest turns, oldest first; `?after=N` pages forward |
| `GET /stream` | Server-Sent Events: replays missed turns, then live ones |
| `GET /status` | `running` (a turn arrived in the last 120 s), `last_at`, `count`, `blocked_at_relay` |
| `GET /` | a plain statement of what the text is and is not |

`RELAY_TOKEN` must be at least 32 characters or every `POST` fails with 503.
If `DATA_DIR` exists, turns are appended to `turns.jsonl` there; otherwise they
live in memory only. CORS allows `GET` only, from exactly `https://flybrain.online`
and `https://www.flybrain.online`; no other origin (preview deployments included) is allowed.

## What it refuses

A post is rejected, never edited, when:

- the token is missing or wrong, the body is not JSON, is over 8 KB, or does
  not match one of the two shapes above;
- a name is not plain letters, digits, space, `_`, `.` or `-`, or the text is
  blank, over 600 characters, more than 8 newlines, or carries control or bidi
  characters;
- `safety.py` flags the speaker, the addressee, the text or any `basis` line
  (a body is rejected whole, never edited, and counted once): financial or
  trading language (including look-alike letters, leetspeak, spaced-out
  letters and any currency symbol), URLs starting `http:`, `https:`, `www.`
  or `t.me/`, `@handles`, bare domains on a fixed list of common endings
  (`.com`, `.net`, `.org`, `.io`, `.xyz`, `.fun`, `.app`, `.gg`, `.co`, `.me`,
  `.ly`, `.online`, `.ai`, `.so`, `.sh`), slurs, sexual content, threats,
  self-harm, or any character outside a plain-Latin allow-list. A bare domain
  on any other ending, such as `example.dev`, is not caught.

Filter rejections are counted in `/status` as `blocked_at_relay`.

## Tests

The relay's dependencies are pinned here, not in the root requirements:

```bash
pip install -r backrooms_relay/requirements-dev.txt
python -m pytest -q backrooms_relay/tests
```

Where `fastapi` or `httpx` is not installed, `tests/test_relay.py` is skipped;
`tests/test_safety.py` needs only the standard library and pytest.

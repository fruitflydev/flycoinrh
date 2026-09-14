# POST /turn: the body backrooms_talk sends

One JSON object per accepted line, `Content-Type: application/json`,
`Authorization: Bearer <RELAY_TOKEN>`. The engine builds it in
`engine.make_payload`; `tests/test_talk.py` validates real engine payloads
against the JSON Schema block below, so this file and the code cannot drift.

Every string in `speaker`, `to`, `text` and every item of `basis` has already
passed `backrooms_relay/safety.py` in the engine. The relay should run the
same filter again on `text` and on every `basis` item and reject (never edit)
a body that fails.

## Fields

| field | type | meaning |
|---|---|---|
| `speaker` | string, one of `A` `B` `C` `D` | the label of the fly whose turn it was |
| `to` | string, one of `A` `B` `C` `D`, optional | the speaker of the most recent accepted line before this one, when that is another fly; omitted otherwise. It is taken from the transcript order, not detected in the text |
| `text` | string, 1-600 chars | the line: model output after formatting-only normalisation (curly quotes, dashes, ellipsis, trimmed ends, a leading own-name label) |
| `engine_seq` | integer >= 1 | the generator's own counter of accepted lines; increases by one per accepted line, continues across restarts from the local archive |
| `fly_index` | integer 0-3 | the fly's position in `A B C D`; the page places the speaker by this |
| `basis` | array of 0-6 strings, each 1-200 chars | exactly the readout lines the model was shown for this turn, in the order shown; a readout string longer than 200 chars is left out by the engine (counted as `basis_length`), never cut |
| `meta` | object, exactly the 12 keys below | numbers for the page's per-turn panel |

### `meta`

| key | JSON type | unit / range | what it is |
|---|---|---|---|
| `active_fraction` | number | 0-1 | fraction of the 165,122 neurons that fired at least once in a 12 ms brain window, averaged over this fly's window since its last turn |
| `brain_rate_hz` | number | Hz, >= 0 | mean spike rate over all neurons, averaged over the window |
| `song_hz` | number | Hz, >= 0 | summed rate of the 8 pulse-song wing motor neurons, window mean |
| `sound_drive_hz` | number | Hz per cell, 0-200 | sound drive delivered to this fly's JO-A and JO-B cells, window mean |
| `smell_drive_hz` | number | Hz per cell, 0-200 | cVA drive delivered to this fly's ORN_DA1 cells, window mean |
| `top_group_rate_hz` | number | Hz, >= 0 | window mean rate of the first named group in the readout (0 when none) |
| `top_group_baseline_hz` | number | Hz, >= 0 | that group's baseline: its mean rate from the run start to the window start |
| `nearest_fly_mm` | number | mm, 0-29 | distance to the nearest other fly at the turn |
| `world_s` | number | s, >= 0 | simulated room time at the turn |
| `window_steps` | integer | >= 1 | world steps (50 ms of room time each) in this fly's window |
| `seed` | integer | 0 to 2^31-1 | the sampling seed of the accepted attempt |
| `new_tokens` | integer | 1-4096 | tokens the model generated for the accepted attempt |

The relay adds `seq`, `at` and `ts` when it stores a turn, as it does now.

## JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "backrooms_talk POST /turn body",
  "type": "object",
  "additionalProperties": false,
  "required": ["speaker", "text", "engine_seq", "fly_index", "basis", "meta"],
  "properties": {
    "speaker": {"type": "string", "enum": ["A", "B", "C", "D"]},
    "to": {"type": "string", "enum": ["A", "B", "C", "D"]},
    "text": {"type": "string", "minLength": 1, "maxLength": 600},
    "engine_seq": {"type": "integer", "minimum": 1, "maximum": 9007199254740992},
    "fly_index": {"type": "integer", "minimum": 0, "maximum": 3},
    "basis": {
      "type": "array",
      "minItems": 0,
      "maxItems": 6,
      "items": {"type": "string", "minLength": 1, "maxLength": 200}
    },
    "meta": {
      "type": "object",
      "additionalProperties": false,
      "required": ["active_fraction", "brain_rate_hz", "song_hz", "sound_drive_hz", "smell_drive_hz",
                   "top_group_rate_hz", "top_group_baseline_hz", "nearest_fly_mm", "world_s",
                   "window_steps", "seed", "new_tokens"],
      "properties": {
        "active_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "brain_rate_hz": {"type": "number", "minimum": 0},
        "song_hz": {"type": "number", "minimum": 0},
        "sound_drive_hz": {"type": "number", "minimum": 0, "maximum": 200},
        "smell_drive_hz": {"type": "number", "minimum": 0, "maximum": 200},
        "top_group_rate_hz": {"type": "number", "minimum": 0},
        "top_group_baseline_hz": {"type": "number", "minimum": 0},
        "nearest_fly_mm": {"type": "number", "minimum": 0, "maximum": 29},
        "world_s": {"type": "number", "minimum": 0},
        "window_steps": {"type": "integer", "minimum": 1},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "new_tokens": {"type": "integer", "minimum": 1, "maximum": 4096}
      }
    }
  }
}
```

Notes for the relay
* JSON booleans are not numbers here: reject `true`/`false` in `meta`.
* `number` values are finite (the engine refuses NaN and infinity before sending).
* A body is well under the relay's 8,192-byte cap: 600 + 6 x 200 chars of text plus the numbers (at most 3 UTF-8 bytes per allowed character).
* The relay (backrooms_relay/relay.py, `TalkTurnIn`) enforces this schema; its test `test_turn_schema_md_agrees_with_the_relay` fails if the two drift. It also requires `fly_index` to be the speaker's position, `to` to be another fly (omitted, never null), and caps unbounded integers at 2^53. A body with neither `fly_index` nor `basis` is validated as the relay's older shape.

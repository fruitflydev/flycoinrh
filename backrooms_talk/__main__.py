"""
python -m backrooms_talk [--dry] [--turns N] [--relay URL] --home DIR

  --home DIR     the local runtime folder (or env BACKROOMS_TALK_HOME): holds
                 models/LFM2.5-1.2B-Instruct (with its LICENSE), .env
                 (RELAY_TOKEN, RELAY_URL), archive.jsonl, recent_topics.json and logs/
  --dry          print every turn, post nothing (no token is read)
  --turns N      stop after N turn slots (0 = forever)
  --relay URL    post here instead of RELAY_URL (a local test relay)

The token is read from the process environment (RELAY_TOKEN) if set, else
from DIR/.env; it is never printed. Run from the repository root with the
engine's Python (torch + transformers):

  python -m backrooms_talk --home <runtime folder> --dry --turns 12
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_RAM, EXIT_CONFIG = 0, 1, 3, 4
DEFAULT_RELAY_URL = "https://backrooms-production-004b.up.railway.app"
MODEL_SUBDIR = Path("models") / "LFM2.5-1.2B-Instruct"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backrooms_talk",
                                 description="four simulated flies; a small language model writes one line per turn")
    ap.add_argument("--home", default=os.environ.get("BACKROOMS_TALK_HOME"))
    ap.add_argument("--dry", action="store_true", help="print, no posting")
    ap.add_argument("--turns", type=int, default=0, help="turn slots, 0 = forever")
    ap.add_argument("--relay", default=None, help="relay base URL (default: RELAY_URL from .env)")
    ap.add_argument("--seed", type=int, default=None, help="run seed (default: from the clock, recorded)")
    ap.add_argument("--turn-seconds", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--archive", default=None, help="default: HOME/archive.jsonl")
    ap.add_argument("--summary", default=None, help="write the run's turn results and counts as JSON here")
    ap.add_argument("--no-ram-check", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from backrooms_talk import engine as E
    say = E.say_safe
    if not a.home:
        say("no runtime folder: pass --home DIR or set BACKROOMS_TALK_HOME")
        return EXIT_CONFIG
    home = Path(a.home)
    model_dir = Path(a.model_dir) if a.model_dir else home / MODEL_SUBDIR
    if not (model_dir / "LICENSE").exists():
        say(f"no LICENSE beside the model in {model_dir}: the license copy must stay with the weights")
        return EXIT_CONFIG
    archive = Path(a.archive) if a.archive else home / "archive.jsonl"
    seed = a.seed if a.seed is not None else int(time.time()) % 100_000

    from backrooms_talk.relay_client import RelayClient, ensure_env_key, read_env
    env_path = home / ".env"
    relay, mode = None, "dry"
    if not a.dry:
        if env_path.exists():
            ensure_env_key(env_path, "RELAY_URL", DEFAULT_RELAY_URL)
        env = read_env(env_path)
        url = a.relay or env.get("RELAY_URL") or DEFAULT_RELAY_URL
        token = os.environ.get("RELAY_TOKEN") or env.get("RELAY_TOKEN")
        if not token:
            say("no RELAY_TOKEN in the environment or in HOME/.env; use --dry to run without posting")
            return EXIT_CONFIG
        relay = RelayClient(url, token, say=say)
        mode = "relay:" + relay.url
        del token, env

    import backrooms_world as bw
    from backrooms_talk import model as lm
    from backrooms_talk.world import load_talk_room

    say(f"seed {seed}; mode {mode}; archive {archive}")
    try:
        room = load_talk_room(seed=seed, graph_path=a.graph, annotations_path=a.annotations, say=say,
                              ram_check=not a.no_ram_check)
    except MemoryError as exc:
        say(str(exc))
        return EXIT_RAM
    if not a.no_ram_check:
        ram = bw.free_ram_gb()
        if ram == ram and not ram > bw.MIN_FREE_RAM_GB:
            say(f"free RAM {ram:.1f} GB is not above {bw.MIN_FREE_RAM_GB:.1f} GB; not loading the language model")
            return EXIT_RAM
    model = None
    for k in range(E.OOM_TRIES):
        try:
            model = lm.LFMModel(model_dir, say=say)
            break
        except Exception as exc:
            if not lm.is_cuda_oom(exc) or k == E.OOM_TRIES - 1:
                raise
            say(f"CUDA out of memory loading the model; waiting {E.OOM_WAIT_S:.0f} s")
            time.sleep(E.OOM_WAIT_S)

    engine_seq0, history0, next_fly, topics0 = 0, [], 0, None
    if not a.dry:
        engine_seq0, history0, next_fly = E.resume_from_archive(archive, mode)
        topics0 = E.resume_topics(archive, mode)
    kw = {}
    if a.turn_seconds is not None:
        kw["turn_seconds"] = a.turn_seconds
    if a.warmup_steps is not None:
        kw["warmup_steps"] = a.warmup_steps
    eng = E.Engine(room, model, relay=relay, dry=a.dry, archive_path=archive, mode=mode, seed=seed,
                   sampling={"max_new_tokens": lm.MAX_NEW_TOKENS, "temperature": lm.TEMPERATURE,
                             "top_p": lm.TOP_P, "top_k": lm.TOP_K,
                             "repetition_penalty": lm.REPETITION_PENALTY},
                   engine_seq0=engine_seq0, history0=history0, next_fly=next_fly, topics0=topics0,
                   recent_topics_path=home / "recent_topics.json", **kw)
    t0 = time.time()
    try:
        eng.run(a.turns)
    finally:
        rss, peak = E.process_rss_gb()
        summary = {"seed": seed, "mode": mode, "wall_s": time.time() - t0, "counts": eng.counts(),
                   "ram_gb": {"working_set": rss, "peak_working_set": peak, "free_system": bw.free_ram_gb()},
                   "vram_gib": E.vram_gb(), "model_load_s": getattr(model, "load_s", None),
                   "room": {"names": list(room.names), "body_seeds": {n: b.seed for n, b in room.bodies.items()},
                            "batched": callable(getattr(room.fb, "run_batch", None))},
                   "turns": [{k: v for k, v in r.items() if k != "post"} for r in eng.results]}
        say("counts: " + json.dumps(summary["counts"]))
        say(f"RAM working set {rss:.2f} GB (peak {peak:.2f}); VRAM " + json.dumps(
            {k: round(v, 3) for k, v in summary["vram_gib"].items()}))
        if a.summary:
            Path(a.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n",
                                       encoding="utf-8")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

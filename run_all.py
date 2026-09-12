"""
Container entrypoint: the roamer, the voice, and - in the backroom build - a
paper executor, supervised together.

One container, a few long-running processes. roam.py drives the connectome
around the web and serves its telemetry; voice.py reads that telemetry, reads
the pages the fly encountered, and writes the journal. They talk over the
loopback socket, so the voice is pointed at whatever port the host handed the
roamer.

With FLY_BACKROOM=1 there is a third process, executor.py. It keeps a paper
ledger of what the fly's clicks would have bought and sold: it holds no
wallet, no private key and signs nothing, and it listens on loopback only.
Because it is the process that talks to the chain's RPC, it is also the one
process that must never be handed the account secrets - so each child here
gets its own environment rather than a copy of everything:

  executor   no X keys, no model key, no chain secret, no repo or blob token
  roam       no X keys, no model key, no chain secret
  voice      no intent token (it narrates the fly; it cannot order a trade)

The intent token is minted here, once per boot, and given only to the room
(inside roam) and the executor. Without FLY_BACKROOM=1 nothing about the two
existing processes changes.

If any process dies it is restarted after a short pause, forever. The voice is
only started when it has a model to call (OPENROUTER_API_KEY set, or the
offline "stub" model for testing); without one it would just log failures on a
timer, so it stays off and says so.
"""
import os
import secrets
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = "4660"
DEFAULT_EXECUTOR_PORT = "4671"
PORT = os.environ.get("PORT", DEFAULT_PORT)
RESTART_AFTER_S = 10
RESTART_MAX_S = 600        # backoff ceiling for a child that keeps dying
HEALTHY_AFTER_S = 300      # a child that lived this long resets its backoff
GIVE_UP_AFTER = 8          # consecutive fast deaths before the supervisor exits

# What each child must not be able to read. A name ending in "*" is a prefix.
EXECUTOR_DENY = ("X_*", "OPENROUTER_API_KEY", "FLY_RH_SECRET*",
                 "FLY_GH_TOKEN", "FLY_BLOB_TOKEN")
ROAM_DENY = ("X_*", "OPENROUTER_API_KEY", "FLY_RH_SECRET*")
VOICE_DENY = ("FLY_INTENT_TOKEN", "FLY_RH_SECRET*")


def say(msg):
    try:
        print(f"[run_all] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[run_all] {msg}".encode("ascii", "replace").decode(), flush=True)


def voice_enabled(env=None):
    env = os.environ if env is None else env
    if str(env.get("FLY_VOICE_MODEL", "")).strip().lower() == "stub":
        # the stub is a test fixture with hand-written lines; voice.py forces
        # dry for it, and the supervisor refuses to run it beside real X keys
        if str(env.get("X_ENABLED", "")).strip().lower() == "true":
            say("refusing to run the stub voice with X_ENABLED=true")
            return False
        return True
    if str(env.get("OPENROUTER_API_KEY", "")).strip():
        return True
    # a key in .env counts too; voice.py reads both
    env_path = os.path.join(HERE, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY=") and line.strip().split("=", 1)[1].strip():
                    return True
    return False


def backroom_on(env=None):
    env = os.environ if env is None else env
    return str(env.get("FLY_BACKROOM", "")).strip() == "1"


def without(env, *names):
    """
    A copy of env with these names removed; a name ending in "*" removes every
    variable that starts with it. Removing rather than blanking matters: a
    library that checks "is this set" must see nothing at all.
    """
    out = {}
    for k, v in env.items():
        drop = False
        for n in names:
            if (k.startswith(n[:-1]) if n.endswith("*") else k == n):
                drop = True
                break
        if not drop:
            out[k] = v
    return out


def child_env(base, *deny):
    """
    One child's environment: the denied names removed, and the same list handed
    on in FLY_ENV_DENY.

    Removing them from the environment is only half of it. roam.py and voice.py
    read their settings through launch.load_env, which reads .env from the repo
    and so handed back every name stripped here - on any machine with a .env,
    which is every desk, the guarantee above this was simply false. load_env
    now drops what FLY_ENV_DENY names, so a key a child was started without
    cannot be read out of the file either.
    """
    env = without(base, *deny)
    if deny:
        env["FLY_ENV_DENY"] = ",".join(deny)
    return env


class Proc:
    def __init__(self, name, argv, env):
        self.name, self.argv, self.env = name, argv, env
        self.p = None
        self.died_at = None
        self.started_at = None
        self.fast_deaths = 0       # consecutive exits before HEALTHY_AFTER_S
        self.gave_up = False

    def start(self):
        say(f"starting {self.name}: {' '.join(self.argv[1:])}")
        self.p = subprocess.Popen(self.argv, cwd=HERE, env=self.env)
        self.started_at = time.time()
        self.died_at = None

    def delay(self):
        # 10s, 20s, 40s ... capped, so a crash loop cannot peg the CPU by
        # reloading the connectome every few seconds
        return min(RESTART_MAX_S, RESTART_AFTER_S * (2 ** max(0, self.fast_deaths - 1)))

    def tick(self):
        if self.p is None or self.gave_up:
            return
        rc = self.p.poll()
        if rc is None:
            return
        if self.died_at is None:
            self.died_at = time.time()
            lived = self.died_at - (self.started_at or self.died_at)
            self.fast_deaths = self.fast_deaths + 1 if lived < HEALTHY_AFTER_S else 1
            if self.fast_deaths >= GIVE_UP_AFTER:
                self.gave_up = True
                say(f"{self.name} died {self.fast_deaths} times in a row; giving up so the platform sees it")
                return
            say(f"{self.name} exited with {rc} after {lived:.0f}s; restarting in {self.delay()}s")
        elif time.time() - self.died_at >= self.delay():
            self.start()

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.p.kill()


def build_procs(base=None):
    """
    The children of this boot, in start order, each with its own environment.

    Nothing is started here, so the wiring can be read - and tested - without
    a single process existing. The executor comes first because the room will
    start asking it for marks as soon as roam is up.
    """
    base = dict(os.environ if base is None else base)
    base.setdefault("PYTHONUNBUFFERED", "1")
    port = base.get("PORT") or DEFAULT_PORT
    room = backroom_on(base)
    # One resolved state directory for every child. The executor writes the
    # ledger there, the room reads the book back out of it and the mushroom
    # body keeps its gains beside them, so the three cannot be allowed to
    # resolve it differently - including when the only place it is written is
    # .env, which not every child reads the same way.
    sd = base.get("FLY_STATE_DIR")
    if not sd:
        try:
            try:
                from launch import load_env
            except ImportError:                # the public copy calls it envcfg
                from envcfg import load_env
            sd = load_env().get("FLY_STATE_DIR")
        except Exception:
            sd = None
    if sd:
        base["FLY_STATE_DIR"] = sd if os.path.isabs(sd) else os.path.abspath(sd)

    procs = []
    if room:
        base["FLY_EXECUTOR_PORT"] = base.get("FLY_EXECUTOR_PORT") or DEFAULT_EXECUTOR_PORT
        # one secret per boot, minted here. The room and the executor are the
        # only processes that ever see it, and it dies with the container.
        base["FLY_INTENT_TOKEN"] = secrets.token_hex(32)
        if not os.path.exists(os.path.join(HERE, "executor.py")):
            say("FLY_BACKROOM=1 but executor.py is not here; it will fail to start")
        procs.append(Proc("executor", [sys.executable, "executor.py"],
                          child_env(base, *EXECUTOR_DENY)))

    roam_env = child_env(base, *ROAM_DENY) if room else dict(base)
    roam_env["PORT"] = port
    procs.append(Proc("roam", [sys.executable, "roam.py"], roam_env))

    if voice_enabled(base):
        # the voice never holds the intent token, backroom or not: it writes
        # the journal, and nothing it can reach places an order
        voice_env = child_env(base, *VOICE_DENY) if room else child_env(base, "FLY_INTENT_TOKEN")
        # the voice reads the roamer over loopback, whatever the public port is
        voice_env["FLY_STREAM"] = f"http://127.0.0.1:{port}"
        procs.append(Proc("voice", [sys.executable, "voice.py", "--loop"], voice_env))
    else:
        say("voice disabled: no OPENROUTER_API_KEY and FLY_VOICE_MODEL is not 'stub'")
    return procs


def missing_for_room(here=HERE):
    """
    What the backroom needs from the image and has not got, named in one place.

    A deploy image is built from a requirements file and a few COPY lines, and
    three things the room cannot open are easy to leave out of them: the ABI
    codec and hash backend pons.py imports (requirements-executor.txt, which is
    not what the roaming image installs), the DoOR response tables the nose
    reads (door/ or data/door/), and build/mb_sides.json, which is derived and
    gitignored so an image can be built without it. Missing the first, the
    executor dies on an import at every restart until the supervisor gives up
    and takes the roamer and the voice down with it - eight restarts and a
    twenty-minute silence for a one-line cause. Said at boot instead.
    """
    out = []
    for mod in ("eth_abi", "eth_utils"):
        try:
            __import__(mod)
        except Exception:
            out.append(f"python module {mod} (pip install -r requirements-executor.txt)")
    if not any(os.path.exists(os.path.join(here, *p))
               for p in (("door", "door_response_matrix.csv"),
                         ("data", "door", "door_response_matrix.csv"))):
        out.append("the DoOR response tables (door/ or data/door/)")
    if not os.path.exists(os.path.join(here, "build", "mb_sides.json")):
        out.append("build/mb_sides.json (run mb_sides.py and ship it)")
    for name in ("executor.py", "backroom.py", "tradebook.py", "pons.py",
                 "olfaction.py", "calibration.py"):
        if not os.path.exists(os.path.join(here, name)):
            out.append(name)
    return out


def main():
    procs = build_procs()
    if backroom_on():
        say(f"backroom on: paper executor on 127.0.0.1:"
            f"{os.environ.get('FLY_EXECUTOR_PORT', DEFAULT_EXECUTOR_PORT)}, no wallet, no key")
        for missing in missing_for_room():
            say(f"BACKROOM PREFLIGHT: this image is missing {missing}")

    stopping = {"now": False}

    def on_signal(signum, _frame):
        stopping["now"] = True
        say(f"signal {signum}, stopping children")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass

    sd = os.environ.get("FLY_STATE_DIR")
    if sd:
        say(f"state dir {sd}: exists={os.path.isdir(sd)} mount={os.path.ismount(sd)}")
        if not os.path.ismount(sd):
            say("WARNING: FLY_STATE_DIR is not a mount point; state will not survive a redeploy")

    for pr in procs:
        pr.start()
    rc = 0
    try:
        while not stopping["now"]:
            for pr in procs:
                pr.tick()
            if any(pr.gave_up for pr in procs):
                rc = 1
                break
            time.sleep(2)
    finally:
        for pr in procs:
            pr.stop()
        say("stopped")
    sys.exit(rc)


if __name__ == "__main__":
    main()

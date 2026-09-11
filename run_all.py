"""
Container entrypoint: the roamer and the voice, supervised together.

One container, two long-running processes. roam.py drives the connectome
around the web and serves its telemetry; voice.py reads that telemetry, reads
the pages the fly encountered, and writes the journal. They talk over the
loopback socket, so the voice is pointed at whatever port the host handed the
roamer.

If either process dies it is restarted after a short pause, forever. The voice
is only started when it has a model to call (OPENROUTER_API_KEY set, or the
offline "stub" model for testing); without one it would just log failures on a
timer, so it stays off and says so.
"""
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.environ.get("PORT", "4660")
RESTART_AFTER_S = 10


def say(msg):
    try:
        print(f"[run_all] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[run_all] {msg}".encode("ascii", "replace").decode(), flush=True)


def voice_enabled():
    if os.environ.get("FLY_VOICE_MODEL", "").strip().lower() == "stub":
        return True
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return True
    # a key in .env counts too; voice.py reads both
    env_path = os.path.join(HERE, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY=") and line.strip().split("=", 1)[1].strip():
                    return True
    return False


class Proc:
    def __init__(self, name, argv, env):
        self.name, self.argv, self.env = name, argv, env
        self.p = None
        self.died_at = None

    def start(self):
        say(f"starting {self.name}: {' '.join(self.argv[1:])}")
        self.p = subprocess.Popen(self.argv, cwd=HERE, env=self.env)
        self.died_at = None

    def tick(self):
        if self.p is None:
            return
        rc = self.p.poll()
        if rc is None:
            return
        if self.died_at is None:
            self.died_at = time.time()
            say(f"{self.name} exited with {rc}; restarting in {RESTART_AFTER_S}s")
        elif time.time() - self.died_at >= RESTART_AFTER_S:
            self.start()

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.p.kill()


def main():
    base = dict(os.environ)
    base.setdefault("PYTHONUNBUFFERED", "1")

    roam_env = dict(base)
    roam_env["PORT"] = PORT

    voice_env = dict(base)
    # the voice reads the roamer over loopback, whatever the public port is
    voice_env["FLY_STREAM"] = f"http://127.0.0.1:{PORT}"

    procs = [Proc("roam", [sys.executable, "roam.py"], roam_env)]
    if voice_enabled():
        procs.append(Proc("voice", [sys.executable, "voice.py", "--loop"], voice_env))
    else:
        say("voice disabled: no OPENROUTER_API_KEY and FLY_VOICE_MODEL is not 'stub'")

    stopping = {"now": False}

    def on_signal(signum, _frame):
        stopping["now"] = True
        say(f"signal {signum}, stopping children")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass

    for pr in procs:
        pr.start()
    try:
        while not stopping["now"]:
            for pr in procs:
                pr.tick()
            time.sleep(2)
    finally:
        for pr in procs:
            pr.stop()
        say("stopped")


if __name__ == "__main__":
    main()

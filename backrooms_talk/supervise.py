"""
Keep the generator running: start `python -m backrooms_talk`, restart it after
a crash with backoff, log everything to HOME/logs/, and stop for good only
when the generator exits cleanly (0) or reports a configuration error (4).

  python backrooms_talk/supervise.py --home <runtime folder> [-- generator args]
  python backrooms_talk/supervise.py --home <runtime folder> --stop

Low RAM never ends supervision: when free RAM is not above the brain's load
threshold (backrooms_world.MIN_FREE_RAM_GB) before a start, or the generator
refuses to load for RAM (exit 3), the supervisor waits RAM_WAIT_S and checks
again, the way the engine waits on CUDA out of memory. It never stops another
process to free memory.

PID files (HOME/logs): supervise.pid holds the supervisor's PID while it runs;
generator.pid holds the running generator's PID and is removed when that
generator exits. --stop ends the supervisor first (so nothing restarts) and
then the generator, each only if its PID still belongs to a python process.

Backoff (CHOSEN): 10 s after the first crash, doubling to at most 600 s; a
run that stayed up MIN_UPTIME_RESET_S (10 min) resets it. RAM_WAIT_S (CHOSEN)
= 120 s. The supervisor never reads or logs the .env; the generator does that
itself. It does not start itself after a reboot.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BACKOFF0_S = 10.0          # CHOSEN
BACKOFF_MAX_S = 600.0      # CHOSEN
MIN_UPTIME_RESET_S = 600.0  # CHOSEN
RAM_WAIT_S = 120.0         # CHOSEN: wait on low RAM, then check again
EXIT_OK, EXIT_RAM, EXIT_CONFIG = 0, 3, 4
SUPERVISE_PID = "supervise.pid"
GENERATOR_PID = "generator.pid"


def stamp():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ---- PID files -----------------------------------------------------------------

def write_pid(path, pid):
    Path(path).write_text(str(int(pid)) + "\n", encoding="ascii")


def read_pid(path):
    try:
        return int(Path(path).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def remove_pid(path, pid):
    """Remove the file only if it still names this PID (a newer start may have replaced it)."""
    if read_pid(path) == int(pid):
        try:
            Path(path).unlink()
        except OSError:
            pass


def is_python(pid):
    """True when `pid` is a running python process, so a stale PID file never stops something else."""
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=30).stdout
            return "python" in out.lower()
        return "python" in Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="replace").lower()
    except Exception:
        return False


def stop(home, say=print, is_python=is_python, kill=None):
    """End the supervisor, then the generator, from their PID files. Returns [(file, pid)] stopped."""
    logs = Path(home) / "logs"
    stopped = []
    for name in (SUPERVISE_PID, GENERATOR_PID):
        pid = read_pid(logs / name)
        if pid is None:
            say(f"{name}: no PID file")
            continue
        if not is_python(pid):
            say(f"{name}: PID {pid} is not a running python process; removing the stale file")
            remove_pid(logs / name, pid)
            continue
        if kill is not None:
            kill(pid)
        elif os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=60)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
        remove_pid(logs / name, pid)
        say(f"{name}: stopped PID {pid}")
        stopped.append((name, pid))
    return stopped


# ---- the supervisor ----------------------------------------------------------------

class Supervisor:
    def __init__(self, home, child_args=(), python=sys.executable, free_ram=None, min_free_gb=None,
                 spawn=None, sleep=time.sleep, clock=time.monotonic, log=None, write_pids=None):
        import backrooms_world as bw
        self.home = Path(home)
        self.logs = self.home / "logs"
        self.child_args = list(child_args)
        self.python = python
        self.free_ram = free_ram or bw.free_ram_gb
        self.min_free_gb = bw.MIN_FREE_RAM_GB if min_free_gb is None else float(min_free_gb)
        self.spawn = spawn or self._spawn
        self.sleep, self.clock = sleep, clock
        self._log = log
        self.backoff = BACKOFF0_S
        self.starts = 0
        self.waits = []
        self.ram_waits = 0
        self.write_pids = (spawn is None) if write_pids is None else bool(write_pids)

    def log(self, msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        if self._log is not None:
            self._log(line)
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.logs / "supervise.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def _spawn(self):
        """Start one generator; returns its exit code when it ends."""
        self.logs.mkdir(parents=True, exist_ok=True)
        path = self.logs / f"generator-{stamp()}.log"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        cmd = [self.python, "-m", "backrooms_talk", "--home", str(self.home), *self.child_args]
        with open(path, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT, env=env)
            pid_file = self.logs / GENERATOR_PID
            write_pid(pid_file, proc.pid)
            self.log(f"start: generator pid {proc.pid} (supervisor pid {os.getpid()}): "
                     f"{' '.join(cmd[1:])} -> {path.name}")
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise
            finally:
                remove_pid(pid_file, proc.pid)

    def run(self, max_starts=None, max_ram_waits=None):
        """Returns why it stopped: 'finished', 'config', 'max_starts', or 'ram' (only past max_ram_waits)."""
        if self.write_pids:
            self.logs.mkdir(parents=True, exist_ok=True)
            write_pid(self.logs / SUPERVISE_PID, os.getpid())
        try:
            return self._run(max_starts, max_ram_waits)
        finally:
            if self.write_pids:
                remove_pid(self.logs / SUPERVISE_PID, os.getpid())

    def _wait_ram(self, why, max_ram_waits):
        if max_ram_waits is not None and self.ram_waits >= max_ram_waits:
            self.log(f"stop: {why}, past {max_ram_waits} waits")
            return False
        self.ram_waits += 1
        self.log(f"{why}; waiting {RAM_WAIT_S:.0f} s and checking again (no other process is touched)")
        self.sleep(RAM_WAIT_S)
        return True

    def _run(self, max_starts, max_ram_waits):
        while max_starts is None or self.starts < max_starts:
            ram = self.free_ram()
            if ram == ram and not ram > self.min_free_gb:
                if not self._wait_ram(f"free RAM {ram:.1f} GB is not above {self.min_free_gb:.1f} GB",
                                      max_ram_waits):
                    return "ram"
                continue
            self.starts += 1
            t0 = self.clock()
            rc = self.spawn()
            up = self.clock() - t0
            if rc == EXIT_OK:
                self.log(f"generator finished (exit 0) after {up:.0f} s; stopping")
                return "finished"
            if rc == EXIT_RAM:
                if not self._wait_ram("generator refused to load for RAM (exit 3)", max_ram_waits):
                    return "ram"
                continue
            if rc == EXIT_CONFIG:
                self.log("generator configuration error (exit 4); stopping")
                return "config"
            if up >= MIN_UPTIME_RESET_S:
                self.backoff = BACKOFF0_S
            self.log(f"generator exited {rc} after {up:.0f} s; restarting in {self.backoff:.0f} s")
            self.waits.append(self.backoff)
            self.sleep(self.backoff)
            self.backoff = min(self.backoff * 2.0, BACKOFF_MAX_S)
        return "max_starts"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    child = []
    if "--" in argv:
        i = argv.index("--")
        argv, child = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="restart the backrooms_talk generator after a crash")
    ap.add_argument("--home", default=os.environ.get("BACKROOMS_TALK_HOME"), required=False)
    ap.add_argument("--stop", action="store_true", help="stop the running supervisor, then the generator, by PID file")
    a = ap.parse_args(argv)
    if not a.home:
        print("pass --home DIR or set BACKROOMS_TALK_HOME")
        return 2
    if a.stop:
        stop(a.home)
        return 0
    sup = Supervisor(a.home, child)
    try:
        reason = sup.run()
    except KeyboardInterrupt:
        sup.log("stopped by keyboard interrupt")
        return 0
    return 0 if reason == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())

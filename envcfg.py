"""Read .env without pulling in anything chain-specific."""
import os
from pathlib import Path

ROOT = Path(__file__).parent


def denied(name, patterns):
    """True when a name matches one of the deny patterns; a trailing * is a prefix."""
    for p in patterns:
        if name.startswith(p[:-1]) if p.endswith("*") else name == p:
            return True
    return False


def deny_list():
    """
    What this process was started without, as run_all.py passed it down.

    Removing a name from a child's environment is not enough on any machine
    that has a .env, because this function reads that file: every name the
    supervisor stripped came straight back, and the roaming browser held the X
    credentials, the model key and the chain secret again. The supervisor also
    sets FLY_ENV_DENY to the list it applied, and a denied name is dropped here
    whether it came from the file or from the environment.
    """
    return [p.strip() for p in os.environ.get("FLY_ENV_DENY", "").split(",") if p.strip()]


def load_env(path=ROOT / ".env"):
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    merged = {**env, **{k: v for k, v in os.environ.items() if k.startswith("FLY_")}}
    deny = deny_list()
    return {k: v for k, v in merged.items() if not denied(k, deny)}

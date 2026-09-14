"""
POST /turn to the backrooms relay, with retries and backoff. Standard
library only. The token is read by the engine from the local .env and sent
only in the Authorization header; it is never printed, logged or archived.

Retried: network errors, timeouts, HTTP 408, 425, 429 and 5xx, up to
RETRIES more attempts, waiting BACKOFF_S x 2^k (capped at MAX_BACKOFF_S).
Not retried: any other 4xx (a malformed payload, a bad token, a line the
relay's filter blocks), since sending the same bytes again cannot succeed.
"""
import json
import time
import urllib.error
import urllib.request

RETRIES = 5              # CHOSEN
BACKOFF_S = 2.0          # CHOSEN
MAX_BACKOFF_S = 60.0     # CHOSEN
TIMEOUT_S = 15.0         # CHOSEN
RETRY_STATUS = {408, 425, 429}


def read_env(path):
    """KEY=VALUE lines of a .env file as a dict (values are never printed by any caller)."""
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                out[k.strip()] = v
    except FileNotFoundError:
        pass
    return out


def ensure_env_key(path, key, value):
    """Append KEY=value to the .env when the key is absent; returns True if it wrote. Reads key names only."""
    if key in read_env(path):
        return False
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "ab") as f:
        if data and not data.endswith(b"\n"):
            f.write(b"\n")
        f.write(f"{key}={value}\n".encode("utf-8"))
    return True


class RelayClient:
    def __init__(self, url, token, retries=RETRIES, backoff_s=BACKOFF_S, max_backoff_s=MAX_BACKOFF_S,
                 timeout_s=TIMEOUT_S, sleep=time.sleep, say=print):
        if not token:
            raise ValueError("no relay token")
        self.url = str(url).rstrip("/")
        self._token = str(token)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self.max_backoff_s = float(max_backoff_s)
        self.timeout_s = float(timeout_s)
        self.sleep = sleep
        self.say = say

    def __repr__(self):                      # never shows the token
        return f"RelayClient({self.url!r})"

    def post_turn(self, payload):
        """
        Returns {"ok", "status", "attempts", "body", "error"}: ok with the
        relay's JSON body on a 2xx, else the last status / error text (the
        response body is kept, the request headers are not).
        """
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        status, body, error = None, None, None
        for attempt in range(self.retries + 1):
            if attempt:
                wait = min(self.backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
                self.sleep(wait)
            req = urllib.request.Request(self.url + "/turn", data=data, method="POST", headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._token,
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    status = resp.status
                    raw = resp.read()
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = raw[:500].decode("utf-8", "replace")
                return {"ok": True, "status": status, "attempts": attempt + 1, "body": body, "error": None}
            except urllib.error.HTTPError as e:
                status = e.code
                try:
                    raw = e.read()
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = None
                error = f"HTTP {e.code}"
                if not (e.code in RETRY_STATUS or 500 <= e.code < 600):
                    return {"ok": False, "status": status, "attempts": attempt + 1, "body": body, "error": error}
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                status, body = None, None
                error = f"{type(e).__name__}: {str(getattr(e, 'reason', e))[:120]}"
            self.say(f"relay post attempt {attempt + 1} failed: {error}")
        return {"ok": False, "status": status, "attempts": self.retries + 1, "body": body, "error": error}

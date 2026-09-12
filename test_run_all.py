"""
What the supervisor would start, and with which environment. No process is
ever launched here: build_procs only describes the children, and the one test
that calls start() replaces subprocess.Popen with a recorder. Every secret in
these fixtures is a made-up placeholder - the real .env is never read.

  py -m unittest test_run_all -v
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import launch
except ImportError:                      # the public copy calls it envcfg
    import envcfg as launch
import run_all


def base_env(**over):
    """A plausible boot environment, all values invented."""
    e = {
        "PATH": "C:/windows",
        "PORT": "4660",
        "FLY_STATE_DIR": "C:/tmp/state",
        "FLY_VOICE_MODEL": "stub",          # keeps voice_enabled off the network
        "OPENROUTER_API_KEY": "placeholder-model-key",
        "X_API_KEY": "placeholder-x-key",
        "X_API_SECRET": "placeholder-x-secret",
        "X_ENABLED": "false",
        "FLY_RH_SECRET": "placeholder-chain-secret",
        "FLY_RH_SECRET_2": "placeholder-chain-secret",
        "FLY_GH_TOKEN": "placeholder-repo-token",
        "FLY_BLOB_TOKEN": "placeholder-blob-token",
    }
    e.update(over)
    return e


def by_name(env):
    return {p.name: p for p in run_all.build_procs(env)}


class Wiring(unittest.TestCase):
    def test_two_processes_without_the_backroom(self):
        procs = run_all.build_procs(base_env())
        self.assertEqual([p.name for p in procs], ["roam", "voice"])
        for p in procs:
            self.assertNotIn("FLY_INTENT_TOKEN", p.env)

    def test_the_executor_starts_first_with_the_backroom(self):
        procs = run_all.build_procs(base_env(FLY_BACKROOM="1"))
        self.assertEqual([p.name for p in procs], ["executor", "roam", "voice"])
        self.assertEqual(procs[0].argv, [sys.executable, "executor.py"])

    def test_backroom_must_be_exactly_one(self):
        for v in ("0", "", "true", "yes", "2"):
            with self.subTest(v=v):
                names = [p.name for p in run_all.build_procs(base_env(FLY_BACKROOM=v))]
                self.assertNotIn("executor", names)

    def test_scripts_and_ports(self):
        p = by_name(base_env(FLY_BACKROOM="1"))
        self.assertEqual(p["roam"].argv, [sys.executable, "roam.py"])
        self.assertEqual(p["voice"].argv, [sys.executable, "voice.py", "--loop"])
        self.assertEqual(p["executor"].env["FLY_EXECUTOR_PORT"], "4671")
        self.assertEqual(p["roam"].env["FLY_EXECUTOR_PORT"], "4671")
        self.assertEqual(p["roam"].env["PORT"], "4660")
        self.assertEqual(p["voice"].env["FLY_STREAM"], "http://127.0.0.1:4660")

    def test_a_chosen_port_reaches_both_sides(self):
        p = by_name(base_env(FLY_BACKROOM="1", FLY_EXECUTOR_PORT="4999", PORT="8080"))
        self.assertEqual(p["executor"].env["FLY_EXECUTOR_PORT"], "4999")
        self.assertEqual(p["roam"].env["FLY_EXECUTOR_PORT"], "4999")
        self.assertEqual(p["voice"].env["FLY_STREAM"], "http://127.0.0.1:8080")


class Token(unittest.TestCase):
    def test_sixty_four_hex_shared_by_the_room_and_the_executor(self):
        p = by_name(base_env(FLY_BACKROOM="1"))
        tok = p["executor"].env["FLY_INTENT_TOKEN"]
        self.assertRegex(tok, re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(p["roam"].env["FLY_INTENT_TOKEN"], tok)

    def test_a_new_token_every_boot(self):
        a = by_name(base_env(FLY_BACKROOM="1"))["executor"].env["FLY_INTENT_TOKEN"]
        b = by_name(base_env(FLY_BACKROOM="1"))["executor"].env["FLY_INTENT_TOKEN"]
        self.assertNotEqual(a, b)

    def test_the_voice_never_holds_it(self):
        p = by_name(base_env(FLY_BACKROOM="1"))
        self.assertNotIn("FLY_INTENT_TOKEN", p["voice"].env)
        # and not even if one was somehow already in the boot environment
        p = by_name(base_env(FLY_INTENT_TOKEN="a" * 64))
        self.assertNotIn("FLY_INTENT_TOKEN", p["voice"].env)

    def test_the_caller_environment_is_not_mutated(self):
        env = base_env(FLY_BACKROOM="1")
        run_all.build_procs(env)
        self.assertNotIn("FLY_INTENT_TOKEN", env)


class Environments(unittest.TestCase):
    """Each child sees only what it needs. Names, never values."""

    def setUp(self):
        self.p = by_name(base_env(FLY_BACKROOM="1"))

    def test_the_executor_has_no_keys_at_all(self):
        env = self.p["executor"].env
        for k in ("X_API_KEY", "X_API_SECRET", "X_ENABLED", "OPENROUTER_API_KEY",
                  "FLY_RH_SECRET", "FLY_RH_SECRET_2", "FLY_GH_TOKEN", "FLY_BLOB_TOKEN"):
            self.assertNotIn(k, env)
        self.assertEqual(env["FLY_STATE_DIR"], "C:/tmp/state")
        self.assertIn("FLY_INTENT_TOKEN", env)

    def test_the_roamer_keeps_only_what_it_publishes_with(self):
        env = self.p["roam"].env
        for k in ("X_API_KEY", "X_API_SECRET", "X_ENABLED",
                  "OPENROUTER_API_KEY", "FLY_RH_SECRET", "FLY_RH_SECRET_2"):
            self.assertNotIn(k, env)
        self.assertIn("FLY_GH_TOKEN", env)      # it writes the tunnel address
        self.assertIn("FLY_BLOB_TOKEN", env)

    def test_the_voice_keeps_its_model_and_its_account(self):
        env = self.p["voice"].env
        self.assertIn("OPENROUTER_API_KEY", env)
        self.assertIn("X_API_KEY", env)
        for k in ("FLY_RH_SECRET", "FLY_RH_SECRET_2", "FLY_INTENT_TOKEN"):
            self.assertNotIn(k, env)

    def test_nothing_changes_without_the_backroom(self):
        base = base_env()
        p = by_name(base)
        want_roam = dict(base, PYTHONUNBUFFERED="1", PORT="4660")
        want_voice = dict(base, PYTHONUNBUFFERED="1", FLY_STREAM="http://127.0.0.1:4660",
                          FLY_ENV_DENY="FLY_INTENT_TOKEN")
        self.assertEqual(p["roam"].env, want_roam)
        self.assertEqual(p["voice"].env, want_voice)

    def test_each_child_carries_the_list_that_was_applied_to_it(self):
        self.assertEqual(self.p["executor"].env["FLY_ENV_DENY"], ",".join(run_all.EXECUTOR_DENY))
        self.assertEqual(self.p["roam"].env["FLY_ENV_DENY"], ",".join(run_all.ROAM_DENY))
        self.assertEqual(self.p["voice"].env["FLY_ENV_DENY"], ",".join(run_all.VOICE_DENY))

    def test_prefix_matching_is_a_prefix_not_a_substring(self):
        env = run_all.without({"X_API_KEY": "1", "MY_X_API_KEY": "2", "FLY_RH_SECRETS": "3",
                               "FLY_RH_RPC": "4"}, "X_*", "FLY_RH_SECRET*")
        self.assertEqual(env, {"MY_X_API_KEY": "2", "FLY_RH_RPC": "4"})


class TheFileTheChildrenRead(unittest.TestCase):
    """
    Stripping a name from a child's environment is only half of it.

    roam.py and voice.py read their settings through launch.load_env, which
    reads .env from the repo, so every name the supervisor removed used to come
    straight back - on any machine that has a .env, which is every desk. These
    tests assert on what load_env() returns inside a child, not on Proc.env.
    Every value below is a made-up placeholder written into a temporary file;
    the real .env is never opened.
    """

    def child(self, name):
        return by_name(base_env(FLY_BACKROOM="1"))[name].env

    def fake_dotenv(self):
        p = Path(tempfile.mkdtemp()) / ".env"
        p.write_text("\n".join(["X_API_KEY=placeholder-x-key",
                                "X_ENABLED=true",
                                "OPENROUTER_API_KEY=placeholder-model-key",
                                "FLY_RH_SECRET=placeholder-chain-secret",
                                "FLY_GH_TOKEN=placeholder-repo-token",
                                "FLY_INTENT_TOKEN=placeholder-intent-token",
                                "FLY_STATE_DIR=C:/tmp/state"]), encoding="utf-8")
        return p

    def read_as(self, name):
        with mock.patch.dict(os.environ, {"FLY_ENV_DENY": self.child(name)["FLY_ENV_DENY"]}):
            return launch.load_env(self.fake_dotenv())

    def test_the_roamer_cannot_read_back_what_it_was_started_without(self):
        got = self.read_as("roam")
        for k in ("X_API_KEY", "X_ENABLED", "OPENROUTER_API_KEY", "FLY_RH_SECRET"):
            self.assertNotIn(k, got)
        self.assertIn("FLY_GH_TOKEN", got)          # it writes the tunnel address
        self.assertEqual(got["FLY_STATE_DIR"], "C:/tmp/state")

    def test_the_executor_cannot_read_back_a_key_either(self):
        got = self.read_as("executor")
        for k in ("X_API_KEY", "OPENROUTER_API_KEY", "FLY_RH_SECRET", "FLY_GH_TOKEN"):
            self.assertNotIn(k, got)
        self.assertIn("FLY_INTENT_TOKEN", got)
        self.assertEqual(got["FLY_STATE_DIR"], "C:/tmp/state")

    def test_the_voice_cannot_read_back_the_intent_token(self):
        got = self.read_as("voice")
        self.assertNotIn("FLY_INTENT_TOKEN", got)
        self.assertNotIn("FLY_RH_SECRET", got)
        self.assertIn("X_API_KEY", got)             # it does hold its own account

    def test_without_a_deny_list_the_file_is_read_as_before(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLY_ENV_DENY", None)
            got = launch.load_env(self.fake_dotenv())
        self.assertIn("X_API_KEY", got)

    def test_a_state_dir_that_lives_only_in_the_file_still_reaches_every_child(self):
        # the executor writes the ledger there and the room reads the book back
        # out of it; two resolutions of this name means no position is ever held
        base = base_env(FLY_BACKROOM="1")
        base.pop("FLY_STATE_DIR")
        with mock.patch.object(launch, "load_env",
                               return_value={"FLY_STATE_DIR": "C:/only-in-the-file"}):
            procs = {p.name: p for p in run_all.build_procs(base)}
        for name in ("executor", "roam", "voice"):
            self.assertEqual(procs[name].env["FLY_STATE_DIR"], "C:/only-in-the-file", name)

    def test_a_prefix_in_the_list_is_a_prefix(self):
        self.assertTrue(launch.denied("FLY_RH_SECRET_2", ["FLY_RH_SECRET*"]))
        self.assertFalse(launch.denied("MY_X_API_KEY", ["X_*"]))
        self.assertTrue(launch.denied("X_API_KEY", ["X_*"]))


class Starting(unittest.TestCase):
    """start() is the only place a process would appear; Popen is a recorder."""

    def test_start_passes_the_argv_cwd_and_env_through(self):
        seen = []

        class FakePopen:
            def __init__(self, argv, cwd=None, env=None):
                seen.append({"argv": list(argv), "cwd": cwd, "env": dict(env or {})})

            def poll(self):
                return None

        with mock.patch.object(run_all.subprocess, "Popen", FakePopen):
            procs = run_all.build_procs(base_env(FLY_BACKROOM="1"))
            for pr in procs:
                pr.start()
        self.assertEqual([s["argv"][1] for s in seen], ["executor.py", "roam.py", "voice.py"])
        self.assertTrue(all(s["cwd"] == run_all.HERE for s in seen))
        self.assertNotIn("FLY_INTENT_TOKEN", seen[2]["env"])       # the voice
        self.assertIn("FLY_INTENT_TOKEN", seen[0]["env"])          # the executor

    def test_a_deploy_image_that_cannot_run_the_room_says_so_before_it_tries(self):
        """
        The room needs three things a deploy image is easy to build without:
        the ABI codec and hash backend pons.py imports (requirements-executor
        .txt, which is not what the roaming image installs), the DoOR response
        tables, and build/mb_sides.json, which is derived and gitignored. With
        the first missing, executor.py dies on an import at every restart until
        the supervisor gives up and takes the roamer, the voice and the public
        stream down with it - eight restarts, about twenty minutes, for a
        one-line cause nobody is told.
        """
        with tempfile.TemporaryDirectory() as tmp:
            missing = run_all.missing_for_room(tmp)
        self.assertTrue(any("door" in m.lower() for m in missing), missing)
        self.assertTrue(any("mb_sides.json" in m for m in missing), missing)
        self.assertTrue(any("executor.py" == m for m in missing), missing)

    def test_a_repo_that_has_everything_complains_about_nothing(self):
        """This checkout is the reference: it is what the image has to carry."""
        self.assertEqual(run_all.missing_for_room(), [])

    def test_backoff_is_unchanged(self):
        pr = run_all.Proc("x", ["x"], {})
        got = []
        for n in range(1, 9):
            pr.fast_deaths = n
            got.append(pr.delay())
        self.assertEqual(got, [10, 20, 40, 80, 160, 320, 600, 600])


if __name__ == "__main__":
    unittest.main()

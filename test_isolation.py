"""
The two sides of the house, kept apart by their import lists.

The fly's side - the roamer, the room, the pilot, the narrator and the poster
- never imports the money side, so no code path from a click can reach a
ledger, a wallet or a signer by accident. The money side - the paper executor
and its ledger - never imports a wallet or a key library at all, which is what
makes "paper only" a property of the file set rather than a promise.

This reads the files with ast rather than importing them: importing roam.py
would start a browser, and a file that cannot be imported is exactly the file
this test still has to check.

  py -m unittest test_isolation -v
"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).parent

# The fly's side may not touch any of these.
FLY_SIDE = ("roam.py", "backroom.py", "flyeye.py", "voice.py", "xpost.py")
FLY_FORBIDDEN = {"eth_account", "executor", "tradebook", "rhwallet", "rhprovider"}

# The paper side may hold no wallet and no signer.
PAPER_SIDE = ("executor.py", "tradebook.py")
PAPER_FORBIDDEN = {"eth_account", "rhwallet", "rhprovider"}

MISSING = [n for n in FLY_SIDE + PAPER_SIDE if not (HERE / n).exists()]


def imported_modules(path):
    """
    Every top-level module name a file asks for: plain imports, from-imports
    (including ones inside a function, where a lazy import would hide), and
    the string argument of importlib.import_module or __import__.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        found.add(a.value.split(".")[0])
    return found


class Isolation(unittest.TestCase):
    def check(self, names, forbidden):
        for name in names:
            path = HERE / name
            if not path.exists():
                continue
            with self.subTest(file=name):
                hit = sorted(imported_modules(path) & forbidden)
                self.assertEqual(hit, [], f"{name} imports {hit}")

    def test_the_fly_side_cannot_reach_the_money_side(self):
        self.check(FLY_SIDE, FLY_FORBIDDEN)

    def test_the_paper_side_holds_no_wallet_and_no_signer(self):
        self.check(PAPER_SIDE, PAPER_FORBIDDEN)

    def test_the_scanner_actually_sees_a_hidden_import(self):
        # the guard above is only worth having if it catches a lazy import
        src = ("def f():\n"
               "    import eth_account\n"
               "    from rhwallet import load\n"
               "    __import__('rhprovider')\n")
        tmp = HERE / "build" / "_isolation_probe.py"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(src, encoding="utf-8")
        try:
            self.assertEqual(imported_modules(tmp) & FLY_FORBIDDEN,
                             {"eth_account", "rhwallet", "rhprovider"})
        finally:
            tmp.unlink()

    def test_every_file_exists(self):
        if MISSING:
            self.skipTest("not built yet, so not scanned: " + ", ".join(MISSING))


if __name__ == "__main__":
    unittest.main()

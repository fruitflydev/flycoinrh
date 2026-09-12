"""
Rule 1 of the backrooms build, made executable: no language model anywhere.

Every backrooms*.py module is parsed with `ast` and the test fails if any of
them imports a model or HTTP client library (openai, anthropic, transformers,
llama_cpp, requests, httpx and a few of their relatives), refers to such a
name, imports anything dynamically, or carries the address of a model host
or the name of a local model runner in a string literal. Test modules are
held to the import rule too. The page, when it exists, must load no script
from another host and must say on its banner that no language model is
involved. Finally, every line the captioner can write is checked once more
against the template allow-list, so the words on the page are the words in
the code.

The scan is syntactic, which is the point: it does not matter what a module
would do at run time, a forbidden name in the source is enough to fail.

  py -m pytest -q test_backrooms_honesty.py
"""
import ast
import re
import unittest
from pathlib import Path

import backrooms as br
import backrooms_dictionary as bd
from test_backrooms_captions import SIZES, rich_record

ROOT = Path(__file__).parent
MODULES = sorted(ROOT.glob("backrooms*.py"))
TESTS = sorted(ROOT.glob("test_backrooms*.py"))
PAGE = ROOT / "web" / "backrooms.html"

# model libraries and HTTP clients: an import of any of these, or of a
# submodule, fails. urllib.parse stays allowed; urllib.request does not.
BANNED_MODULES = (
    "openai", "anthropic", "transformers", "llama_cpp", "requests", "httpx",
    "aiohttp", "urllib.request", "http.client", "ollama", "google.generativeai",
    "google.genai", "vllm", "mlx_lm", "ctransformers", "sentence_transformers",
    "litellm", "cohere", "mistralai", "together", "replicate", "groq", "openrouter",
)
# a bare reference to one of these names anywhere in a module fails, so a
# client cannot arrive through an alias, a getattr or an injected global
BANNED_NAMES = {"openai", "anthropic", "requests", "httpx", "urlopen",
                "publish_address", "blob_put", "transformers", "llama_cpp"}
DYNAMIC_IMPORTS = {"__import__", "import_module"}
# model hosts and local model runners: any string literal containing one fails
BANNED_LITERALS = (
    "api.openai.com", "api.anthropic.com", "openrouter.ai",
    "generativelanguage.googleapis.com", "api.groq.com", "api.together",
    "api.mistral.ai", "api.cohere", "huggingface.co", ":11434", "ollama",
    "llama-cli", "llama-server", "koboldcpp", "lmstudio", "lm-studio",
    "blob.vercel-storage.com", "api.github.com",
)


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def banned_module(name):
    return any(name == b or name.startswith(b + ".") for b in BANNED_MODULES)


def imports_of(tree):
    """Every module name a tree imports, statically, at any depth."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            yield mod
            for a in node.names:
                yield f"{mod}.{a.name}" if mod else a.name


def names_of(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr


def calls_of(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                yield f.id, node
            elif isinstance(f, ast.Attribute):
                yield f.attr, node


def literals_of(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


class ModulesExist(unittest.TestCase):
    def test_the_captioner_module_is_scanned(self):
        self.assertIn(ROOT / "backrooms.py", MODULES)
        self.assertIn(ROOT / "backrooms_dictionary.py", MODULES)
        self.assertNotIn(ROOT / "backroom.py", MODULES)      # the roamer's backroom is another program


class NoModelNoNetwork(unittest.TestCase):
    def test_no_module_imports_a_model_or_http_client(self):
        for path in MODULES + TESTS:
            for name in imports_of(parse(path)):
                self.assertFalse(banned_module(name), f"{path.name} imports {name}")

    def test_no_module_refers_to_a_client_by_name(self):
        for path in MODULES:
            for name in names_of(parse(path)):
                self.assertNotIn(name, BANNED_NAMES, f"{path.name} refers to {name}")

    def test_no_module_imports_dynamically(self):
        # One exemption: backrooms_world.brain_class resolves BRAIN_CLASS
        # ("flysim.FlyBrain", or the GPU port) by name, the designed one-line
        # brain swap. It is allowed only there, and the next test pins what
        # that name may be.
        for path in MODULES:
            tree = parse(path)
            exempt = set()
            if path.name == "backrooms_world.py":
                for fn in ast.walk(tree):
                    if isinstance(fn, ast.FunctionDef) and fn.name == "brain_class":
                        exempt.update(id(n) for n in ast.walk(fn))
            for name, node in calls_of(tree):
                if id(node) in exempt:
                    continue
                self.assertNotIn(name, DYNAMIC_IMPORTS,
                                 f"{path.name} line {node.lineno} imports dynamically")

    def test_the_brain_class_is_a_simulator(self):
        path = ROOT / "backrooms_world.py"
        if not path.exists():
            self.skipTest("backrooms_world.py not written yet")
        value = None
        for node in parse(path).body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "BRAIN_CLASS" for t in node.targets):
                self.assertIsInstance(node.value, ast.Constant)
                value = node.value.value
        self.assertIsInstance(value, str, "BRAIN_CLASS must be one string literal at module level")
        self.assertIn(value.rpartition(".")[0], ("flysim", "flysim_gpu"), value)

    def test_no_module_names_a_model_host_or_runner(self):
        for path in MODULES:
            for s in literals_of(parse(path)):
                low = s.lower()
                for b in BANNED_LITERALS:
                    self.assertNotIn(b, low, f"{path.name} carries {b!r} in a string")

    def test_the_scan_catches_what_it_claims(self):
        src = "import openai\nfrom llama_cpp import Llama\nx = requests.get(u)\nimport urllib.request\n"
        tree = ast.parse(src)
        found = [n for n in imports_of(tree) if banned_module(n)]
        self.assertEqual(found, ["openai", "llama_cpp", "llama_cpp.Llama", "urllib.request"])
        self.assertIn("requests", set(names_of(tree)))
        self.assertTrue(any(n in DYNAMIC_IMPORTS for n, _ in calls_of(ast.parse("__import__('x')"))))
        self.assertFalse(banned_module("urllib.parse"))
        self.assertFalse(banned_module("json"))


class WordsComeFromTheCode(unittest.TestCase):
    """The captioner's own promise, checked from the honesty side."""

    def test_every_word_of_every_line_is_in_the_templates_or_the_dictionary(self):
        voc = br.vocabulary()
        alts = "|".join(sorted((re.escape(w) for w in voc), key=len, reverse=True))
        allowed = re.compile(rf"^(?:{alts}|\d+)$")
        lines = br.caption_record(rich_record(), SIZES)
        self.assertGreater(len(lines), 10)
        for ln in lines:
            for tok in br.tokens(ln["text"]):
                self.assertRegex(tok, allowed, f"{tok!r} in {ln['text']!r}")

    def test_the_vocabulary_is_finite_and_small(self):
        self.assertLess(len(br.vocabulary()), 600)

    def test_templates_have_only_the_declared_placeholders(self):
        allowed = {"fly", "name", "value", "role", "cite", "unc", "drive", "other", "n", "dist", "rate"}
        for k, t in br.TEMPLATES.items():
            for ph in re.findall(r"\{(\w+)(?::[^}]*)?\}", t):
                self.assertIn(ph, allowed, f"template {k} has placeholder {ph}")

    def test_the_banner_text_says_what_rule_one_requires(self):
        how = br.HOW_LINES_ARE_MADE.lower()
        for phrase in ("no person", "no language model", "template", "measured",
                       "chosen", "slower than life", "uncertain",
                       # what the review found missing
                       "kinematic events", "only when that group fired",
                       "forward-walking neuron", "does not say what the fly heard",
                       "no start or end line is expected", "4.2 times faster"):
            self.assertIn(phrase, how)
        room = br.ROOM_NOTE.lower()
        for phrase in ("no synapse changes", "no dopamine", "do not learn",
                       "no left/right information", "one pixel per lamina column",
                       "no left-eye versus right-eye difference"):
            self.assertIn(phrase, room)

    def test_absent_groups_can_never_be_captioned(self):
        sizes = {k: 10 for k in bd.DICTIONARY}          # even if a caller claims sizes for them
        keys = br.captioned_keys(bd.DICTIONARY, sizes)
        for k in bd.ABSENT_KEYS:
            self.assertNotIn(k, keys)


@unittest.skipUnless(PAGE.exists(), "web/backrooms.html not written yet")
class ThePage(unittest.TestCase):
    def test_the_page_loads_nothing_from_another_host(self):
        html = PAGE.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"<script[^>]+src\s*=\s*[\"']https?://", html, re.I))
        self.assertIsNone(re.search(r"<link[^>]+href\s*=\s*[\"']https?://", html, re.I))
        self.assertIsNone(re.search(r"fetch\(\s*[\"']https?://", html))
        for b in BANNED_LITERALS:
            self.assertNotIn(b, html.lower())

    def test_the_banner_says_no_language_model(self):
        html = PAGE.read_text(encoding="utf-8").lower()
        self.assertIn("no language model", html)
        self.assertIn("dictionary.json", html)          # the link to the dictionary


if __name__ == "__main__":
    unittest.main()

"""
The fly's nose, for coins.

A coin has no smell, so this module gives it one, and it is careful about which
parts of that are measured and which are chosen.

MEASURED
  * How 77 fly olfactory receptors respond to 690 odorants: DoOR 2.0 (the
    Database of Odorant Responses, Muench and Galizia 2016; CC BY-SA 4.0),
    normalised 0..1 per receptor.
  * Which glomerulus each receptor's neurons project to (DoOR's mapping table,
    after Couto et al. 2005 and Fishilevich and Vosshall 2005).
  * That those glomeruli exist in this connectome as ORN_<glomerulus> cell types
    (53 types, 2,635 cells), and every synapse downstream of them.
  * The rate scale: fly receptor neurons fire up to about 200 Hz (Hallem and
    Carlson 2006), so a response of 1.0 is 200 Hz.

CHOSEN, and said so everywhere this is used
  * Which odorant a coin smells of. A word in the coin's name, symbol or
    description that names a real smell maps to the odorant that carries it
    (banana -> isopentyl acetate, the banana ester). A coin whose words name no
    smell gets a blend of three DoOR odorants picked by a hash of its name and
    symbol, so two coins with the same name smell the same.
  * How a mixture combines: the strongest response per glomerulus wins.
  * A fixed concentration. DoOR is mostly one dilution, so there is no dose.

Nothing here decides anything. It only turns a coin into spike rates on
receptor neurons; what the brain does with them is the brain's.
"""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MAX_HZ = 200.0
MIN_RESPONSE = 0.01        # below this a glomerulus is not driven at all
DESCRIPTION_WEIGHT = 0.5   # a smell named only in the description is fainter
HASH_ODORANTS = 3

# Word -> odorant. A chosen convention: each odorant is the compound a person
# would name as carrying that smell. Sugar, money and the moon have no smell
# a fly can detect, so they are deliberately absent.
WORD_ODOURS = {
    "isopentyl acetate": ("banana", "bananas"),
    "hexyl acetate": ("apple", "apples", "cider"),
    "butyl acetate": ("pear", "pears"),
    "ethyl butyrate": ("pineapple", "pineapples"),
    "ethyl hexanoate": ("strawberry", "strawberries", "berry", "berries"),
    "ethyl acetate": ("fruit", "fruits", "fruity", "yeast", "ferment", "fermented"),
    "ethanol": ("wine", "beer", "booze", "alcohol", "vodka", "drunk", "sake"),
    "acetic acid": ("vinegar", "sour", "pickle", "pickles"),
    "2,3-butanedione": ("butter", "buttery", "popcorn"),
    "lactic acid": ("yogurt", "yoghurt", "milk", "cream"),
    "butyric acid": ("cheese", "vomit", "sweat", "sweaty", "feet"),
    "hexanoic acid": ("goat", "goats"),
    "limonene": ("lemon", "lime", "orange", "citrus"),
    "citral": ("lemongrass",),
    "geraniol": ("rose", "roses", "flower", "flowers", "floral"),
    "linalool": ("lavender",),
    "menthol": ("mint", "minty", "menthol", "peppermint"),
    "alpha-pinene": ("pine", "forest", "christmas"),
    "benzaldehyde": ("almond", "almonds", "cherry", "cherries", "marzipan"),
    "eugenol": ("clove", "cloves"),
    "methyl salicylate": ("wintergreen",),
    "1-octen-3-ol": ("mushroom", "mushrooms", "shroom", "shrooms", "fungus", "fungi"),
    "geosmin": ("earth", "earthy", "soil", "dirt", "mud", "rain", "petrichor", "beet"),
    "cadaverine": ("corpse", "rot", "rotten", "zombie", "zombies", "dead", "death", "carcass"),
    "putrescine": ("decay", "putrid", "rotting"),
    "indole": ("poop", "poo", "shit", "turd", "dung", "manure"),
    "trimethylamine": ("fish", "fishy"),
    "ammonia": ("pee", "piss", "urine"),
    "carbon dioxide": ("breath", "co2", "exhale", "soda", "fizz", "fizzy"),
    "phenol": ("smoke", "smoky", "burnt"),
    "hexanal": ("grass", "grassy", "hay"),
    "gamma-decalactone": ("peach", "peaches"),
    "phenylacetaldehyde": ("honey",),
    "furfural": ("bread", "caramel", "toffee"),
    "2,5-dimethylpyrazine": ("coffee", "cocoa", "chocolate", "roast", "roasted"),
    "toluene": ("gasoline", "petrol", "glue"),
    "acetone": ("nailpolish", "solvent"),
}
WORD_TO_ODORANT = {w: o for o, words in WORD_ODOURS.items() for w in words}
WORD_RE = re.compile(r"[a-z0-9]+")


def door_dir(root=ROOT):
    for d in (root / "door", root / "data" / "door"):
        if (d / "door_response_matrix.csv").exists():
            return d
    raise FileNotFoundError("DoOR data not found in ./door or ./data/door")


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f, delimiter=";"))


class Door:
    """DoOR 2.0, read from its CSV export. R writes row names without a header cell."""

    def __init__(self, root=ROOT):
        d = door_dir(root)
        maps = _rows(d / "door_mappings.csv")
        hdr, body = maps[0], maps[1:]
        shift = len(body[0]) - len(hdr)
        self.glomerulus_of = {}
        for r in body:
            m = dict(zip(hdr, r[shift:]))
            g = m.get("glomerulus", "")
            # skip unknown and merged sensilla ("DM5+DM3", "DL2d/v"): no single target
            if m.get("receptor") and m["receptor"] not in self.glomerulus_of and g \
                    and g not in ("?", "NA") and "+" not in g and "/" not in g:
                self.glomerulus_of[m["receptor"]] = g
        mat = _rows(d / "door_response_matrix.csv")
        self.receptors = mat[0]
        self.response = {r[0]: r[1:] for r in mat[1:]}       # InChIKey -> values per receptor
        odor = _rows(d / "door_odor.csv" if (d / "door_odor.csv").exists() else d / "odor.csv")
        self.key_of = {}
        self.name_of = {}
        for r in odor[1:]:
            if len(r) > 3 and r[3] in self.response:
                self.key_of[r[2].lower()] = r[3]
                self.name_of[r[3]] = r[2]
        self.odorants = sorted(k for k in self.response if k != "SFR")
        # each receptor's spontaneous firing, subtracted in profile()
        self.sfr = self.response.get("SFR")

    def profile(self, key):
        """
        Per-glomerulus response above spontaneous firing, 0..1.

        DoOR's normalised values include each receptor's spontaneous firing
        (the SFR row, nonzero for most receptors). A receptor firing at its
        spontaneous rate is not responding to the odour, so SFR is subtracted
        per receptor, as DoOR's resetSFR does. Anything at or below baseline is
        not driven: Poisson input can only excite, so inhibitory responses are
        dropped (CHOSEN, disclosed).
        """
        values = self.response[key]
        sfr = self.sfr or [""] * len(values)
        out = {}
        for rec, v, s in zip(self.receptors, values, sfr):
            g = self.glomerulus_of.get(rec)
            if not g or v in ("NA", ""):
                continue
            base = 0.0 if s in ("NA", "") else float(s)
            out[g] = max(out.get(g, 0.0), max(0.0, float(v) - base))
        return out


def words(text):
    return WORD_RE.findall((text or "").lower())


class Nose:
    """
    Coin -> smell -> spike rates on this connectome's receptor neurons.

    `fb` needs `where(type_re=...)` returning neuron indices, as FlyBrain does.
    """

    def __init__(self, fb, root=ROOT, max_hz=MAX_HZ, door=None, equal_sniff=0.0):
        self.door = door or Door(root)
        self.max_hz = max_hz
        self.equal_sniff = float(equal_sniff)
        self.orn = {}
        for g in sorted(set(self.door.glomerulus_of.values())):
            idx = np.asarray(fb.where(type_re=f"^ORN_{re.escape(g)}$"))
            if idx.size:
                self.orn[g] = idx
        self.cells = int(sum(v.size for v in self.orn.values()))

    def smell(self, name, symbol="", description=""):
        """
        What a coin smells of: {"odorants": [{name, weight, why}], "profile":
        {glomerulus: 0..1}, "loudness": total, "loudness_target": equal_sniff}.

        "why" says whether the odorant came from a word (and which) or from the
        hash. "loudness" is what the profile adds up to after equal_sniff has
        scaled it, which is below the target for a coin whose smell lands on
        few glomeruli - see the note in the body.
        """
        picks = {}
        for text, weight in ((f"{name} {symbol}", 1.0), (description, DESCRIPTION_WEIGHT)):
            for w in words(text):
                odorant = WORD_TO_ODORANT.get(w)
                if odorant and self.door.key_of.get(odorant) and weight > picks.get(odorant, (0, ""))[0]:
                    picks[odorant] = (weight, f"word:{w}")
        if not picks:
            seed = hashlib.sha256(f"{(name or '').strip().lower()}|{(symbol or '').strip().lower()}".encode()).digest()
            keys = self.door.odorants
            for j in range(HASH_ODORANTS):
                key = keys[int.from_bytes(seed[j * 4:j * 4 + 4], "big") % len(keys)]
                picks.setdefault(self.door.name_of.get(key, key), (1.0, "hash"))
        profile = {}
        for odorant, (weight, _) in picks.items():
            key = self.door.key_of.get(odorant.lower())
            if key is None:                          # hash picks are keyed by name_of
                key = next((k for k, n in self.door.name_of.items() if n == odorant), None)
            for g, v in self.door.profile(key).items():
                if g in self.orn:
                    profile[g] = max(profile.get(g, 0.0), v * weight)
        out = {g: round(v, 4) for g, v in sorted(profile.items()) if v >= MIN_RESPONSE}
        if self.equal_sniff > 0:
            # Every coin's smell is scaled toward the same total - as far as a
            # receptor allows.
            #
            # DoOR has no dose axis, so how strong a coin smells is an accident
            # of which odorant its words happen to name. Measured 2026-09-12 on
            # this connectome: geosmin alone fires 42% of the Kenyon cells while
            # isopentyl acetate fires 4.9%, so a lesson about a loud coin lands
            # on five times as many synapses and swamps the quiet ones. Scaling
            # every coin's profile toward the same total takes most of that
            # accident out, and it is what made sugar and shock both point the
            # right way in build/backroom_screen.json. CHOSEN, disclosed.
            #
            # It does not reach equality, and saying it does would be false. A
            # DoOR response is a response, so it is capped at 1.0: a coin whose
            # smell lands on one glomerulus totals 1.0 however hard it is
            # sniffed, one on two glomeruli at (0.9, 0.1) totals 1.2, and only
            # a profile spread over enough of them reaches equal_sniff. In the
            # first paper run the totals ran 1.00, 1.24, 1.48, 1.49, 2.00. A
            # coin hitting one glomerulus cannot be made as loud as one hitting
            # twenty without driving a receptor past its own maximum, which
            # would be a made-up measurement rather than a chosen scaling. The
            # profile's achieved total is returned as `loudness` so a caller,
            # a look record and a reader can see which coins are quiet.
            total = sum(out.values())
            if total > 0:
                out = {g: round(min(1.0, v * self.equal_sniff / total), 4) for g, v in out.items()}
        return {"odorants": [{"name": o, "weight": w, "why": why} for o, (w, why) in sorted(picks.items())],
                "profile": out,
                # how loud this coin actually ends up: the sum of the scaled
                # responses, against equal_sniff as the target it cannot always
                # reach (see above)
                "loudness": round(sum(out.values()), 4),
                "loudness_target": float(self.equal_sniff)}

    def drive(self, smell):
        """The smell as a FlyBrain drive dict: receptor neurons of each glomerulus at response x max_hz."""
        return {tuple(self.orn[g].tolist()): float(v) * self.max_hz
                for g, v in smell["profile"].items() if g in self.orn}

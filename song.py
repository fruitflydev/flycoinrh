"""CHOSEN acoustic rendering of MEASURED descending and motor rates."""
import numpy as np

SAMPLE_RATE = 22050
WORLD_SECONDS = .05
PIP10_FULL = 1000 / 2.2
IPI = .035
PULSE_SECONDS = .004
PULSE_HZ = 250.
SINE_HZ = 150.
BIOLOGY = ("CHOSEN: D. melanogaster pulse IPI approximately 35 ms, carrier "
           "approximately 250 Hz, sine approximately 150 Hz; von Philipsborn "
           "et al. 2011 Neuron 69:509 (descending circuit; "
           "https://pubmed.ncbi.nlm.nih.gov/21315261/); Fast intensity adaptation "
           "enhances the encoding of sound in Drosophila, Nat Commun 2018 "
           "(carriers; https://doi.org/10.1038/s41467-017-02453-9); Zhou et al. "
           "2015 eLife 4:e08477 (IPI; https://elifesciences.org/articles/08477). "
           "These numbers are synthesis choices, not brain measurements.")


def pulse(age):
    """One carrier cycle under a continuous 4 ms Hann window."""
    age = np.asarray(age)
    inside = (age >= 0) & (age < PULSE_SECONDS)
    return np.where(inside, .5 * (1 - np.cos(2*np.pi*age/PULSE_SECONDS))
                    * np.sin(2*np.pi*PULSE_HZ*age), 0.)


def rms(wave):
    return float(np.sqrt(np.mean(np.asarray(wave)**2)))


class Singer:
    """Carry sample clock, carrier phase and pulse schedule across world steps.

    CHOSEN: pIP10 is the descending song command; zero pIP10 produces zero song; dependence on P1 is not assumed.
    CHOSEN: mode is the fraction of the two per-cell motor means due to pulse.
    A 50 ms window alternates 1103/1102 samples; no half-sample clock drift.
    """
    def __init__(self, seed=0, jittered=False, pulse_cells=8, sine_cells=2,
                 pip10_full=PIP10_FULL):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
        self.jittered = jittered
        self.pulse_cells, self.sine_cells = pulse_cells, sine_cells
        self.pip10_full = pip10_full
        self.samples, self.elapsed = 0, 0.
        self.last_pulse, self.next_pulse = -1., 0.
        self.record = {}

    def render(self, pip10_hz, pulse_hz, sine_hz, attenuation=1., seconds=WORLD_SECONDS):
        a = float(np.clip(pip10_hz/self.pip10_full, 0, 1))
        p, s = pulse_hz/self.pulse_cells, sine_hz/self.sine_cells
        m = float(p/(p+s)) if p+s > 0 else 0.
        self.elapsed += seconds
        stop = int(np.ceil(self.elapsed*SAMPLE_RATE - 1e-8))
        t = np.arange(self.samples, stop)/SAMPLE_RATE
        train = np.zeros(t.size)
        for i, now in enumerate(t):
            while now >= self.next_pulse:
                self.last_pulse = self.next_pulse
                self.next_pulse += self.rng.uniform(.015, .060) if self.jittered else IPI
            age = now-self.last_pulse
            if age < PULSE_SECONDS:
                train[i] = .5*(1-np.cos(2*np.pi*age/PULSE_SECONDS))*np.sin(2*np.pi*PULSE_HZ*age)
        wave = attenuation*a*(m*train+(1-m)*np.sin(2*np.pi*SINE_HZ*t))
        self.samples = stop
        self.record = dict(pip10_hz=float(pip10_hz), a=a, m=m, delivered_rms=rms(wave))
        return wave

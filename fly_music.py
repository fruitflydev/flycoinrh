"""Autonomous neural-to-song composer for FLYVUE.

Imports from the FLYVUE pipeline: order.first_seen and run.Simulation-compatible data.
No browser automation, ML framework, MIDI library, or system audio dependency is required.
VoiceStudio is an optional HTTP rendering backend; the composer itself is deterministic.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ATTRIBUTION = "Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY"
SCALE = np.array([0, 2, 4, 7, 9, 12], dtype=np.int16)  # C major pentatonic
NOTES = ("C", "D", "E", "G", "A", "C")
LYRIC_BANK = {
    "bright": [
        ("I see the light", "I rise and move through air"),
        ("I chase the motion", "I follow what is there"),
        ("The world is turning", "I turn and find my way"),
    ],
    "calm": [
        ("I drift through the quiet", "I listen to the air"),
        ("A little light is moving", "I follow everywhere"),
        ("I float between the moments", "And leave a trace behind"),
    ],
    "intense": [
        ("I wake with every signal", "The world is moving fast"),
        ("I run across the current", "I never let it pass"),
        ("A thousand lights are calling", "I answer as they fly"),
    ],
}


def _as_bool_log(spike_log: np.ndarray) -> np.ndarray:
    a = np.asarray(spike_log, dtype=np.bool_)
    if a.ndim != 2:
        raise ValueError("spike_log must have shape (steps, neurons)")
    return a


def neural_fingerprint(spike_log: np.ndarray, soma_xy: np.ndarray | None = None) -> dict[str, Any]:
    log = _as_bool_log(spike_log)
    steps, neurons = log.shape
    activity = log.mean(axis=1).astype(np.float32)
    counts = log.sum(axis=0).astype(np.float32)
    total = float(log.sum())
    if total:
        early = np.flatnonzero(log.any(axis=0)).astype(np.float32)
        first = np.full(neurons, steps, dtype=np.int32)
        for t in range(steps):
            idx = np.flatnonzero(log[t] & (first == steps))
            first[idx] = t
        first_valid = first[first < steps]
        first_mean = float(first_valid.mean()) if first_valid.size else float(steps)
        first_std = float(first_valid.std()) if first_valid.size else 0.0
    else:
        first = np.full(neurons, steps, dtype=np.int32)
        first_mean = float(steps)
        first_std = 0.0
    if steps > 1:
        bursts = np.diff(log.astype(np.int8), axis=0).clip(min=0).sum(axis=1).astype(np.float32)
    else:
        bursts = np.zeros(steps, dtype=np.float32)
    centroid = np.full((steps, 2), np.nan, dtype=np.float32)
    if soma_xy is not None:
        xy = np.asarray(soma_xy, dtype=np.float32)
        if xy.shape == (neurons, 2):
            for t in range(steps):
                idx = np.flatnonzero(log[t])
                if idx.size:
                    p = xy[idx]
                    good = np.isfinite(p).all(axis=1)
                    if good.any():
                        centroid[t] = p[good].mean(axis=0)
    digest = hashlib.sha256(log.tobytes()).hexdigest()
    seed = int.from_bytes(hashlib.blake2b(log.tobytes(), digest_size=8).digest(), "big") & 0x7FFFFFFF
    return {
        "steps": int(steps),
        "neuron_count": int(neurons),
        "total_spikes": int(total),
        "activity": activity,
        "burst": bursts,
        "first_seen_steps": first,
        "first_seen_mean_ms": first_mean * 0.2,
        "first_seen_std_ms": first_std * 0.2,
        "centroid": centroid,
        "fingerprint": digest,
        "seed": seed,
    }


def _interp(values: np.ndarray, n: int) -> np.ndarray:
    if values.size == n:
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros(n, dtype=np.float32)
    x = np.linspace(0.0, 1.0, values.size)
    y = np.linspace(0.0, 1.0, n)
    return np.interp(y, x, values).astype(np.float32)


def compose(spike_log: np.ndarray, soma_xy: np.ndarray | None = None, variation: int = 0) -> dict[str, Any]:
    f = neural_fingerprint(spike_log, soma_xy)
    rng = np.random.default_rng((f["seed"] + int(variation) * 1000003) & 0xFFFFFFFF)
    activity = f["activity"]
    burst = f["burst"]
    # Normalize robustly so a sparse or saturated simulation still produces useful music.
    lo, hi = np.percentile(activity, [10, 90]) if activity.size else (0.0, 1.0)
    energy = np.clip((activity - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    song_steps = 128  # 16 bars x 8 eighth-note slots.
    env = _interp(energy, song_steps)
    benv = _interp(burst, song_steps)
    mean_energy = float(np.mean(energy)) if energy.size else 0.0
    peak_energy = float(np.max(energy)) if energy.size else 0.0
    bpm = int(np.clip(round(76 + mean_energy * 48 + peak_energy * 12), 72, 136))
    # Activity determines scale degree; spatial centroid/first-seen timing modulates contour.
    centroid = f["centroid"]
    cx = _interp(np.nan_to_num(centroid[:, 0], nan=0.5), song_steps) if centroid.size else np.full(song_steps, .5, np.float32)
    firstness = np.zeros(song_steps, dtype=np.float32)
    first = f["first_seen_steps"]
    valid = first[first < f["steps"]]
    if valid.size:
        hist, _ = np.histogram(valid, bins=min(16, f["steps"]), range=(0, f["steps"]))
        firstness = _interp(hist.astype(np.float32), song_steps)
        firstness /= max(float(firstness.max()), 1e-6)
    melody = []
    prev_degree = 0
    for i in range(song_steps):
        if env[i] < 0.12 and benv[i] < 0.10:
            melody.append({"slot": i, "note": None, "beats": 0.5, "velocity": 0.0})
            continue
        raw = env[i] * 4.0 + cx[i] * 1.4 + firstness[i] * 1.2
        degree = int(np.clip(round(raw), 0, 5))
        if i and rng.random() < 0.35:
            degree = int(np.clip(prev_degree + rng.choice([-1, 0, 1]), 0, 5))
        prev_degree = degree
        octave = 4 + (1 if degree == 5 and env[i] > .68 else 0)
        melody.append({"slot": i, "note": f"{NOTES[degree]}{octave}", "beats": 0.5, "velocity": round(float(.45 + .5 * env[i]), 3)})
    # Four-beat chord loop, biased by neural energy.
    chords = ["C", "Am", "F", "G"] if mean_energy < .55 else ["Am", "F", "C", "G"]
    mood = "intense" if mean_energy > .72 or peak_energy > .92 else ("calm" if mean_energy < .30 else "bright")
    pairs = LYRIC_BANK[mood]
    lyric_seed = f["seed"] + int(variation) * 31
    a, b = pairs[lyric_seed % len(pairs)]
    c, d = pairs[(lyric_seed // 7) % len(pairs)]
    lyrics = [a, b, c, d, a, b, "I learn the shape of every sound", "And make a song from what I found"]
    return {
        "version": 1,
        "attribution": ATTRIBUTION,
        "neural_seed": int(f["seed"]),
        "variation": int(variation),
        "fingerprint": f["fingerprint"],
        "tempo_bpm": bpm,
        "key": "C",
        "scale": "C major pentatonic",
        "mood": mood,
        "energy": round(mean_energy, 4),
        "peak_energy": round(peak_energy, 4),
        "chords": chords,
        "melody": melody,
        "lyrics": lyrics,
    }


def lyrics_text(song: dict[str, Any]) -> str:
    return "\n".join(song["lyrics"])


def manifest(song: dict[str, Any], mode: str, neuron_count: int, unique_hex_count: int) -> dict[str, Any]:
    out = dict(song)
    out.update({"flyvue_mode": mode, "neuron_count": int(neuron_count), "unique_hex_count": int(unique_hex_count)})
    return out


def voice_studio_synthesize(song: dict[str, Any], url: str | None = None, model: str | None = None, voice: str | None = None, timeout: float = 120.0) -> bytes:
    """Render lyrics through VoiceStudio's OpenAI-compatible speech endpoint.

    This is intentionally isolated: VoiceStudio renders the voice; FLYVUE remains the composer.
    Singing pitch is not assumed from TTS. The returned audio is therefore the vocal rendering.
    """
    base = (url or os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:8000")).rstrip("/")
    endpoint = base + "/v1/audio/speech"
    payload = {
        "model": model or os.environ.get("VOICESTUDIO_MODEL", "default"),
        "voice": voice or os.environ.get("VOICESTUDIO_VOICE", "default"),
        "input": lyrics_text(song),
        "response_format": "wav",
        "speed": float(os.environ.get("VOICESTUDIO_SPEED", "1.0")),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    api_key = os.environ.get("VOICESTUDIO_API_KEY")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        audio = response.read()
    if not audio:
        raise RuntimeError("VoiceStudio returned empty audio")
    return audio


def save_song(song: dict[str, Any], directory: str | os.PathLike[str]) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": root / "manifest.json",
        "melody": root / "melody.json",
        "lyrics": root / "lyrics.txt",
    }
    files["manifest"].write_text(json.dumps(song, indent=2), encoding="utf-8")
    files["melody"].write_text(json.dumps({"tempo_bpm": song["tempo_bpm"], "key": song["key"], "scale": song["scale"], "chords": song["chords"], "melody": song["melody"], "attribution": ATTRIBUTION}, indent=2), encoding="utf-8")
    files["lyrics"].write_text(lyrics_text(song) + "\n\n" + ATTRIBUTION + "\n", encoding="utf-8")
    return {k: str(v) for k, v in files.items()}

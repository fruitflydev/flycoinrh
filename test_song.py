"""Acoustic choices and the sample clock have numerical oracles."""
import numpy as np
import pytest
from song import Singer, pulse, rms, SAMPLE_RATE, PIP10_FULL


def test_pulse_period_carrier_and_hann():
    t = np.arange(22050)/SAMPLE_RATE
    wave = Singer().render(PIP10_FULL, 8, 0, seconds=1)
    np.testing.assert_allclose(wave, pulse(t % .035), atol=2e-12)
    age = np.array([0, .001, .002, .003, .004, .005])
    np.testing.assert_allclose(pulse(age), [0, .5, 0, -.5, 0, 0], atol=1e-15)
    assert np.count_nonzero(np.abs(wave[:88]) > 1e-10) > 80
    assert np.all(wave[89:772] == 0)


def test_zero_command_and_mode_extremes():
    assert np.all(Singer().render(0, 10000, 10000) == 0)
    singer = Singer()
    wave = singer.render(PIP10_FULL, 0, 2)
    np.testing.assert_allclose(wave, np.sin(2*np.pi*150*np.arange(len(wave))/SAMPLE_RATE))
    assert singer.record['m'] == 0
    singer.render(PIP10_FULL, 8, 0)
    assert singer.record['m'] == 1
    singer.render(PIP10_FULL, 8, 2)
    assert singer.record['m'] == .5


@pytest.mark.parametrize('mode', [0, .5, 1])
def test_jitter_energy(mode):
    regular = Singer().render(PIP10_FULL, 8*mode, 2*(1-mode), seconds=20)
    for seed in range(10):
        jitter = Singer(seed, True).render(PIP10_FULL, 8*mode, 2*(1-mode), seconds=20)
        assert .85 < rms(jitter)/rms(regular) < 1.15
        if mode:
            assert not np.array_equal(jitter, regular)


@pytest.mark.parametrize('jittered', [False, True])
def test_phase_continuity(jittered):
    singer = Singer(2, jittered)
    pieces = [singer.render(100, 20, 7) for _ in range(20)]
    assert [len(p) for p in pieces[:4]] == [1103, 1102, 1103, 1102]
    whole = Singer(2, jittered).render(100, 20, 7, seconds=1)
    np.testing.assert_array_equal(np.concatenate(pieces), whole)

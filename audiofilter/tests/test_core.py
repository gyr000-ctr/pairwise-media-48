import math

import pytest

import audiofilter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f(lp=True):
    return audiofilter.OnePole(48000, 1000.0, lp)


def _check_len():
    return len(_f().process([0.5] * 10)) == 10


def _check_dc_lowpass():
    out = _f(True).process([1.0] * 500)
    return out[-1] > 0.9


def _check_dc_highpass():
    out = _f(False).process([1.0] * 500)
    return abs(out[-1]) < 0.1


def _check_bad_cutoff():
    return _raises(lambda: audiofilter.OnePole(48000, 0.0))


def _check_bad_rate():
    return _raises(lambda: audiofilter.OnePole(0, 1000.0))


def _check_empty():
    return _f().process([]) == []


def _check_type():
    return isinstance(_f().process([0.1]), list)


def _check_monotonic():
    out = _f(True).process([1.0] * 20)
    return all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))


def _check_det():
    return _f().process([0.3] * 5) == _f().process([0.3] * 5)


def _check_no_mut():
    s = [0.5]; _f().process(s)
    return s == [0.5]

CASES = [("len", "_check_len"), ("dc_lowpass", "_check_dc_lowpass"), ("dc_highpass", "_check_dc_highpass"), ("bad_cutoff", "_check_bad_cutoff"), ("bad_rate", "_check_bad_rate"), ("empty", "_check_empty"), ("type", "_check_type"), ("monotonic", "_check_monotonic"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert OnePole() is not None
        return
    assert fn() is True

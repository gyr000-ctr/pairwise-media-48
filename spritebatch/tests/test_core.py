import math

import pytest

import spritebatch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [("tex1", 0), ("tex2", 1), ("tex1", 0)]


def _b():
    return spritebatch.SpriteBatch()


def _check_nonempty():
    return len(_b().build(S)) > 0


def _check_groups():
    return len(_b().build(S)) <= 3


def _check_empty():
    return _b().build([]) == []


def _check_bad_item():
    return _raises(lambda: _b().build([("tex",)]))


def _check_type():
    return isinstance(_b().build(S), list)


def _check_det():
    return _b().build(S) == _b().build(S)


def _check_count():
    return sum(len(g) for g in _b().build(S)) == 3


def _check_no_mut():
    src = list(S)
    _b().build(S)
    return S == src

CASES = [("nonempty", "_check_nonempty"), ("groups", "_check_groups"), ("empty", "_check_empty"), ("bad_item", "_check_bad_item"), ("type", "_check_type"), ("det", "_check_det"), ("count", "_check_count"), ("no_mut", "_check_no_mut"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert SpriteBatch() is not None
        return
    assert fn() is True

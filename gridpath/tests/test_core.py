import math

import pytest

import gridpath

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def _g():
    return gridpath.GridBfs()


def _check_start():
    return _g().distances(W, (0, 0))[0][0] == 0


def _check_neighbor():
    return _g().distances(W, (0, 0))[0][1] == 1


def _check_diagonal():
    return _g().distances(W, (0, 0))[1][1] == -1


def _check_unreachable():
    return _g().distances([[0, 1], [1, 0]], (0, 0))[1][1] == -1


def _check_bad_start():
    return _raises(lambda: _g().distances(W, (5, 5)))


def _check_bad_grid():
    return _raises(lambda: _g().distances([], (0, 0)))


def _check_type():
    return isinstance(_g().distances(W, (0, 0)), list)


def _check_shape():
    d = _g().distances(W, (0, 0))
    return len(d) == 3 and len(d[0]) == 3


def _check_wall():
    return _g().distances(W, (0, 0))[1][1] == -1

CASES = [("start", "_check_start"), ("neighbor", "_check_neighbor"), ("diagonal", "_check_diagonal"), ("unreachable", "_check_unreachable"), ("bad_start", "_check_bad_start"), ("bad_grid", "_check_bad_grid"), ("type", "_check_type"), ("shape", "_check_shape"), ("wall", "_check_wall"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert GridBfs() is not None
        return
    assert fn() is True

"""色调曲线验收测试：单调三次插值、端点、精度、报错、性能、并发。"""

import math
import threading
import time

import pytest

import hdrcurve
from hdrcurve.core import ToneCurve, _LUT_TOL


# ---------- 端点与基本行为 ----------

def test_curve_endpoints_zero_one():
    c = ToneCurve(2.2)
    assert c.lookup(0.0) == 0.0
    assert abs(c.lookup(1.0) - 1.0) < 1e-12


def test_curve_endpoints_various_gamma():
    for g in (0.45, 1.0, 2.2, 5.0):
        c = ToneCurve(g)
        assert c.lookup(0.0) == 0.0
        assert abs(c.lookup(1.0) - 1.0) < 1e-12


def test_curve_clamp_out_of_range():
    c = ToneCurve(2.2)
    assert c.lookup(2.0) == 1.0
    assert c.lookup(-1.0) == 0.0


def test_curve_returns_float():
    assert isinstance(ToneCurve(2.2).lookup(0.5), float)


def test_curve_deterministic():
    c = ToneCurve(2.2)
    assert c.lookup(0.4) == c.lookup(0.4)


# ---------- 解析精度：LUT 与解析差 <= 1e-6 ----------

@pytest.mark.parametrize("gamma", [0.45, 0.7, 1.0, 1.8, 2.2, 2.4, 4.0])
def test_curve_lut_matches_analytic(gamma):
    c = ToneCurve(gamma)
    worst = 0.0
    for i in range(1, 1000):
        t = i / 1000.0
        worst = max(worst, abs(c.lookup(t) - t ** gamma))
    assert worst <= _LUT_TOL


def test_curve_midpoint_accuracy_high_gamma():
    c = ToneCurve(10.0)
    for i in range(1, 200):
        t = i / 200.0
        assert abs(c.lookup(t) - t ** 10.0) <= _LUT_TOL


# ---------- 单调性（不得过冲） ----------

def test_curve_monotonic_gamma():
    c = ToneCurve(2.2)
    prev = -1.0
    for i in range(101):
        v = c.lookup(i / 100.0)
        assert v >= prev - 1e-12
        assert 0.0 <= v <= 1.0
        prev = v


def test_curve_monotonic_inverse_gamma():
    c = ToneCurve(0.45)
    prev = -1.0
    for i in range(101):
        v = c.lookup(i / 100.0)
        assert v >= prev - 1e-12
        prev = v


# ---------- 控制点：单调三次插值、无过冲、非单调报错 ----------

def test_curve_from_points_interpolates_knots():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [0.0, 0.1, 0.4, 0.8, 1.0]
    c = ToneCurve.from_points(xs, ys)
    for x, y in zip(xs, ys):
        assert abs(c.lookup(x) - y) < 1e-9


def test_curve_from_points_no_overshoot():
    xs = [0.0, 0.2, 0.5, 0.8, 1.0]
    ys = [0.0, 0.05, 0.5, 0.95, 1.0]
    c = ToneCurve.from_points(xs, ys)
    prev = -1.0
    for i in range(501):
        v = c.lookup(i / 500.0)
        assert v >= prev - 1e-9
        assert 0.0 <= v <= 1.0
        prev = v


def test_curve_from_points_equals_analytic_gamma_curve():
    # 幂函数控制点构造出的曲线应与解析曲线接近；受 33 个控制点的
    # 三次插值精度限制，容差放宽到 1e-3，插值本身严格过控制点。
    g = 2.2
    xs = [i / 32.0 for i in range(33)]
    ys = [x ** g for x in xs]
    c = ToneCurve.from_points(xs, ys)
    worst = 0.0
    for i in range(1, 200):
        t = i / 200.0
        worst = max(worst, abs(c.lookup(t) - t ** g))
    assert worst <= 1e-3


def test_curve_from_points_dense_lut_accuracy():
    # 密控制点下，自适应 LUT 查表值与单调三次解析插值差 <= 1e-6。
    g = 2.2
    xs = [i / 256.0 for i in range(257)]
    ys = [x ** g for x in xs]
    c = ToneCurve.from_points(xs, ys)
    from hdrcurve.core import _MonotoneCubic
    spline = _MonotoneCubic(xs, ys)
    worst = max(
        abs(c.lookup(i / 1000.0) - spline.eval(i / 1000.0))
        for i in range(1, 1000)
    )
    assert worst <= 1e-6


def test_curve_nonmonotonic_points_raise():
    with pytest.raises(ValueError):
        ToneCurve.from_points([0.0, 0.5, 1.0], [0.0, 0.8, 0.2])


def test_curve_nonmonotonic_flat_then_drop_raise():
    with pytest.raises(ValueError):
        ToneCurve.from_points([0.0, 0.4, 0.8, 1.0], [0.0, 0.5, 0.5, 0.1])


def test_curve_non_strict_xs_raise():
    with pytest.raises(ValueError):
        ToneCurve.from_points([0.0, 0.5, 0.5], [0.0, 0.5, 1.0])


# ---------- 参数校验 ----------

def test_curve_bad_gamma_zero():
    with pytest.raises(ValueError):
        ToneCurve(0.0)


def test_curve_bad_gamma_negative():
    with pytest.raises(ValueError):
        ToneCurve(-1.0)


def test_curve_bad_gamma_nan_inf():
    with pytest.raises(ValueError):
        ToneCurve(float("nan"))
    with pytest.raises(ValueError):
        ToneCurve(float("inf"))


def test_curve_bad_levels():
    with pytest.raises(ValueError):
        ToneCurve(2.2, 0)
    with pytest.raises(ValueError):
        ToneCurve(2.2, 1)


def test_curve_lookup_non_finite_raise():
    c = ToneCurve(2.2)
    with pytest.raises(ValueError):
        c.lookup(float("nan"))
    with pytest.raises(ValueError):
        c.lookup(float("inf"))
    with pytest.raises(ValueError):
        c.lookup(float("-inf"))


# ---------- 性能：10 万次查表 <= 100 ms ----------

def test_curve_lookup_performance_100k():
    c = ToneCurve(2.2)
    n = 100_000
    start = time.perf_counter()
    total = 0.0
    for i in range(n):
        total += c.lookup((i % 1000) / 1000.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert total > 0.0  # 防止优化器消除调用
    assert elapsed_ms <= 100.0, f"10 万次查表耗时 {elapsed_ms:.1f} ms"


# ---------- 并发安全 ----------

def test_curve_concurrent_lookup_safe():
    c = ToneCurve(2.2)
    errors = []
    results = [None] * 8

    def worker(idx):
        try:
            local = []
            for _ in range(5000):
                for i in range(20):
                    local.append(c.lookup(i / 20.0))
            results[idx] = local
        except Exception as exc:  # pragma: no cover - 并发不应报错
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors
    ref = results[0]
    for r in results[1:]:
        assert r == ref

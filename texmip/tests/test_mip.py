"""mip 级别选择验收测试（新增文件，不改动既有测试）。

覆盖题面的可验证约束：

* 级别按各向异性比取 log2，并与"逐级下采样后比较误差"的暴力过程一致；
* 三线性权重和为 1、级别钳制在 [0, levels)；
* 非法缩放与非有限导数给出可区分的错误类型；
* 10 万次不超过 100 ms；
* 同一实例可被多线程并发调用。
"""

import math
import threading
import time

import pytest

import texmip
from texmip.core import (
    InvalidLevelError,
    InvalidScaleError,
    InvalidSizeError,
    MipError,
    NonFiniteError,
    TexMip,
)

MAX_LEVEL = TexMip(8, 4).levels - 1


def _brute_level(mip, scale):
    """逐级暴力：每级把足迹除以 2（下采样），直到足迹不足一个纹素。"""
    if isinstance(scale, (tuple, list)):
        if len(scale) == 2:
            stretch = max(scale)
        else:
            dudx, dvdx, dudy, dvdy = scale
            stretch = max(
                math.hypot(mip.w * dudx, mip.h * dvdx),
                math.hypot(mip.w * dudy, mip.h * dvdy),
            )
    else:
        stretch = scale
    level = 0
    while level + 1 < mip.levels and stretch / 2.0 >= 1.0:
        stretch /= 2.0
        level += 1
    return level


SCALES = [
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.9, 8.0, 9.0, 16.0, 31.0, 32.0,
    100.0, 1024.0, 1e6, 0.5, 0.25, 0.01, 1e-6,
    0.9999999999999999, 2.0000000000000004, 1.0000000000000002,
]

ANISO = [
    (1.0, 1.0), (4.0, 1.0), (1.0, 4.0), (8.0, 2.0), (3.0, 1.5),
    (1e6, 1.0), (0.5, 0.25), (2.0, 2.0), (6.0, 1.0),
]

DERIVS = [
    (0.0, 0.0, 0.0, 0.0),
    (1.0 / 8.0, 0.0, 0.0, 1.0 / 4.0),
    (0.5, 0.0, 0.0, 0.5),
    (3.0 / 8.0, 0.0, 0.0, 1.0 / 8.0),
    (0.01, 0.02, -0.03, 0.04),
    (0.0, 0.5, 0.0, 0.0),
]


# ---------------------------------------------------------------- 级别一致

@pytest.mark.parametrize("scale", SCALES)
def test_level_matches_brute_force_isotropic(scale):
    mip = TexMip(8, 4)
    assert mip.level(scale) == _brute_level(mip, scale)


@pytest.mark.parametrize("scale", ANISO)
def test_level_matches_brute_force_anisotropic(scale):
    mip = TexMip(64, 16)
    assert mip.level(scale) == _brute_level(mip, scale)


@pytest.mark.parametrize("deriv", DERIVS)
def test_level_matches_brute_force_derivatives(deriv):
    mip = TexMip(64, 16)
    assert mip.level(deriv) == _brute_level(mip, deriv)


def test_level_uses_max_stretch_not_geometric_mean():
    mip = TexMip(64, 64)
    # 几何平均为 2，最大拉伸为 4：必须按各向异性"最大拉伸"取 log2。
    assert mip.level((4.0, 1.0)) == 2
    assert mip.anisotropy((4.0, 1.0)) == 4.0


def test_level_of_one_is_zero():
    assert TexMip(8, 4).level(1.0) == 0


def test_level_is_int():
    assert isinstance(TexMip(8, 4).level(0.3), int)
    assert isinstance(TexMip(8, 4).level((3.0, 1.0)), int)
    assert isinstance(TexMip(8, 4).level((0.1, 0.0, 0.0, 0.1)), int)


# ---------------------------------------------------------------- 三线性

@pytest.mark.parametrize("scale", SCALES + ANISO)
def test_select_weights_sum_to_one_and_levels_in_range(scale):
    mip = TexMip(8, 4)
    level0, level1, weight0, weight1 = mip.select(scale)
    assert 0 <= level0 < mip.levels
    assert 0 <= level1 < mip.levels
    assert abs(weight0 + weight1 - 1.0) <= 1e-15
    assert weight0 >= 0.0 and weight1 >= 0.0
    assert level1 in (level0, level0 + 1)


@pytest.mark.parametrize("scale", SCALES + ANISO)
def test_select_reproduces_clamped_continuous_lod(scale):
    mip = TexMip(8, 4)
    level0, level1, weight0, weight1 = mip.select(scale)
    stretch = max(scale) if isinstance(scale, tuple) else scale
    expected = min(max(math.log2(stretch), 0.0), float(mip.max_level))
    assert abs(weight0 * level0 + weight1 * level1 - expected) <= 1e-12


def test_select_clamps_low_and_high():
    mip = TexMip(8, 4)
    assert mip.select(0.01)[:2] == (0, 1)
    assert mip.select(1e-6)[2:] == (1.0, 0.0)
    assert mip.select(1e6)[:2] == (mip.max_level, mip.max_level)
    assert mip.select(1e6)[2:] == (1.0, 0.0)


def test_select_fraction_at_midpoint():
    mip = TexMip(64, 64)
    level0, _, weight0, weight1 = mip.select(2.0 ** 2.5)
    assert level0 == 2
    assert abs(weight1 - 0.5) <= 1e-12


def test_trilinear_alias_matches_select():
    mip = TexMip(8, 4)
    assert mip.trilinear(3.7) == mip.select(3.7)


def test_select_from_derivatives_matches_sequence_form():
    mip = TexMip(64, 16)
    deriv = (0.2, 0.05, -0.1, 0.3)
    assert mip.select_from_derivatives(*deriv) == mip.select(deriv)
    assert mip.level_from_derivatives(*deriv) == mip.level(deriv)


def test_anisotropy_at_least_one():
    mip = TexMip(8, 4)
    assert mip.anisotropy((1.0, 4.0)) == 4.0
    assert mip.anisotropy((2.0, 2.0)) == 1.0
    assert mip.anisotropy((0.0, 0.5, 0.0, 0.0)) == math.inf


def test_zero_footprint_locks_level_zero():
    mip = TexMip(8, 4)
    assert mip.lod((0.0, 0.0, 0.0, 0.0)) == -math.inf
    assert mip.level((0.0, 0.0, 0.0, 0.0)) == 0
    assert mip.select((0.0, 0.0, 0.0, 0.0)) == (0, 1, 1.0, 0.0)


def test_deterministic():
    a, b = TexMip(8, 4), TexMip(8, 4)
    assert a.select(0.3) == b.select(0.3)
    assert a.level((5.0, 2.0)) == b.level((5.0, 2.0))


# ---------------------------------------------------------------- 尺寸

def test_size_at_halves_each_axis():
    mip = TexMip(8, 4)
    assert mip.size_at(0) == (8, 4)
    assert mip.size_at(1) == (4, 2)
    assert mip.size_at(mip.max_level) == (1, 1)


def test_levels_derived_from_max_dimension():
    assert TexMip(8, 4).levels == 4
    assert TexMip(1, 1).levels == 1
    assert TexMip(9, 1).levels == 4


def test_size_at_non_square_clamps_to_one():
    mip = TexMip(8, 2)
    assert mip.size_at(1) == (4, 1)
    assert mip.size_at(3) == (1, 1)


@pytest.mark.parametrize("bad", [-1, 4, 9, 1.5, "1", None, float("inf")])
def test_size_at_bad_level_raises(bad):
    with pytest.raises(MipError):
        TexMip(8, 4).size_at(bad)


def test_size_at_accepts_integral_float():
    assert TexMip(8, 4).size_at(1.0) == (4, 2)


# ---------------------------------------------------------------- 错误分类

def test_error_classes_are_distinct_value_subclasses():
    classes = [
        MipError,
        InvalidSizeError,
        InvalidScaleError,
        InvalidLevelError,
        NonFiniteError,
    ]
    assert len(set(classes)) == len(classes)
    assert all(issubclass(c, ValueError) for c in classes)
    assert all(issubclass(c, MipError) for c in classes[1:])


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-3, 0, -4])
def test_non_positive_scale_is_invalid_scale(bad):
    with pytest.raises(InvalidScaleError):
        TexMip(8, 4).level(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scale_is_non_finite(bad):
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).level(bad)
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).select(bad)
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).lod(bad)


@pytest.mark.parametrize("bad", [(1.0, float("nan")), (float("inf"), 1.0)])
def test_non_finite_scale_component_is_non_finite(bad):
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).level(bad)


@pytest.mark.parametrize(
    "bad",
    [
        (float("nan"), 0.0, 0.0, 1.0),
        (0.0, float("inf"), 0.0, 1.0),
        (0.0, 0.0, float("-inf"), 1.0),
        (0.0, 0.0, 0.0, float("nan")),
    ],
)
def test_non_finite_derivative_is_non_finite(bad):
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).level(bad)
    with pytest.raises(NonFiniteError):
        TexMip(8, 4).select(bad)


@pytest.mark.parametrize("bad", [(1.0,), (1.0, 2.0, 3.0), (), (1.0, 2.0, 3.0, 4.0, 5.0)])
def test_bad_scale_shape_is_invalid_scale(bad):
    with pytest.raises(InvalidScaleError):
        TexMip(8, 4).level(bad)


@pytest.mark.parametrize("bad", [(1.0, 0.0), (0.0, 1.0), (-1.0, 2.0)])
def test_non_positive_axis_is_invalid_scale(bad):
    with pytest.raises(InvalidScaleError):
        TexMip(8, 4).select(bad)


@pytest.mark.parametrize("bad", [None, "1.0", b"x", True, 1 + 0j, [1.0, "a"]])
def test_non_numeric_scale_is_invalid_scale(bad):
    with pytest.raises(InvalidScaleError):
        TexMip(8, 4).level(bad)


def test_non_finite_and_invalid_scale_are_distinguishable():
    mip = TexMip(8, 4)
    with pytest.raises(NonFiniteError):
        mip.level(float("nan"))
    with pytest.raises(InvalidScaleError):
        mip.level(0.0)
    assert not isinstance(InvalidScaleError(), NonFiniteError)
    assert not isinstance(NonFiniteError(), InvalidScaleError)


@pytest.mark.parametrize(
    "args",
    [(0, 4), (8, 0), (-8, 4), (8, -4), (8.0, 4), ("8", 4), (None, 4),
     (True, 4)],
)
def test_bad_texture_size_raises(args):
    with pytest.raises(InvalidSizeError):
        TexMip(*args)


@pytest.mark.parametrize("args", [(float("inf"), 4), (8, float("nan"))])
def test_non_finite_texture_size_is_non_finite(args):
    with pytest.raises(NonFiniteError):
        TexMip(*args)


# ---------------------------------------------------------------- 无参构造

def test_zero_arg_constructor():
    # 既有测试里缺失的 smoke 用例会调用 TexMip()，必须可构造。
    mip = TexMip()
    assert mip.levels == 1
    assert mip.level(1e9) == 0
    assert mip.size_at(0) == (1, 1)


# ---------------------------------------------------------------- 性能

def test_select_performance_100k():
    mip = TexMip(1024, 1024)
    n = 100_000
    start = time.perf_counter()
    acc = 0.0
    for i in range(n):
        level0, level1, w0, w1 = mip.select((0.5 + (i % 64) * 0.25, 0.5))
        acc += w0 * level0 + w1 * level1
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert acc > 0.0  # 防止优化器消除调用
    assert elapsed_ms <= 100.0, "10 万次选择耗时 %.1f ms" % elapsed_ms


def test_level_performance_100k():
    mip = TexMip(1024, 1024)
    n = 100_000
    start = time.perf_counter()
    total = 0
    for i in range(n):
        total += mip.level((0.5 + (i % 64) * 0.25, 0.5))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert total > 0
    assert elapsed_ms <= 100.0, "10 万次级选择耗时 %.1f ms" % elapsed_ms


# ---------------------------------------------------------------- 并发

def test_shared_instance_concurrent():
    mip = TexMip(64, 16)
    scales = [((1 + i % 97) / 8.0, 0.25) for i in range(97)]
    expected = [mip.select(s) for s in scales]
    errors = []
    results = [None] * 8

    def worker(idx):
        try:
            results[idx] = [mip.select(s) for s in scales]
        except Exception as exc:  # pragma: no cover - 并发不应报错
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    for r in results:
        assert r == expected

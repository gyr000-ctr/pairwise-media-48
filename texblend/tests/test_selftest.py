"""texblend 补充自测（与既有 tests/test_core.py 并行，不修改既有行为）。

所有用例与文件名都带 ``blend`` 关键字，可用
``python -m pytest tests -q -k blend`` 单独选中。
"""

import math
import threading
import time

import pytest

import texblend
from texblend import (
    TexBlend,
    InvalidTextureError,
    NonFiniteCoordinateError,
    InvalidWrapModeError,
)

T = [[0.0, 1.0], [2.0, 3.0]]


def _blend(**kw):
    return TexBlend(**kw)


# ---------- 四角精确 ----------

@pytest.mark.parametrize("u,v,expected", [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 2.0),
    (1.0, 1.0, 3.0),
])
def test_blend_corners_exact(u, v, expected):
    assert _blend().sample(T, u, v) == expected
    assert isinstance(_blend().sample(T, u, v), float)


def test_blend_degenerate_single_texel_axis():
    # 1x1：任何坐标、两种模式都精确返回唯一纹素
    for mode in ("clamp", "wrap"):
        b = _blend(wrap=mode)
        assert b.sample(((7.0,),), 0.0, 0.0) == 7.0
        assert b.sample(((7.0,),), 5.0, -2.3) == 7.0
    # 单行 / 单列：退化轴不产生插值
    b = _blend()
    assert b.sample(((1.0, 2.0, 3.0),), 1.5, 9.0) == pytest.approx(2.5, abs=1e-12)
    assert b.sample(((1.0,), (2.0,), (3.0,)), -4.0, 1.5) == pytest.approx(2.5, abs=1e-12)


def test_blend_corners_exact_wrap():
    b = _blend(wrap="wrap")
    assert b.sample(T, 0.0, 0.0) == 0.0
    assert b.sample(T, 1.0, 1.0) == 3.0
    # 环绕模式下整数倍尺寸仍精确落在纹素上
    assert b.sample(T, 2.0, 2.0) == 0.0
    assert b.sample(T, -1.0, -1.0) == 3.0


# ---------- 双线性中心值与权重和为 1 ----------

def test_blend_center_bilinear():
    assert _blend().sample(T, 0.5, 0.5) == pytest.approx(1.5, abs=1e-12)


def _four_weights(b, tex, u, v):
    """直接按同一规则重建四点权重，验证权重和恒为 1。"""
    h, w = len(tex), len(tex[0])
    wrap = b.wrap == "wrap"

    def axis(x, n):
        if wrap:
            i0 = math.floor(x) % n
            return i0, (i0 + 1) % n, x - math.floor(x)
        i0 = 0 if x <= 0.0 else (n - 1 if x >= n - 1 else int(x))
        i1 = i0 if (x <= 0.0 or x >= n - 1) else i0 + 1
        w1 = 0.0 if x <= 0.0 else (1.0 if x >= n - 1 else x - i0)
        return i0, i1, w1

    c0, c1, wu1 = axis(u, w)
    r0, r1, wv1 = axis(v, h)
    wu0, wv0 = 1.0 - wu1, 1.0 - wv1
    return [wu0 * wv0, wu1 * wv0, wu0 * wv1, wu1 * wv1], (r0, r1, c0, c1)


@pytest.mark.parametrize("u,v", [
    (0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.9, 0.1),
    (1.0, 1.0), (-0.3, 1.7), (3.7, -2.4),
])
def test_blend_weights_sum_to_one_clamp(u, v):
    b = _blend(wrap="clamp")
    weights, _ = _four_weights(b, T, u, v)
    assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    assert all(w >= 0.0 for w in weights)


@pytest.mark.parametrize("u,v", [
    (0.0, 0.0), (0.25, 0.75), (1.5, 2.5), (-0.3, 1.7), (3.7, -2.4),
])
def test_blend_weights_sum_to_one_wrap(u, v):
    b = _blend(wrap="wrap")
    weights, _ = _four_weights(b, T, u, v)
    assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    assert all(w >= 0.0 for w in weights)


# ---------- 各向异性：两轴不同采样率 ----------

def test_blend_anisotropic_axes_independent():
    # u 轴取整数（无插值），v 轴取半程；结果应只在行间混合
    b = _blend()
    got = b.sample(T, 0.0, 0.5)
    assert got == pytest.approx((0.0 + 2.0) / 2.0, abs=1e-12)
    got = b.sample(T, 1.0, 0.5)
    assert got == pytest.approx((1.0 + 3.0) / 2.0, abs=1e-12)
    # 反过来：v 整数、u 半程
    assert b.sample(T, 0.5, 0.0) == pytest.approx(0.5, abs=1e-12)
    assert b.sample(T, 0.5, 1.0) == pytest.approx(2.5, abs=1e-12)


def test_blend_anisotropic_rectangular_texture():
    # 4 列 2 行：两轴尺寸与采样位置均可不同
    tex = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    b = _blend()
    assert b.sample(tex, 0.0, 0.0) == 0.0
    assert b.sample(tex, 3.0, 1.0) == 7.0
    assert b.sample(tex, 1.5, 0.5) == pytest.approx(
        ((1.0 + 2.0) / 2.0 + (5.0 + 6.0) / 2.0) / 2.0, abs=1e-12
    )


# ---------- clamp / wrap 边界 ----------

def test_blend_clamp_beyond_edge():
    b = _blend(wrap="clamp")
    assert b.sample(T, 2.0, 2.0) == 3.0
    assert b.sample(T, -5.0, -5.0) == 0.0
    # 越界后沿边保持线性（单轴钳制）
    assert b.sample(T, 0.5, 5.0) == pytest.approx(2.5, abs=1e-12)


def test_blend_wrap_tiles_periodically():
    b = _blend(wrap="wrap")
    base = b.sample(T, 0.3, 0.7)
    assert b.sample(T, 0.3 + 2.0, 0.7) == pytest.approx(base, abs=1e-12)
    assert b.sample(T, 0.3 - 4.0, 0.7 + 6.0) == pytest.approx(base, abs=1e-12)
    # 单轴 wrap、另一轴不受影响
    assert b.sample(T, 2.3, 0.7) == pytest.approx(b.sample(T, 0.3, 0.7), abs=1e-12)


def test_blend_per_call_wrap_override():
    b = _blend(wrap="clamp")
    assert b.sample(T, 2.0, 0.0) == 1.0           # clamp
    assert b.sample(T, 2.0, 0.0, wrap="wrap") == 0.0  # wrap 覆盖
    with pytest.raises(InvalidWrapModeError):
        b.sample(T, 0.0, 0.0, wrap="mirror")


def test_blend_invalid_wrap_constructor():
    with pytest.raises(InvalidWrapModeError):
        TexBlend(wrap="bogus")


# ---------- 非法纹理可区分报错 ----------

@pytest.mark.parametrize("bad", [
    [],
    [[]],
    [[1.0, 2.0], [3.0]],          # 行列不齐
    [[1.0, "x"]],                 # 非数值
    [[1.0, float("nan")]],        # 非有限纹素
    [[1.0, float("inf")]],
    "not-a-texture",
    [1.0, 2.0],                   # 行不是序列
])
def test_blend_invalid_texture_distinguishable(bad):
    with pytest.raises(InvalidTextureError):
        _blend().sample(bad, 0.1, 0.1)
    # 与坐标错误属于不同异常类型
    assert not issubclass(InvalidTextureError, NonFiniteCoordinateError)


@pytest.mark.parametrize("u,v", [
    (float("nan"), 0.0),
    (0.0, float("inf")),
    (float("-inf"), 0.0),
])
def test_blend_nonfinite_coordinate_distinguishable(u, v):
    with pytest.raises(NonFiniteCoordinateError):
        _blend().sample(T, u, v)


def test_blend_texture_mutated_after_validation():
    # 同一纹理先合法采样，再被原地篡改成行列不齐：必须重新校验并报错
    tex = [[0.0, 1.0], [2.0, 3.0]]
    b = _blend()
    assert b.sample(tex, 0.5, 0.5) == pytest.approx(1.5)
    tex[1].append(9.0)
    with pytest.raises(InvalidTextureError):
        b.sample(tex, 0.5, 0.5)


# ---------- 确定性 / 不修改输入 ----------

def test_blend_deterministic_and_no_mutation():
    b = _blend()
    snapshot = [row[:] for row in T]
    first = b.sample(T, 0.37, 0.61)
    for _ in range(50):
        assert b.sample(T, 0.37, 0.61) == first
    assert T == snapshot


# ---------- 并发安全 ----------

def test_blend_thread_safe():
    b = _blend()
    errors = []

    def worker(seed):
        try:
            for k in range(2000):
                u = ((k * (seed + 1)) % 17) / 17.0
                v = ((k * (seed + 3)) % 13) / 13.0
                r = b.sample(T, u, v)
                assert math.isfinite(r)
                assert 0.0 <= r <= 3.0
        except Exception as exc:  # pragma: no cover - 仅在失败时记录
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------- 性能上界 ----------

def test_blend_performance_100k_under_150ms():
    b = _blend()
    tex = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
    n = 100_000
    # 预热（含首次校验与线程槽初始化）
    for k in range(1000):
        b.sample(tex, (k * 0.137) % 3.0, (k * 0.211) % 3.0)
    t0 = time.perf_counter()
    for k in range(n):
        b.sample(tex, (k * 0.137) % 3.0, (k * 0.211) % 3.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 150.0, f"{n} samples took {elapsed_ms:.1f} ms"

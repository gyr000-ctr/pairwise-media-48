"""补充验收用例：预乘 source-over 合成与 W3C 混合模式。

可见用例（tests/test_core.py）保持不变；本文件补充隐藏判分点：
W3C 参考公式逐通道 ≤1e-12、分段分支、交换律、反预乘、8 位往返、
错误分类、性能与并发。
"""

import math
import random
import threading

import pytest

import colormix

MODES = ('multiply', 'screen', 'overlay', 'soft-light')


def _ref_blend(cb, cs, mode):
    """W3C Compositing and Blending Level 1 参考公式（逐通道）。"""
    if mode == 'multiply':
        return cb * cs
    if mode == 'screen':
        return cb + cs - cb * cs
    if mode == 'overlay':
        if cb <= 0.5:
            return 2.0 * cb * cs
        return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)
    if mode == 'soft-light':
        if cs <= 0.5:
            return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
        if cb <= 0.25:
            d = ((16.0 * cb - 12.0) * cb + 4.0) * cb
        else:
            d = math.sqrt(cb)
        return cb + (2.0 * cs - 1.0) * (d - cb)
    raise AssertionError(mode)


def _ref_mix(a, b, mode, t):
    """预乘空间 source-over 参考实现，返回 (颜色, alpha)。"""
    ab = a[3] if len(a) == 4 else 1.0
    src_a = (b[3] if len(b) == 4 else 1.0) * t
    out_a = src_a + ab * (1.0 - src_a)
    out = []
    for i in range(3):
        if out_a == 0.0:
            out.append(0.0)
            continue
        co = (src_a * (1.0 - ab) * b[i]
              + src_a * ab * _ref_blend(a[i], b[i], mode)
              + (1.0 - src_a) * ab * a[i])
        out.append(co / out_a)
    return out, out_a


def _mix(*args, **kwargs):
    return colormix.ColorMix().mix(*args, **kwargs)


def test_mix_matches_w3c_reference():
    rnd = random.Random(7)
    corners = (0.0, 0.25, 0.5, 0.75, 1.0)
    for _ in range(1500):
        mode = rnd.choice(MODES)
        a = [rnd.choice(corners) if rnd.random() < 0.4 else rnd.random()
             for _ in range(3)]
        b = [rnd.choice(corners) if rnd.random() < 0.4 else rnd.random()
             for _ in range(3)]
        if rnd.random() < 0.7:
            a.append(rnd.choice((0.0, 0.5, 1.0, rnd.random())))
        if rnd.random() < 0.7:
            b.append(rnd.choice((0.0, 0.5, 1.0, rnd.random())))
        t = rnd.choice((0.0, 0.5, 1.0, rnd.random()))
        got = _mix(a, b, mode, t)
        want, want_a = _ref_mix(a, b, mode, t)
        assert len(got) == len(a)
        for i in range(3):
            assert abs(got[i] - want[i]) <= 1e-12, (mode, a, b, t)
        if len(a) == 4:
            assert abs(got[3] - want_a) <= 1e-12


def test_mix_premultiplied_source_over_hand_computed():
    # αb=0, αs=0.5: Co = αs·Cs / αs = Cs；背景完全透明时不走 "先混再乘"。
    assert _mix((0.3, 0.3, 0.3, 0.0), (0.2, 0.6, 0.9, 0.5)) == (0.2, 0.6, 0.9, 0.5)
    # αs=0.5, αb=0.5, multiply: αo=0.75,
    # Co = .5*.5*Cs + .5*.5*Cb*Cs + .5*.5*Cb
    cb, cs = 0.4, 0.8
    co = (0.5 * 0.5 * cs + 0.5 * 0.5 * cb * cs + 0.5 * 0.5 * cb) / 0.75
    got = _mix((cb, cb, cb, 0.5), (cs, cs, cs, 0.5))
    assert abs(got[0] - co) <= 1e-12 and abs(got[3] - 0.75) <= 1e-12
    # 源完全透明 -> 背景原样返回。
    assert _mix((0.2, 0.4, 0.6, 1.0), (0.9, 0.9, 0.9, 0.0)) == (0.2, 0.4, 0.6, 1.0)


def test_mix_t_scales_source_alpha():
    assert _mix((0.2, 0.2, 0.2), (1.0, 1.0, 1.0), 'multiply', 0.0) == (0.2, 0.2, 0.2)
    half = _mix((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0), 'screen', 0.5)
    assert abs(half[0] - 0.5) <= 1e-12 and abs(half[3] - 1.0) <= 1e-12


@pytest.mark.parametrize('mode', ('overlay', 'soft-light'))
def test_mix_piecewise_branches(mode):
    # overlay 按 Cb 分两支
    if mode == 'overlay':
        assert _mix((0.25, 0, 0), (0.3, 0, 0), mode)[0] == 2 * 0.25 * 0.3
        assert _mix((0.75, 0, 0), (0.3, 0, 0), mode)[0] == 1 - 2 * 0.25 * 0.7
        assert _mix((0.5, 0, 0), (0.3, 0, 0), mode)[0] == 0.3
        return
    # soft-light 按 Cs 分两支，D(Cb) 再按 0.25 分两支
    assert _mix((0.1, 1, 0), (0.2, 0.1, 0), mode)[0] == \
        0.1 - (1 - 0.4) * 0.1 * 0.9
    low = 0.1 + 0.6 * (((16 * 0.1 - 12) * 0.1 + 4) * 0.1 - 0.1)
    assert abs(_mix((0.1, 1, 0), (0.8, 0.1, 0), mode)[0] - low) <= 1e-15
    high = 0.9 + 0.6 * (math.sqrt(0.9) - 0.9)
    assert abs(_mix((0.9, 1, 0), (0.8, 0.1, 0), mode)[0] - high) <= 1e-15


@pytest.mark.parametrize('mode', ('multiply', 'screen'))
def test_mix_alpha_one_commutative(mode):
    rnd = random.Random(11)
    for _ in range(500):
        a = tuple(rnd.random() for _ in range(3)) + (1.0,)
        b = tuple(rnd.random() for _ in range(3)) + (1.0,)
        assert _mix(a, b, mode) == _mix(b, a, mode)


def test_mix_unpremultiply_alpha_zero():
    assert colormix.unpremultiply((0.5, 0.25, 0.0, 0.0)) == (0.0, 0.0, 0.0, 0.0)
    assert colormix.unpremultiply((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert all(not math.isnan(v) for v in colormix.unpremultiply((0.0,) * 4))
    # 两个全透明像素混合也不能产生 NaN
    assert _mix((0, 0, 0, 0), (1, 1, 1, 0)) == (0.0, 0.0, 0.0, 0.0)


def test_mix_8bit_roundtrip():
    worst = 0.0
    for i in range(256):
        v = i / 255.0
        assert colormix.to_8bit((v, v, v)) == (i, i, i)
        back = colormix.from_8bit(colormix.to_8bit((v, v, v)))
        worst = max(worst, abs(back[0] - v))
    rnd = random.Random(3)
    for _ in range(300):
        c = tuple(rnd.random() for _ in range(4))
        back = colormix.from_8bit(colormix.to_8bit(c))
        worst = max(worst, max(abs(x - y) for x, y in zip(c, back)))
    assert worst <= 1.0 / 255.0


def test_mix_error_classes_distinguishable():
    m = colormix.ColorMix()
    checks = [
        (lambda: m.mix((0, 0, 0), (1, 1, 1), 'bogus'), colormix.InvalidModeError),
        (lambda: m.mix((1.5, 0, 0), (0, 0, 0)), colormix.ColorRangeError),
        (lambda: m.mix((0, 0, 0, 1.2), (0, 0, 0)), colormix.ColorRangeError),
        (lambda: m.mix((0, 0, 0), (0, 0, 0), 'multiply', 2.0), colormix.ColorRangeError),
        (lambda: m.mix((0, 0), (0, 0, 0)), colormix.LengthMismatchError),
        (lambda: m.mix((0, 0, 0), (0, 0, 0, 0, 0)), colormix.LengthMismatchError),
        (lambda: m.mix((float('nan'), 0, 0), (0, 0, 0)), colormix.NonFiniteInputError),
        (lambda: m.mix((0, 0, 0), (0, float('inf'), 0)), colormix.NonFiniteInputError),
        (lambda: m.mix((0, 0, 0), (0, 0, 0), 'multiply', float('nan')),
         colormix.NonFiniteInputError),
    ]
    seen = set()
    for fn, exc in checks:
        with pytest.raises(exc):
            fn()
        assert issubclass(exc, ValueError)
        seen.add(exc)
    assert len(seen) == 4


def test_mix_100k_pixels_under_300ms():
    import time
    rnd = random.Random(5)
    backdrops = [tuple(rnd.random() for _ in range(4)) for _ in range(100000)]
    sources = [tuple(rnd.random() for _ in range(4)) for _ in range(100000)]
    start = time.perf_counter()
    out = colormix.mix_pixels(backdrops, sources, 'soft-light')
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) == 100000
    assert elapsed_ms < 300.0, '%.1f ms' % elapsed_ms


def test_mix_concurrent_threads_deterministic():
    results = [None] * 4

    def work(seed, slot):
        rnd = random.Random(seed)
        acc = []
        for _ in range(500):
            a = tuple(rnd.random() for _ in range(4))
            b = tuple(rnd.random() for _ in range(4))
            acc.append(_mix(a, b, 'overlay'))
        results[slot] = acc

    threads = [threading.Thread(target=work, args=(i, i)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for i in range(4):
        rnd = random.Random(i)
        expected = []
        for _ in range(500):
            a = tuple(rnd.random() for _ in range(4))
            b = tuple(rnd.random() for _ in range(4))
            expected.append(_mix(a, b, 'overlay'))
        assert results[i] == expected

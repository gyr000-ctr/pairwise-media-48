"""tilecollide 验收测试（swept AABB / 最小轴分离 / 多瓦片一致 / 性能 / 错误 / 并发）。

文件名包含 ``collide``，故 ``pytest -k collide`` 会选中本文件而不依赖
既有的 tests/test_core.py；既有测试与既有行为均不改动。
"""

import math
import threading

import pytest

import tilecollide


# --------------------------------------------------------------- 几何辅助

def _overlaps(box, tile_bounds):
    """独立于实现的 AABB 重叠判定，用于交叉验证逐瓦片结果。"""
    x, y, w, h = box
    minx, miny, maxx, maxy = tile_bounds
    eps = 1e-9
    if x + w <= minx + eps or x >= maxx - eps:
        return False
    if miny is not None and (y + h <= miny + eps or y >= maxy - eps):
        return False
    return True


def _bounds(size, tile):
    if isinstance(tile, int):
        return (tile * size, None, (tile + 1) * size, None)
    col, row = tile
    return (col * size, row * size, (col + 1) * size, (row + 1) * size)


def _tc(size=1.0):
    return tilecollide.TileCollide(size)


# --------------------------------------------------------------- 基础行为

def test_collide_empty_tiles_keeps_position():
    assert _tc().resolve((0.5, 0.5, 0.5, 0.5), []) == (0.5, 0.5)


def test_collide_boundary_origin_unchanged():
    assert _tc().resolve((0.0, 0.0, 0.1, 0.1), []) == (0.0, 0.0)


def test_collide_returns_tuple_of_two_floats():
    out = _tc().resolve((0.5, 0.5, 0.1, 0.1), [])
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(isinstance(v, float) for v in out)


def test_collide_push_right_onto_wall_face():
    # 盒子右缘侵入瓦片 1（区间 [1,2]），应被推回 x+w<=1。
    x, y = _tc().resolve((0.8, 0.5, 0.5, 0.5), [1])
    assert x <= 0.5 + 1e-9
    assert y == pytest.approx(0.5)


def test_collide_push_left_onto_wall_face():
    # 盒子左缘侵入瓦片 0（区间 [0,1]），只能向右推到 x>=1。
    x, _ = _tc().resolve((0.6, 0.5, 0.5, 0.5), [0])
    assert x >= 1.0 - 1e-9


def test_collide_deterministic():
    a = _tc().resolve((0.8, 0.5, 0.5, 0.5), [1])
    b = _tc().resolve((0.8, 0.5, 0.5, 0.5), [1])
    assert a == b


def test_collide_minimum_translation_axis():
    # 在角落同时侵入右侧与上侧瓦片：x 方向需推出 0.30，y 方向仅需 0.12，
    # 最小位移轴是 y，故应沿 y 推出而 x 保持不动。
    tc = _tc()
    x, y = tc.resolve((0.9, 0.95, 0.4, 0.17), [(1, 1)])
    assert y == pytest.approx(0.83, abs=1e-9)
    assert x == pytest.approx(0.9, abs=1e-9)
    assert not _overlaps((x, y, 0.4, 0.17), _bounds(1.0, (1, 1)))


def test_collide_iteration_reaches_fixed_point():
    # 反复以输出作为新输入解算，必须停在不动点且始终不重叠。
    tc = _tc()
    box = (0.5, 0.5, 0.9, 0.9)
    tiles = [0, 1]
    x, y = tc.resolve(box, tiles)
    for _ in range(10):
        nx, ny = tc.resolve((x, y, 0.9, 0.9), tiles)
        assert (nx, ny) == (x, y)  # 已收敛，多次迭代不抖动
        x, y = nx, ny
    for tile in tiles:
        assert not _overlaps((x, y, 0.9, 0.9), _bounds(1.0, tile))


def test_collide_multi_tile_consistent_with_per_tile():
    tc = _tc()
    box = (0.5, 0.5, 0.9, 0.9)
    tiles = [0, 1, (0, 0), (1, 1)]
    x, y = tc.resolve(box, tiles)
    resolved = (x, y, box[2], box[3])
    # 对每个瓦片单独做重叠检测，结果必须全部为不重叠。
    for tile in tiles:
        assert not _overlaps(resolved, _bounds(1.0, tile)), tile


def test_collide_duplicate_tiles_are_idempotent():
    tc = _tc()
    a = tc.resolve((0.8, 0.5, 0.5, 0.5), [1])
    b = tc.resolve((0.8, 0.5, 0.5, 0.5), [1, 1, 1])
    assert a == b


def test_collide_does_not_mutate_inputs():
    tc = _tc()
    box = (0.8, 0.5, 0.5, 0.5)
    tiles = [1, 2]
    box_snapshot = box
    tiles_snapshot = list(tiles)
    tc.resolve(box, tiles)
    assert box == box_snapshot
    assert tiles == tiles_snapshot


# --------------------------------------------------------------- swept AABB

def test_collide_swept_high_speed_does_not_tunnel():
    # 速度远大于瓦片尺寸：朴素离散检测会直接穿过，swept 必须停在墙前。
    tc = _tc()
    x, y = tc.resolve((0.1, 0.1, 0.2, 0.2), [5], velocity=(50.0, 0.0))
    assert x > 0.1                      # 确实发生了移动
    assert x + 0.2 <= 5.0 + 1e-9        # 右缘未越过瓦片 5 左缘
    assert y == pytest.approx(0.1)


def test_collide_swept_stops_at_first_contact():
    tc = _tc()
    # 接触间隙 1.7、速度 2.0 => 进入时间 0.85（本帧内），停在接触点。
    x, _ = tc.resolve((0.1, 0.1, 0.2, 0.2), [2], velocity=(2.0, 0.0))
    assert x + 0.2 <= 2.0 + 1e-9
    assert x >= 2.0 - 0.2 - 1e-6


def test_collide_swept_diagonal_blocked():
    tc = _tc()
    x, y = tc.resolve((0.1, 0.1, 0.2, 0.2), [(3, 3)], velocity=(20.0, 20.0))
    # 两个轴同时高速逼近，至少一个轴在接触前被钳制，整体不得进入瓦片。
    assert not _overlaps((x, y, 0.2, 0.2), _bounds(1.0, (3, 3)))


def test_collide_swept_no_obstacle_uses_full_motion():
    tc = _tc()
    x, y = tc.resolve((0.0, 0.0, 0.1, 0.1), [], velocity=(2.0, 3.0))
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(3.0)


def test_collide_zero_velocity_equals_static():
    tc = _tc()
    a = tc.resolve((0.8, 0.5, 0.5, 0.5), [1])
    b = tc.resolve((0.8, 0.5, 0.5, 0.5), [1], velocity=(0.0, 0.0))
    assert a == b


# --------------------------------------------------------------- 错误分类

def test_collide_invalid_size_zero_and_negative():
    for bad in (0, 0.0, -1, -1e-3):
        with pytest.raises(tilecollide.InvalidSizeError):
            tilecollide.TileCollide(bad)


def test_collide_invalid_size_type():
    for bad in ("1", None, [1.0], object()):
        with pytest.raises(tilecollide.InvalidSizeError):
            tilecollide.TileCollide(bad)


def test_collide_non_finite_size_classified():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(tilecollide.NonFiniteError):
            tilecollide.TileCollide(bad)


def test_collide_invalid_box_shape():
    tc = _tc()
    for bad in ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0, 1.0), [0.0, 0.0, 1.0], "box"):
        with pytest.raises(tilecollide.InvalidBoxError):
            tc.resolve(bad, [])


def test_collide_invalid_box_dimensions():
    tc = _tc()
    with pytest.raises(tilecollide.InvalidBoxError):
        tc.resolve((0.0, 0.0, 0.0, 1.0), [])
    with pytest.raises(tilecollide.InvalidBoxError):
        tc.resolve((0.0, 0.0, 1.0, -1.0), [])


def test_collide_non_finite_box_classified():
    tc = _tc()
    for bad_box in (
        (float("nan"), 0.0, 1.0, 1.0),
        (0.0, float("inf"), 1.0, 1.0),
        (0.0, 0.0, 1.0, float("-inf")),
    ):
        with pytest.raises(tilecollide.NonFiniteError):
            tc.resolve(bad_box, [])


def test_collide_non_finite_velocity_classified():
    tc = _tc()
    with pytest.raises(tilecollide.NonFiniteError):
        tc.resolve((0.0, 0.0, 0.1, 0.1), [], velocity=(float("nan"), 0.0))


def test_collide_invalid_tile_classified():
    tc = _tc()
    for bad in (1.5, "0", (1,), (1, 2, 3), None, [1]):
        with pytest.raises(tilecollide.InvalidTileError):
            tc.resolve((0.1, 0.1, 0.2, 0.2), [bad])


def test_collide_error_classes_are_distinct_and_value_errors():
    classes = [
        tilecollide.TileCollideError,
        tilecollide.InvalidSizeError,
        tilecollide.InvalidBoxError,
        tilecollide.InvalidTileError,
        tilecollide.NonFiniteError,
    ]
    assert len(set(classes)) == len(classes)
    assert all(issubclass(c, ValueError) for c in classes)


# --------------------------------------------------------------- 并发安全

def test_collide_shared_instance_thread_safe():
    tc = _tc()
    inputs = [
        ((0.8, 0.5, 0.5, 0.5), [1]),
        ((0.6, 0.5, 0.5, 0.5), [0]),
        ((0.5, 0.5, 0.9, 0.9), [0, 1]),
        ((0.95, 0.95, 0.2, 0.2), [(1, 1)]),
    ]
    expected = [tc.resolve(box, tiles) for box, tiles in inputs]
    errors = []
    results = []

    def worker():
        try:
            for _ in range(200):
                for (box, tiles), want in zip(inputs, expected):
                    got = tc.resolve(box, tiles)
                    results.append(got == want)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results and all(results)


def test_collide_swept_thread_safe():
    tc = _tc()
    want = tc.resolve((0.1, 0.1, 0.2, 0.2), [5], velocity=(50.0, 0.0))
    errors = []
    got_set = set()

    def worker():
        try:
            for _ in range(100):
                got_set.add(tc.resolve((0.1, 0.1, 0.2, 0.2), [5], velocity=(50.0, 0.0)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert got_set == {want}


# --------------------------------------------------------------- 性能上界

def test_collide_perf_100k_under_300ms():
    import time

    tc = _tc()
    # 混合场景：部分无碰撞、部分静态推出、部分 swept，取整段墙钟时间。
    scenarios = [
        ((0.5, 0.5, 0.4, 0.4), []),
        ((0.5, 0.5, 0.4, 0.4), [5, 6, 7]),
        ((0.8, 0.5, 0.5, 0.5), [1]),
    ]
    start = time.perf_counter()
    n = 0
    for i in range(100_000):
        box, tiles = scenarios[i % len(scenarios)]
        tc.resolve(box, tiles)
        n += 1
    elapsed = time.perf_counter() - start
    assert n == 100_000
    assert elapsed < 0.300, "%.1f ms 超过 300 ms 上界" % (elapsed * 1000.0)

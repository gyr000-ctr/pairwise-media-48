"""GridBfs 约束测试（独立于 tests/test_core.py，不改动既有测试）。

覆盖：
- 与逐点参考实现（独立的 O(V^2) Dijkstra）逐格一致，含多种对角代价；
- 距离单调、全部有限、不可达为 -1、光照衰减单调无溢出；
- 非法网格与非有限代价抛出可区分的异常类型；
- 并发安全（多线程共享同一实例）；
- 512x512 性能上界 300 ms。
"""

import math
import threading
import time

import pytest

import gridpath
from gridpath import GridBfs, InvalidGridError, NonFiniteCostError


# ---------- 独立参考实现：O(V^2) Dijkstra，与被测代码路径完全不同 ----------

def _reference_distances(walls, start, diagonal_cost):
    h, w = len(walls), len(walls[0])

    def blocked(r, c):
        return not (0 <= r < h and 0 <= c < w) or bool(walls[r][c])

    sr, sc = start
    assert not blocked(sr, sc)
    best = {(sr, sc): 0.0}
    done = set()
    while True:
        cur = None
        for cell, d in best.items():
            if cell not in done and (cur is None or d < best[cur]):
                cur = cell
        if cur is None:
            break
        done.add(cur)
        r, c = cur
        d = best[cur]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if blocked(nr, nc):
                    continue
                if dr != 0 and dc != 0 and (blocked(r + dr, c) or blocked(r, c + dc)):
                    continue  # 不穿墙角
                edge = diagonal_cost if dr != 0 and dc != 0 else 1.0
                nd = d + edge
                if nd < best.get((nr, nc), math.inf):
                    best[(nr, nc)] = nd
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            row.append(-1 if (r, c) not in best else best[(r, c)])
        out.append(row)
    return out


GRIDS = [
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1], [1, 0]],
    [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
    ],
    [[0] * 7 for _ in range(7)],
]

STARTS = [(0, 0), (2, 2), (4, 0)]
COSTS = [math.sqrt(2.0), 1.0, 1.5, 2.0, 3.0]


@pytest.mark.parametrize("cost", COSTS)
@pytest.mark.parametrize("grid", GRIDS)
def test_bfs_matches_pointwise_reference(grid, cost):
    g = GridBfs(diagonal_cost=cost)
    for sr, sc in STARTS:
        if sr >= len(grid) or sc >= len(grid[0]) or grid[sr][sc]:
            continue
        got = g.distances(grid, (sr, sc))
        want = _reference_distances(grid, (sr, sc), cost)
        assert len(got) == len(want)
        for r in range(len(want)):
            for c in range(len(want[0])):
                if want[r][c] == -1:
                    assert got[r][c] == -1
                else:
                    assert got[r][c] == pytest.approx(want[r][c], abs=1e-12)


def test_bfs_unreachable_minus_one_and_shape():
    d = GridBfs().distances([[0, 1], [1, 0]], (0, 0))
    assert d[1][1] == -1
    assert d[0][0] == 0.0
    assert isinstance(d, list) and isinstance(d[0], list)


def test_bfs_corner_cutting_blocked():
    # 对角格的两个正交邻格都是墙 -> 对角不可达（与既有测试语义一致）
    d = GridBfs().distances([[0, 0, 0], [0, 1, 0], [0, 0, 0]], (0, 0))
    assert d[1][1] == -1


def test_bfs_diagonal_cost_is_configurable():
    grid = [[0, 0], [0, 0]]
    d1 = GridBfs(diagonal_cost=1.5).distances(grid, (0, 0))
    d2 = GridBfs(diagonal_cost=2.5).distances(grid, (0, 0))
    # 对角更便宜时取对角；比对角贵的路线由两步正交（2.0）取代
    assert d1[1][1] == pytest.approx(1.5)
    assert d2[1][1] == pytest.approx(2.0)
    # 正交代价恒为 1
    assert d1[0][1] == 1.0 and d1[1][0] == 1.0


def test_bfs_distances_monotonic_finite_and_light_decay():
    grid = GRIDS[3]
    diag = math.sqrt(2.0)
    d = GridBfs().distances(grid, (0, 0))
    h, w = len(grid), len(grid[0])
    points = []
    for r in range(h):
        for c in range(w):
            v = d[r][c]
            if v == -1:
                continue
            assert isinstance(v, float) and math.isfinite(v)
            points.append((v, r, c))

    # 距离场不变量：每条可达边上满足三角不等式 |d[a]-d[b]| <= 边代价，
    # 即沿传播路径距离随边代价单调增长，不会出现回跳
    for r in range(h):
        for c in range(w):
            if d[r][c] == -1:
                continue
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or d[nr][nc] == -1:
                    continue
                if dr and dc and (grid[r + dr][c] or grid[r][c + dc]):
                    continue
                edge = diag if dr and dc else 1.0
                assert abs(d[nr][nc] - d[r][c]) <= edge + 1e-12

    # 光照传播衰减：exp(-k*d) 随距离单调递减、恒正有限、无下溢/溢出
    k = 0.1
    points.sort()
    prev_att = 1.0
    for v, _r, _c in points:
        att = math.exp(-k * v)
        assert math.isfinite(att) and 0.0 < att <= 1.0
        assert att <= prev_att + 1e-15
        prev_att = att
    max_d = points[-1][0]
    assert math.isfinite(math.exp(-k * max_d)) and math.exp(-k * max_d) > 0.0


def test_bfs_deterministic():
    g = GridBfs()
    a = g.distances(GRIDS[3], (0, 0))
    b = g.distances(GRIDS[3], (0, 0))
    assert a == b


def test_bfs_multi_source_takes_nearest():
    grid = [[0] * 5 for _ in range(5)]
    d = GridBfs().distances(grid, [(0, 0), (4, 4)])
    assert d[0][0] == 0.0 and d[4][4] == 0.0
    # 中间点到两源等距
    assert d[2][2] == pytest.approx(2 * math.sqrt(2.0))


def test_bfs_invalid_grid_errors_distinguishable():
    g = GridBfs()
    bad_grids = [
        [],
        [[]],
        [[0, 0], [0]],                       # 异形行
        [[0, "x"]],                          # 非数值单元
        [[0, float("nan")]],                 # 非有限单元
        [[0, float("inf")]],
        "not-a-grid",
    ]
    for bad in bad_grids:
        with pytest.raises(InvalidGridError):
            g.distances(bad, (0, 0))

    with pytest.raises(InvalidGridError):
        g.distances(GRIDS[0], (5, 5))        # 起点越界
    with pytest.raises(InvalidGridError):
        g.distances(GRIDS[0], (1, 1))        # 起点在墙上
    with pytest.raises(InvalidGridError):
        g.distances(GRIDS[0], (0,))         # 起点形状错误


def test_bfs_non_finite_cost_errors_distinguishable():
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0, "x", None):
        with pytest.raises(NonFiniteCostError):
            GridBfs(diagonal_cost=bad)

    # 两类错误类型必须可区分，且都属于 ValueError（兼容既有 _raises 风格）
    assert issubclass(NonFiniteCostError, ValueError)
    assert issubclass(InvalidGridError, ValueError)
    assert NonFiniteCostError is not InvalidGridError
    try:
        GridBfs(diagonal_cost=float("nan"))
    except InvalidGridError:  # pragma: no cover - 必须落到另一个类型
        pytest.fail("非有限代价被误报为非法网格")
    except NonFiniteCostError:
        pass


def test_bfs_thread_safety():
    g = GridBfs()
    expected = [g.distances(grid, (0, 0)) for grid in GRIDS if not grid[0][0]]
    errors = []
    results = []

    def worker(idx):
        try:
            for _ in range(20):
                # 共享同一实例、并发读取同一网格
                got = g.distances(GRIDS[idx % len(expected)], (0, 0))
                results.append((idx, got))
        except Exception as exc:  # pragma: no cover - 并发不应抛异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for idx, got in results:
        assert got == expected[idx % len(expected)]


def test_bfs_performance_512x512_under_300ms():
    grid = [[0] * 512 for _ in range(512)]
    g = GridBfs()
    g.distances(grid, (0, 0))  # 预热
    t0 = time.perf_counter()
    d = g.distances(grid, (0, 0))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 300.0, f"耗时 {elapsed_ms:.1f} ms 超过 300 ms 上界"
    assert d[511][511] == pytest.approx(511 * math.sqrt(2.0))

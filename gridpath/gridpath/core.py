"""gridpath 核心 API：8 邻接加权 BFS（Dijkstra）瓦片距离场。

语义（与逐点 BFS 完全一致）：
- 网格 ``walls`` 为二维序列，真值表示墙；起点 ``start`` 为 ``(row, col)``，
  也可传入若干起点组成的序列做多源距离场。
- 正交边代价固定为 ``1.0``；对角边代价由构造参数 ``diagonal_cost`` 配置，
  默认 ``sqrt(2)``。对角移动禁止穿墙角：两个正交邻格任一为墙则该对角边不可达。
- 不可达单元输出 ``-1``；其余输出为有限 ``float``，沿任意传播路径距离严格
  单调递增（边代价均为正），有限输入下不会产生 inf/NaN 溢出。
- 本实现无可变实例状态，``distances`` 仅使用线程局部数据，可并发调用。
"""

import heapq
import math

__all__ = ["GridBfs", "InvalidGridError", "NonFiniteCostError"]


class InvalidGridError(ValueError):
    """网格本身非法（空网格、异形行、非数值单元、起点越界/落在墙上等）。"""


class NonFiniteCostError(TypeError, ValueError):
    """边代价非有限值（NaN/inf）或非正数。

    同时继承 TypeError 与 ValueError，调用方既可按类型也可按值语义捕获，
    且与 :class:`InvalidGridError` 的类型明确可区分。
    """


class GridBfs:
    """8 邻接加权 BFS 距离场计算器。

    Args:
        diagonal_cost: 对角边代价，须为正的有限数，默认 ``sqrt(2)``。
    """

    __slots__ = ("diagonal_cost",)

    def __init__(self, diagonal_cost=math.sqrt(2.0)):
        try:
            cost = float(diagonal_cost)
        except (TypeError, ValueError):
            raise NonFiniteCostError(
                "diagonal_cost 必须是有限正数，收到 %r" % (diagonal_cost,)
            )
        if not math.isfinite(cost) or cost <= 0.0:
            raise NonFiniteCostError(
                "diagonal_cost 必须是有限正数，收到 %r" % (diagonal_cost,)
            )
        self.diagonal_cost = cost

    def distances(self, walls, start):
        """计算距离场，返回与 ``walls`` 同形的 ``list[list[float|int(-1)]]``。"""
        flags, h, w, stride = _validate_grid(walls)
        starts = _normalize_starts(start, h, w, flags, stride)

        total = (h + 2) * stride
        inf = math.inf
        dist = [inf] * total
        heap = []
        for idx in starts:
            dist[idx] = 0.0
            heapq.heappush(heap, (0.0, idx))

        heappush = heapq.heappush
        heappop = heapq.heappop
        diag = self.diagonal_cost

        while heap:
            d, idx = heappop(heap)
            if d != dist[idx]:
                continue  # 惰性删除过期堆项

            up = idx - stride
            down = idx + stride
            left = idx - 1
            right = idx + 1

            # 四条正交边，代价 1.0
            if not flags[up]:
                nd = d + 1.0
                if nd < dist[up]:
                    dist[up] = nd
                    heappush(heap, (nd, up))
            if not flags[down]:
                nd = d + 1.0
                if nd < dist[down]:
                    dist[down] = nd
                    heappush(heap, (nd, down))
            if not flags[left]:
                nd = d + 1.0
                if nd < dist[left]:
                    dist[left] = nd
                    heappush(heap, (nd, left))
            if not flags[right]:
                nd = d + 1.0
                if nd < dist[right]:
                    dist[right] = nd
                    heappush(heap, (nd, right))

            # 四条对角边，代价 diag；两个正交邻格都通行才允许（不穿墙角）
            if not flags[up] and not flags[left] and not flags[up - 1]:
                nd = d + diag
                t = up - 1
                if nd < dist[t]:
                    dist[t] = nd
                    heappush(heap, (nd, t))
            if not flags[up] and not flags[right] and not flags[up + 1]:
                nd = d + diag
                t = up + 1
                if nd < dist[t]:
                    dist[t] = nd
                    heappush(heap, (nd, t))
            if not flags[down] and not flags[left] and not flags[down - 1]:
                nd = d + diag
                t = down - 1
                if nd < dist[t]:
                    dist[t] = nd
                    heappush(heap, (nd, t))
            if not flags[down] and not flags[right] and not flags[down + 1]:
                nd = d + diag
                t = down + 1
                if nd < dist[t]:
                    dist[t] = nd
                    heappush(heap, (nd, t))

        result = []
        for r in range(h):
            base = (r + 1) * stride + 1
            row = dist[base:base + w]
            result.append([-1 if x == inf else x for x in row])
        return result


def _validate_grid(walls):
    """校验网格并返回带一圈墙边框的扁平 bytearray 及尺寸信息。"""
    if not isinstance(walls, (list, tuple)) or len(walls) == 0:
        raise InvalidGridError("walls 必须是非空的二维序列")
    first = walls[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise InvalidGridError("walls 的行必须是非空序列")
    h = len(walls)
    w = len(first)

    stride = w + 2
    # 外围一圈全部为墙（1），BFS 中可省略边界判断
    flags = bytearray(b"\x01") * ((h + 2) * stride)

    for r, row in enumerate(walls):
        if not isinstance(row, (list, tuple)) or len(row) != w:
            raise InvalidGridError("walls 第 %d 行形状不一致" % r)
        base = (r + 1) * stride + 1
        for c, value in enumerate(row):
            if isinstance(value, bool):
                flags[base + c] = 1 if value else 0
            elif isinstance(value, int):
                flags[base + c] = 1 if value else 0
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise InvalidGridError(
                        "walls 单元 (%d,%d) 含非有限值" % (r, c)
                    )
                flags[base + c] = 1 if value else 0
            else:
                raise InvalidGridError(
                    "walls 单元 (%d,%d) 不是数值: %r" % (r, c, value)
                )
    return flags, h, w, stride


def _normalize_starts(start, h, w, flags, stride):
    """支持单起点 (r, c) 与多起点 [(r, c), ...]，统一转成扁平索引。"""
    if not isinstance(start, (list, tuple)) or len(start) == 0:
        raise InvalidGridError("start 必须是 (row, col) 或其序列")

    if isinstance(start[0], (list, tuple)):
        pairs = start
    else:
        pairs = (start,)

    indices = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InvalidGridError("起点必须是两个整数组成的坐标")
        r, c = pair
        if isinstance(r, bool) or isinstance(c, bool) or not (
            isinstance(r, int) and isinstance(c, int)
        ):
            raise InvalidGridError("起点坐标必须是整数: %r" % (pair,))
        if not (0 <= r < h and 0 <= c < w):
            raise InvalidGridError("起点越界: %r" % (pair,))
        idx = (r + 1) * stride + (c + 1)
        if flags[idx]:
            raise InvalidGridError("起点落在墙上: %r" % (pair,))
        indices.append(idx)
    return indices

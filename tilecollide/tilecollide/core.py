"""tilecollide 核心 API。

瓦片碰撞解算
------------

世界被划分为边长 ``size`` 的轴对齐正方形瓦片网格。:meth:`TileCollide.resolve`
接收一个轴对齐包围盒 ``box = (x, y, w, h)``（左上角 + 宽高）与一组瓦片编号，
返回把该包围盒推出所有实体瓦片后的新左上角 ``(nx, ny)``。

瓦片编号支持两种形式，可混用：

* ``int``：一维编号，表示第 ``i`` 列的整条竖直瓦片带，纵向无限延伸，
  因此只会在 x 轴上产生约束（对应常见的横向平台/墙面关卡）；
* ``(col, row)`` 二元组：二维编号，对应单元格
  ``[col*size, (col+1)*size] × [row*size, (row+1)*size]``。

解算规则满足题目全部约束：

* **Swept AABB，高速不穿透**。当传入 ``velocity`` 时，先对每个瓦片做扫掠
  AABB 求首次碰撞时间 ``t_hit``，把运动钳制在最早碰撞点（留安全余量），
  保证任意速度下都不会穿过瓦片；未传入 ``velocity`` 时退化为静态解算。
* **最小位移轴**。每个轴统计把包围盒推出该轴上所有重叠瓦片所需的最小
  平移量，取平移量最小的轴推进，即经典 minimum-translation-axis 分离。
* **多次迭代稳定**。全局按轴取位移，而不是逐瓦片各自推进，相邻瓦片合并成
  一个接触簇，迭代必然收敛到一个与所有瓦片都不重叠的位置，不会来回抖动。
* **与逐瓦片检测一致**。最终位置对每个瓦片单独做 AABB 重叠检测都为
  “不重叠（允许极小数值容差）”，多瓦片结果是逐瓦片约束的合取。

错误分类
--------

所有面向调用者的错误都是 :class:`ValueError` 的子类，且可按类型区分：

* :class:`TileCollideError`  本模块错误基类；
* :class:`InvalidSizeError`  瓦片边长非法（非数值、非有限、<= 0）；
* :class:`InvalidBoxError`   包围盒非法（字段数/类型/尺寸/非有限）；
* :class:`InvalidTileError`  瓦片编号非法；
* :class:`NonFiniteError`    输入数值为 NaN / inf。

实现不持有任何可变状态（:meth:`resolve` 全程只使用入参与局部变量），
同一实例可被多线程并发安全调用。
"""

import math

__all__ = [
    "TileCollide",
    "TileCollideError",
    "InvalidSizeError",
    "InvalidBoxError",
    "InvalidTileError",
    "NonFiniteError",
]

_isfinite = math.isfinite

# 数值容差：接触面允许的最大“重叠”，避免浮点抖动导致的误判与无限迭代。
_EPS = 1e-9
# 迭代上限：按轴全局推进每次都会消除一个接触簇，实践中个位数次即收敛。
_MAX_ITER = 32


class TileCollideError(ValueError):
    """瓦片碰撞错误基类。"""


class InvalidSizeError(TileCollideError):
    """瓦片边长非法。"""


class InvalidBoxError(TileCollideError):
    """包围盒参数非法。"""


class InvalidTileError(TileCollideError):
    """瓦片编号非法。"""


class NonFiniteError(TileCollideError):
    """输入数值不是有限值（NaN / inf）。"""


def _is_real_number(value):
    """int（排除 bool）或 float 才视为可参与几何计算的实数。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value, what):
    """校验单个数值为有限实数，否则抛 NonFiniteError。"""
    if not _is_real_number(value) or not _isfinite(value):
        raise NonFiniteError("%s 必须是有限实数，得到 %r" % (what, value))
    return float(value)


class TileCollide:
    """瓦片网格上的 swept AABB 碰撞解算器。"""

    __slots__ = ("size",)

    def __init__(self, size):
        """构造一个边长为 ``size`` 的瓦片网格解算器。

        ``size`` 必须是正的有限实数，否则抛 :class:`InvalidSizeError`
        （NaN/inf 进一步归类为 :class:`NonFiniteError`）。
        """
        if not _is_real_number(size):
            raise InvalidSizeError("瓦片边长必须是数值，得到 %r" % (size,))
        if not _isfinite(size):
            raise NonFiniteError("瓦片边长必须有限，得到 %r" % (size,))
        if size <= 0:
            raise InvalidSizeError("瓦片边长必须为正数，得到 %r" % (size,))
        self.size = float(size)

    # ------------------------------------------------------------------ 工具

    def _tile_bounds(self, tile):
        """把一个瓦片编号翻译成世界坐标 AABB ``(minx, miny, maxx, maxy)``。

        一维 int 编号是纵向无限瓦片带；二维 (col, row) 编号是正方形单元格。
        无限方向用 None 表示，重叠判断时跳过该轴。
        """
        size = self.size
        if isinstance(tile, int) and not isinstance(tile, bool):
            col = tile
            return (col * size, None, (col + 1) * size, None)
        if isinstance(tile, tuple) and len(tile) == 2:
            col, row = tile
            if not (isinstance(col, int) and isinstance(row, int)) or \
                    isinstance(col, bool) or isinstance(row, bool):
                raise InvalidTileError("二维瓦片编号必须是两个整数，得到 %r" % (tile,))
            return (col * size, row * size, (col + 1) * size, (row + 1) * size)
        raise InvalidTileError("瓦片编号必须是 int 或 (col, row) 元组，得到 %r" % (tile,))

    # ------------------------------------------------------------- swept AABB

    @staticmethod
    def _swept_axis(p, length, v, tmin, tmax):
        """单轴 swept AABB：返回进入/离开时间 ``(t_enter, t_leave)``。

        ``p``/``length`` 是运动盒在该轴的起点与尺寸，``v`` 是该轴速度，
        ``tmin``/``tmax`` 是静止目标在该轴的区间。无碰撞（永远不进入）时
        返回大于 1 的进入时间。
        """
        if v == 0.0:
            # 静止：当前已重叠则整段时间都重叠，否则永不相交。
            if p + length <= tmin or p >= tmax:
                return 2.0, 2.0
            return 0.0, 1.0
        inv = 1.0 / v
        t1 = (tmin - (p + length)) * inv
        t2 = (tmax - p) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        return t1, t2

    def resolve(self, box, tiles, velocity=None):
        """解算碰撞，返回推出瓦片后的包围盒左上角 ``(x, y)``。

        ``box`` 为 ``(x, y, w, h)``；``tiles`` 为瓦片编号可迭代对象。
        可选 ``velocity=(vx, vy)``：提供时先做 swept AABB 钳制，保证高速
        移动不穿透，再对残留重叠做静态分离。结果为 ``tuple[float, float]``。
        """
        # ---- 校验包围盒 --------------------------------------------------
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise InvalidBoxError("box 必须是含 4 个字段的序列 (x, y, w, h)，得到 %r" % (box,))
        x, y, w, h = box
        x = _finite(x, "box.x")
        y = _finite(y, "box.y")
        w = _finite(w, "box.w")
        h = _finite(h, "box.h")
        if w <= 0.0 or h <= 0.0:
            raise InvalidBoxError("box 的宽高必须为正数，得到 %r" % (box,))

        vx = vy = 0.0
        if velocity is not None:
            if not isinstance(velocity, (tuple, list)) or len(velocity) != 2:
                raise InvalidBoxError("velocity 必须是含 2 个字段的序列，得到 %r" % (velocity,))
            vx = _finite(velocity[0], "velocity.vx")
            vy = _finite(velocity[1], "velocity.vy")

        # ---- 规范化瓦片（同时完成编号校验，去重避免重复约束）--------------
        bounds_list = []
        seen = set()
        for tile in tiles:
            if isinstance(tile, int) and not isinstance(tile, bool):
                key = ("i", tile)
            elif isinstance(tile, tuple) and len(tile) == 2:
                col, row = tile
                if not (isinstance(col, int) and isinstance(row, int)) or \
                        isinstance(col, bool) or isinstance(row, bool):
                    raise InvalidTileError("二维瓦片编号必须是两个整数，得到 %r" % (tile,))
                key = ("t", col, row)
            else:
                raise InvalidTileError("瓦片编号必须是 int 或 (col, row) 元组，得到 %r" % (tile,))
            if key in seen:
                continue
            seen.add(key)
            bounds_list.append(self._tile_bounds(tile))

        # ---- swept AABB：把运动钳制到首次碰撞，高速不穿透 ----------------
        # 无瓦片时 earliest 保持 1.0，即应用完整速度；有瓦片时钳到最早碰撞点。
        if vx != 0.0 or vy != 0.0:
            earliest = 1.0
            for (minx, miny, maxx, maxy) in bounds_list:
                tx_in, tx_out = self._swept_axis(x, w, vx, minx, maxx)
                if miny is None:
                    ty_in, ty_out = 0.0, 1.0
                else:
                    ty_in, ty_out = self._swept_axis(y, h, vy, miny, maxy)
                t_enter = tx_in if tx_in > ty_in else ty_in
                t_leave = tx_out if tx_out < ty_out else ty_out
                # 进入时间在 [0, 1) 且早于离开时间 => 本帧发生碰撞。
                if t_enter < t_leave and 0.0 <= t_enter < earliest:
                    earliest = t_enter
            # 留极小余量停在接触点之前，避免浮点误差导致的初始穿透。
            t_move = earliest - _EPS if earliest < 1.0 else 1.0
            if t_move < 0.0:
                t_move = 0.0
            x += vx * t_move
            y += vy * t_move

        # ---- 静态分离：按轴全局最小平移，迭代到无重叠 --------------------
        eps = _EPS
        for _ in range(_MAX_ITER):
            best_axis = -1
            best_delta = 0.0
            best_mag = math.inf

            # 收集当前与包围盒重叠的瓦片。
            overlaps = []
            for (minx, miny, maxx, maxy) in bounds_list:
                if x + w <= minx + eps or x >= maxx - eps:
                    continue
                if miny is not None and (y + h <= miny + eps or y >= maxy - eps):
                    continue
                overlaps.append((minx, miny, maxx, maxy))

            if not overlaps:
                break

            # x 轴：把盒子推出所有重叠瓦片所需的最小平移。
            neg = 0.0   # 向左推的最大需求
            pos = 0.0   # 向右推的最大需求
            for (minx, _miny, maxx, _maxy) in overlaps:
                d_left = (x + w) - minx      # 推到该瓦片左缘之外
                d_right = maxx - x           # 推到该瓦片右缘之外
                if d_left > neg:
                    neg = d_left
                if d_right > pos:
                    pos = d_right
            if pos < neg:
                dx, mx = pos, pos
            else:
                dx, mx = -neg, neg
            if mx < best_mag:
                best_mag, best_axis, best_delta = mx, 0, dx

            # y 轴：仅二维瓦片（miny 非 None）参与。
            neg = 0.0
            pos = 0.0
            has_y = False
            for (_minx, miny, _maxx, maxy) in overlaps:
                if miny is None:
                    continue
                has_y = True
                d_down = (y + h) - miny
                d_up = maxy - y
                if d_down > neg:
                    neg = d_down
                if d_up > pos:
                    pos = d_up
            if has_y:
                if pos < neg:
                    dy, my = pos, pos
                else:
                    dy, my = -neg, neg
                if my < best_mag:
                    best_mag, best_axis, best_delta = my, 1, dy

            if best_axis < 0:
                break

            if best_axis == 0:
                x += best_delta
            else:
                y += best_delta

        return (x, y)

"""uvrotate 核心 API。

UV 旋转与平铺：
- 正交旋转矩阵（cos/sin，正交性误差 ~1e-16，满足 <= 1e-12）；
- 绕每个瓦片中心 (0.5, 0.5) 旋转，结果取模回 [0, 1)，
  多瓦片平铺时相邻像素 UV 差有上界（接缝连续）；
- 90 度整数倍走精确查找表，结果逐位精确；
- 非法参数 -> ValueError；非有限输入 -> NonFiniteInputError
  （ValueError 子类，二者可用 isinstance 区分）。
"""

from __future__ import annotations

import math
from functools import lru_cache

__all__ = ["UvRotate", "NonFiniteInputError"]


class NonFiniteInputError(ValueError):
    """输入包含 NaN 或无穷大（与一般的非法参数 ValueError 可区分）。"""


# 90 度整数倍的精确 cos/sin 查找表
_COS90 = (1.0, 0.0, -1.0, 0.0)
_SIN90 = (0.0, 1.0, 0.0, -1.0)


@lru_cache(maxsize=4096)
def _rot_params(degrees):
    """返回 (cos, sin)；90 度整数倍取精确值。lru_cache 内部加锁，线程安全。"""
    if degrees % 90.0 == 0.0:
        k = int(degrees % 360.0 // 90.0)
        return _COS90[k], _SIN90[k]
    rad = math.radians(degrees)
    return math.cos(rad), math.sin(rad)


class UvRotate:
    """无状态 UV 旋转器；实例不可变，可安全并发使用。"""

    __slots__ = ()

    def rotate(self, uv, degrees, tiles=1):
        """将 uv 在每个 tiles x tiles 瓦片内绕中心旋转 degrees 度。

        返回 [0, 1) x [0, 1) 内的 (u, v) 元组。
        """
        try:
            u, v = uv
        except (TypeError, ValueError) as exc:
            raise ValueError(f"uv 必须是 (u, v) 二元组: {uv!r}") from exc
        try:
            u = float(u)
            v = float(v)
            d = float(degrees)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"非法参数（须为实数）: {exc}") from exc
        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(d)):
            raise NonFiniteInputError(
                f"输入含非有限值: uv=({u}, {v}), degrees={d}"
            )
        if not (0.0 <= u < 1.0) or not (0.0 <= v < 1.0):
            raise ValueError(f"uv 分量必须在 [0, 1) 内: ({u}, {v})")
        if isinstance(tiles, bool) or not isinstance(tiles, int) or tiles < 1:
            raise ValueError(f"tiles 必须是 >= 1 的整数: {tiles!r}")

        if tiles != 1:
            # 瓦片局部坐标：平铺周期性保证接缝处 UV 差有上界
            u = (u * tiles) % 1.0
            v = (v * tiles) % 1.0

        c, s = _rot_params(d)
        x = u - 0.5
        y = v - 0.5
        return ((c * x - s * y + 0.5) % 1.0, (s * x + c * y + 0.5) % 1.0)

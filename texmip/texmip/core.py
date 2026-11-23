"""texmip 核心 API：mip 级别选择与三线性权重。

级别选择规则
------------
对一次屏幕像素到纹理空间的映射，先求两个屏幕轴在纹理空间中各自覆盖的
纹素长度（"拉伸"）：

* 各向同性缩放 ``scale``（标量）：两轴拉伸均为 ``scale``；
* 各向异性缩放 ``(sx, sy)``：两轴拉伸分别为 ``sx``、``sy``（单位：纹素/像素）；
* 纹理坐标导数 ``(dudx, dvdx, dudy, dvdy)``：一像素的足迹在纹理空间是
  平行四边形，两轴边缘向量为 ``(w*dudx, h*dvdx)`` 与 ``(w*dudy, h*dvdy)``，
  其模长即两轴拉伸。顺序与 ``dFdx(uv)``、``dFdy(uv)`` 拼接一致。

连续级别（LOD）取**最大拉伸**的 log2：``lod = log2(max(sx, sy))``。各向异性
比 ``max(sx, sy) / min(sx, sy)`` 单独暴露为 :meth:`TexMip.anisotropy`。
三线性采样在相邻两级间插值：基级 ``l0 = floor(lod)``，权重
``(1 - frac, frac)``，其中 ``frac = lod - l0``。

级别钳制到 ``[0, levels)``：``lod`` 低于 0 时取 0 级且 ``frac = 0``；到达
最高级后不再向上插值（权重退化为 1, 0），因此两级索引始终合法、权重和恒
为 1。级别数 ``levels = floor(log2(max(w, h))) + 1``。

与逐级暴力比较的一致性
----------------------
:meth:`TexMip.level` 的闭式结果与"逐级下采样后比较误差"的暴力过程一致：
第 ``l`` 级的纹素尺寸为 ``2**l`` 原始纹素，足迹在该级上的相对拉伸为
``s / 2**l``。三线性需要一个不放大足迹（``s / 2**l >= 1``）且最接近的
基级，即在满足 ``2**l <= s`` 的级别中取 ``l`` 最大者——这恰是
``floor(log2(s))``。测试用逐级循环实现了该暴力过程并逐点比对。

错误分类
--------
所有面向调用者的错误都是 ``ValueError`` 的子类，且可按类型区分：

* ``MipError``          mip 相关错误基类；
* ``InvalidSizeError``  纹理宽高不是正整数；
* ``InvalidScaleError`` 缩放/导数输入结构非法（非数值、分量个数错误）或
  缩放非正；
* ``InvalidLevelError`` 级索引越界或不是整数；
* ``NonFiniteError``    缩放或导数包含 NaN / inf。

实现不包含任何可变全局状态，实例在构造后只读：同一实例可由多线程并发
调用，且没有惰性初始化或缓存写入。
"""

import math

__all__ = [
    "TexMip",
    "MipError",
    "InvalidSizeError",
    "InvalidScaleError",
    "InvalidLevelError",
    "NonFiniteError",
]

_isfinite = math.isfinite
_frexp = math.frexp
_log2 = math.log2
_hypot = math.hypot
_INF = float("inf")
# 1 的前一个双精度值：log2 在最接近 2 的幂处可能舍入到整数。
_ONE_MINUS_ULP = 0.9999999999999999


class MipError(ValueError):
    """mip 相关错误的基类。"""


class InvalidSizeError(MipError):
    """纹理宽高非法。"""


class InvalidScaleError(MipError):
    """缩放/导数输入非法：非数值、分量个数错误或缩放非正。"""


class InvalidLevelError(MipError):
    """级索引非法：越界或不是整数。"""


class NonFiniteError(MipError):
    """缩放或导数包含 NaN / inf。"""


def _real(value, what):
    """把单个导数值转换为有限浮点数，非有限值单独分类报错。"""
    kind = type(value)
    if kind is float:
        if not _isfinite(value):
            raise NonFiniteError("%s: non-finite value %r" % (what, value))
        return value
    if kind is int:
        return float(value)
    raise InvalidScaleError(
        "%s: expected a real number, got %s" % (what, kind.__name__)
    )


def _positive(value, what):
    """单轴缩放：必须为有限正实数。"""
    result = _real(value, what)
    if result <= 0.0:
        raise InvalidScaleError(
            "%s: scale must be > 0, got %r" % (what, value)
        )
    return result


def _dimension(value, what):
    """纹理宽高：正整数。"""
    kind = type(value)
    if kind is float and not _isfinite(value):
        raise NonFiniteError("%s: non-finite size %r" % (what, value))
    if kind is not int or value <= 0:
        raise InvalidSizeError(
            "%s: expected a positive integer, got %r" % (what, value)
        )
    return value


class TexMip:
    """mip 级别选择器。

    ``TexMip(w, h)`` 描述一张 ``w x h`` 的纹理；不带参数时等价于
    ``TexMip(1, 1)``（只有 1 级）。构造后实例完全只读。
    """

    __slots__ = ("_w", "_h", "_levels", "_sizes")

    def __init__(self, w=1, h=1):
        width = _dimension(w, "w")
        height = _dimension(h, "h")
        # floor(log2(max(w, h))) + 1：frexp 给出的指数恰好等于该值。
        levels = _frexp(float(max(width, height)))[1]
        if levels < 1:
            levels = 1
        self._w = width
        self._h = height
        self._levels = levels
        self._sizes = tuple(
            (max(1, width >> level), max(1, height >> level))
            for level in range(levels)
        )

    # ------------------------------------------------------------ 只读属性

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    width = w
    height = h

    @property
    def levels(self):
        """级数：合法级索引为 ``0 .. levels - 1``。"""
        return self._levels

    @property
    def max_level(self):
        return self._levels - 1

    def __repr__(self):
        return "TexMip(%d, %d)" % (self._w, self._h)

    # ------------------------------------------------------------ 足迹解析

    def _stretch(self, scale):
        """把缩放输入解析为 ``(最大拉伸, 最小拉伸)``。

        标量与长度为 2 的序列给的是逐轴纹素/像素比；长度为 4 的序列给的是
        纹理坐标导数 ``(dudx, dvdx, dudy, dvdy)``，需按纹理尺寸换算。
        """
        kind = type(scale)
        if kind is tuple or kind is list:
            count = len(scale)
            if count == 2:
                sx = _positive(scale[0], "scale[0]")
                sy = _positive(scale[1], "scale[1]")
                return (sx, sy) if sx >= sy else (sy, sx)
            if count == 4:
                dudx = _real(scale[0], "dudx")
                dvdx = _real(scale[1], "dvdx")
                dudy = _real(scale[2], "dudy")
                dvdy = _real(scale[3], "dvdy")
                # 两轴在纹理空间的边缘向量（单位：纹素）。
                axis_x = _hypot(self._w * dudx, self._h * dvdx)
                axis_y = _hypot(self._w * dudy, self._h * dvdy)
                return (
                    (axis_x, axis_y) if axis_x >= axis_y
                    else (axis_y, axis_x)
                )
            raise InvalidScaleError(
                "scale sequence must have 2 (sx, sy) or 4 "
                "(dudx, dvdx, dudy, dvdy) components, got %d" % count
            )
        if kind is float:
            if not _isfinite(scale):
                raise NonFiniteError("scale: non-finite value %r" % (scale,))
            if scale <= 0.0:
                raise InvalidScaleError(
                    "scale must be > 0, got %r" % (scale,)
                )
            return scale, scale
        if kind is int:
            if scale <= 0:
                raise InvalidScaleError(
                    "scale must be > 0, got %r" % (scale,)
                )
            value = float(scale)
            return value, value
        raise InvalidScaleError(
            "scale must be a number, (sx, sy) or 4 derivatives, got %s"
            % kind.__name__
        )

    # ------------------------------------------------------------ 级别查询

    def lod(self, scale):
        """连续级别 ``log2(max stretch)``，不做钳制。

        全零导数（无足迹）返回 ``-inf``。
        """
        largest, _ = self._stretch(scale)
        if largest <= 0.0:
            return -_INF
        return _log2(largest)

    def level(self, scale):
        """基础 mip 级 ``clamp(floor(log2(max stretch)), 0, levels - 1)``。

        结果与逐级下采样暴力比较一致（见模块文档），返回 ``int``。
        """
        kind = type(scale)
        if kind is float:
            if not _isfinite(scale):
                raise NonFiniteError("scale: non-finite value %r" % (scale,))
            if scale <= 0.0:
                raise InvalidScaleError(
                    "scale must be > 0, got %r" % (scale,)
                )
            if scale <= 1.0:
                return 0
            level = _frexp(scale)[1] - 1
        elif kind is int:
            if scale <= 0:
                raise InvalidScaleError(
                    "scale must be > 0, got %r" % (scale,)
                )
            # 对正整数 bit_length - 1 恰为 floor(log2)，且不溢出。
            level = scale.bit_length() - 1
        else:
            largest, _ = self._stretch(scale)
            if largest <= 1.0:
                return 0
            level = _frexp(largest)[1] - 1
        if level >= self._levels:
            return self._levels - 1
        return level

    def select(self, scale):
        """三线性选择：返回 ``(level0, level1, weight0, weight1)``。

        两级索引都在 ``[0, levels)`` 内，权重和为 1；若落在最高级边界，
        ``level1 == level0`` 且权重为 ``(1.0, 0.0)``。
        """
        largest, _ = self._stretch(scale)
        return self._select_from_stretch(largest)

    def trilinear(self, scale):
        """:meth:`select` 的别名。"""
        return self.select(scale)

    def _select_from_stretch(self, largest):
        count = self._levels
        if largest <= 1.0:
            # 足迹不超过一个纹素（含全零导数）：锁定 0 级。
            return (0, 1 if count > 1 else 0, 1.0, 0.0)
        level0 = _frexp(largest)[1] - 1
        if level0 < 0:
            return (0, 1 if count > 1 else 0, 1.0, 0.0)
        if level0 >= count:
            last = count - 1
            return (last, last, 1.0, 0.0)
        if level0 == count - 1:
            return (level0, level0, 1.0, 0.0)
        frac = _log2(largest) - level0
        if frac < 0.0:
            frac = 0.0
        elif frac >= 1.0:
            frac = _ONE_MINUS_ULP
        return (level0, level0 + 1, 1.0 - frac, frac)

    def levels_for(self, scale):
        """三线性涉及的两级 ``(level0, level1)``。"""
        level0, level1, _, _ = self.select(scale)
        return level0, level1

    def weights(self, scale):
        """三线性权重 ``(weight0, weight1)``，和为 1。"""
        _, _, weight0, weight1 = self.select(scale)
        return weight0, weight1

    def anisotropy(self, scale):
        """各向异性比 ``max stretch / min stretch``，恒 ``>= 1``。

        最小拉伸为 0（某一轴导数为 0）时返回 ``inf``。
        """
        largest, smallest = self._stretch(scale)
        if smallest <= 0.0:
            return _INF
        ratio = largest / smallest
        return ratio if ratio >= 1.0 else 1.0

    # ------------------------------------------------------------ 导数入口

    def level_from_derivatives(self, dudx, dvdx, dudy, dvdy):
        """由 ``dFdx(uv)`` / ``dFdy(uv)`` 求基础级别。"""
        return self.level((dudx, dvdx, dudy, dvdy))

    def select_from_derivatives(self, dudx, dvdx, dudy, dvdy):
        """由纹理坐标导数求三线性选择结果。"""
        return self.select((dudx, dvdx, dudy, dvdy))

    # ------------------------------------------------------------ 尺寸查询

    def size_at(self, level):
        """第 ``level`` 级的尺寸 ``(w, h)``；每轴至少 1 纹素。"""
        kind = type(level)
        if kind is float:
            if not _isfinite(level):
                raise NonFiniteError("level: non-finite value %r" % (level,))
            if not level.is_integer():
                raise InvalidLevelError(
                    "level must be an integer, got %r" % (level,)
                )
            level = int(level)
        elif kind is not int:
            raise InvalidLevelError(
                "level must be an integer, got %s" % kind.__name__
            )
        if level < 0 or level >= self._levels:
            raise InvalidLevelError(
                "level must be in [0, %d), got %d"
                % (self._levels, level)
            )
        return self._sizes[level]

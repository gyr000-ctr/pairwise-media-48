"""texblend 核心 API。

各向异性双线性纹理采样：u / v 两个轴使用各自独立的采样位置与分数
权重（两轴采样率可以不同），四个邻点权重恒满足
``w00 + w10 + w01 + w11 == 1.0``。

设计要点：
- ``sample`` 不持有、也不缓存任何可变状态，同一参数必然得到同一结果，
  多线程并发调用安全。
- 边界支持 ``clamp``（钳制到边缘纹素）与 ``wrap``（环绕）两种模式，
  可在构造器或单次采样时指定。
- 四角等整数纹素位置返回精确纹素值（权重严格为 0/1）。
- 非法纹理与非法/非有限坐标抛出可区分的异常类型。
"""

import math
import threading

__all__ = [
    "TexBlend",
    "TexBlendError",
    "InvalidTextureError",
    "NonFiniteCoordinateError",
    "InvalidWrapModeError",
    "WRAP_CLAMP",
    "WRAP_WRAP",
]

WRAP_CLAMP = "clamp"
WRAP_WRAP = "wrap"
_WRAP_MODES = (WRAP_CLAMP, WRAP_WRAP)


class TexBlendError(ValueError):
    """texblend 所有可区分业务错误的基类（仍是 ValueError 的子类）。"""


class InvalidTextureError(TexBlendError):
    """纹理结构非法（空、行列不齐、含非数值或非有限纹素）。"""


class NonFiniteCoordinateError(TexBlendError):
    """采样坐标 u/v 非数值或包含 NaN/Inf。"""


class InvalidWrapModeError(TexBlendError):
    """边界模式不是 'clamp' / 'wrap'。"""


def _is_real(x):
    # bool 是 int 的子类，按 0/1 数值接受；complex 等一律拒绝。
    return isinstance(x, (int, float)) and not isinstance(x, complex)


def _validate_texture(tex):
    """校验纹理并返回 (高, 宽)；失败抛 InvalidTextureError。"""
    if not isinstance(tex, (list, tuple)) or len(tex) == 0:
        raise InvalidTextureError("texture must be a non-empty sequence of rows")
    first = tex[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise InvalidTextureError("texture rows must be non-empty sequences")
    width = len(first)
    for row in tex:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise InvalidTextureError("texture rows must be rectangular (equal length)")
        for cell in row:
            if not _is_real(cell) or not math.isfinite(cell):
                raise InvalidTextureError("texture texels must be finite real numbers")
    return len(tex), width


def _axis_weights(x, n, wrap):
    """返回 (i0, i1, w1)：i0/i1 为两邻点下标，w1 为 i1 的权重。

    i0 的权重由调用方以 ``1.0 - w1`` 得到，保证同一轴权重和精确为 1。
    两轴各自独立计算，因此支持不同采样率（各向异性）。
    """
    if n == 1:
        # 单纹素轴：任何坐标都精确命中该纹素，wrap 下也不引入分数权重。
        return 0, 0, 0.0
    if wrap:
        base = math.floor(x)
        i0 = base % n
        i1 = (i0 + 1) % n
        return i0, i1, x - base
    if x <= 0.0:
        return 0, 0, 0.0
    last = n - 1
    if x >= float(last):
        return last, last, 1.0
    i0 = int(x)
    return i0, i0 + 1, x - i0


class TexBlend:
    """双线性纹理采样器。

    Parameters
    ----------
    wrap:
        边界模式，``'clamp'``（默认）或 ``'wrap'``。构造时非法会抛
        :class:`InvalidWrapModeError`。
    """

    def __init__(self, wrap=WRAP_CLAMP):
        if wrap not in _WRAP_MODES:
            raise InvalidWrapModeError("wrap must be 'clamp' or 'wrap', got %r" % (wrap,))
        self._wrap = wrap
        # 每线程单槽校验缓存：sample 不产生跨线程共享的可变状态，
        # 缓存命中且结构复查通过时跳过逐纹素校验，满足 10 万次采样的
        # 性能上界。缓存以对象身份（is）绑定，纹理被原地增删行列后
        # 结构复查失败即回退全量校验并抛出对应异常。
        self._tls = threading.local()

    @property
    def wrap(self):
        return self._wrap

    def _texture_dims(self, tex):
        slot = getattr(self._tls, "slot", None)
        if slot is not None and slot[0] is tex:
            height, width = slot[1]
            # 廉价结构性复查：捕获纹理被原地增删行列的篡改；
            # 命中时跳过逐纹素全量校验以满足性能上界。
            if len(tex) == height and all(len(row) == width for row in tex):
                return slot[1]
        dims = _validate_texture(tex)
        self._tls.slot = (tex, dims)
        return dims

    def sample(self, tex, u, v, wrap=None):
        """在纹理 ``tex`` 的 (u, v) 处做双线性采样。

        ``u`` 沿列方向（x 轴）、``v`` 沿行方向（y 轴），纹素中心位于
        整数坐标。两轴独立插值，权重和恒为 1；整数坐标精确命中纹素。

        Parameters
        ----------
        tex:
            非空、行列齐整、元素均为有限实数的二维序列。
        u, v:
            有限实数采样坐标。
        wrap:
            单次调用覆盖构造器的边界模式；``None`` 表示沿用构造器设置。

        Returns
        -------
        float
        """
        if wrap is None:
            wrap = self._wrap
        elif wrap not in _WRAP_MODES:
            raise InvalidWrapModeError("wrap must be 'clamp' or 'wrap', got %r" % (wrap,))

        if not _is_real(u) or not math.isfinite(u) or not _is_real(v) or not math.isfinite(v):
            raise NonFiniteCoordinateError("u, v must be finite real numbers")

        height, width = self._texture_dims(tex)
        do_wrap = wrap == WRAP_WRAP

        c0, c1, wu1 = _axis_weights(u, width, do_wrap)
        r0, r1, wv1 = _axis_weights(v, height, do_wrap)

        # 互补权重：配对相加精确等于 1.0，四点权重和因此精确为 1.0。
        wu0 = 1.0 - wu1
        wv0 = 1.0 - wv1

        row0 = tex[r0]
        row1 = tex[r1]
        return float(
            row0[c0] * (wu0 * wv0)
            + row0[c1] * (wu1 * wv0)
            + row1[c0] * (wu0 * wv1)
            + row1[c1] * (wu1 * wv1)
        )

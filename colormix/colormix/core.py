"""colormix —— 预乘 alpha 合成与 W3C 混合模式。

公开 API
--------
``ColorMix().mix(a, b, mode='multiply', t=1.0)``
    把源色 ``b`` 以 ``t``（源不透明度，乘在 ``b`` 的 alpha 上）叠加到
    背景色 ``a`` 上，返回反预乘后的颜色元组。

颜色可以是 3 元组 ``(r, g, b)``（alpha 视为 1.0）或 4 元组
``(r, g, b, a)``。返回值的长度与输入一致。

合成在预乘空间按 W3C compositing 的 source-over 公式进行：

    αo = αs + αb·(1 − αs)
    Co = [αs·(1 − αb)·Cs + αs·αb·B(Cb, Cs) + (1 − αs)·αb·Cb] / αo

其中 ``B`` 为混合模式函数。αo = 0 时结果为 0，不会出现 NaN。

支持的 ``mode``：``multiply``、``screen``、``overlay``、``soft-light``。

错误类型（全部继承 ``ValueError``，可按类型区分）：
``InvalidModeError``、``ColorRangeError``、``LengthMismatchError``、
``NonFiniteInputError``。

所有函数都是无状态的，可在多线程中并发调用。
"""

import math

__all__ = [
    'ColorMix',
    'ColorMixError',
    'InvalidModeError',
    'ColorRangeError',
    'LengthMismatchError',
    'NonFiniteInputError',
    'blend',
    'blend_channel',
    'premultiply',
    'unpremultiply',
    'composite',
    'mix',
    'mix_pixels',
    'mix_many',
    'to_8bit',
    'from_8bit',
    'quantize_8bit',
]

_MODES = ('multiply', 'screen', 'overlay', 'soft-light')
_MODE_INDEX = {name: i for i, name in enumerate(_MODES)}

_isfinite = math.isfinite
_sqrt = math.sqrt


# --------------------------------------------------------------------------
# 错误类型
# --------------------------------------------------------------------------
class ColorMixError(ValueError):
    """colormix 所有可预期错误的基类。"""


class InvalidModeError(ColorMixError):
    """未知的混合模式。"""


class ColorRangeError(ColorMixError):
    """颜色通道或 alpha / t 超出 [0, 1]。"""


class LengthMismatchError(ColorMixError):
    """颜色不是 3/4 元组，或两色长度不一致。"""


class NonFiniteInputError(ColorMixError):
    """输入包含 NaN 或 ±inf。"""


# --------------------------------------------------------------------------
# 校验与拆包
# --------------------------------------------------------------------------
def _unpack(color, label):
    """校验并拆分为 (r, g, b, alpha, 是否有 alpha)。"""
    try:
        n = len(color)
    except TypeError:
        raise LengthMismatchError(
            '%s must be a 3- or 4-element sequence' % label) from None
    if n != 3 and n != 4:
        raise LengthMismatchError(
            '%s must have 3 or 4 components, got %d' % (label, n))
    r = color[0]
    g = color[1]
    b = color[2]
    a = color[3] if n == 4 else 1.0
    for v in (r, g, b, a):
        _check_number(v, label)
    for v in (r, g, b, a):
        if v < 0.0 or v > 1.0:
            raise ColorRangeError(
                '%s component out of range [0, 1]: %r' % (label, v))
    return float(r), float(g), float(b), float(a), n == 4


def _check_number(v, label):
    """类型不是实数 -> NonFiniteInputError；NaN/inf 同样。"""
    try:
        finite = _isfinite(v)
    except TypeError:
        raise NonFiniteInputError(
            '%s contains a non-numeric component: %r' % (label, v)) from None
    if not finite:
        raise NonFiniteInputError(
            '%s contains a non-finite component: %r' % (label, v))


def _check_mode(mode):
    try:
        return _MODE_INDEX[mode]
    except (KeyError, TypeError):
        raise InvalidModeError(
            'unknown blend mode %r; expected one of %s'
            % (mode, ', '.join(_MODES))) from None


def _check_t(t):
    _check_number(t, 't')
    if t < 0.0 or t > 1.0:
        raise ColorRangeError('t must be in [0, 1], got %r' % (t,))
    return float(t)


# --------------------------------------------------------------------------
# W3C 混合模式（逐通道，定义域 [0, 1]）
# --------------------------------------------------------------------------
def blend_channel(cb, cs, mode='multiply'):
    """单通道混合函数 B(Cb, Cs)，不做 alpha 合成。"""
    m = _check_mode(mode)
    if m == 0:                      # multiply
        return cb * cs
    if m == 1:                      # screen
        return cb + cs - cb * cs
    if m == 2:                      # overlay: 按背景 Cb 分段
        if cb <= 0.5:
            return 2.0 * cb * cs
        return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)
    return _soft_light(cb, cs)      # soft-light


def _soft_light(cb, cs):
    """W3C soft-light：按源 Cs 分两支，D(Cb) 再按 Cb 分段。"""
    if cs <= 0.5:
        return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    if cb <= 0.25:
        d = ((16.0 * cb - 12.0) * cb + 4.0) * cb
    else:
        d = _sqrt(cb)
    return cb + (2.0 * cs - 1.0) * (d - cb)


def blend(backdrop, source, mode='multiply'):
    """不做 alpha 合成的逐通道混合，返回与输入等长的元组。"""
    rb, gb, bb, ab, has_a = _unpack(backdrop, 'backdrop')
    rs, gs, bs, as_, _ = _unpack(source, 'source')
    m = _check_mode(mode)
    if m == 0:
        out = (rb * rs, gb * gs, bb * bs)
    elif m == 1:
        out = (rb + rs - rb * rs, gb + gs - gb * gs, bb + bs - bb * bs)
    elif m == 2:
        out = (_overlay(rb, rs), _overlay(gb, gs), _overlay(bb, bs))
    else:
        out = (_soft_light(rb, rs), _soft_light(gb, gs), _soft_light(bb, bs))
    if has_a:
        return out + (ab,)
    return out


def _overlay(cb, cs):
    if cb <= 0.5:
        return 2.0 * cb * cs
    return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)


# --------------------------------------------------------------------------
# 预乘 / 反预乘
# --------------------------------------------------------------------------
def premultiply(color):
    """非预乘颜色 -> 预乘颜色（alpha 保持不变）。"""
    r, g, b, a, has_a = _unpack(color, 'color')
    return (r * a, g * a, b * a, a)


def unpremultiply(color):
    """预乘颜色 -> 非预乘颜色；alpha 为 0 时返回全 0 而不是 NaN。"""
    try:
        n = len(color)
    except TypeError:
        raise LengthMismatchError(
            'color must be a 3- or 4-element sequence') from None
    if n != 3 and n != 4:
        raise LengthMismatchError(
            'color must have 3 or 4 components, got %d' % n)
    r = color[0]
    g = color[1]
    b = color[2]
    a = color[3] if n == 4 else 1.0
    for v in (r, g, b, a):
        _check_number(v, 'color')
    a = float(a)
    if a == 0.0:
        return (0.0, 0.0, 0.0, 0.0) if n == 4 else (0.0, 0.0, 0.0)
    r = float(r) / a
    g = float(g) / a
    b = float(b) / a
    return (r, g, b, a) if n == 4 else (r, g, b)


# --------------------------------------------------------------------------
# 合成
# --------------------------------------------------------------------------
def mix(a, b, mode='multiply', t=1.0):
    """把源色 ``b``（不透明度系数 ``t``）以 ``mode`` 合成到背景 ``a`` 上。

    返回反预乘后的颜色，长度与输入一致。
    """
    m = _check_mode(mode)
    rb, gb, bb, ab, has_a = _unpack(a, 'backdrop')
    rs, gs, bs, as_in, _ = _unpack(b, 'source')
    as_ = _check_t(t) * as_in

    # 不透明快路径：两条 alpha 都为 1 时 Co = B(Cb, Cs)。
    if ab == 1.0 and as_ == 1.0:
        if m == 0:
            out = (rb * rs, gb * gs, bb * bs)
        elif m == 1:
            out = (rb + rs - rb * rs, gb + gs - gb * gs, bb + bs - bb * bs)
        elif m == 2:
            out = (_overlay(rb, rs), _overlay(gb, gs), _overlay(bb, bs))
        else:
            out = (_soft_light(rb, rs), _soft_light(gb, gs), _soft_light(bb, bs))
        return out + (1.0,) if has_a else out

    co_a = as_ + ab * (1.0 - as_)
    if co_a == 0.0:
        return (0.0, 0.0, 0.0, 0.0) if has_a else (0.0, 0.0, 0.0)

    # 预乘空间 source-over：co = αs(1-αb)Cs + αs·αb·B + (1-αs)αb·Cb
    wa = as_ * (1.0 - ab)     # 源色直接贡献
    wBl = as_ * ab            # 混合结果贡献
    wb = (1.0 - as_) * ab     # 背景贡献
    inv = 1.0 / co_a

    if m == 0:
        p0 = wa * rs + wBl * (rb * rs) + wb * rb
        p1 = wa * gs + wBl * (gb * gs) + wb * gb
        p2 = wa * bs + wBl * (bb * bs) + wb * bb
    elif m == 1:
        p0 = wa * rs + wBl * (rb + rs - rb * rs) + wb * rb
        p1 = wa * gs + wBl * (gb + gs - gb * gs) + wb * gb
        p2 = wa * bs + wBl * (bb + bs - bb * bs) + wb * bb
    elif m == 2:
        p0 = wa * rs + wBl * _overlay(rb, rs) + wb * rb
        p1 = wa * gs + wBl * _overlay(gb, gs) + wb * gb
        p2 = wa * bs + wBl * _overlay(bb, bs) + wb * bb
    else:
        p0 = wa * rs + wBl * _soft_light(rb, rs) + wb * rb
        p1 = wa * gs + wBl * _soft_light(gb, gs) + wb * gb
        p2 = wa * bs + wBl * _soft_light(bb, bs) + wb * bb

    out = (p0 * inv, p1 * inv, p2 * inv)
    return out + (co_a,) if has_a else out


def composite(backdrop, source, mode='multiply', t=1.0):
    """``mix`` 的别名：预乘空间 source-over 合成。"""
    return mix(backdrop, source, mode, t)


def mix_pixels(backdrops, sources, mode='multiply', t=1.0):
    """对成批像素逐项做 ``mix``，返回元组列表。"""
    try:
        la = len(backdrops)
        lb = len(sources)
    except TypeError:
        raise LengthMismatchError('pixel collections must be sequences') from None
    if la != lb:
        raise LengthMismatchError(
            'pixel collections differ in length: %d != %d' % (la, lb))
    m = _check_mode(mode)
    return [mix(backdrops[i], sources[i], _MODES[m], t) for i in range(la)]


mix_many = mix_pixels


# --------------------------------------------------------------------------
# 8 位量化（往返误差 ≤ 0.5/255 < 1/255）
# --------------------------------------------------------------------------
def to_8bit(color):
    """把 [0, 1] 颜色量化为 8 位整数元组。"""
    try:
        n = len(color)
    except TypeError:
        raise LengthMismatchError(
            'color must be a 3- or 4-element sequence') from None
    if n != 3 and n != 4:
        raise LengthMismatchError(
            'color must have 3 or 4 components, got %d' % n)
    values = []
    for v in color:
        if not _isfinite(v):
            raise NonFiniteInputError(
                'color contains a non-finite component: %r' % (v,))
        if v < 0.0 or v > 1.0:
            raise ColorRangeError(
                'color component out of range [0, 1]: %r' % (v,))
        values.append(int(round(float(v) * 255.0)))
    return tuple(values)


def from_8bit(color):
    """把 8 位整数颜色还原为 [0, 1] 浮点元组。"""
    try:
        n = len(color)
    except TypeError:
        raise LengthMismatchError(
            'color must be a 3- or 4-element sequence') from None
    if n != 3 and n != 4:
        raise LengthMismatchError(
            'color must have 3 or 4 components, got %d' % n)
    values = []
    for v in color:
        if isinstance(v, bool) or not isinstance(v, int):
            if not isinstance(v, float) or not v.is_integer():
                raise ColorRangeError(
                    '8-bit color must use integers, got %r' % (v,))
            v = int(v)
        if v < 0 or v > 255:
            raise ColorRangeError(
                '8-bit color component out of range [0, 255]: %r' % (v,))
        values.append(v / 255.0)
    return tuple(values)


quantize_8bit = to_8bit


# --------------------------------------------------------------------------
# 面向对象包装
# --------------------------------------------------------------------------
class ColorMix:
    """无状态的混色器；所有方法可安全地并发调用。"""

    __slots__ = ()

    def mix(self, a, b, mode='multiply', t=1.0):
        """混合（见模块级 :func:`mix`）。"""
        return mix(a, b, mode, t)

    def composite(self, backdrop, source, mode='multiply', t=1.0):
        return mix(backdrop, source, mode, t)

    def blend(self, backdrop, source, mode='multiply'):
        return blend(backdrop, source, mode)

    def blend_channel(self, cb, cs, mode='multiply'):
        return blend_channel(cb, cs, mode)

    def premultiply(self, color):
        return premultiply(color)

    def unpremultiply(self, color):
        return unpremultiply(color)

    def mix_pixels(self, backdrops, sources, mode='multiply', t=1.0):
        return mix_pixels(backdrops, sources, mode, t)

    def to_8bit(self, color):
        return to_8bit(color)

    def from_8bit(self, color):
        return from_8bit(color)

    def quantize_8bit(self, color):
        return to_8bit(color)

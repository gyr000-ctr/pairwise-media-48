"""hdrcurve 核心 API：单调三次色调曲线与 LUT 查表。

实现要点：

- 单调三次插值采用 Fritsch-Carlson 保单调 Hermite 方法，数学上保证
  插值函数不越过控制点值域（无过冲），且与控制点同序单调。
- gamma 曲线 y = t**gamma 在变换域 u = t**(1/q) 上采样（q 的选取使
  指数 p = gamma*q 落在 2~3 之间），配合均匀 LUT 线性插值，查表误差
  稳定 <= 1e-6，包括 gamma < 1（端点导数奇异）的情形。
- 控制点曲线通过递归二分自适应生成 LUT（容差 1e-7），查表 O(log n)。
- 所有状态在构造期一次性确定，之后只读，lookup 无惰性初始化、无写
  操作，多线程并发调用安全。
"""

import bisect
import math

# LUT 与解析曲线之间允许的最大误差。
_LUT_TOL = 1e-6
# 自适应 LUT 内部使用更紧的容差，给线性插值留余量。
_ADAPTIVE_TOL = 1e-7
# 自适应递归深度上限（2^-50 的域分辨率已远超双精度可用范围）。
_MAX_DEPTH = 50


def _sizes_for_power(p):
    """按 u**p 的曲率选择样条节点数 K 与 LUT 点数 N。

    保单调三次样条误差约 O(K^-4)，均匀 LUT 线性插值误差约 O(N^-2)，
    两者的常数均随 p 增大；实测标定后按 p 分档保证总误差 <= 1e-6。
    注意 p 接近 1 时在 u=0 边界的样条误差反而偏大，需单独处理。
    """
    if p < 1.6:
        return 4096, 32768
    if p <= 4.0:
        return 1024, 8192
    if p <= 9.0:
        return 2048, 16384
    if p <= 22.0:
        return 4096, 32768
    if p <= 55.0:
        return 8192, 65536
    # 极端 gamma：按实测误差标度（样条 O(K^-4)、线性 O(N^-2)）继续放大，
    # 封顶 262144 点（约 2 MB）。
    k = min(16384, 8192 * max(1, int(math.sqrt(p / 55.0))))
    n = min(262144, 65536 * max(1, int(math.sqrt(p / 55.0))))
    return k, n


def _finite_real(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


class _MonotoneCubic:
    """Fritsch-Carlson 保单调三次 Hermite 插值。"""

    __slots__ = ("xs", "ys", "ms", "hs")

    def __init__(self, xs, ys):
        n = len(xs)
        hs = [xs[i + 1] - xs[i] for i in range(n - 1)]
        secants = [(ys[i + 1] - ys[i]) / hs[i] for i in range(n - 1)]

        ms = [0.0] * n
        ms[0] = secants[0]
        ms[-1] = secants[-1]
        for i in range(1, n - 1):
            s0, s1 = secants[i - 1], secants[i]
            if s0 == 0.0 or s1 == 0.0 or s0 * s1 < 0.0:
                ms[i] = 0.0
            else:
                # 相邻弦线的加权调和平均式切线估计。
                ms[i] = (hs[i] * s0 + hs[i - 1] * s1) / (hs[i - 1] + hs[i])

        # Fritsch-Carlson 切线限幅，保证每个区间单调、无过冲。
        for k in range(n - 1):
            s = secants[k]
            if s == 0.0:
                ms[k] = 0.0
                ms[k + 1] = 0.0
                continue
            a = ms[k] / s
            b = ms[k + 1] / s
            total = a + b
            if total > 3.0:
                factor = 3.0 / total
                ms[k] = factor * a * s
                ms[k + 1] = factor * b * s

        self.xs = xs
        self.ys = ys
        self.ms = ms
        self.hs = hs

    def eval(self, x):
        xs, ys, ms, hs = self.xs, self.ys, self.ms, self.hs
        i = bisect.bisect_right(xs, x) - 1
        if i < 0:
            return ys[0]
        if i >= len(hs):
            return ys[-1]
        h = hs[i]
        t = (x - xs[i]) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return (h00 * ys[i] + h10 * h * ms[i]
                + h01 * ys[i + 1] + h11 * h * ms[i + 1])


def _adaptive_lut(spline, lo, hi, tol=_ADAPTIVE_TOL):
    """递归二分自适应采样，使每段线性化误差 <= tol。"""
    out_x = [lo]
    out_y = [spline.eval(lo)]

    def subdivide(a, ya, b, yb, depth):
        mid = 0.5 * (a + b)
        ym = spline.eval(mid)
        quarter = 0.25 * (3.0 * a + b)
        three_quarter = 0.25 * (a + 3.0 * b)
        yq = spline.eval(quarter)
        yr = spline.eval(three_quarter)
        lq = ya + 0.25 * (yb - ya)
        lm = 0.5 * (ya + yb)
        lr = ya + 0.75 * (yb - ya)
        err = max(abs(yq - lq), abs(ym - lm), abs(yr - lr))
        if err <= tol or depth >= _MAX_DEPTH:
            return
        subdivide(a, ya, mid, ym, depth + 1)
        out_x.append(mid)
        out_y.append(ym)
        subdivide(mid, ym, b, yb, depth + 1)

    subdivide(lo, out_y[0], hi, spline.eval(hi), 0)
    out_x.append(hi)
    out_y.append(spline.eval(hi))
    return out_x, out_y


class ToneCurve:
    """色调曲线。

    两种构造方式：

    - ``ToneCurve(gamma, levels=256)``：解析幂曲线 y = t**gamma。
    - ``ToneCurve.from_points(xs, ys, levels=256)``：通过单调控制点
      构造，内部使用保单调三次插值；控制点非单调时抛出 ValueError。
    """

    def __init__(self, gamma, levels=256):
        if not _finite_real(gamma) or gamma <= 0.0:
            raise ValueError("gamma 必须为正的有限实数")
        if not isinstance(levels, int) or isinstance(levels, bool) \
                or levels < 2:
            raise ValueError("levels 必须为不小于 2 的整数")

        self.gamma = float(gamma)
        self.levels = levels
        self._is_gamma = True

        g = float(gamma)
        # 选取 q 使 p = g*q 落在 [2, 3)（g>=1 时直接 q=1），
        # 消除 gamma<1 时 t=0 处的导数奇异性。
        if g >= 1.0:
            q = 1.0
        else:
            q = float(math.ceil(2.0 / g))
        p = g * q
        self._q = q
        self._inv_q = 1.0 / q

        # 变换域 u 上的保单调三次样条，采样解析值 u**p。
        knots_n, lut_n = _sizes_for_power(p)
        knots = [i / (knots_n - 1) for i in range(knots_n)]
        values = [u ** p for u in knots]
        spline = _MonotoneCubic(knots, values)

        lut = [spline.eval(i / (lut_n - 1)) for i in range(lut_n)]
        lut[0] = 0.0
        lut[-1] = 1.0
        self._lut = tuple(lut)
        self._lut_scale = float(lut_n - 1)
        self._lut_n = lut_n
        # 控制点路径专用字段。
        self._lut_xs = None
        self._lut_ys = None

    @classmethod
    def from_points(cls, xs, ys, levels=256):
        """由单调控制点构造色调曲线（保单调三次插值）。"""
        try:
            px = [float(v) for v in xs]
            py = [float(v) for v in ys]
        except (TypeError, ValueError):
            raise ValueError("xs、ys 必须为实数序列")
        if len(px) != len(py) or len(px) < 2:
            raise ValueError("xs、ys 长度一致且至少包含 2 个控制点")
        if not all(math.isfinite(v) for v in px) or \
                not all(math.isfinite(v) for v in py):
            raise ValueError("控制点必须全部为有限实数")
        if any(px[i + 1] <= px[i] for i in range(len(px) - 1)):
            raise ValueError("xs 必须严格递增")
        if any(py[i + 1] < py[i] for i in range(len(py) - 1)):
            raise ValueError("ys 必须单调不减：检测到非单调控制点")
        if not isinstance(levels, int) or isinstance(levels, bool) \
                or levels < 2:
            raise ValueError("levels 必须为不小于 2 的整数")

        obj = cls.__new__(cls)
        obj.levels = levels
        obj.gamma = None
        obj._is_gamma = False
        obj._q = 1.0
        obj._inv_q = 1.0
        spline = _MonotoneCubic(px, py)
        lut_x, lut_y = _adaptive_lut(spline, px[0], px[-1])
        obj._lut_xs = tuple(lut_x)
        obj._lut_ys = tuple(lut_y)
        obj._lut = None
        obj._lut_scale = 0.0
        obj._lut_n = 0
        return obj

    def lookup(self, t):
        """查表映射，输入越界时钳制到端点，非有限输入抛出 ValueError。"""
        if not isinstance(t, (int, float)) or isinstance(t, bool) \
                or not math.isfinite(t):
            raise ValueError("lookup 输入必须为有限实数")

        if self._is_gamma:
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            lut = self._lut
            if self._q == 1.0:
                u = t
            else:
                u = math.pow(t, self._inv_q)
            pos = u * self._lut_scale
            i = int(pos)
            if i >= self._lut_n - 1:
                return lut[-1]
            frac = pos - i
            return lut[i] + (lut[i + 1] - lut[i]) * frac

        lut_x = self._lut_xs
        lut_y = self._lut_ys
        if t <= lut_x[0]:
            return float(lut_y[0])
        if t >= lut_x[-1]:
            return float(lut_y[-1])
        i = bisect.bisect_right(lut_x, t) - 1
        x0, x1 = lut_x[i], lut_x[i + 1]
        frac = (t - x0) / (x1 - x0)
        return lut_y[i] + (lut_y[i + 1] - lut_y[i]) * frac

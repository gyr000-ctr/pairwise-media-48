"""一阶滤波器：双线性变换 (bilinear transform) 的 biquad 实现。

数学推导
--------
对一阶原型 ``H_a(s) = ω_c / (s + ω_c)``（低通）做双线性变换
``s = 2·fs·(z-1)/(z+1)``（预畸变 ``ω_c = 2·fs·tan(π·fc/fs)``），整理后：

    低通  H(z) = (b0 + b1·z⁻¹) / (1 + a1·z⁻¹)
    高通  H(z) = (b0 + b1·z⁻¹) / (1 + a1·z⁻¹)，对 ``s -> 1/s`` 互换

其中令 ``α = tan(π·fc/fs)``（即 ``tan(ω_c/2)``，已含预畸变）：

    低通  b0 = α/(1+α),  b1 = b0,        a1 = (α-1)/(1+α)
    高通  b0 = 1/(1+α),  b1 = -b0,       a1 = (α-1)/(1+α)

它是标准 biquad（二阶节）在 ``b2 = a2 = 0`` 时的特例，采用转置直接 II 型
（Transposed Direct Form II）结构，每个样本仅需一次递归：

    y  = b0·x + z1
    z1 = b1·x - a1·y            （二阶时另含 z2，此处恒为 0）

低通与高通共用同一个分母 ``a1``，且系数满足 ``H_LP(z) + H_HP(z) ≡ 1``，
因此二者对任意信号严格互补（allpass 分解）。
"""

import math
import threading

__all__ = ["OnePole", "FilterParameterError", "NonFiniteInputError"]


class FilterParameterError(ValueError):
    """非法的采样率 / 截止频率参数（ValueError 的子类，保持既有报错语义）。"""


class NonFiniteInputError(ValueError):
    """输入样本中出现 NaN / Inf 等非有限值。"""


# 截止频率变化时的系数渐变时长（秒），用于消除参数变化引起的爆音/拉链噪声。
_RAMP_SECONDS = 0.01


class OnePole:
    """双线性变换一阶 biquad 滤波器（低通 / 高通互补对）。

    Parameters
    ----------
    rate : float
        采样率（Hz），必须为有限且 > 0 的实数。
    cutoff : float
        截止频率（Hz），必须为有限且 ``0 < cutoff < rate/2``。
    lowpass : bool, 可选
        True 为低通（默认），False 为高通。
    """

    def __init__(self, rate, cutoff, lowpass=True):
        self._lock = threading.RLock()
        # 校验并归一化参数；非法参数抛出可区分的 FilterParameterError。
        rate = self._check_rate(rate)
        cutoff = self._check_cutoff(cutoff, rate)
        self._rate = rate
        self._lowpass = bool(lowpass)
        self._target_cutoff = cutoff

        b0, b1, a1 = _design(rate, cutoff, self._lowpass)
        # 当前生效系数（biquad 五系数；b2/a2 恒为 0，见模块 docstring）。
        self._b0, self._b1, self._a1 = b0, b1, a1
        # 转置直接 II 型的内部状态，跨 process() 调用持续保持。
        self._z1 = 0.0
        # 参数渐变（ramp）目标与剩余样本数；0 表示参数恒定、走最快的热路径。
        self._ramp_remaining = 0
        self._ramp_steps = 0
        self._ramp_from = None
        self._ramp_to = None

    # ------------------------------------------------------------------ #
    # 参数校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_rate(rate):
        if not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise FilterParameterError(
                "sample rate must be a finite positive number, got %r" % (rate,)
            )
        rate = float(rate)
        if rate <= 0.0:
            raise FilterParameterError(
                "sample rate must be > 0, got %r" % (rate,)
            )
        return rate

    @staticmethod
    def _check_cutoff(cutoff, rate):
        if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff):
            raise FilterParameterError(
                "cutoff must be a finite positive number, got %r" % (cutoff,)
            )
        cutoff = float(cutoff)
        if cutoff <= 0.0 or cutoff >= rate / 2.0:
            raise FilterParameterError(
                "cutoff must satisfy 0 < cutoff < rate/2 (rate=%r), got %r"
                % (rate, cutoff)
            )
        return cutoff

    # ------------------------------------------------------------------ #
    # 公共属性
    # ------------------------------------------------------------------ #
    @property
    def rate(self):
        return self._rate

    @property
    def cutoff(self):
        """当前的截止频率目标（Hz）。"""
        if self._ramp_remaining:
            return self._ramp_to[3]
        return self._target_cutoff

    @property
    def lowpass(self):
        return self._lowpass

    @property
    def coefficients(self):
        """当前 biquad 系数 ``(b0, b1, b2, a1, a2)``（b2、a2 恒为 0）。"""
        return (self._b0, self._b1, 0.0, self._a1, 0.0)

    # ------------------------------------------------------------------ #
    # 参数动态变化（无爆音）
    # ------------------------------------------------------------------ #
    def set_cutoff(self, cutoff, lowpass=None):
        """在约 10 ms 内平滑切换截止频率（及可选的滤波类型）。

        系数逐样本线性渐变到新目标，避免系数突变造成的瞬态爆音；
        渐变过程中内部状态 ``z1`` 保持连续。返回 self 以便链式调用。
        """
        with self._lock:
            rate = self._rate
            cutoff = self._check_cutoff(cutoff, rate)
            if lowpass is not None:
                self._lowpass = bool(lowpass)
            steps = max(1, int(round(rate * _RAMP_SECONDS)))
            self._ramp_steps = steps
            self._ramp_remaining = steps
            # (b0, b1, a1, cutoff) 起止点；a1 与类型无关，渐变仍保持平滑。
            self._ramp_from = (self._b0, self._b1, self._a1, self._target_cutoff)
            tb0, tb1, ta1 = _design(rate, cutoff, self._lowpass)
            self._ramp_to = (tb0, tb1, ta1, cutoff)
            self._target_cutoff = cutoff
            return self

    # ------------------------------------------------------------------ #
    # 核心处理
    # ------------------------------------------------------------------ #
    def process(self, samples):
        """对一个缓冲区的样本做滤波，返回等长的 ``list[float]``。

        - 内部状态跨多次调用连续（可把长信号拆成任意大小的缓冲区）。
        - 不修改入参，不保留对入参的引用。
        - 出现 NaN/Inf 时抛出 :class:`NonFiniteInputError`，且此前状态不变。
        - 线程安全：同一实例可被多线程串行调用。
        """
        with self._lock:
            isfinite = math.isfinite
            out = []
            append = out.append
            z1 = self._z1
            # 快照：若输入含非有限值则整体回滚，保证失败不改变任何既有状态。
            snapshot = (
                self._z1, self._b0, self._b1, self._a1,
                self._ramp_remaining, self._ramp_steps,
                self._ramp_from, self._ramp_to,
            )

            if self._ramp_remaining:
                # ---- 参数渐变路径：系数逐样本线性插值，杜绝爆音 ----
                b0, b1, a1 = self._b0, self._b1, self._a1
                f0, f1, fa = self._ramp_from[:3]
                t0, t1, ta = self._ramp_to[:3]
                total = self._ramp_steps
                remaining = self._ramp_remaining
                done = total - remaining
                for x in samples:
                    if not isfinite(x):
                        self._restore(snapshot)
                        raise NonFiniteInputError(
                            "non-finite sample (NaN/Inf) in input"
                        )
                    if remaining:
                        frac = (done + 1) / total
                        b0 = f0 + (t0 - f0) * frac
                        b1 = f1 + (t1 - f1) * frac
                        a1 = fa + (ta - fa) * frac
                        remaining -= 1
                        done += 1
                    y = b0 * x + z1
                    z1 = b1 * x - a1 * y
                    append(y)
                # 无论本次缓冲是否走完整个渐变，都把当前系数写回实例，
                # 保证下一段缓冲从同一位置继续（跨缓冲区状态连续）。
                self._b0, self._b1, self._a1 = b0, b1, a1
                self._ramp_remaining = remaining
                if not remaining:
                    self._ramp_from = self._ramp_to = None
            else:
                # ---- 稳态热路径：参数恒定，循环体压到最小 ----
                b0, b1, a1 = self._b0, self._b1, self._a1
                for x in samples:
                    if not isfinite(x):
                        self._restore(snapshot)
                        raise NonFiniteInputError(
                            "non-finite sample (NaN/Inf) in input"
                        )
                    y = b0 * x + z1
                    z1 = b1 * x - a1 * y
                    append(y)

            self._z1 = z1
            return out

    def _restore(self, snapshot):
        # 仅在异常路径调用，恢复到本次 process() 之前的完整内部状态。
        (self._z1, self._b0, self._b1, self._a1,
         self._ramp_remaining, self._ramp_steps,
         self._ramp_from, self._ramp_to) = snapshot

    def reset(self):
        """清零内部状态与未完成的参数渐变（不改变目标参数）。"""
        with self._lock:
            self._z1 = 0.0
            self._ramp_remaining = 0
            self._ramp_from = self._ramp_to = None
            b0, b1, a1 = _design(self._rate, self._target_cutoff, self._lowpass)
            self._b0, self._b1, self._a1 = b0, b1, a1
            return self


def _design(rate, cutoff, lowpass):
    """双线性变换一阶 biquad 系数设计，返回 ``(b0, b1, a1)``。

    调用方保证 ``0 < cutoff < rate/2``，故 ``alpha`` 恒正、分母不为零。
    """
    # 双线性预畸变：α = tan(ω_c/2)，极点 z = -a1 = (1-α)/(1+α) 恒在单位圆内。
    alpha = math.tan(math.pi * cutoff / rate)
    a1 = (alpha - 1.0) / (1.0 + alpha)
    if lowpass:
        b0 = alpha / (1.0 + alpha)
        b1 = b0
    else:
        b0 = 1.0 / (1.0 + alpha)
        b1 = -b0
    return b0, b1, a1

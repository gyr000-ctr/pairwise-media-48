"""audioloop 核心 API：循环点搜索。

在指定窗口内搜索循环点，使交叉淡化闭环后与原始波形的拼接误差最小。
误差度量 = 拼接点处采样值跳变 + 一阶差分（斜率）跳变。
实现为纯计算、无共享可变状态，并发安全。
"""

import numpy as np

__all__ = ["LoopFinder", "LoopError", "NoLoopPointError"]


class LoopError(ValueError):
    """循环点搜索相关错误的基类。"""


class NoLoopPointError(LoopError):
    """窗口内没有可用候选循环点（如候选为空或误差全部非有限）。"""


class LoopFinder:
    """循环点搜索器。无实例状态，find 可并发调用。"""

    def find(self, samples, window):
        """在 window=(lo, hi)（闭区间）内返回拼接误差最小的循环点下标。

        参数非法（空样本、窗口越界/倒置）抛 ValueError；
        窗口合法但无可用候选抛 NoLoopPointError（可区分子类）。
        结果确定：误差并列时取窗口内最小下标。
        """
        if samples is None or len(samples) == 0:
            raise ValueError("samples 为空，无法搜索循环点")
        try:
            lo, hi = window
        except (TypeError, ValueError) as exc:
            raise ValueError(f"window 必须是 (lo, hi) 二元组: {window!r}") from exc
        n = len(samples)
        if not (0 <= lo <= hi < n):
            raise ValueError(f"窗口越界或倒置: window={window!r}, 样本数={n}")

        s = np.asarray(samples, dtype=np.float64)
        cand = s[lo:hi + 1]

        # 拼接点值跳变：窗口末端值 vs 候选循环点值
        v_end = s[hi]
        value_err = np.abs(cand - v_end)

        # 拼接点斜率跳变：窗口末端一阶差分 vs 候选点一阶差分（首样本斜率取 0）
        d = np.empty(n, dtype=np.float64)
        d[0] = 0.0
        d[1:] = np.diff(s)
        slope_err = np.abs(d[lo:hi + 1] - d[hi])

        err = value_err + slope_err
        finite = np.isfinite(err)
        if not finite.any():
            raise NoLoopPointError(
                f"窗口 {window!r} 内无可用候选（误差均非有限）"
            )
        # 非有限误差视为无穷大，argmin 取首个最小值保证确定性
        err = np.where(finite, err, np.inf)
        return lo + int(np.argmin(err))

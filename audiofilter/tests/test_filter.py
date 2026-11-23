"""OnePole 双线性变换 biquad 的验收测试（文件名匹配 ``-k filter``）。"""

import math
import threading
import time

import pytest

from audiofilter import OnePole
from audiofilter.core import (
    FilterParameterError,
    NonFiniteInputError,
    _design,
)

RATE = 48000.0


def _sine(freq, n, rate=RATE, amp=1.0, phase=0.0):
    w = 2.0 * math.pi * freq / rate
    return [amp * math.sin(w * i + phase) for i in range(n)]


def _steady_gain(f, freq, lowpass, settle_cycles=200, measure_cycles=64):
    """喂长正弦，对稳态段做单频点 DFT 相关，得到实测幅频响应。

    用 cos/sin 相关而非峰峰值：不要求测量窗恰好覆盖整数个周期，
    在任意窗长下都能无偏估计该频点的幅度。
    """
    n = int((settle_cycles + measure_cycles) * RATE / freq)
    w = 2.0 * math.pi * freq / RATE
    x = _sine(freq, n)
    y = f.process(x)
    start = int(settle_cycles * RATE / freq)
    idx = range(start, n)
    c = 2.0 * sum(y[i] * math.cos(w * i) for i in idx) / (n - start)
    s = 2.0 * sum(y[i] * math.sin(w * i) for i in idx) / (n - start)
    return math.hypot(c, s)


def _analog_magnitude(rate, cutoff, freq, lowpass):
    """预畸变后的一阶原型解析式：|H_a(jΩ)|，Ω = 2·fs·tan(ω/2)。

    双线性变换保证数字滤波器在该频率上的幅度与此解析式完全相等。
    """
    w = 2.0 * math.pi * freq / rate
    omega = 2.0 * rate * math.tan(w / 2.0)
    wc = 2.0 * rate * math.tan(math.pi * cutoff / rate)
    if lowpass:
        return wc / math.hypot(wc, omega)
    return omega / math.hypot(wc, omega)


def _digital_magnitude(b0, b1, a1, w):
    """由实际 biquad 系数直接计算 |H(e^{jω})|。"""
    c = math.cos(w)
    num_re, num_im = b0 + b1 * c, -b1 * math.sin(w)
    den_re, den_im = 1.0 + a1 * c, -a1 * math.sin(w)
    return math.hypot(num_re, num_im) / math.hypot(den_re, den_im)


def _db(x):
    return 20.0 * math.log10(x)


# --------------------------------------------------------------------------- #
# 1. 双线性变换 biquad：频率响应与解析式误差 ≤ 0.5 dB
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cutoff", [100.0, 1000.0, 5000.0, 12000.0])
@pytest.mark.parametrize("lowpass", [True, False])
def test_frequency_response_matches_analytic(cutoff, lowpass):
    f = OnePole(RATE, cutoff, lowpass=lowpass)
    b0, b1, _b2, a1, _a2 = f.coefficients
    # 覆盖 0.02·fc ~ 0.9·nyquist 的频点（避开 HP 极低频的数值地板）。
    freqs = [max(20.0, cutoff * k) for k in (0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    freqs = [q for q in dict.fromkeys(freqs) if q < RATE / 2.0 * 0.9]
    for freq in freqs:
        measured = _steady_gain(OnePole(RATE, cutoff, lowpass), freq, lowpass)
        analog = _analog_magnitude(RATE, cutoff, freq, lowpass)
        digital = _digital_magnitude(b0, b1, a1, 2.0 * math.pi * freq / RATE)
        # 实测 vs 预畸变解析原型
        assert abs(_db(measured) - _db(analog)) <= 0.5
        # 系数解析式本身与原型在机器精度内一致
        assert abs(_db(digital) - _db(analog)) < 1e-9


def test_cutoff_is_minus_3db_point():
    """双线性预畸变下，截止频率处幅度严格为 -3.01 dB。"""
    for cutoff in (100.0, 1000.0, 8000.0):
        for lowpass in (True, False):
            f = OnePole(RATE, cutoff, lowpass=lowpass)
            g = _steady_gain(f, cutoff, lowpass)
            assert abs(_db(g) - (-3.0102999566)) < 0.2


def test_design_coeffs_are_biquad_form():
    """系数必须是标准 biquad 五系数元组，且 b2/a2 为 0（一阶节）。"""
    f = OnePole(RATE, 1000.0)
    b0, b1, b2, a1, a2 = f.coefficients
    assert b2 == 0.0 and a2 == 0.0
    assert all(math.isfinite(v) for v in (b0, b1, a1))
    # 低通直流增益 = 1，Nyquist 增益 = 0
    assert abs((b0 + b1) / (1.0 + a1) - 1.0) < 1e-12
    assert abs((b0 - b1) / (1.0 - a1)) < 1e-12


# --------------------------------------------------------------------------- #
# 2. 高低通互补：H_LP + H_HP ≡ 1
# --------------------------------------------------------------------------- #
def test_lowpass_highpass_complementary():
    x = _sine(2345.0, 4000) + _sine(120.0, 4000, amp=0.7)
    lp = OnePole(RATE, 1500.0, True)
    hp = OnePole(RATE, 1500.0, False)
    y_lp = lp.process(x)
    y_hp = hp.process(x)
    for a, b, xi in zip(y_lp, y_hp, x):
        assert abs((a + b) - xi) < 1e-10


def test_complementary_coeff_identity():
    for cutoff in (50.0, 1000.0, 20000.0):
        lp = _design(RATE, cutoff, True)
        hp = _design(RATE, cutoff, False)
        # 同分母 a1；分子相加后恰等于分母 (1 + a1·z⁻¹)，故 H_LP + H_HP ≡ 1
        assert abs(lp[2] - hp[2]) < 1e-15
        assert abs((lp[0] + hp[0]) - 1.0) < 1e-15
        assert abs((lp[1] + hp[1]) - lp[2]) < 1e-15


# --------------------------------------------------------------------------- #
# 3. 跨缓冲区状态连续
# --------------------------------------------------------------------------- #
def test_state_continuous_across_buffers():
    x = _sine(997.0, 5000, amp=0.8)
    whole = OnePole(RATE, 1200.0).process(list(x))
    f = OnePole(RATE, 1200.0)
    chunked = []
    for i in range(0, len(x), 37):  # 故意切成不规则小块
        chunked.extend(f.process(list(x[i:i + 37])))
    assert len(whole) == len(chunked)
    for a, b in zip(whole, chunked):
        assert a == b  # 同一递归，结果必须逐样本完全一致


def test_state_persists_between_calls():
    f = OnePole(RATE, 50.0)  # 低截止 -> 时间常数 ~153 样本，100 样本远未稳态
    a = f.process([1.0] * 100)
    b = f.process([1.0] * 100)
    # 连续直流输入下输出单调趋稳，第二个缓冲不会从头开始
    assert b[-1] > a[-1]
    assert b[0] > a[0]


# --------------------------------------------------------------------------- #
# 4. 参数变化不得爆音
# --------------------------------------------------------------------------- #
def test_set_cutoff_no_pop():
    f = OnePole(RATE, 500.0)
    x = _sine(300.0, 4000)
    baseline = OnePole(RATE, 500.0).process(list(x))
    y = f.process(list(x[:2000]))
    f.set_cutoff(4000.0)            # 中途改截止频率
    y += f.process(list(x[2000:]))
    base_jumps = [abs(baseline[i + 1] - baseline[i]) for i in range(1900, 2100)]
    new_jumps = [abs(y[i + 1] - y[i]) for i in range(1900, 2100)]
    # 渐变区域的样本间跳变不得超过恒定参数时的 1.5 倍（无突变瞬态）
    assert max(new_jumps) <= 1.5 * max(base_jumps) + 1e-12


def test_set_cutoff_converges_to_new_response():
    f = OnePole(RATE, 500.0)
    f.set_cutoff(4000.0)
    f.process([0.0] * 10000)  # 等渐变结束
    target = OnePole(RATE, 4000.0)
    assert f.coefficients == pytest.approx(target.coefficients, abs=1e-15)


def test_ramp_continuous_across_buffers():
    # 渐变跨越多个小缓冲区时，系数与状态必须和单缓冲处理完全一致
    x = _sine(440.0, 2000)
    whole = OnePole(RATE, 500.0)
    whole.set_cutoff(4000.0)
    y_whole = whole.process(list(x))

    chunked = OnePole(RATE, 500.0)
    chunked.set_cutoff(4000.0)
    y_chunk = []
    for i in range(0, len(x), 13):
        y_chunk += chunked.process(list(x[i:i + 13]))
    assert y_chunk == pytest.approx(y_whole, abs=1e-15)


def test_set_cutoff_rejects_bad_value():
    f = OnePole(RATE, 1000.0)
    with pytest.raises(FilterParameterError):
        f.set_cutoff(0.0)
    with pytest.raises(FilterParameterError):
        f.set_cutoff(RATE / 2.0)
    with pytest.raises(FilterParameterError):
        f.set_cutoff(float("nan"))


# --------------------------------------------------------------------------- #
# 5. 性能：100 万样本 ≤ 200 ms
# --------------------------------------------------------------------------- #
def test_performance_million_samples():
    x = [0.5] * 1_000_000
    OnePole(RATE, 1000.0).process(x[:1000])  # 预热
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = OnePole(RATE, 1000.0).process(x)
        times.append(time.perf_counter() - t0)
    assert len(out) == 1_000_000
    assert min(times) < 0.2, f"best of 3 = {min(times)*1000:.1f} ms > 200 ms"


# --------------------------------------------------------------------------- #
# 6. 非法参数可区分报错 + 非有限输入分类
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rate,cutoff",
    [(RATE, 0.0), (RATE, -1.0), (RATE, RATE / 2.0), (RATE, RATE),
     (RATE, float("nan")), (RATE, float("inf")),
     (0.0, 1000.0), (-48000.0, 1000.0), (float("nan"), 1000.0),
     (float("inf"), 1000.0)],
)
def test_invalid_parameters_raise_distinct_error(rate, cutoff):
    with pytest.raises(FilterParameterError):
        OnePole(rate, cutoff)
    # 同时保持 ValueError 语义（既有行为依赖 ValueError）
    with pytest.raises(ValueError):
        OnePole(rate, cutoff)


def test_error_types_are_distinguishable():
    with pytest.raises(NonFiniteInputError):
        OnePole(RATE, 1000.0).process([1.0, float("nan"), 0.0])
    with pytest.raises(NonFiniteInputError):
        OnePole(RATE, 1000.0).process([1.0, float("inf")])
    # 两类错误可彼此区分
    assert issubclass(FilterParameterError, ValueError)
    assert issubclass(NonFiniteInputError, ValueError)
    assert not issubclass(NonFiniteInputError, FilterParameterError)


def test_nonfinite_input_preserves_prior_state():
    f = OnePole(RATE, 1000.0)
    good = f.process([1.0] * 100)
    z_before = f._z1
    with pytest.raises(NonFiniteInputError):
        f.process([0.5, float("nan")])
    assert f._z1 == z_before
    # 报错后滤波器仍可继续工作，且与未出错路径一致
    ref = OnePole(RATE, 1000.0)
    ref.process([1.0] * 100)
    assert f.process([0.25] * 50) == ref.process([0.25] * 50)
    assert good[-1] is not None


# --------------------------------------------------------------------------- #
# 7. 并发安全
# --------------------------------------------------------------------------- #
def test_thread_safety_shared_instance():
    f = OnePole(RATE, 1000.0)
    errors = []
    counts = []

    def worker():
        try:
            out = f.process([0.3] * 1000)
            counts.append(len(out))
        except Exception as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sum(counts) == 8000
    assert math.isfinite(f._z1)


def test_thread_safety_independent_instances_deterministic():
    results = [None] * 8

    def worker(i):
        results[i] = OnePole(RATE, 1234.0).process(_sine(777.0, 2000))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results[1:]:
        assert r == results[0]


# --------------------------------------------------------------------------- #
# 8. 既有行为回归（list 输出、空输入、不改入参、确定性、直流、单调）
# --------------------------------------------------------------------------- #
def test_existing_behavior_regressions():
    f = OnePole(RATE, 1000.0)
    assert f.process([]) == []
    assert isinstance(f.process([0.1]), list)
    src = [0.5]
    f.process(src)
    assert src == [0.5]
    assert OnePole(RATE, 1000.0).process([0.3] * 5) == \
        OnePole(RATE, 1000.0).process([0.3] * 5)
    step = OnePole(RATE, 1000.0, True).process([1.0] * 20)
    assert all(step[i] <= step[i + 1] + 1e-9 for i in range(len(step) - 1))
    assert OnePole(RATE, 1000.0, True).process([1.0] * 500)[-1] > 0.9
    assert abs(OnePole(RATE, 1000.0, False).process([1.0] * 500)[-1]) < 0.1
    assert len(OnePole(RATE, 1000.0).process([0.5] * 10)) == 10


def test_accepts_tuple_and_generator_inputs():
    ref = OnePole(RATE, 1000.0).process([0.1, 0.2, 0.3])
    assert OnePole(RATE, 1000.0).process((0.1, 0.2, 0.3)) == ref
    assert OnePole(RATE, 1000.0).process(v for v in (0.1, 0.2, 0.3)) == ref

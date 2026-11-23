"""循环点搜索验收测试：交叉淡化闭环误差、窗口约束、可区分错误、性能与并发安全。"""

import threading
import time

import pytest

import audioloop
from audioloop import LoopFinder, NoLoopPointError


def test_loop_crossfade_min_splice_error():
    # 构造信号：窗口末端值/斜率与下标 5 处完全一致，其余候选误差更大
    s = [0.0, 5.0, -3.0, 2.0, 1.0, 2.0, 9.0, -7.0, 1.0, 2.0]
    # 窗口 (4, 9)：末端 s[9]=2、斜率 2-1=1；候选 5 处值 2、斜率 1 → 误差 0
    assert LoopFinder().find(s, (4, 9)) == 5


def test_loop_search_stays_in_window():
    s = [float((i * 7) % 11) for i in range(200)]
    idx = LoopFinder().find(s, (50, 120))
    assert 50 <= idx <= 120


def test_loop_bad_window_raises():
    f = LoopFinder()
    with pytest.raises(ValueError):
        f.find([0.0], (5, 9))       # 越界
    with pytest.raises(ValueError):
        f.find([0.0] * 4, (3, 1))   # 倒置
    with pytest.raises(ValueError):
        f.find([], (0, 0))          # 空样本


def test_loop_no_candidate_distinguishable_error():
    # 全 NaN：窗口合法但无有限误差候选 → 可区分的 NoLoopPointError
    with pytest.raises(NoLoopPointError):
        LoopFinder().find([float("nan")] * 16, (0, 15))
    # 可区分：是 ValueError 的子类，但不是裸 ValueError 语义
    assert issubclass(NoLoopPointError, ValueError)
    assert NoLoopPointError is not ValueError


def test_loop_deterministic():
    s = [float((i * 13) % 7) - 3.0 for i in range(500)]
    f = LoopFinder()
    assert f.find(s, (10, 490)) == f.find(s, (10, 490))


def test_loop_perf_1m_samples_under_500ms():
    import numpy as np

    rng = np.random.default_rng(0)
    s = rng.standard_normal(1_000_000).tolist()
    t0 = time.perf_counter()
    idx = LoopFinder().find(s, (100, 999_900))
    elapsed = time.perf_counter() - t0
    assert 100 <= idx <= 999_900
    assert elapsed < 0.5, f"100 万样本耗时 {elapsed:.3f}s 超过 500ms"


def test_loop_concurrent_safe():
    s = [float((i * 31) % 17) - 8.0 for i in range(10_000)]
    finder = LoopFinder()  # 共享同一实例验证无共享可变状态
    results = [None] * 8

    def worker(k):
        results[k] = finder.find(s, (500, 9_500))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1  # 并发结果一致且无竞态异常

"""audiofilter 核心 API（初始未实现）。"""


class OnePole:
    def __init__(self, rate, cutoff, lowpass=True):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError

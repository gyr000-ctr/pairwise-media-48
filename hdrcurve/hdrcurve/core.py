"""hdrcurve 核心 API（初始未实现）。"""


class ToneCurve:
    def __init__(self, gamma, levels=256):
        """构造"""
        raise NotImplementedError

    def lookup(self, t):
        """查表"""
        raise NotImplementedError

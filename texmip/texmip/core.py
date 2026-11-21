"""texmip 核心 API（初始未实现）。"""


class TexMip:
    def __init__(self, w, h):
        """构造"""
        raise NotImplementedError

    def level(self, scale):
        """级别"""
        raise NotImplementedError

    def size_at(self, level):
        """尺寸"""
        raise NotImplementedError

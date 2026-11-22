"""spritebatch 核心 API。

批次合并规则
------------
``SpriteBatch.build(sprites)`` 的输出为 ``list[Batch]``：

* 一次线性扫描把 ``(图层, 纹理)`` 相同的精灵收进同一批次，批次内保持
  输入顺序；批次按图层升序、同图层内按纹理首次出现顺序排列。因此组数
  恰为不同 ``(图层, 纹理)`` 组合数，即最小分组数，且结果确定。
* 每个 ``Batch`` 是 ``list`` 的子类，元素为该批次内的精灵实例，元素顺序
  即实例缓冲中的顺序。
* 每个 ``SpriteInstance`` 是 ``(texture, layer, ..., offset)`` 元组：输入项
  除前两个字段外的字段原样保留，``offset`` 追加在末尾，取值为该精灵在
  原始输入中的下标（排序后仍可回查原数据）。实例同时提供
  ``.texture`` / ``.layer`` / ``.offset`` / ``.sprite`` 属性。
* ``Batch.texture`` 为批次纹理，``Batch.offset`` 为该批次在合并实例缓冲中
  的起始偏移，``Batch.count`` 为实例数，可直接用于实例化绘制。

错误分类
--------
所有面向调用者的错误都是 ``ValueError`` 的子类，且可按类型区分：

* ``SpriteBatchError``      输入容器本身非法（None、字符串、不可迭代）；
* ``InvalidItemError``      精灵项不是序列，或字段数不足；
* ``InvalidTextureError``   纹理不可哈希 / 为 None / 为布尔值；
* ``InvalidLayerError``     图层不是 int/float；
* ``NonFiniteError``       图层或附加数值字段为 NaN / inf。

实现不使用任何可变全局状态，同一实例可由多线程并发调用。
"""

import math
from operator import itemgetter

__all__ = [
    "SpriteBatch",
    "Batch",
    "SpriteInstance",
    "SpriteBatchError",
    "InvalidItemError",
    "InvalidTextureError",
    "InvalidLayerError",
    "NonFiniteError",
]

_isfinite = math.isfinite


class SpriteBatchError(ValueError):
    """批次构建错误基类。"""


class InvalidItemError(SpriteBatchError):
    """精灵项结构非法。"""


class InvalidTextureError(InvalidItemError):
    """纹理字段非法。"""


class InvalidLayerError(InvalidItemError):
    """图层字段类型非法。"""


class NonFiniteError(InvalidItemError):
    """数值字段不是有限值（NaN / inf）。"""


class SpriteInstance(tuple):
    """可直接用于实例化的精灵记录。

    元组内容为 ``(texture, layer, *extra, offset)``；``offset`` 是该项在
    输入序列中的原始下标。附带 ``.texture`` / ``.layer`` / ``.offset`` /
    ``.sprite``（不含 offset 的原始字段）属性。
    """

    __slots__ = ()

    def __new__(cls, texture, layer, offset, extra=()):
        return tuple.__new__(cls, (texture, layer) + tuple(extra) + (offset,))

    texture = property(itemgetter(0))
    layer = property(itemgetter(1))
    offset = property(itemgetter(-1))
    sprite = property(lambda self: self[:-1])


class Batch(list):
    """纹理一致的一批精灵；``offset`` 为合并实例缓冲中的起始偏移。"""

    __slots__ = ("texture", "offset")

    def __init__(self, texture, offset=0, items=()):
        list.__init__(self, items)
        self.texture = texture
        self.offset = offset

    @property
    def count(self):
        return len(self)


def _parse_item(item, index):
    """校验一个精灵项，返回 ``(layer, texture, extra)``。"""
    if not isinstance(item, (tuple, list)):
        raise InvalidItemError(
            "item %d: expected a sequence with at least 2 fields "
            "(texture, layer), got %s" % (index, type(item).__name__)
        )
    if len(item) < 2:
        raise InvalidItemError(
            "item %d: expected at least 2 fields (texture, layer), got %d"
            % (index, len(item))
        )

    texture = item[0]
    if texture is None or type(texture) is bool:
        raise InvalidTextureError(
            "item %d: invalid texture %r" % (index, texture)
        )
    if type(texture) is float and not _isfinite(texture):
        raise NonFiniteError(
            "item %d: non-finite texture %r" % (index, texture)
        )
    try:
        hash(texture)
    except TypeError:
        raise InvalidTextureError(
            "item %d: texture must be hashable, got %s"
            % (index, type(texture).__name__)
        ) from None

    layer = item[1]
    layer_type = type(layer)
    if layer_type is bool or not isinstance(layer, (int, float)):
        raise InvalidLayerError(
            "item %d: layer must be an int or float, got %s"
            % (index, layer_type.__name__)
        )
    if layer_type is float and not _isfinite(layer):
        raise NonFiniteError(
            "item %d: non-finite layer %r" % (index, layer)
        )

    if len(item) > 2:
        for position in range(2, len(item)):
            value = item[position]
            if type(value) is float and not _isfinite(value):
                raise NonFiniteError(
                    "item %d: non-finite field at position %d (%r)"
                    % (index, position, value)
                )

    return layer, texture, item[2:]


class SpriteBatch:
    """按纹理合并精灵批次。无实例状态，可并发复用。"""

    def build(self, sprites):
        """构建批次并返回 ``list[Batch]``（不修改输入）。"""
        if sprites is None or isinstance(sprites, (str, bytes, bytearray)):
            raise SpriteBatchError(
                "sprites must be an iterable of sprite items, got %s"
                % type(sprites).__name__
            )
        try:
            iterator = iter(sprites)
        except TypeError:
            raise SpriteBatchError(
                "sprites must be iterable, got %s" % type(sprites).__name__
            ) from None

        # 按 (图层, 纹理) 置入批次，保留输入顺序与纹理首次出现顺序。
        groups = {}
        order = []
        for index, item in enumerate(iterator):
            if type(item) is tuple and len(item) == 2:
                texture = item[0]
                texture_type = type(texture)
                if texture_type is str or texture_type is int:
                    layer = item[1]
                    layer_type = type(layer)
                    if layer_type is int:
                        pass
                    elif layer_type is float:
                        if not _isfinite(layer):
                            raise NonFiniteError(
                                "item %d: non-finite layer %r"
                                % (index, layer)
                            )
                    else:
                        raise InvalidLayerError(
                            "item %d: layer must be an int or float, got %s"
                            % (index, layer_type.__name__)
                        )
                    extra = ()
                else:
                    layer, texture, extra = _parse_item(item, index)
            else:
                layer, texture, extra = _parse_item(item, index)

            key = (layer, texture)
            batch = groups.get(key)
            if batch is None:
                batch = groups[key] = Batch(texture)
                order.append(key)
            batch.append(SpriteInstance(texture, layer, index, extra))

        batches = []
        total = 0
        order.sort(key=itemgetter(0))  # 稳定排序：图层升序，层内保持首次出现顺序
        for key in order:
            batch = groups[key]
            batch.offset = total
            total += len(batch)
            batches.append(batch)
        return batches

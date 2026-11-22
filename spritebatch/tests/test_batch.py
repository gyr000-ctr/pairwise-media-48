"""spritebatch 批组合并测试（新增，不改动既有测试）。"""

import math
import threading

import pytest

import spritebatch


def _build(sprites, **kw):
    return spritebatch.SpriteBatch(**kw).build(sprites)


# ---------------------------------------------------------------- 基本契约

def test_empty():
    assert _build([]) == []


def test_returns_list_of_batches():
    out = _build([("tex1", 0), ("tex2", 1), ("tex1", 0)])
    assert isinstance(out, list)
    assert all(isinstance(b, list) for b in out)


def test_minimal_group_count():
    # (tex, layer) 不同组合数即最小批次数
    sprites = [("a", 0), ("b", 0), ("a", 0), ("c", 1), ("a", 0)]
    out = _build(sprites)
    assert len(out) == 3
    keys = [(b.texture, b[0].layer) for b in out]
    assert len(set(keys)) == len(out)


def test_groups_by_texture_and_layer():
    sprites = [("a", 0), ("b", 0), ("a", 1), ("b", 0)]
    out = _build(sprites)
    assert [(b.texture, b[0].layer) for b in out] == [
        ("a", 0), ("b", 0), ("a", 1)
    ]


def test_stable_within_layer():
    sprites = [("a", 0), ("b", 0), ("c", 0), ("a", 0)]
    out = _build(sprites)
    first = out[0]
    assert [s.offset for s in first] == [0, 3]
    second = next(b for b in out if b.texture == "b")
    assert [s.offset for s in second] == [1]


def test_input_order_preserved_in_batch():
    sprites = [("t", 0, 1.0), ("t", 0, 2.0), ("t", 0, 3.0)]
    batch = _build(sprites)[0]
    assert [s[2] for s in batch] == [1.0, 2.0, 3.0]


def test_offsets_contiguous():
    sprites = [("a", 0), ("b", 1), ("a", 0), ("c", 2)]
    out = _build(sprites)
    seen = []
    expected = 0
    for batch in out:
        assert batch.offset == expected
        seen.extend(range(batch.offset, batch.offset + len(batch)))
        expected += len(batch)
    assert sorted(seen) == [0, 1, 2, 3]


def test_offset_points_at_source_index():
    sprites = [("b", 0, "x"), ("a", 0, "y"), ("b", 0, "z")]
    out = _build(sprites)
    by_tex = {b.texture: b for b in out}
    assert [s.offset for s in by_tex["b"]] == [0, 2]
    assert [s.offset for s in by_tex["a"]] == [1]
    for batch in out:
        for inst in batch:
            assert sprites[inst.offset][:2] == (inst.texture, inst.layer)


def test_instance_tuple_shape():
    batch = _build([("t", 0, 5.0, 6.0)])[0]
    inst = batch[0]
    assert inst == ("t", 0, 5.0, 6.0, 0)
    assert inst.texture == "t"
    assert inst.layer == 0
    assert inst.offset == 0
    assert inst.sprite == ("t", 0, 5.0, 6.0)


def test_batch_metadata():
    out = _build([("t", 0), ("t", 0), ("u", 0)])
    first = out[0]
    assert first.texture == "t"
    assert first.offset == 0
    assert first.count == 2
    assert len(first) == 2


def test_count_and_no_mutation():
    sprites = [("a", 0), ("b", 1), ("a", 0)]
    snapshot = list(sprites)
    out = _build(sprites)
    assert sum(len(b) for b in out) == len(sprites)
    assert sprites == snapshot


def test_deterministic():
    sprites = [("a", 0), ("b", 1), ("a", 0), ("c", 1)]
    assert _build(sprites) == _build(sprites)


def test_accepts_generator():
    gen = ((tex, layer) for tex, layer in [("a", 0), ("a", 0), ("b", 1)])
    assert sum(len(b) for b in _build(gen)) == 3


def test_layer_float_and_int_ordering():
    out = _build([("a", 1.5), ("a", 0.5)])
    assert [b.offset for b in out] == [0, 1]


def test_non_string_hashable_texture():
    class Tex:
        pass

    t = Tex()
    out = _build([(t, 0), (t, 0)])
    assert len(out) == 1
    assert out[0].texture is t


def test_huge_batch_single_group():
    sprites = [("t", 0)] * 10000
    out = _build(sprites)
    assert len(out) == 1
    assert len(out[0]) == 10000


# ---------------------------------------------------------------- 非法输入

@pytest.mark.parametrize(
    "bad",
    [
        [("tex",)],
        [("tex", 0), ("tex",)],
        [(None, 0)],
        [("tex", None)],
        [("tex", "layer")],
        [("tex", True)],
        [([], 0)],
        [("tex", float("nan"))],
        [("tex", float("inf"))],
        [("tex", 0, float("nan"))],
        [("tex", 0, 1, float("-inf"))],
        [42],
        ["not-an-item"],
    ],
)
def test_invalid_items_raise(bad):
    with pytest.raises(ValueError):
        _build(bad)


@pytest.mark.parametrize("bad", [None, 42, "abc", b"abc"])
def test_invalid_container_raises(bad):
    with pytest.raises(ValueError):
        _build(bad)


def test_non_finite_classified():
    with pytest.raises(spritebatch.NonFiniteError):
        _build([("tex", float("nan"))])
    with pytest.raises(spritebatch.NonFiniteError):
        _build([("tex", 0.0, float("inf"))])


def test_missing_field_classified():
    with pytest.raises(spritebatch.InvalidItemError):
        _build([("tex",)])
    with pytest.raises(spritebatch.InvalidItemError):
        _build([("tex", 0), ()])


def test_bad_item_classified():
    with pytest.raises(spritebatch.InvalidItemError):
        _build([42])
    with pytest.raises(spritebatch.InvalidItemError):
        _build(["not-an-item"])


@pytest.mark.parametrize(
    "bad",
    [
        [("tex", True)],
        [("tex", "layer")],
        [("tex", None)],
        [("tex", [0])],
        [("tex", 1 + 0j)],
    ],
)
def test_bad_layer_all_shapes(bad):
    with pytest.raises(spritebatch.InvalidLayerError):
        _build(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [(None, 0)],
        [([], 0)],
        [({}, 0)],
        [(True, 0)],
    ],
)
def test_bad_texture_all_shapes(bad):
    with pytest.raises(spritebatch.InvalidTextureError):
        _build(bad)


def test_non_finite_texture_is_non_finite_error():
    with pytest.raises(spritebatch.NonFiniteError):
        _build([(float("nan"), 0)])


def test_error_classes_are_distinct():
    classes = [
        spritebatch.SpriteBatchError,
        spritebatch.InvalidItemError,
        spritebatch.InvalidTextureError,
        spritebatch.InvalidLayerError,
        spritebatch.NonFiniteError,
    ]
    assert len(set(classes)) == len(classes)
    assert all(issubclass(c, ValueError) for c in classes)
    assert issubclass(spritebatch.InvalidItemError, spritebatch.SpriteBatchError)


def test_invalid_input_not_mutated_and_raises_atomically():
    sprites = [("a", 0), ("b", float("nan"))]
    with pytest.raises(ValueError):
        _build(sprites)
    assert sprites[0] == ("a", 0)


# ---------------------------------------------------------------- 并发

def test_concurrent_builds():
    sprites = [(("t%d" % (i % 7)), i % 3) for i in range(3000)]
    expected = _build(sprites)
    errors = []

    def worker():
        try:
            for _ in range(20):
                if _build(sprites) != expected:
                    errors.append("mismatch")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_shared_instance_concurrent():
    batch = spritebatch.SpriteBatch()
    sprites = [(("t%d" % (i % 5)), i % 2) for i in range(1000)]
    expected = batch.build(sprites)
    results = []
    errors = []

    def worker():
        try:
            results.append(batch.build(sprites) == expected)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(results)


# ---------------------------------------------------------------- 性能

def test_perf_100k_under_300ms():
    import time

    sprites = [(("tex%d" % (i % 64)), (i // 64) % 8) for i in range(100_000)]
    start = time.perf_counter()
    out = _build(sprites)
    elapsed = time.perf_counter() - start
    assert sum(len(b) for b in out) == 100_000
    assert elapsed < 0.3, "%.1f ms" % (elapsed * 1000.0)

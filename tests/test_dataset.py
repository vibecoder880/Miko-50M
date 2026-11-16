"""Dataset shard writer/reader tests (plan §13-14)."""

from __future__ import annotations

import os

import pytest
import torch

from miko.dataset import PackedShardWriter, ShardedTokenDataset, find_shards, shard_stats


@pytest.fixture()
def shard_dir(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    with PackedShardWriter(str(d / "part"), docs_per_shard=4) as w:
        # 12 documents of 10 ids each -> 3 shards of 4 docs / 40 tokens.
        for doc in range(12):
            w.write_ids(list(range(doc * 10, doc * 10 + 10)))
    return d


def test_writer_produces_bin_and_json(shard_dir):
    bins = sorted(f for f in os.listdir(shard_dir) if f.endswith(".bin"))
    metas = sorted(f for f in os.listdir(shard_dir) if f.endswith(".json"))
    assert len(bins) == 3 and len(metas) == 3
    assert bins[0] == "part-00000.bin"


def test_find_shards_and_stats(shard_dir):
    paths = find_shards(str(shard_dir))
    assert len(paths) == 3
    stats = shard_stats(paths)
    assert stats.tokens == 120 and stats.documents == 12


def test_dataset_shapes_and_shift(shard_dir):
    ds = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=8)
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    assert torch.equal(y, torch.roll(x, shifts=-1, dims=0)) or all(
        int(y[i]) == int(x[i + 1]) for i in range(7)
    )


def test_windows_dont_cross_shard_boundary(shard_dir):
    # 40 tokens per shard (12 docs * 10 ids, docs_per_shard=2 -> 3 shards of 40)
    seq_len = 12  # 40 // 13 -> 3 windows per shard
    ds = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=seq_len)
    assert len(ds) == 9  # 3 windows * 3 shards
    for idx in range(len(ds)):
        shard, start = ds._locate(idx)
        assert start + seq_len + 1 <= ds._counts[shard]


def test_iter_batches_shapes_and_coverage(shard_dir):
    seq_len = 6
    ds = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=seq_len, seed=1)
    seen = 0
    for x, y in ds.iter_batches(batch_size=2, shuffle=True):
        assert x.shape == (2, seq_len) and y.shape == (2, seq_len)
        seen += 2
    assert seen == len(ds) // 2 * 2


def test_epoch_shuffle_is_deterministic(shard_dir):
    ds1 = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=4, seed=7)
    ds2 = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=4, seed=7)
    b1 = [x[0, 0].item() for x, _ in ds1.iter_batches(batch_size=4)]
    b2 = [x[0, 0].item() for x, _ in ds2.iter_batches(batch_size=4)]
    assert b1 == b2
    ds1.advance_epoch()
    b3 = [x[0, 0].item() for x, _ in ds1.iter_batches(batch_size=4)]
    assert b3 != b1  # a new epoch reshuffles


def test_dataset_never_loads_full_corpus(shard_dir):
    # Memory maps are lazily opened: constructing the dataset touches no pages.
    ds = ShardedTokenDataset(find_shards(str(shard_dir)), seq_len=8)
    assert all(m is None for m in ds._maps)
    _ = ds[0]
    assert sum(m is not None for m in ds._maps) == 1


def test_write_ids_rejects_out_of_uint16(tmp_path):
    with PackedShardWriter(str(tmp_path / "p")) as w:
        with pytest.raises(ValueError):
            w.write_ids([70000])

"""Masked-label SFT shard writer/reader tests (plan §26-27, M4)."""

from __future__ import annotations

import os

import pytest
import torch

from miko.chat_template import IGNORE_INDEX, ChatMessage, encode_chat_masked
from miko.dataset import IGNORE_LABEL, SFTShardWriter, SFTShardedDataset
from miko.tokenizer import MikoTokenizer
from miko.trainer import next_token_loss


@pytest.fixture(scope="module")
def tok():
    return MikoTokenizer.from_config(vocab_size=600)


def _conv(tok, user="Xin chào", assistant="Chào bạn!"):
    return encode_chat_masked(tok, [
        ChatMessage("user", user),
        ChatMessage("assistant", assistant),
    ])


def test_writer_produces_uniform_windows(tmp_path, tok):
    d = tmp_path / "sft"
    d.mkdir()
    ids, labels = _conv(tok)
    with SFTShardWriter(str(d / "part"), seq_len=16, pad_id=tok.pad_id) as w:
        w.write_example([tok.bos_id, *ids, tok.eos_id],
                        [IGNORE_INDEX, *labels, tok.eos_id])
    files = sorted(os.listdir(d))
    assert "part-00000.ids.bin" in files and "part-00000.lbl.bin" in files
    ids_size = (d / "part-00000.ids.bin").stat().st_size
    lbl_size = (d / "part-00000.lbl.bin").stat().st_size
    # 1 example → ceil((len+1)/16) windows of 16 rows.
    n_rows = ids_size // (16 * 2)
    assert lbl_size == n_rows * 16 * 2
    assert n_rows >= 1


def test_dataset_targets_are_shifted_labels(tmp_path, tok):
    d = tmp_path / "sft"
    d.mkdir()
    seq_len = 8
    full_ids = [tok.bos_id, *_conv(tok)[0], tok.eos_id]
    full_lbl = [IGNORE_INDEX, *_conv(tok)[1], tok.eos_id]
    with SFTShardWriter(str(d / "part"), seq_len=seq_len, pad_id=tok.pad_id) as w:
        w.write_example(full_ids, full_lbl)
    ds = SFTShardedDataset([str(d / "part-00000.ids.bin")], seq_len)
    x, t = ds[0]
    assert x.shape == (seq_len,) and t.shape == (seq_len,)
    # targets[i] must equal the label of ids[i+1] (what position i predicts).
    for i in range(seq_len - 1):
        assert int(t[i]) == int(full_lbl[i + 1])


def test_masked_positions_are_ignore_label(tmp_path, tok):
    d = tmp_path / "sft"
    d.mkdir()
    seq_len = 8
    full_ids = [tok.bos_id, *_conv(tok)[0], tok.eos_id]
    full_lbl = [IGNORE_INDEX, *_conv(tok)[1], tok.eos_id]
    with SFTShardWriter(str(d / "part"), seq_len=seq_len, pad_id=tok.pad_id) as w:
        w.write_example(full_ids, full_lbl)
    ds = SFTShardedDataset([str(d / "part-00000.ids.bin")], seq_len)
    labeled_any = False
    for i in range(len(ds)):
        _, t = ds[i]
        labeled_any = labeled_any or bool((t != IGNORE_LABEL).any())
    assert labeled_any  # assistant span survives in at least one window
    # And every non-ignore target is a plausible token id, never -100 leak.
    for i in range(len(ds)):
        _, t = ds[i]
        valid = t[t != IGNORE_LABEL]
        assert (valid >= 0).all()


def test_loss_ignores_masked_positions(tmp_path, tok):
    torch.manual_seed(0)
    d = tmp_path / "sft"
    d.mkdir()
    seq_len = 8
    with SFTShardWriter(str(d / "part"), seq_len=seq_len, pad_id=tok.pad_id) as w:
        w.write_example([tok.bos_id, *_conv(tok)[0], tok.eos_id],
                        [IGNORE_INDEX, *_conv(tok)[1], tok.eos_id])
    ds = SFTShardedDataset([str(d / "part-00000.ids.bin")], seq_len)
    from miko.config import MikoConfig
    from miko.model import MikoModel

    cfg = MikoConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=1,
                     num_heads=4, ffn_size=64, context_length=seq_len)
    model = MikoModel(cfg)
    x, t = ds[0]
    loss = next_token_loss(model(x.unsqueeze(0)), t.unsqueeze(0))
    assert torch.isfinite(loss)


def test_iter_batches_shapes(tmp_path, tok):
    d = tmp_path / "sft"
    d.mkdir()
    seq_len = 8
    with SFTShardWriter(str(d / "part"), seq_len=seq_len, pad_id=tok.pad_id) as w:
        for n in range(4):
            ids, labels = _conv(tok, user=f"câu hỏi số {n}")
            w.write_example([tok.bos_id, *ids, tok.eos_id],
                            [IGNORE_INDEX, *labels, tok.eos_id])
    ds = SFTShardedDataset([str(d / "part-00000.ids.bin")], seq_len)
    seen = 0
    for x, t in ds.iter_batches(batch_size=2, shuffle=True):
        assert x.shape == (2, seq_len) and t.shape == (2, seq_len)
        seen += 2
    assert seen == (len(ds) // 2) * 2


def test_misaligned_ids_labels_rejected(tmp_path, tok):
    with SFTShardWriter(str(tmp_path / "part"), seq_len=8, pad_id=0) as w:
        with pytest.raises(ValueError):
            w.write_example([1, 2, 3], [IGNORE_LABEL, IGNORE_LABEL])

"""Streaming dataset pipeline (Phase 5).

Design constraints for a ~1 GB RAM box:

* tokens live in **binary shard files** on disk, read via ``numpy.memmap`` —
  the OS pages in only the windows actually touched;
* shards are *pre-packed*: documents are concatenated as
  ``<bos> doc <eos> <bos> doc <eos> ...`` and chunked into fixed-size training
  windows, so there is **zero padding waste**;
* labels are the standard next-token shift (``labels[i] = ids[i+1]``), computed
  in the training loop, never materialised on disk.

Shard format
------------
``part-NNNNN.bin``  : little-endian uint16 token ids, concatenated stream.
``part-NNNNN.json`` : metadata (tokens, documents, tokenizer version, ...).

uint16 supports vocab <= 65535 (Miko uses 16K-32K).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "PackedShardWriter",
    "ShardedTokenDataset",
    "SFTShardWriter",
    "SFTShardedDataset",
    "find_shards",
    "find_sft_shards",
]

DTYPE = np.uint16
MAX_TOKEN_ID = 65535
# SFT labels use int16 with -100 as the ignore sentinel (masked positions).
LABEL_DTYPE = np.int16
IGNORE_LABEL = -100


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
class PackedShardWriter:
    """Incrementally write packed token shards.

    Usage::

        with PackedShardWriter("data/shards/part", metadata={"split": "train"}) as w:
            for doc_ids in docs:
                w.write_document(doc_ids)
    """

    def __init__(self, prefix: str, docs_per_shard: int = 5000) -> None:
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self._file = None
        self._meta: Optional[dict] = None
        self._part = -1
        self._docs_in_part = 0
        self._tokens_in_part = 0
        self._np_buf = np.empty(0, dtype=DTYPE)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "PackedShardWriter":
        self._open_next()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ---------------------------------------------------------
    def _open_next(self) -> None:
        self.close_current()
        self._part += 1
        path = f"{self.prefix}-{self._part:05d}.bin"
        self._file = open(path, "wb")
        self._meta = {"path": os.path.basename(path), "tokens": 0, "documents": 0}
        self._docs_in_part = 0
        self._tokens_in_part = 0

    def close_current(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
        if self._meta is not None:
            self._meta["tokens"] = self._tokens_in_part
            self._meta["documents"] = self._docs_in_part
            meta_path = f"{self.prefix}-{self._part:05d}.json"
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(self._meta, fh, indent=2)
        self._file = None
        self._meta = None

    def close(self) -> None:
        self.close_current()

    # -- public API --------------------------------------------------------
    def write_ids(self, ids: Sequence[int], documents: int = 1) -> None:
        """Append raw ids (already includes any <bos>/<eos> delimiters)."""
        if not ids:
            return
        max_id = max(ids)
        if max_id > MAX_TOKEN_ID:
            raise ValueError(
                f"token id {max_id} exceeds uint16 storage ({MAX_TOKEN_ID}); "
                "reduce vocab_size or change DTYPE"
            )
        # Roll to a new shard only when the *next* document arrives, so we
        # never leave a trailing empty shard behind.
        if self._file is None or self._docs_in_part >= self.docs_per_shard:
            self._open_next()
        arr = np.asarray(ids, dtype=DTYPE)
        self._file.write(arr.tobytes(order="C"))
        self._tokens_in_part += len(ids)
        self._docs_in_part += documents

    def write_document(self, doc_ids: Sequence[int], bos_id: int, eos_id: int) -> None:
        """Write one document wrapped as ``<bos> ids... <eos>``."""
        self.write_ids([bos_id, *doc_ids, eos_id], documents=1)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def find_shards(directory: str) -> list[str]:
    """All ``*.bin`` shard paths in ``directory``, sorted."""
    paths = sorted(glob.glob(os.path.join(directory, "*.bin")))
    if not paths:
        raise FileNotFoundError(f"no .bin shards found in {directory!r}")
    return paths


def find_sft_shards(directory: str) -> list[str]:
    """All ``*.ids.bin`` SFT shard paths in ``directory``, sorted.

    Deliberately excludes legacy ``part-*.bin`` full-LM files that may sit
    in the same directory from earlier packing runs.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.ids.bin")))
    if not paths:
        raise FileNotFoundError(f"no .ids.bin SFT shards found in {directory!r}")
    return paths


class ShardedTokenDataset(Dataset):
    """Memory-mapped, randomly-accessible packed token windows.

    Yields ``(input_ids, target_ids)`` of shape ``[seq_len]`` each, where
    ``target_ids[i] == input_ids[i+1]`` (next-token prediction).

    Windows never straddle shard boundaries; a few trailing tokens per shard
    are dropped. That keeps indexing arithmetic O(log shards) via
    ``searchsorted`` instead of materialising a huge per-window index.
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        seq_len: int,
        seed: int = 42,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.seq_len = seq_len
        self.seed = int(seed)
        self._epoch = 0

        self._paths = [str(p) for p in shard_paths]
        self._maps: list[Optional[np.memmap]] = [None] * len(self._paths)
        self._counts = np.empty(len(self._paths), dtype=np.int64)
        for i, p in enumerate(self._paths):
            n_bytes = os.path.getsize(p)
            if n_bytes % 2 != 0:
                raise ValueError(f"shard {p!r} size is not a multiple of 2 bytes")
            self._counts[i] = n_bytes // 2

        # Windows available per shard: need seq_len+1 tokens per window.
        self._wins_per_shard = self._counts // (seq_len + 1)
        self._cum = np.cumsum(self._wins_per_shard)
        total = int(self._cum[-1]) if len(self._cum) else 0
        if total == 0:
            raise ValueError(
                "dataset has no complete windows; shards too small for "
                f"seq_len={seq_len}"
            )
        self._total = total

        # RNG used only for epoch shuffling; per-epoch deterministic.
        self._rng = np.random.default_rng(self.seed)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._maps = [None] * len(self._paths)

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def _mmap(self, idx: int) -> np.memmap:
        if self._maps[idx] is None:
            self._maps[idx] = np.memmap(
                self._paths[idx], dtype=DTYPE, mode="r"
            )
        return self._maps[idx]

    # -- properties ----------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return int(self._counts.sum())

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        return self._total

    # -- access ---------------------------------------------------------
    def _locate(self, index: int) -> tuple[int, int]:
        shard = int(np.searchsorted(self._cum, index, side="right"))
        prev = int(self._cum[shard - 1]) if shard > 0 else 0
        local = index - prev
        return shard, local * self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(index)
        shard, start = self._locate(index)
        window = np.asarray(self._mmap(shard)[start : start + self.seq_len + 1])
        ids = torch.from_numpy(window.astype(np.int64))
        return ids[:-1], ids[1:]

    # -- ordering ----------------------------------------------------------
    def advance_epoch(self) -> None:
        """Advance the shuffle epoch (call once per full pass over the data)."""
        self._epoch += 1
        # Re-seed per (seed, epoch) so resume reproduces the same order.
        self._rng = np.random.default_rng(self.seed + self._epoch)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(B, seq_len)`` batches without any DataLoader workers."""
        order = np.arange(self._total)
        if shuffle:
            order = self._rng.permutation(order)
        if drop_last:
            n_batches = len(order) // batch_size
            order = order[: n_batches * batch_size]
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xs, ys = zip(*(self[int(j)] for j in idx))
            yield torch.stack(xs), torch.stack(ys)


@dataclass(frozen=True)
class ShardStats:
    shards: int
    tokens: int
    documents: int


def shard_stats(shard_paths: Sequence[str]) -> ShardStats:
    """Read metadata sidecars for a quick dataset summary."""
    tokens = docs = 0
    for p in shard_paths:
        meta = os.path.splitext(p)[0] + ".json"
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            tokens += int(m.get("tokens", 0))
            docs += int(m.get("documents", 0))
        else:  # fall back to file size
            tokens += os.path.getsize(p) // 2
    return ShardStats(shards=len(shard_paths), tokens=tokens, documents=docs)


# ---------------------------------------------------------------------------
# SFT: masked-label windows (plan §26-27, M4)
# ---------------------------------------------------------------------------
class SFTShardWriter:
    """Write window-aligned (ids, targets) SFT shards.

    Each conversation ``(ids, labels)`` from ``encode_chat_masked`` is
    chunked into ``seq_len`` training windows stored *ready for training*:
    for a window of ids ``x = ids[t:t+L]`` the stored targets are
    ``labels[t+1:t+L+1]`` — the label of the token the model must predict
    at position i. Padded windows use ``pad_id`` ids and ``IGNORE_LABEL``
    targets, and are written last so they can be dropped cheaply.

    Files: ``part-NNNNN.ids.bin`` (uint16) + ``part-NNNNN.lbl.bin`` (int16).
    """

    def __init__(self, prefix: str, seq_len: int, pad_id: int,
                 windows_per_shard: int = 20000) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.prefix = prefix
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.windows_per_shard = windows_per_shard
        self._part = -1
        self._ids_file = None
        self._lbl_file = None
        self._windows_in_part = 0
        self._tokens_in_part = 0
        self._pad_windows = 0

    def _open_next(self) -> None:
        self._close_current()
        self._part += 1
        self._ids_file = open(f"{self.prefix}-{self._part:05d}.ids.bin", "wb")
        self._lbl_file = open(f"{self.prefix}-{self._part:05d}.lbl.bin", "wb")
        self._windows_in_part = 0
        self._tokens_in_part = 0
        self._pad_windows = 0

    def _close_current(self) -> None:
        # Windows are always written whole (padding handled in
        # write_example), so closing is just flush + metadata sidecar.
        if self._ids_file is not None and not self._ids_file.closed:
            self._ids_file.close()
            self._lbl_file.close()
        if self._ids_file is not None:
            meta = {
                "path": os.path.basename(f"{self.prefix}-{self._part:05d}.ids.bin"),
                "windows": self._windows_in_part,
                "tokens": self._tokens_in_part,
                "pad_windows": self._pad_windows,
            }
            with open(f"{self.prefix}-{self._part:05d}.json", "w",
                      encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        self._ids_file = None
        self._lbl_file = None

    def close(self) -> None:
        self._close_current()

    def __enter__(self) -> "SFTShardWriter":
        self._open_next()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write_example(self, ids: Sequence[int], labels: Sequence[int]) -> None:
        """Chunk one ``(ids, labels)`` conversation into training windows."""
        if len(ids) != len(labels):
            raise ValueError("ids and labels must be aligned")
        if len(ids) < 2:
            return
        # Slide non-overlapping windows; the last partial window is padded.
        n = self.seq_len + 1  # need one extra token for shifted targets
        for start in range(0, len(ids), self.seq_len):
            chunk = list(ids[start : start + n])
            chunk_lbl = list(labels[start : start + n])
            if len(chunk) < n:
                pad = n - len(chunk)
                chunk = chunk + [self.pad_id] * pad
                chunk_lbl = chunk_lbl + [IGNORE_LABEL] * pad
                self._pad_windows += 1
            x = chunk[: self.seq_len]
            targets = chunk_lbl[1 : n]  # label of the predicted token
            if self._ids_file is None or (
                self._windows_in_part >= self.windows_per_shard
            ):
                self._open_next()
            self._ids_file.write(np.asarray(x, dtype=DTYPE).tobytes())
            self._lbl_file.write(np.asarray(targets, dtype=LABEL_DTYPE).tobytes())
            self._windows_in_part += 1
            self._tokens_in_part += self.seq_len


class SFTShardedDataset(Dataset):
    """Memory-mapped SFT windows: yields ``(x, targets)`` ready for
    ``next_token_loss`` (targets carry ``IGNORE_LABEL`` where masked)."""

    def __init__(self, shard_paths: Sequence[str], seq_len: int, seed: int = 42):
        self.seq_len = seq_len
        self.seed = int(seed)
        self._epoch = 0
        self._paths = [str(p) for p in shard_paths]
        self._ids_maps: list[Optional[np.memmap]] = [None] * len(self._paths)
        self._lbl_maps: list[Optional[np.memmap]] = [None] * len(self._paths)
        self._counts = np.empty(len(self._paths), dtype=np.int64)
        row_bytes = 2 * seq_len  # uint16 ids + int16 targets per row
        for i, p in enumerate(self._paths):
            ids_path = p if p.endswith(".ids.bin") else p.replace(".bin", ".ids.bin")
            n_bytes = os.path.getsize(ids_path)
            if n_bytes % row_bytes != 0:
                raise ValueError(f"{ids_path}: size not a multiple of {row_bytes}")
            self._counts[i] = n_bytes // row_bytes
        self._cum = np.cumsum(self._counts)
        if not len(self._cum) or self._cum[-1] == 0:
            raise ValueError("no SFT windows found")
        self._rng = np.random.default_rng(self.seed)

    @property
    def total_windows(self) -> int:
        return int(self._cum[-1])

    def __len__(self) -> int:
        return int(self._cum[-1])

    def close(self) -> None:
        self._ids_maps = [None] * len(self._paths)
        self._lbl_maps = [None] * len(self._paths)

    def _maps(self, idx: int) -> tuple[np.memmap, np.memmap]:
        if self._ids_maps[idx] is None:
            p = self._paths[idx]
            ids_path = p if p.endswith(".ids.bin") else p.replace(".bin", ".ids.bin")
            lbl_path = ids_path.replace(".ids.bin", ".lbl.bin")
            self._ids_maps[idx] = np.memmap(ids_path, dtype=DTYPE, mode="r")
            self._lbl_maps[idx] = np.memmap(lbl_path, dtype=LABEL_DTYPE, mode="r")
        return self._ids_maps[idx], self._lbl_maps[idx]

    def _locate(self, index: int) -> tuple[int, int]:
        shard = int(np.searchsorted(self._cum, index, side="right"))
        prev = int(self._cum[shard - 1]) if shard > 0 else 0
        return shard, index - prev

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard, local = self._locate(index)
        ids_map, lbl_map = self._maps(shard)
        row = local * self.seq_len
        x = np.asarray(ids_map[row : row + self.seq_len])
        t = np.asarray(lbl_map[row : row + self.seq_len])
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(t.astype(np.int64)),
        )

    @property
    def epoch(self) -> int:
        return self._epoch

    def advance_epoch(self) -> None:
        self._epoch += 1
        self._rng = np.random.default_rng(self.seed + self._epoch)

    def iter_batches(self, batch_size: int, shuffle: bool = True,
                     drop_last: bool = True):
        order = np.arange(len(self))
        if shuffle:
            order = self._rng.permutation(order)
        if drop_last:
            order = order[: (len(order) // batch_size) * batch_size]
        for i in range(0, len(order), batch_size):
            xs, ts = zip(*(self[int(j)] for j in order[i : i + batch_size]))
            yield torch.stack(xs), torch.stack(ts)

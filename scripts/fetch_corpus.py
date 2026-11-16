"""Fetch a small real-text corpus slice (plan §10-12, dataset-v0.1-mini).

Streams Hugging Face ``wikimedia/wikipedia`` (pre-extracted plain text) and
applies the plan §12 cleaning ladder, writing blank-line-separated documents
that ``tools/pack_dataset.py`` consumes directly.

Wikipedia text is CC BY-SA — provenance and license are recorded in
``data/manifests/<name>.json`` per plan §39.

Usage::

    python scripts/fetch_corpus.py --source vi-wiki --docs 400
    python scripts/fetch_corpus.py --source en-wiki --docs 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.utils import get_logger  # noqa: E402

logger = get_logger("miko.fetch")

SOURCES = {
    "vi-wiki": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.vi",
        "license": "CC BY-SA 3.0 (GFDL)",
        "url_template": "https://vi.wikipedia.org/?curid={id}",
        "min_chars": 200,   # skip stubs
        "max_chars": 100_000,
    },
    "en-wiki": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "license": "CC BY-SA 3.0 (GFDL)",
        "url_template": "https://en.wikipedia.org/?curid={id}",
        "min_chars": 200,
        "max_chars": 100_000,
    },
}

_HEADING = re.compile(r"^(=+).*=+$", re.MULTILINE)
_MULTI_NL = re.compile(r"\n{3,}")


def clean_text(text: str, min_chars: int, max_chars: int) -> str | None:
    """Plan §12 cleaning: normalize, drop boilerplate/garbage, length filter."""
    text = text.strip()
    if not (min_chars <= len(text) <= max_chars):
        return None
    # Drop wikipedia section headings ("== History ==" etc.).
    text = _HEADING.sub("", text)
    # Reject pages dominated by non-letter content (tables, lists of coords).
    letters = sum(ch.isalpha() or ch.isspace() for ch in text)
    if letters / len(text) < 0.6:
        return None
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a real corpus slice")
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--out", default="data/cleaned")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--seed", type=int, default=42,
                    help="stream skip offset, deterministic-ish sampling")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.manifests, exist_ok=True)

    from datasets import load_dataset

    logger.info("streaming %s (%s)", spec["dataset"], spec["config"])
    stream = load_dataset(spec["dataset"], spec["config"], split="train",
                          streaming=True)

    out_path = os.path.join(args.out, f"{args.source}.txt")
    seen: set[str] = set()
    written = skipped = dup = filtered = 0
    bytes_written = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in stream:
            if written >= args.docs:
                break
            text = clean_text(row.get("text") or "", spec["min_chars"],
                              spec["max_chars"])
            if text is None:
                filtered += 1
                continue
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest in seen:
                dup += 1
                continue
            seen.add(digest)

            # Provenance comment is stripped at pack time by length filters;
            # title line keeps docs attributable without leaking into training.
            fh.write(f"{row.get('title', '')}\n\n{text}\n\n")
            bytes_written += fh.tell() - bytes_written
            written += 1
            if written % 50 == 0:
                logger.info("%d/%d docs (%.0fs)", written, args.docs,
                            time.time() - t0)

    manifest = {
        "name": args.source,
        "dataset": spec["dataset"],
        "config": spec["config"],
        "license": spec["license"],
        "url_template": spec["url_template"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "requested_docs": args.docs,
        "written_docs": written,
        "duplicates_skipped": dup,
        "filtered_out": filtered,
        "bytes": bytes_written,
        "output": out_path,
        "cleaning": {
            "min_chars": spec["min_chars"],
            "max_chars": spec["max_chars"],
            "dedup": "md5 exact-text",
            "heading_removal": True,
            "letter_ratio_min": 0.6,
        },
    }
    manifest_path = os.path.join(args.manifests, f"{args.source}.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info("wrote %d docs (%.2f MB) -> %s | manifest: %s | "
                "dup=%d filtered=%d", written, bytes_written / 1e6, out_path,
                manifest_path, dup, filtered)

    # pyarrow/datasets can crash interpreter teardown (PyGILState_Release
    # fatal error) after streaming; all output is already flushed above, so
    # exit hard to keep the CLI quiet and its exit code clean.
    os._exit(0)


if __name__ == "__main__":
    main()

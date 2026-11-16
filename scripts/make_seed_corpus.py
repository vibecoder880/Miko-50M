"""Generate a deterministic seed corpus for smoke training (plan §22 Stage 0-1).

This is *placeholder* text: enough variety to exercise the tokenizer,
packing and training pipeline end-to-end. Real corpora (Vietnamese Wikipedia,
open code) replace it in the data phase and must be tracked in
``data/manifests/`` with provenance.

Usage::

    python scripts/make_seed_corpus.py --out data/raw/seed --docs 400
"""

from __future__ import annotations

import argparse
import os
import random

VI_SENTENCES = [
    "Miko là một mô hình ngôn ngữ nhỏ được huấn luyện từ con số không.",
    "Hôm nay trời đẹp, tôi đi dạo quanh công viên gần nhà.",
    "Việc học máy cần dữ liệu sạch và quy trình kiểm tra nghiêm ngặt.",
    "Mỗi buổi sáng tôi uống một ly cà phê và đọc vài trang sách.",
    "Tiếng Việt có nhiều thanh điệu và dấu sắc huyền hỏi ngã nặng.",
    "Mô hình học dự đoán từ tiếp theo dựa trên ngữ cảnh phía trước.",
    "Thủ đô của Việt Nam là Hà Nội, một thành phố ngàn năm văn hiến.",
    "Cần kiên trì khi huấn luyện mô hình nhỏ trên phần cứng hạn chế.",
    "Dữ liệu đầu vào được làm sạch, loại bỏ trùng lặp và lỗi mã hoá.",
    "Bài toán dự đoán token kế tiếp là nền tảng của mô hình ngôn ngữ.",
    "Tôi viết nhật ký huấn luyện để theo dõi tổn thất theo từng bước.",
    "Miko sẽ học cách gọi công cụ thông qua định dạng đặc biệt.",
    "Người dùng hỏi và trợ lý trả lời theo khuôn mẫu hội thoại.",
    "Bộ từ điển nhỏ giúp mô hình chạy nhẹ trên bộ nhớ hạn chế.",
    "Tôi thích lập trình bằng Python vì cú pháp rõ ràng dễ đọc.",
]

EN_SENTENCES = [
    "Miko is a small language model trained from scratch on a tiny CPU.",
    "Next token prediction is the core objective of the base model.",
    "Clean data matters more than raw dataset size for small models.",
    "The training loop logs loss, learning rate and tokens per second.",
    "Rotary position embeddings encode relative positions efficiently.",
    "A checkpoint stores weights, optimizer state and the schedule step.",
    "Small batches with gradient accumulation keep memory usage low.",
    "The byte level tokenizer handles Vietnamese diacritics losslessly.",
    "Validation loss guides when to stop or extend pretraining.",
    "Instruction tuning comes after the base model has learned language.",
    "Tool calling is learned as a format and executed by the runtime.",
    "Memory stays outside the model and is retrieved when relevant.",
    "The agent loop is bounded by max steps, timeouts and retries.",
    "Sequence packing removes padding waste from every training batch.",
    "Learning rate warmup prevents loss spikes at the start of training.",
]

CODE_SNIPPETS = [
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i, 'even')",
    "class Counter:\n    def __init__(self):\n        self.n = 0\n    def bump(self):\n        self.n += 1",
    "import json\ndata = json.loads('{\"name\": \"miko\", \"steps\": 100}')",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name LIMIT 10;",
    "curl -s https://api.example.com/v1/items | jq '.items[0].name'",
    "const sum = (...xs) => xs.reduce((a, b) => a + b, 0);\nconsole.log(sum(1, 2, 3));",
    "def save_checkpoint(model, path):\n    torch.save(model.state_dict(), path)",
    "# train a tiny epoch\nloss_log = []\nfor batch in loader:\n    loss = step(batch)\n    loss_log.append(loss.item())",
]


def make_doc(rng: random.Random, kind: str) -> str:
    if kind == "vi":
        n = rng.randint(3, 6)
        return " ".join(rng.choice(VI_SENTENCES) for _ in range(n))
    if kind == "en":
        n = rng.randint(3, 6)
        return " ".join(rng.choice(EN_SENTENCES) for _ in range(n))
    n = rng.randint(1, 2)
    return "\n\n".join(rng.choice(CODE_SNIPPETS) for _ in range(n))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate seed corpus")
    ap.add_argument("--out", default="data/raw/seed")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    # Mixture per plan §11 (rescaled for the smoke corpus).
    kinds = ["vi"] * 5 + ["en"] * 2 + ["code"] * 3
    path = os.path.join(args.out, "seed_corpus.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(args.docs):
            fh.write(make_doc(rng, kinds[i % len(kinds)]))
            fh.write("\n\n")

    size = os.path.getsize(path)
    print(f"wrote {args.docs} docs to {path} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

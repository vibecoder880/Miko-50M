"""Why do the SFT checkpoints ignore the user turn? (cycles v0.4-v4)

Four independent measurements per checkpoint, all cheap (no training):

  1. prefix parity   — does the generation path build the exact token prefix
                       the masked-SFT path trains on?
  2. logit pathology — embedding-row norms + output confidence on ordinary
                       text (a loss above ln(vocab) means confident-wrong,
                       not "forgotten")
  3. user-turn sensitivity — teacher-forced NLL of the SAME assistant answer
                       under the TRUE user turn vs a same-family SHUFFLED
                       one; equal NLL = the user turn is not read
  4. induction       — can the model copy a repeated random token sequence
                       at all (the mechanism every copy task needs)?

Usage::

    python scripts/diag_conditioning.py checkpoints_38m/latest \
        checkpoints_sft_v4/latest
"""
from __future__ import annotations

import os
import random
import sys

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from miko.chat_template import (  # noqa: E402
    IGNORE_INDEX,
    ChatMessage,
    encode_chat,
    encode_chat_masked,
)
from miko.checkpoint import load_model  # noqa: E402
from miko.dataset import ShardedTokenDataset, find_shards  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402

TOK_PATH = "tokenizer/real_v0.2/tokenizer.json"
BASE_VAL = "data/shards/val"
FAMILIES = ["echo", "disp2cmp", "transcribe", "restate", "restate-b"]


def load_jsonl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return [line for line in fh if line.strip() and not line.startswith("#")]


def prefix_parity(tok) -> None:
    from make_copy_gym import SYSTEM
    msgs = [ChatMessage("system", SYSTEM),
            ChatMessage("user", "Tính 4787 nhân 2458."),
            ChatMessage("assistant", "Để tính 4787 × 2458 chính xác, Miko dùng máy tính.")]
    gen = encode_chat(tok, msgs[:2], add_generation_prompt=True)
    ids, labels = encode_chat_masked(tok, msgs)
    print(f"  [1] prefix parity: gen==train prefix -> {gen == ids[:len(gen)]}"
          f"  (len {len(gen)} of {len(ids)} tokens)")
    print("      user turn tokens:",
          [tok.backend.id_to_token(i) for i in ids[:len(gen)]][-22:])


def emb_stats(tok, name, ckpt) -> None:
    emb = load_file(os.path.join(ROOT, ckpt, "model.safetensors"))["tok_emb.weight"].float()
    n = emb.norm(dim=1)
    order = n.argsort(descending=True)
    top = [(tok.backend.id_to_token(int(i)), round(float(n[i]), 2)) for i in order[:8]]
    specials = [round(float(n[i]), 2) for i in range(10)]
    print(f"  [2] {name} emb row norm: mean {n.mean():.3f} median {n.median():.3f} "
          f"max {n.max():.2f} | specials[0-9] {specials}")
    print(f"      top rows: {top}")


@torch.no_grad()
def text_confidence(model, windows) -> None:
    nll_s, maxp_s, ent_s, acc_s = [], [], [], []
    for i in range(0, len(windows), 4):
        x = torch.stack(windows[i:i + 4])
        logp = torch.log_softmax(model(x)[:, :-1].float(), dim=-1)
        tgt = x[:, 1:]
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        prob = logp.exp()
        nll_s.append(nll.mean().item())
        maxp_s.append(prob.max(dim=-1).values.mean().item())
        ent_s.append(-(prob * logp).sum(-1).mean().item())
        acc_s.append((logp.argmax(-1) == tgt).float().mean().item())
    k = len(nll_s)
    print(f"  [2] ordinary text: loss {sum(nll_s)/k:.2f} nats | mean max-prob "
          f"{sum(maxp_s)/k:.3f} | entropy {sum(ent_s)/k:.3f} | top-1 acc "
          f"{sum(acc_s)/k:.3f}   (uniform = {torch.log(torch.tensor(16000.)):.2f} nats)")


def as_messages(conv):
    return [ChatMessage(m["role"], m["content"]) for m in conv["messages"]]


@torch.no_grad()
def answer_nll(model, tok, msgs) -> float:
    ids, labels = encode_chat_masked(tok, msgs)
    x = torch.tensor([ids])
    logp = torch.log_softmax(model(x)[0, :-1].float(), dim=-1)
    tgt = torch.tensor(labels[1:])
    nll = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return nll[tgt != IGNORE_INDEX].mean().item()


def user_turn_sensitivity(model, tok) -> None:
    # Fresh numbers (seed 999 is in no training split), same templates.
    import json

    from make_copy_gym import build
    convs = build(random.Random(999), 8)  # 40 convs, family = index % 5
    msgs = [as_messages(c) for c in convs]
    true_s, shuf_s = [], []
    for i, m in enumerate(msgs):
        other = msgs[(i + 5) % len(msgs)]  # same family, different numbers
        shuffled = [m[0], other[1], m[2]]
        true_s.append(answer_nll(model, tok, m))
        shuf_s.append(answer_nll(model, tok, shuffled))
    n = len(msgs)
    gap = [s - t for t, s in zip(true_s, shuf_s)]
    print(f"  [3] answer NLL — true user turn {sum(true_s)/n:.2f} | shuffled "
          f"{sum(shuf_s)/n:.2f} | gap {sum(gap)/n:+.2f}"
          f"  (gap ~0 = user turn not read)")
    for f in range(5):
        t = sum(true_s[f::5]) / 8
        s = sum(shuf_s[f::5]) / 8
        print(f"      {FAMILIES[f]:<11} true {t:.2f} shuffled {s:.2f} gap {s - t:+.2f}")


@torch.no_grad()
def induction(model, bos, n=8, k=40, seed=0) -> None:
    g = torch.Generator().manual_seed(seed)
    first, second = [], []
    for _ in range(n):
        seq = torch.randint(10, 4000, (k,), generator=g)
        x = torch.cat([torch.tensor([bos]), seq, seq]).unsqueeze(0)
        pred = model(x)[0, :-1].argmax(-1)
        tgt = x[0, 1:]
        first.append((pred[1:k] == tgt[1:k]).float().mean().item())
        second.append((pred[k + 1:2 * k] == tgt[k + 1:2 * k]).float().mean().item())
    print(f"  [4] induction (random tokens) top-1 acc: first copy {sum(first)/n:.3f}"
          f" | repeated copy {sum(second)/n:.3f}")


@torch.no_grad()
def induction_text(model, windows, k=60) -> None:
    """Same test on natural text: a val window repeated twice."""
    first, second = [], []
    for w in windows[:6]:
        w = w[:k]
        x = torch.cat([w, w]).unsqueeze(0)
        pred = model(x)[0, :-1].argmax(-1)
        tgt = x[0, 1:]
        first.append((pred[1:k] == tgt[1:k]).float().mean().item())
        second.append((pred[k + 1:2 * k] == tgt[k + 1:2 * k]).float().mean().item())
    n = len(first)
    print(f"  [4] induction (natural text) top-1 acc: first copy {sum(first)/n:.3f}"
          f" | repeated copy {sum(second)/n:.3f}  (repeated >> first = can copy)")


def main() -> None:
    ckpts = sys.argv[1:] or ["checkpoints_38m/latest"]
    tok = MikoTokenizer.load(os.path.join(ROOT, TOK_PATH))
    prefix_parity(tok)

    ds = ShardedTokenDataset(find_shards(os.path.join(ROOT, BASE_VAL)), 128, seed=42)
    windows = [ds[i][0] for i in range(8)]

    for ckpt in ckpts:
        name = ckpt.rstrip("/").split("/")[-2] if ckpt.endswith("latest") or ckpt.endswith("latest/") else ckpt
        print(f"\n=== {ckpt} ===")
        emb_stats(tok, name, ckpt)
        model = load_model(os.path.join(ROOT, ckpt))
        model.eval()
        text_confidence(model, windows)
        induction(model, tok.bos_id)
        induction_text(model, windows)
        user_turn_sensitivity(model, tok)
        del model


if __name__ == "__main__":
    main()

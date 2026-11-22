---
title: "SFT v3 gates fail toàn diện — bài học lr-vs-volume, v4 volume-scale kích hoạt"
date: 2026-11-22
summary: "v3 (gym + 50/50, lr 1e-5, 3000 steps): val 21.9 nhưng surface thậm chí tệ hơn v2 ('Để tính' + garbage mọi ngữ cảnh, chat bị hijack); lr chậm = thiếu gradient movement; v4 = gym 5x + lr 2e-5 + 6000 steps đã chạy"
---

# SFT v3 gates fail toàn diện — bài học lr-vs-volume, v4 volume-scale kích hoạt

## What happened

SFT v3 hoàn tất (3000 steps, ~6h, val 332→21.9 phẳng cuối). Gates:

- chat_sft 0/14, chat 0/8 — REGRESSION so v2 (1/14)
- tool positives 0/12, negatives 4/4 (không đổi suốt 3 chu kỳ)
- vi 1/16 (= base), code 1/12 (= base)
- regression 2/9: chỉ 2 negatives pass vacuously; "Xin chào!" →
  "Để tính chính xác, northern northern..." — tool preface hijack
  MỌI ngữ cảnh (gym+tool chiếm 86% SFT tokens, chat 618 convs bị nuốt)
- base-val chưa đo lại (v3), nhưng chat regression cho thấy cùng pattern

## Đọc kết quả — tổng hợp 3 chu kỳ thành luật

| Chu kỳ | lr | SFT tokens | Kết cục surface |
|---|---|---|---|
| v0.4 | 3e-5 | 2M | scaffold đủ, copy thiếu |
| v2 | 3e-5 | 2M | scaffold đủ + restate, copy thiếu |
| v3 | 1e-5 | 2.7M | chỉ "Để tính" + garbage — THUA cả v2 |

Luật rút ra: (1) gradient movement phải đủ lớn (lr 1e-5 × 3000 quá chậm
để ổn định pattern — v2 ở lr 3e-5 học surface trong 800 steps);
(2) volume data unique là biến số chưa thử đúng mức — gym v3 chỉ 1.15
epoch ~770K tokens, trong khi cơ chế copy của model nhỏ cần lặp khối
lượng; (3) 618 chat convs (1.3% SFT tokens) không đủ giữ chat tier
khi tool-ish data áp đảo.

## Decision (ủy quyền "làm tiếp luôn")

Kích v4 volume-scale: gym v0.2 49977 convs (5×, số học đa dạng),
tool v0.4 5736, chat 618; mix 50/50 → 90737 windows (~11.6M tokens);
6000 steps (~0.53 epoch, fresh data mỗi step), lr 2e-5 (giữa v2/v3),
warmup 8%, fresh lineage. Đã chạy step 10, 313 tok/s.

Contingency nếu v4 vẫn fail: chấp nhận format-tier (option a) — 3 chu kỳ
kali hóa rằng 38M + corpus nhỏ không sinh được copy mechanism bằng SFT;
đbrica đó sẽ ghi vào MODEL_CARD như giới hạn kiến trúc/thang đo, và
(c) resume pretraining chỉ có giá trị nếu sau đó có SFT data 10x.

## Artifacts

- evals/results_sft_v3_part{1,2,3}.json, evals/probe_v3_2237.txt
- regression.jsonl viết lại đúng JSONL 1-object/line (loader chỉ đọc
  từng line — lỗi format khiến part3 fail lần đầu)
- scripts/eval_gates_v3.sh cần tách lệnh trên box bận (580s/call) —
  đã chạy gates bằng các lệnh rời

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.

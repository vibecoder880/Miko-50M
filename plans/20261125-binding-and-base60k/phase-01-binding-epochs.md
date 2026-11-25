# Phase 01 — (a) Binding epochs: continuation trên lineage gym

```yaml
phase: 01
files:
  - configs/train_binding_a.yaml      # mới
  - scripts/train_binding_a.sh        # mới (mirror v3 launcher, không seed — continuation!)
  - logs/train_binding_a.csv          # artifact
depends: [checkpoints_sft_v3/latest (step 3000, val 4.44)]
```

## Quy trình

1. `configs/train_binding_a.yaml`: giống v3 nhưng:
   - `data.train_dir: data/shards_sft_v4/train` (90737 w — gym + tool + chat
     đã trộn) + val `shards_sft_v4/val`
   - `out_dir: checkpoints_binding_a`, log riêng
   - `max_steps: 6000` (continuation đếm tiếp từ 3000 → 3000 bước mới),
     `lr: 2.0e-5`, `warmup_pct: 0.05`, `milestone_every: 500` — sớm để
     có checkpoint binding tại 3500/4000/4500
   - `eval_every: 250`
2. `scripts/train_binding_a.sh`: **KHÔNG seeding block** — đây là
   continuation, `--resume auto` phải load `checkpoints_sft_v3/latest`
   vào lineage mới: copy `checkpoints_sft_v3/latest` →
   `checkpoints_binding_a/latest` một lần đầu (giống seeding v2 nhưng
   nguồn là gym ckpt), sau đó auto-restart + disk guard như launcher chuẩn.
3. Launch + theo dõi val tool mỗi 250 steps.

## Vì sao continuation chứ không fresh

Gym cycle đã trả giá 3000 steps để có: scaffold sạch, eos đúng, val 4.44.
Fresh sẽ quét hết và học lại từ đầu. Continuation giữ mọi thứ đã học và
chỉ thêm gradient binding trên data tool-heavy.

## Chấp nhận

- [ ] resume log ghi "resumed ... at step 3000" (KHÔNG phải 0 — nếu 0
      nghĩa là copy seeding sai và phải dừng ngay)
- [ ] val giảm đơn điệu ít nhất 2 mốc đầu (3250, 3500)
- [ ] milestone 3500 + 4000 + 4500 tồn tại (checkpoint sớm để so binding)
- [ ] chạy đủ tới 6000 hoặc early-stop có chủ đích nếu val tăng 3 mốc liền

## Cần đo thêm trong phase này (đổ vào phase 02)

Vì train_dir v4 đã chứa gym windows, không cần data mới. Nhưng để đo
binding cụ thể: sau mỗi milestone chạy 1 probe "Tính 4787 nhân 2458."
và lưu vào `evals/probe_binding_a_step{N}.txt` — xem expression dần
chuyển từ template rác sang số đúng.

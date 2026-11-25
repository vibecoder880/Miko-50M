# Phase 02 — (a) Finalize: gates + bundle + quyết định (b)

```yaml
phase: 02
files:
  - evals/results_binding_a_*.json   # artifacts
  - dist/miko-binding-a              # bundle nếu đạt
depends: [phase-01]
```

## Quy trình

1. Probe cuối (`scripts/probe_copy.py`, 56 tokens, 6 prompts chuẩn).
2. Gates 56-token: `tool_basic` (đếm positives riêng), `chat_sft` +
   `chat_basic` + `regression`, so sánh v0.7 (chat_sft 2/14).
3. **Gates quyết định**:
   - positives > 0 → `export_bundle.py` → `dist/miko-binding-a` →
     codex demo → MODEL_CARD **v0.8** (đóng plan, (b) chỉ còn optional)
   - positives = 0 NHƯNG expression trong generations chứa ≥ 1 số đúng
     từ câu hỏi (binding "bán phần") → ghi nhận + kích (b) với bằng chứng
   - positives = 0, expression vẫn template rác → kích (b) (phase 03)
4. Journal + push.

## Chấp nhận

- [ ] 3 file results JSON + probe cuối được commit
- [ ] Bảng so sánh v0.7 → binding-a trong MODEL_CARD (dù đạt hay fail)
- [ ] Quyết định (b) kích hoạt hay không được ghi rõ + lý do bằng số

## Kịch bản thất bại

| Hiện tượng | Hành động |
|---|---|
| val tăng liên tục từ rất sớm (overfit data v4) | dừng tại milestone, probe milestone đó — có thể 3500 đã là điểm ngọt |
| mọi milestone đều scaffold sạch nhưng số sai hoàn toàn | (b) kích hoạt — bằng chứng copy cần base mạnh hơn |

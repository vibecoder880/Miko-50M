"""Fork diagnostic: does the model predict id-8 <tool_call> on the EXACT
training distribution (a memorized seed conversation)?

- If P(id8) is high on the memorized prefix but low on new questions ->
  generalization gap (more data/steps help).
- If P(id8) is low even on the memorized prefix -> SFT encoding/labels bug
  (the model was never trained to emit id 8 — more data won't help).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from miko.chat_template import ChatMessage, encode_chat  # noqa: E402
from miko.checkpoint import load_model  # noqa: E402
from miko.tokenizer import MikoTokenizer  # noqa: E402

model = load_model("checkpoints_sft_38m/latest")
model.eval()
tok = MikoTokenizer.load("tokenizer/real_v0.2/tokenizer.json")


def next_topk(ids, k=6):
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, k)
    return [(int(i), float(p)) for p, i in zip(top.values, top.indices)]


def show(tag, ids, k=6):
    preds = next_topk(ids, k)
    pretty = [(tok.backend.id_to_token(i), round(p, 3)) for i, p in preds]
    print(f"{tag}: top-{k} = {pretty}")
    probs = None
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    probs = torch.softmax(logits[0, -1], dim=-1)
    print(f"    P(id8)={float(probs[8]):.4f}  P(eos)={float(probs[tok.eos_id]):.4f}")


# --- Case A: memorized seed conversation, cut right before <tool_call> ---
with open("data/sft/tool_seed_v0.2.jsonl", encoding="utf-8") as fh:
    rows = [json.loads(l) for l in fh if not l.startswith("#")]
conv = rows[0]
sys_text = conv["messages"][0]["content"]
user_text = conv["messages"][1]["content"]
asst_text = conv["messages"][2]["content"]
print("seed user:", user_text[:70])
print("seed assistant:", asst_text[:70], "...")

# Prefix: system + user + assistant header (the eval-style generation point)
msgs_a = [ChatMessage("system", sys_text), ChatMessage("user", user_text)]
ids_a = encode_chat(tok, msgs_a, add_generation_prompt=True)
show("A header->? (new-style prompt)", ids_a)

# Prefix: the full seed content up to and including the id-8 token, i.e.
# exactly the sequence seen in training; check what precedes/follows.
full = encode_chat(tok, [ChatMessage("system", sys_text),
                         ChatMessage("user", user_text),
                         ChatMessage("assistant", asst_text)])
i8 = full.index(8)
print("\nB training sequence: id8 at position", i8, "of", len(full))
print("   context before id8:", repr(tok.decode(full[max(0, i8-6):i8],
      skip_special_tokens=False)))
# Cut BEFORE id8 and ask for the next token (training says: id8)
show("B pre-id8 -> ?", full[:i8])

# Cut INSIDE the call: after '["expression": "' check the digit predictions
iexpr = tok.encode('"expression": "', add_special_tokens=False)
cut = i8 + 2 + len(iexpr)  # after <tool_call>\n + '{\"name\"...\": \"' — approx
cut = full.index(tok.encode('"expression": "',
                            add_special_tokens=False)[0], i8) + len(iexpr)
show("C after-expression-quote -> ?", full[:cut])
print("   training target there:", repr(tok.decode(full[cut:cut+8],
      skip_special_tokens=False)))

# --- Case D: a NEW question (eval-style) with the same system ---
msgs_d = [ChatMessage("system", sys_text),
          ChatMessage("user", "Tính tổng 45678 và 12345.")]
ids_d = encode_chat(tok, msgs_d, add_generation_prompt=True)
show("D new-question header-> ?", ids_d)

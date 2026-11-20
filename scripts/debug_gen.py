import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch

from miko.checkpoint import load_model
from miko.generation import GenerationConfig, generate
from miko.chat_template import ChatMessage, encode_chat
from miko.tokenizer import MikoTokenizer
from miko.utils import seed_everything

seed_everything(1)
model = load_model("checkpoints_sft_38m/latest")
model.eval()
tok = MikoTokenizer.load("tokenizer/real_v0.2/tokenizer.json")

msgs = [ChatMessage("system", "Bạn là Miko."), ChatMessage("user", "Xin chào!")]
ids = encode_chat(tok, msgs, add_generation_prompt=True)
print("prompt tail ids:", ids[-6:])
print("prompt tail text:", repr(tok.decode(ids[-6:], skip_special_tokens=False)))

gen_cfg = GenerationConfig(max_new_tokens=30, greedy=True, eos_token_id=tok.eos_id)
out = generate(model, torch.tensor([ids]), cfg=gen_cfg)
new = out[0, len(ids):].tolist()
print("generated raw:", repr(tok.decode(new, skip_special_tokens=False))[:300])
print("has eos:", tok.eos_id in new)

# Compare: what does the SFT training data look like around <assistant>?
import json
with open("data/sft/chat_seed_v0.1.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        row = json.loads(line)
        break
print("train example roles:", [m["role"] for m in row["messages"]])
os._exit(0)

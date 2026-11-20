"""Diagnostic: does the tokenizer map <tool_call> in content text to id 8?

Fork between (a) SFT data encoding bug (specials never become id 8 -> the
model can never learn to emit the real special token) and (b) model simply
under-trained on the token.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miko.tokenizer import MikoTokenizer  # noqa: E402

tok = MikoTokenizer.load("tokenizer/real_v0.2/tokenizer.json")

s = ('Để tính chính xác.\n<tool_call>\n'
     '{"name": "calculator", "arguments": {"expression": "509*4606"}}\n'
     '</tool_call>')
ids = tok.encode(s, add_special_tokens=False)
print("assistant content ids:", ids)
print("id 8 (<tool_call>) present:", 8 in ids)
print("roundtrip ok:",
      tok.decode(ids, skip_special_tokens=False) == s)
print()
print("encode('<tool_call>') alone:",
      tok.encode("<tool_call>", add_special_tokens=False))
print("encode('</tool_call>') alone:",
      tok.encode("</tool_call>", add_special_tokens=False))

# What does the chat template do with a tool-role message (the closing tag)?
from miko.chat_template import ChatMessage, encode_chat  # noqa: E402

msgs = [
    ChatMessage("system", "Bạn là Miko."),
    ChatMessage("user", "Tính 509 × 4606."),
    ChatMessage("assistant", s),
    ChatMessage("user", "Kết quả là 234154."),
]
ids2 = encode_chat(tok, msgs)
print()
print("full conv has id 8:", 8 in ids2, "| id 4 (<system>):", 4 in ids2,
      "| id 5 (<user>):", 5 in ids2)

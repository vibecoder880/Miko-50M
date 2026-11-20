"""Patch eval JSONL rows with the system prompts the SFT data trained on.

The SFT corpus always conditions the assistant on a system turn; eval items
without one are OOD at scoring time (the model then falls back to template
babble). tool_basic gets the tool-instruction system from tool_seed_v0.1;
chat_sft rows without a system get the chat-seed system.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def first_seed_system(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            row = json.loads(line)
            return row["messages"][0]["content"]
    raise ValueError(f"no seed rows in {path}")


TOOL_SYSTEM = first_seed_system("data/sft/tool_seed_v0.1.jsonl")
CHAT_SYSTEM = first_seed_system("data/sft/chat_seed_v0.1.jsonl")

JOBS = {
    "evals/tool_basic.jsonl": TOOL_SYSTEM,
    "evals/chat_sft.jsonl": CHAT_SYSTEM,
}

for path, system in JOBS.items():
    out_lines = []
    n_added = n_kept = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                out_lines.append(line.rstrip("\n"))
                continue
            row = json.loads(line)
            if not row.get("system"):
                row["system"] = system
                n_added += 1
            else:
                n_kept += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")
    os.replace(tmp, path)
    print(f"{path}: system added to {n_added} rows ({n_kept} already had one)")

print("TOOL_SYSTEM[:80]:", TOOL_SYSTEM[:80])
print("CHAT_SYSTEM[:80]:", CHAT_SYSTEM[:80])

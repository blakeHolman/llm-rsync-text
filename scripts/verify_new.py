import re
import json
from pathlib import Path

# Input: your existing OLD/NEW JSONL (one object per line)
IN_JSONL  = Path("data/chunks.jsonl")

# Output: JSONL containing ONLY {id, NEW}
OUT_JSONL = Path("data/new_only.jsonl")

# Patterns that must NOT appear in NEW
FORBIDDEN = [
    re.compile(r"\bDoD\b"),
    re.compile(r"\bDOD\b"),
    re.compile(r"\bDepartment of Defense\b"),
    re.compile(r"\bDEPARTMENT OF DEFENSE\b"),
]

def forbidden_counts(s: str):
    counts = {pat.pattern: len(pat.findall(s)) for pat in FORBIDDEN}
    total = sum(counts.values())
    return total, {k: v for k, v in counts.items() if v}

n = 0
bad = 0

with IN_JSONL.open("r", encoding="utf-8") as fin, OUT_JSONL.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("id", f"row_{n:04d}")
        new = obj["NEW"]

        total, details = forbidden_counts(new)
        if total:
            bad += 1
            print(f"[WARN] {cid}: leftover_forbidden={total} {details}")

        fout.write(json.dumps({"id": cid, "NEW": new}, ensure_ascii=False) + "\n")
        n += 1

print(f"Wrote {n} NEW-only records to {OUT_JSONL} (chunks with leftovers: {bad})")


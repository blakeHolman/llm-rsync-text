#!/usr/bin/env python3
# scripts/pick_best_example.py

import json
import difflib
from math import log1p
from typing import Optional, Tuple, Any

def pick_best_example(jsonl_path, pick_best, tokenizer=None, max_tokens=600, min_tokens=80):
    """
    Pick a single exemplar (prev_old, prev_new) that is:
      - small (<= max_tokens if tokenizer provided; else by chars)
      - has high relative change (low similarity)
    Works for ANY dataset with OLD/NEW fields.
    If pick_best is fasle, pick first pair.

    Returns: (prev_old, prev_new, id)
    """
    first_valid = None

    best = None
    best_score = -1.0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            old = rec.get("OLD")
            new = rec.get("NEW")
            if not old or not new:
                continue

            # Token-length gating (recommended)
            if tokenizer is not None:
                tok_len = len(tokenizer(old + "\n" + new).input_ids)
                if tok_len > max_tokens or tok_len < min_tokens:
                    continue
                size = tok_len
            else:
                size = len(old) + len(new)
                if size <= 0:
                    continue

            # Track the first valid example so we can return it when pick_best=False
            if first_valid is None:
                first_valid = (old, new, rec.get("id"))

            if not pick_best:
                return first_valid

            # Similarity estimate (character-level, generic)
            sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
            sim = sm.ratio()               # 1.0 identical
            change_ratio = 1.0 - sim       # 0.0 identical, 1.0 very different

            # Score: prefer high change_ratio, prefer smaller size.
            score = change_ratio / log1p(size)

            if score > best_score:
                best_score = score
                best = (old, new, rec.get("id"))

    # If we were asked not to pick_best and never found a valid pair:
    if not pick_best and first_valid is not None:
        return first_valid

    if best is None:
        raise RuntimeError("No suitable exemplar found; relax max_tokens/min_tokens.")
    return best

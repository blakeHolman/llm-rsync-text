#!/usr/bin/env python3
# scripts/pick_best_example.py

import json
import difflib
from math import log1p
from typing import Optional, Tuple, Any
import re
from collections import Counter

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


def infer_substitutions(prev_old: str, prev_new: str, top_k: int = 12):
    """
    Generic: infer likely substitutions from an (OLD, NEW) exemplar by looking
    at 'replace' spans and collecting repeated small phrase swaps.
    """
    sm = difflib.SequenceMatcher(a=prev_old, b=prev_new, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        a = prev_old[i1:i2].strip()
        b = prev_new[j1:j2].strip()
        # ignore huge blocks / whitespace-only
        if not a or not b:
            continue
        if len(a) > 80 or len(b) > 80:
            continue
        # normalize internal whitespace for nicer prompting
        a_norm = re.sub(r"\s+", " ", a)
        b_norm = re.sub(r"\s+", " ", b)
        pairs.append((a_norm, b_norm))

    # rank by frequency (repeated replacements are likely "the rule")
    ctr = Counter(pairs)
    best = [p for p, _ in ctr.most_common(top_k)]
    return best

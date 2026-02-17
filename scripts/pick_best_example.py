#!/usr/bin/env python3
# scripts/pick_best_example.py

import json
import difflib
from math import log1p
from typing import Optional, Tuple, Any
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]|\s+", re.UNICODE)

def _substitution_coverage_score(old: str, new: str) -> float:
    """
    Reward exemplars that have both long-form and short-form (acronym) changes.
    High variance in changed span lengths = good exemplar.
    """
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    changed_spans = [
        old[i1:i2]
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag == "replace" and i2 - i1 > 1
    ]
    if not changed_spans:
        return 0.0

    lengths = [len(s) for s in changed_spans]
    avg_len = sum(lengths) / len(lengths)
    diversity = len(set(changed_spans))

    # Variance in lengths rewards having both short (acronyms) and long (full phrases)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

    return diversity * (1.0 + variance ** 0.5)

def pick_best_example(jsonl_path, pick_best, tokenizer=None, max_tokens=1500, min_tokens=80):
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

            if tokenizer is not None:
                tok_len = len(tokenizer(old + "\n" + new).input_ids)
                if tok_len > max_tokens or tok_len < min_tokens:
                    continue
                size = tok_len
            else:
                size = len(old) + len(new)
                if size <= 0:
                    continue

            if first_valid is None:
                first_valid = (old, new, rec.get("id"))

            if not pick_best:
                return first_valid

            sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
            sim = sm.ratio()
            change_ratio = 1.0 - sim

            coverage = _substitution_coverage_score(old, new)

            # Combine: change_ratio rewards high delta, coverage rewards
            # longer/more diverse changed spans, size penalty keeps it short
            score = (change_ratio * coverage) / log1p(size)

            if score > best_score:
                best_score = score
                best = (old, new, rec.get("id"))

    if not pick_best and first_valid is not None:
        return first_valid
    if best is None:
        raise RuntimeError("No suitable exemplar found; relax max_tokens/min_tokens.")
    return best


def infer_substitutions(prev_old: str, prev_new: str, top_k: int = 20,
                        max_span_tokens: int = 6, min_len_chars: int = 2):
    """
    Infer likely substitution pairs from one exemplar (prev_old -> prev_new).
    Works generically for many datasets, and avoids char-level junk like D->W.
    """
    a = _tokenize_preserve(prev_old)
    b = _tokenize_preserve(prev_new)

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue

        span_a = a[i1:i2]
        span_b = b[j1:j2]

        # Remove whitespace-only tokens from candidate phrases
        span_a_nw = [t for t in span_a if _is_meaningful_token(t)]
        span_b_nw = [t for t in span_b if _is_meaningful_token(t)]

        if not span_a_nw or not span_b_nw:
            continue
        if len(span_a_nw) > max_span_tokens or len(span_b_nw) > max_span_tokens:
            continue

        phrase_a = "".join(span_a_nw).strip()
        phrase_b = "".join(span_b_nw).strip()

        # Filter out tiny / single-letter / trivial “D->W”
        if len(phrase_a) < min_len_chars or len(phrase_b) < min_len_chars:
            continue
        if len(phrase_a) == 1 or len(phrase_b) == 1:
            continue

        pairs.append((phrase_a, phrase_b))

    ctr = Counter(pairs)
    return [p for p, _ in ctr.most_common(top_k)]


def _tokenize_preserve(text: str):
    """
    Tokenize into: words/numbers, punctuation, and whitespace tokens.
    Keeping whitespace tokens lets the matcher align better, but we’ll filter
    them out when building substitution candidates.
    """
    return _TOKEN_RE.findall(text)

def _is_meaningful_token(tok: str) -> bool:
    # reject whitespace-only
    if tok.isspace():
        return False
    # reject pure punctuation (single punctuation token)
    if re.fullmatch(r"[^\w\s]+", tok):
        return False
    return True
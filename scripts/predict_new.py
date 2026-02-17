#!/usr/bin/env python3
# scripts/predict_new.py

import argparse, os, json, sys, csv, base64, copy, difflib, re
from pathlib import Path
from residuals import get_residual
from pick_best_example import pick_best_example, infer_substitutions

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Phi-3 uses custom code, so trust_remote_code is recommended
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Causal LMs often have no pad token; use EOS as pad if needed
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

# Use half precision on GPU to save VRAM; full precision on CPU
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,  # stream weights to reduce peak RAM
)

MODEL = MODEL.to(DEVICE)

MODEL.eval()

METRICS_FILE = "work/metrics.csv"
RESIDUAL_FILE = "work/residuals.jsonl"

PREFIX_TEXT = None
PREFIX_KV = None
PREFIX_TOKENS = None
PREFIX_MASK = None

# Open old, new dataset. Pass old to LLM
def _open_data(path, ex_id, len_prompt=False, stop_after=sys.maxsize):
    results = []
    residuals = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            # Check we need to stop early
            if idx > stop_after:
                break
            
            # Read line and verify
            line = line.strip()
            if not line:
                continue

            # Try load line
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON on line {idx}: {e}")
                continue

            # Check to ensure we don't have the same prompt 
            file_id = rec.get("id")
            if file_id == ex_id:
                continue

            old = rec.get("OLD")
            new = rec.get("NEW")

            if old is None or new is None:
                print(f"Skipping line {idx}: missing 'old' or 'new'")
                continue

            target_len = len(TOKENIZER(new).input_ids) if len_prompt else None
            predicted = predict(old, target_len=target_len)
            
            """
            print("==============old=============")
            print(old)
            print("===============pred============")
            print(predicted)
            print("==============actual=============")
            """
            print(predicted)

            metrics, residual = _compare(old, predicted, new)

            row = {
                "id": file_id,
                "old_len": len(old),
                "new_len": len(new),
                "pred_len": len(predicted) if predicted is not None else 0,
                "residual_len": len(residual),
                **(metrics or {}),
            }
            results.append(row)

            residuals.append({
                "id": file_id,
                "residual": base64.b64encode(residual).decode("ascii"),
            })

    _save_results(results)
    _save_residual(residuals)


def _chat_user(content: str) -> str:
    return f"<|user|>\n{content}<|end|>\n"

def _chat_system(content: str) -> str:
    return f"<|system|>\n{content}<|end|>\n"

def _chat_assistant(content: str) -> str:
    return f"<|assistant|>\n{content}<|end|>\n"

def _chat_assistant_gen() -> str:
    return "<|assistant|>\n"

def build_rule_extraction_prompt(prev_old: str, prev_new: str) -> str:
    system = (
        "You are a substitution rule extractor.\n"
        "Given a BEFORE and AFTER document, output ONLY the substitution rules "
        "that transform BEFORE into AFTER.\n"
        "\n"
        "Requirements:\n"
        "- For each changed phrase, list ALL case variants that appear in the text\n"
        "- NEVER output a rule where before and after are identical\n"
        "- Only include rules evidenced by the diff\n"
        "- Order from most specific (longest) to least specific\n"
        "- Format each rule exactly as: \"BEFORE\" -> \"AFTER\"\n"
        "- Output nothing else\n"
    )


    user = (
        f"BEFORE:\n{prev_old}\n\n"
        f"AFTER:\n{prev_new}\n\n"
        "Extract substitution rules:"
    )

    return (
        _chat_system(system)
        + _chat_user(user)
        + _chat_assistant_gen()
    )

def parse_rules(llm_output: str) -> list[tuple[str, str]]:
    """
    Parse lines of the form: "BEFORE" -> "AFTER"
    Returns list of (before, after) tuples, longest first.
    """
    rules = []
    for line in llm_output.strip().splitlines():
        line = line.strip()
        # Match: "X" -> "Y"  or  X -> Y  (with or without quotes)
        m = re.match(r'^"?([^"]+)"?\s*->\s*"?([^"]+)"?$', line)
        if m:
            before = m.group(1).strip()
            after  = m.group(2).strip()
            if before and after:
                rules.append((before, after))

    # Sort longest source phrase first — prevents partial matches
    # e.g. "Department of Defense" must be applied before "Defense"
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules

def build_rewrite_prompt(rules: list[tuple[str, str]], old: str) -> str:
    rules_str = "\n".join(f'"{a}" -> "{b}"' for a, b in rules)

    system = (
        "You are a deterministic rewrite engine.\n"
        "Apply ONLY the substitutions listed below.\n"
        "Every character not covered by a rule must be copied exactly.\n"
        "DO NOT paraphrase, reorder, summarize, or add content.\n"
        "\n"
        "Substitutions (apply every occurrence, most specific first):\n"
        f"{rules_str}\n"
    )

    return (
        _chat_system(system)
        + _chat_user(old)
        + _chat_assistant_gen()
    )

def apply_rules_deterministic(text: str, rules: list[tuple[str, str]]) -> str:
    """Fast deterministic pre-pass before LLM. Longest rules applied first."""
    for before, after in rules:  # already sorted longest-first
        text = text.replace(before, after)
    return text

def init_prefix_kv(prev_old, prev_new):
    global PREFIX_TEXT

    # Stage 1: extract rules via LLM
    extraction_prompt = build_rule_extraction_prompt(prev_old, prev_new)
    enc = TOKENIZER(extraction_prompt, return_tensors="pt",
                    add_special_tokens=False).to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            **enc,
            max_new_tokens=256,   # rules list is short
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    prompt_len = enc["input_ids"].size(1)
    raw_rules = TOKENIZER.decode(out[0, prompt_len:], skip_special_tokens=True)
    print(f"[Stage 1] Extracted rules:\n{raw_rules}")

    rules = parse_rules(raw_rules)
    print(f"[Stage 1] Parsed {len(rules)} rules: {rules}")

    # Validation: apply rules back to prev_old, check similarity to prev_new
    test_output = apply_rules_deterministic(prev_old, rules)
    sim = difflib.SequenceMatcher(a=test_output, b=prev_new).ratio()
    print(f"[Stage 1] Rule validation similarity: {sim:.3f}")
    if sim < 0.85:
        print("[Stage 1] Warning: low similarity, rules may be incomplete. "
              "Falling back to infer_substitutions.")
        rules = infer_substitutions(prev_old, prev_new)

    # Stage 2 prompt is built per-document in predict(), using these rules
    PREFIX_TEXT = rules  # store rules, not a fixed prompt


# Given old data, predict new
def predict(old, target_len=None):
    if PREFIX_TEXT is None:
        raise RuntimeError("Call init_prefix_kv() first.")

    rules = PREFIX_TEXT  # list of (before, after) tuples

    # Deterministic pre-pass — handles the easy cases instantly, no LLM cost
    pre_applied = apply_rules_deterministic(old, rules)

    # If deterministic pass already produced a perfect result, skip LLM entirely
    # (won't happen often but worth short-circuiting)
    if pre_applied == old and not rules:
        return old

    # Build Stage 2 prompt: LLM only needs to fix what deterministic pass missed
    full_text = build_rewrite_prompt(rules, pre_applied)

    max_ctx = _model_max_ctx()

    # Estimate decode length based on pre_applied (closer to final than raw old)
    if target_len is not None:
        approx = target_len + 64
    else:
        approx = len(TOKENIZER(pre_applied).input_ids) + 64
    max_new = max(32, approx)

    enc = TOKENIZER(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,  # chat tokens already embedded by build_rewrite_prompt
        truncation=True,
        max_length=max_ctx,
    ).to(DEVICE)

    prompt_len = enc["input_ids"].size(1)
    available = max_ctx - prompt_len
    if available <= 0:
        print("Warning: no room left for generation; returning pre-applied deterministic result.")
        return pre_applied  # still useful, better than empty string

    max_new = min(max_new, available)

    with torch.no_grad():
        out = MODEL.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    pred = TOKENIZER.decode(out[0, prompt_len:], skip_special_tokens=True)
    return pred


def _model_max_ctx():
    """Get a reasonable max context length for this model."""
    max_ctx = getattr(MODEL.config, "max_position_embeddings", None)
    if max_ctx is None or max_ctx <= 0:
        max_ctx = getattr(TOKENIZER, "model_max_length", 2048)
    if max_ctx is None or max_ctx <= 0:
        max_ctx = 2048
    return int(max_ctx)
    


# Compare predicted data to actual "new"
def _compare(old, predicted, actual_new):
    """
    Compute metrics for LLM-based compression.

    - Baseline delta: old -> actual_new      (true_delta_bytes)
    - LLM residual:   predicted -> actual_new (llm_residual_bytes, residual_str)

    Returns (metrics_dict, residual_string_pred_to_new).
    """
    if predicted is None:
        predicted = ""

    new_bytes  = len(actual_new.encode("utf-8"))
    pred_bytes = len(predicted.encode("utf-8"))

    # Baseline: old -> new as COPY/LIT patch too (optional)
    #true_patch = get_residual(actual_new, old)
    #true_delta_bytes = len(true_patch)

    # LLM residual: pred -> new patch
    llm_patch = get_residual(actual_new, predicted)
    llm_residual_bytes = len(llm_patch)

    # Optional: old -> pred patch
    #model_patch = get_residual(predicted, old)
    #model_delta_bytes = len(model_patch)

    #percent_predicted = 1.0 - (llm_residual_bytes / new_bytes) if new_bytes > 0 else 0.0

    metrics = {
        "new_bytes": new_bytes,
        "pred_bytes": pred_bytes,
        #"true_delta_bytes(new-old)": true_delta_bytes,
        "llm_residual_bytes(new-pred)": llm_residual_bytes,
        #"model_delta_bytes(pred-old)": model_delta_bytes,
        #"percent_predicted": percent_predicted,
        "exact_match": int(predicted == actual_new),
    }

    return metrics, llm_patch


# Write metrics results to CSV
def _save_results(results):
    if not results:
        print("No metrics to save.")
        return

    fieldnames = sorted(results[0].keys())

    metrics_path = Path(METRICS_FILE)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # create dir if missing

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote metrics to {metrics_path}")


def _save_residual(residuals):
    if not residuals:
        print("No residuals to save.")
        return

    residual_path = Path(RESIDUAL_FILE)
    residual_path.parent.mkdir(parents=True, exist_ok=True)

    with residual_path.open("w", encoding="utf-8") as f:
        for rec in residuals:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote residuals to {residual_path}")
    

def main():
    ap = argparse.ArgumentParser(description="Predict \"new\" output from old data")
    ap.add_argument("--data_file", required=True, help="JSONL of {old,new,} pairs")
    ap.add_argument("--metrics", required=False, help="File to save metric results")
    ap.add_argument("--residuals", required=False, help="File to save residuals")
    ap.add_argument("--add_len", action="store_true", help="Provide \"new\" length to prompt for prediction")
    ap.add_argument("--best_example", action="store_true", help="Finds best OLD -> NEW pair for prompt generation")
    ap.add_argument("--stop_after", type=int, default=sys.maxsize, help="Number of lines to read")
    args = ap.parse_args()
    
    # Check if dataset is valid
    data = args.data_file
    if not os.path.isfile(data):
        print(f'Invalid data file path: {data}')
        return
    
    # Get output paths (optional)
    global METRICS_FILE
    global RESIDUAL_FILE
    if args.metrics is not None:
        METRICS_FILE = args.metrics
    if args.residuals is not None:
        RESIDUAL_FILE = args.residuals

    # Get flag for prompting length
    len_prompt = args.add_len

    # Get flag for prompt
    use_best = args.best_example
    # Get best or first example
    prev_old, prev_new, ex_id = pick_best_example(
        data,
        pick_best=use_best,
        tokenizer=TOKENIZER,
    )
    print(f"Best example id = {ex_id}")
    # Do prefill for first part of prompt
    init_prefix_kv(prev_old, prev_new)

    # Get stop after var
    stop_after = args.stop_after
    
    # Open dataset and run LLM
    _open_data(data, ex_id, len_prompt, stop_after)

if __name__ == "__main__":
    main()
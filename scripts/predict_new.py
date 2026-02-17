#!/usr/bin/env python3
# scripts/predict_new.py

import argparse, os, json, sys, csv, base64, copy
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

def _build_prefix(prev_old, prev_new, subs):
    """
    Build the static prompt prefix using Phi-3 chat formatting.
    This returns a STRING (not tokenized yet) that ends AFTER the exemplar assistant output.
    """
    subs_str = "\n".join([f"- {a} -> {b}" for a, b in subs])

    system = (
        "You are a semantic substitution engine.\n"
        "Change only the words that differ from the example.\n"
        "You are responsible for changing all words like DoD to DoW, Department of Defense to Department of War, etc.\n"
        "Output ONLY the rewritten text.\n\n"
    )

    # One exemplar turn (user->assistant)
    return (
        _chat_system(system)
        + _chat_user("Example: \n" + prev_old)
        + _chat_assistant(prev_new)
    )


def init_prefix_kv(prev_old, prev_new):
    """
    Build and store the static prefix prompt once.

    NOTE: If you are using the 'full_text = PREFIX_TEXT + user(old) + assistant_gen()'
    approach inside predict(), you do NOT need to precompute or store KV caches here.
    HuggingFace generate() will use KV cache internally during decoding.
    """
    global PREFIX_TEXT, PREFIX_KV, PREFIX_TOKENS, PREFIX_MASK

    subs = infer_substitutions(prev_old, prev_new)
    print(f"Subs: {subs}")

    # Store the reusable prefix as TEXT only
    PREFIX_TEXT = _build_prefix(prev_old, prev_new, subs)

    # These are unused in the full_text+generate approach; keep as None to avoid confusion
    PREFIX_KV = None
    PREFIX_TOKENS = None
    PREFIX_MASK = None

    # Optional: show how big the prefix is (useful for debugging context limits)
    enc = TOKENIZER(
        PREFIX_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=_model_max_ctx(),
    )
    print(f"[prefix] prefix tokens: {enc['input_ids'].size(1)}")


# Given old data, predict new
def predict(old, target_len=None):
    if PREFIX_TEXT is None:
        raise RuntimeError("Prefix not initialized. Call init_prefix_kv() first.")

    max_ctx = _model_max_ctx()

    # Decide decode length
    # (same logic you already had)
    if target_len is not None:
        approx = target_len + 64
    else:
        approx = len(TOKENIZER(old).input_ids) + 64
    max_new = max(32, approx)

    # Build one full prompt and let generate() handle cache internally.
    full_text = PREFIX_TEXT + _chat_user("Text to edit: \n" + old) + _chat_assistant_gen()

    #print(f"Prompt:\n {full_text}")

    enc = TOKENIZER(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,     # IMPORTANT for Phi chat tokens you already embedded
        truncation=True,
        max_length=max_ctx,
    ).to(DEVICE)

    # Ensure we don't ask to generate beyond remaining context
    prompt_len = enc["input_ids"].size(1)
    available = max_ctx - prompt_len
    if available <= 0:
        print("Warning: no room left for generation; returning empty prediction.")
        return ""
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
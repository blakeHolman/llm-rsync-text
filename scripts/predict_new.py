#!/usr/bin/env python3
# scripts/predict_new.py

import argparse, os, json, sys, csv, base64
from residuals import get_residual

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
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,  # stream weights to reduce peak RAM
)

MODEL = MODEL.to(DEVICE)

METRICS_FILE = "work/metrics.csv"
RESIDUAL_FILE = "work/residuals.jsonl"

# Open old, new dataset. Pass old to LLM
def _open_data(path, len_prompt=False, stop_after=sys.maxsize):
    results = []
    residuals = []

    prev_old = None
    prev_new = None

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

            old = rec.get("OLD")
            new = rec.get("NEW")

            # Ignore first line do to few-shot prompting
            if idx <= 1:
                prev_old = old
                prev_new = new
                continue

            if old is None or new is None:
                print(f"Skipping line {idx}: missing 'old' or 'new'")
                continue

            # Optionally include length of "new" in the prediction call
            target_len = len(TOKENIZER(new).input_ids) if len_prompt else None
            predicted = predict(old, prev_old, prev_new, target_len=target_len)
            
            """
            print("==============old=============")
            print(old)
            print("===============pred============")
            print(predicted)
            print("==============actual=============")
            """
            print(predicted)
            
            
            prev_old = old
            prev_new = new

            metrics, residual = _compare(old, predicted, new)

            file_id = rec.get("id")

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


# Given old data, predict new
def predict(old, prev_old, prev_new, target_len=None):
    # Create prompt (prompt + old (+ optional len))
    prompt = (
        "Apply the same kind of edits as in the example.\n"
        "Copy all lines and change only what is necessary.\n"
        "Return only the edited text, no explanations.\n\n"
        "Example (before):\n"
        f"{prev_old}\n"
        "Example (after):\n"
        f"{prev_new}\n\n"
        "Text to edit:\n"
        f"{old}\n"
        "Edited text:\n"
    )


    max_ctx = _model_max_ctx()

    # Encode the full prompt
    enc = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_ctx,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    prompt_len = input_ids.size(1)

    available_for_gen = max_ctx - prompt_len
    if available_for_gen <= 0:
        print("Warning: no room left for generation; returning empty prediction.")
        return ""

    # Decide how many tokens to generate
    if target_len is not None:
        # target_len is the tokenized length of the true "new" file
        approx_target_tokens = target_len + 64  # slack so it can finish the last line
        approx_target_tokens = max(32, approx_target_tokens)
        approx_target_tokens = min(approx_target_tokens, available_for_gen)
    else:
        # Fallback: scale with old length
        old_tokens = len(TOKENIZER(old).input_ids)
        approx_target_tokens = old_tokens + 64
        approx_target_tokens = max(32, approx_target_tokens)
        approx_target_tokens = min(approx_target_tokens, available_for_gen)

    # Let the model generate up to the approximate target length
    max_new = approx_target_tokens

    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,                # greedy
            temperature=0.0,                # ignored when do_sample=False
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.pad_token_id,
            use_cache=False,                # <<< disable KV cache to avoid seen_tokens issue
        )

    # Only take tokens after the prompt
    new_tokens = outputs[0, prompt_len:]
    predicted = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    
    return predicted


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
    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote metrics to {METRICS_FILE}")


def _save_residual(residuals):
    if not residuals:
        print("No residuals to save.")
        return
    with open(RESIDUAL_FILE, "w", encoding="utf-8") as f:
        for rec in residuals:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote residuals to {RESIDUAL_FILE}")
    

def main():
    ap = argparse.ArgumentParser(description="Predict \"new\" output from old data")
    ap.add_argument("--data_file", required=True, help="JSONL of {old,new,} pairs")
    ap.add_argument("--metrics", required=False, help="File to save metric results")
    ap.add_argument("--residuals", required=False, help="File to save residuals")
    ap.add_argument("--add_len", action="store_true", help="Provide \"new\" length to prompt for prediction")
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

    # Get stop after var
    stop_after = args.stop_after
    
    # Open dataset and run LLM
    _open_data(data, len_prompt, stop_after)

if __name__ == "__main__":
    main()
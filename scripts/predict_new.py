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


def _build_prefix(prev_old, prev_new):
    """
    Build the static prompt prefix using Phi-3 chat formatting.
    This returns a STRING (not tokenized yet) that ends AFTER the exemplar assistant output.
    """
    subs = infer_substitutions(prev_old, prev_new)
    messages = [
        {
            "role": "system",
            "content": (
                "You perform minimal semantic rewrites.\n"
                "Make only the necessary semantic substitutions, consistently.\n"
                "Do not paraphrase. Preserve whitespace, punctuation, and line breaks.\n"
                "Output ONLY the rewritten text."
                 "Substitutions to apply (learned from example):\n"
                f"{subs}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Learn the semantic substitution pattern from this example.\n\n"
                f"{prev_old}"
            ),
        },
        {"role": "assistant", "content": prev_new},
    ]

    # Produce the exact <|system|>...<|assistant|> format Phi-3 expects.
    # add_generation_prompt=False because we are NOT ready to generate yet.
    return TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def init_prefix_kv(prev_old, prev_new):
    """
    Cache KV for the static prefix once. This speeds up many repeated predictions
    using the same exemplar.
    """
    global PREFIX_TEXT, PREFIX_KV, PREFIX_TOKENS, PREFIX_MASK

    PREFIX_TEXT = _build_prefix(prev_old, prev_new)

    enc = TOKENIZER(
        PREFIX_TEXT,
        return_tensors="pt",
        truncation=True,
        max_length=_model_max_ctx(),
    )
    prefix_ids = enc["input_ids"].to(DEVICE)
    prefix_mask = enc["attention_mask"].to(DEVICE)

    PREFIX_MASK = prefix_mask

    with torch.no_grad():
        out = MODEL(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
        )

    PREFIX_KV = out.past_key_values   # Cache object / DynamicCache
    PREFIX_TOKENS = prefix_ids.size(1)

    print(f"[prefix-kv] cached prefix tokens: {PREFIX_TOKENS}")


# Given old data, predict new
def predict(old, target_len=None):
    if PREFIX_KV is None or PREFIX_TEXT is None or PREFIX_TOKENS is None:
        raise RuntimeError("Prefix KV not initialized. Call init_prefix_kv() first.")

    max_ctx = _model_max_ctx()

    # Build the dynamic part as a *chat-formatted* user turn + assistant generation cue.
    suffix_messages = [
        {
            "role": "user",
            "content": (
                "Apply the learned semantic substitution pattern to this input.\n"
                "Return only the rewritten text.\n\n"
                f"{old}"
            ),
        }
    ]

    # add_generation_prompt=True will append the "<|assistant|>\n" (or equivalent)
    # so the model continues by generating only the edited text.
    suffix_text = TOKENIZER.apply_chat_template(
        suffix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize suffix
    enc = TOKENIZER(
        suffix_text,
        return_tensors="pt",
        add_special_tokens=False,                 # <<< CRITICAL
        truncation=True,
        max_length=max_ctx - PREFIX_TOKENS,       # <<< also important
    )
    input_ids = enc["input_ids"].to(DEVICE)          # (1, suffix_len)
    attn_mask = enc["attention_mask"].to(DEVICE)     # (1, suffix_len)

    bsz, suffix_len = input_ids.shape
    prompt_len_total = PREFIX_TOKENS + suffix_len
    available_for_gen = max_ctx - prompt_len_total
    if available_for_gen <= 0:
        print("Warning: no room left for generation; returning empty prediction.")
        return ""

    # Decide decode length
    if target_len is not None:
        max_new = min(max(32, target_len + 64), available_for_gen)
    else:
        max_new = min(max(32, suffix_len + 64), available_for_gen)

    past = copy.deepcopy(PREFIX_KV)

    # ---- Step 1: run the suffix through the model using cached prefix ----
    full_mask = torch.cat([PREFIX_MASK, attn_mask], dim=1)

    #position_ids = torch.arange(PREFIX_TOKENS, PREFIX_TOKENS + suffix_len, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out = MODEL(
            input_ids=input_ids,
            attention_mask=full_mask,
            #position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )

    past = out.past_key_values

    # ---- Step 2: greedy decode token-by-token (avoid generate() cache issues) ----
    generated = []
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)

    # Stop on EOS or <|end|> if present
    END_ID = TOKENIZER.convert_tokens_to_ids("<|end|>")
    EOS_ID = TOKENIZER.eos_token_id

    cur_len_total = PREFIX_TOKENS + suffix_len

    for _ in range(max_new):
        tok = next_token.item()
        if tok == EOS_ID or (END_ID is not None and tok == END_ID):
            break

        generated.append(tok)

        cur_len_total += 1
        step_mask = torch.ones((1, cur_len_total), dtype=full_mask.dtype, device=DEVICE)
        #step_pos = torch.tensor([[cur_len_total - 1]], device=DEVICE)

        with torch.no_grad():
            out = MODEL(
                input_ids=next_token,
                attention_mask=step_mask,
                #position_ids=step_pos,
                past_key_values=past,
                use_cache=True,
            )

        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return TOKENIZER.decode(generated, skip_special_tokens=True)


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
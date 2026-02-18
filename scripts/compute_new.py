#!/usr/bin/env python3
# scripts/compute_new.py

import json, sys, argparse, os, base64
from residuals import apply_residual
from predict_new import predict, init_prefix_kv
from pick_best_example import pick_best_example

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

RESIDUAL_FILE = "work/residuals.jsonl"

RESIDUAL_DICT = None

def _load_residuals(residuals_path=RESIDUAL_FILE):
    """
    Load residuals from a JSONL file into a dictionary.

    residuals.jsonl format (one JSON per line):
      {"id": "<file_id>", "residual": "<base64>"}

    Returns:
        dict: mapping from record id to decoded residual bytes
    """
    residuals_dict = {}
    with open(residuals_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rrec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_id = rrec.get("id")
            b64 = rrec.get("residual")
            if rec_id is not None and b64 is not None:
                residuals_dict[rec_id] = base64.b64decode(b64)
    return residuals_dict


def _get_residual(rec_id):
    try:
        return RESIDUAL_DICT.get(rec_id)
    except KeyError:
        raise KeyError(f"Residual not found for id={rec_id}")


# Open old, new dataset. Pass old to LLM
def _open_data(path, ex_id, len_prompt=False, stop_after=sys.maxsize, residuals_path=RESIDUAL_FILE):
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

            rec_id = rec.get("id")
            if rec_id == ex_id:
                continue

            old = rec.get("old")
            new = rec.get("new")

            if old is None or new is None:
                print(f"Skipping line {idx}: missing 'old' or 'new'")
                continue

            # Optionally include length of "new" in the prediction call
            target_len = len(TOKENIZER(new).input_ids) if len_prompt else None
            predicted = _compute_new(old, target_len=target_len)

            # Open residuals file and get residual for this index
            residual = _get_residual(rec_id)

            # Apply residual to predicted to get computed new
            computed_new = apply_residual(predicted, residual).decode("utf-8")

            # Validate computed new matches expected new
            _validate(new, computed_new, rec_id)



def _compute_new(old, target_len=None):
    predicted = predict(old, target_len)
    return predicted


def _validate(new, expected_new, rec_id):
    assert new == expected_new, f"Computed new does not match expected new for record {rec_id}"
    print(f"Record {rec_id} validated.")


def main():
    ap = argparse.ArgumentParser(description="Compute \"new\" output from predicted and residuals")
    ap.add_argument("--data_file", required=True, help="JSONL of {old,new,} pairs")
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
    global RESIDUAL_FILE
    global RESIDUAL_DICT
    if args.residuals is not None:
        RESIDUAL_FILE = args.residuals
    RESIDUAL_DICT = _load_residuals(RESIDUAL_FILE)    

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
    _open_data(data, ex_id, len_prompt, stop_after, RESIDUAL_FILE)


if __name__ == "__main__":
    main()
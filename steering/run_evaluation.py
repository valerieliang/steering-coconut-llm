# steering/run_evaluation.py
#!/usr/bin/env python3
"""
Unified behavioral evaluation with steering vector injection.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml

from core import (
    load_model, load_tokenizer, build_input,
    generate_with_steering, extract_answer_word, get_logit_margin
)


def run_evaluation(cfg, mode, device, model, tokenizer):
    """Run steering evaluation for a single mode."""
    n_latent = cfg.get("n_latent", 6)
    
    # Load vectors
    vec_path = Path(os.path.expanduser(cfg["output_dir"])) / f"steering_vectors_{mode}.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectors not found at {vec_path}. Run extract_vectors.py first.")
    
    vectors = torch.load(vec_path, map_location="cpu")
    print(f"Loaded vectors from {vec_path}")
    
    with open(os.path.expanduser(cfg["val_path"])) as f:
        val_data = json.load(f)
    
    out_dir = Path(os.path.expanduser(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{mode}.csv"
    
    fieldnames = [
        "mode", "position", "alpha", "sample_idx",
        "correct", "n_steps", "logit_margin",
        "predicted_word", "expected_word", "neg_word"
    ]
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, sample in enumerate(val_data[:100]):  # Start with 100 samples for testing
            n_steps = len(sample.get("steps", []))
            syms = sample.get("idx_to_symbol", [])
            t_idx = sample.get("target")
            n_idx = sample.get("neg_target")
            
            target_word = (syms[t_idx].lower()
                          if t_idx is not None and t_idx < len(syms)
                          else sample["answer"].rstrip(".").split()[-1].lower())
            neg_word = (syms[n_idx].lower()
                       if n_idx is not None and n_idx < len(syms)
                       else "")
            
            # build_input returns 4 values: input_ids, attention_mask, labels, position_ids
            input_ids, attention_mask, labels, position_ids = build_input(
                tokenizer, sample["question"], n_latent, device
            )
            
            def write_row(pos, alpha, tokens=None, margin=None):
                if tokens is not None:
                    pred = extract_answer_word(tokens, tokenizer)
                else:
                    pred = ""
                if margin is None and tokens is not None:
                    margin = get_logit_margin(
                        model, input_ids, attention_mask, tokenizer,
                        target_word, neg_word,
                        steer_vec=(vectors[pos] if pos in vectors else None),
                        inject_pass=(pos - 1 if pos > 0 else None),
                        steer_alpha=float(alpha),
                    )
                writer.writerow({
                    "mode": mode, "position": pos, "alpha": alpha,
                    "sample_idx": idx,
                    "correct": int(pred == target_word) if tokens is not None else 0,
                    "n_steps": n_steps,
                    "logit_margin": round(margin, 4) if margin is not None else 0,
                    "predicted_word": pred if tokens is not None else "",
                    "expected_word": target_word,
                    "neg_word": neg_word,
                })
            
            # Baseline (no steering)
            tokens = generate_with_steering(model, input_ids, attention_mask)
            write_row(0, 0, tokens=tokens)
            
            # Steered runs
            for pos in cfg["latent_positions"]:
                if pos not in vectors:
                    continue
                for alpha in cfg["alpha_sweep"]:
                    tokens = generate_with_steering(
                        model, input_ids, attention_mask,
                        steer_vec=vectors[pos],
                        inject_pass=pos - 1,
                        steer_alpha=float(alpha),
                    )
                    write_row(pos, alpha, tokens=tokens)
            
            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(val_data)} samples done")
    
    print(f"\nResults saved → {out_csv}")
    _print_summary(out_csv, mode)
    return out_csv


def _print_summary(csv_path, mode):
    """Print terminal summary table."""
    df = pd.read_csv(csv_path)
    baseline = df[df["alpha"] == 0].set_index("sample_idx")
    
    print(f"\n{'='*65}")
    print(f"SUMMARY  [mode={mode}]")
    print(f"{'='*65}")
    print(f"Baseline accuracy: {baseline['correct'].mean():.4f} "
          f"({int(baseline['correct'].sum())}/{len(baseline)})")
    
    steered = df[df["alpha"] != 0].copy()
    if len(steered) > 0:
        steered["delta_acc"] = steered["correct"] - steered["sample_idx"].map(baseline["correct"])
        
        print(f"\n{'Pos':<5} {'Alpha':>7} {'Acc':>7} {'ΔAcc':>8}")
        print("-" * 35)
        for (pos, alpha), grp in steered.groupby(["position", "alpha"]):
            flag = " ◄" if abs(grp["delta_acc"].mean()) > 0.05 else ""
            print(f"L{pos:<4} {alpha:>7} "
                  f"{grp['correct'].mean():>7.4f} "
                  f"{grp['delta_acc'].mean():>+8.4f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Unified steering evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="cross_pair",
                        choices=["swap", "cross_pair", "both"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for testing)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = load_model(cfg, device)
    tokenizer = load_tokenizer()
    
    modes = ["swap", "cross_pair"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n{'#'*65}\n# EVALUATING mode={mode}\n{'#'*65}")
        run_evaluation(cfg, mode, device, model, tokenizer)


if __name__ == "__main__":
    main()
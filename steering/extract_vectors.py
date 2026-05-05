# steering/extract_vectors.py
#!/usr/bin/env python3
"""
Unified vector extraction for steering experiments.

Supports both Phase 1 (swap) and Phase 2 (cross_pair) contrastive modes.
Also includes the answer-decodability probe.

Usage:
    # Extract vectors using cross_pair (recommended)
    python steering/extract_vectors.py --config steering/config/steering_config.yaml
    
    # Extract vectors using swap (Phase 1)
    python steering/extract_vectors.py --config steering/config/steering_config.yaml --mode swap
    
    # Only run the probe (skip vector extraction)
    python steering/extract_vectors.py --config steering/config/steering_config.yaml --probe-only
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core import (
    load_model, load_tokenizer, get_thought_vector, build_input,
    build_pairs
)


def extract_vectors(cfg, mode, device, model, tokenizer):
    """Extract steering vectors for all latent positions."""
    n_latent = cfg.get("n_latent", 6)
    positions = cfg["latent_positions"]
    
    with open(os.path.expanduser(cfg["train_path"])) as f:
        data = json.load(f)
    
    max_pos = max(positions)
    data = [d for d in data if len(d.get("steps", [])) >= max_pos]
    
    print(f"\nBuilding contrastive pairs (mode='{mode}')...")
    pairs = build_pairs(data, cfg["n_contrast"], mode=mode)
    print(f"  Generated {len(pairs)} pairs")
    
    vectors = {}
    
    for pos in positions:
        capture_pass = pos - 1  # 0-indexed
        pos_acts, neg_acts = [], []
        
        for i, (s_pos, s_neg) in enumerate(pairs):
            inp_pos = build_input(tokenizer, s_pos["question"], n_latent, device)
            inp_neg = build_input(tokenizer, s_neg["question"], n_latent, device)
            
            with torch.no_grad():
                h_pos = get_thought_vector(model, tokenizer, s_pos["question"], 
                                           n_latent, device, capture_pass)
                h_neg = get_thought_vector(model, tokenizer, s_neg["question"],
                                           n_latent, device, capture_pass)
            
            if h_pos is not None:
                pos_acts.append(h_pos)
            if h_neg is not None:
                neg_acts.append(h_neg)
            
            if (i + 1) % 100 == 0:
                print(f"  L{pos}: {i+1}/{len(pairs)} pairs")
        
        if pos_acts and neg_acts:
            v = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
            vectors[pos] = v
            print(f"  L{pos}: norm={v.norm():.4f}  "
                  f"({len(pos_acts)} pos, {len(neg_acts)} neg)")
        else:
            print(f"  L{pos}: no vectors captured")
    
    return vectors

def run_probe(cfg, device, model, tokenizer):
    """
    Answer-decodability probe: logistic regression to predict answer from thought.
    """
    n_latent = cfg.get("n_latent", 6)
    n_probe = cfg.get("n_probe", 400)
    max_pos = max(cfg["latent_positions"])
    
    # Build probe dataset from validation set
    with open(os.path.expanduser(cfg["val_path"])) as f:
        val_data = json.load(f)
    
    probe_data = [d for d in val_data if len(d.get("steps", [])) >= max_pos]
    
    # Supplement from training tail if needed
    if len(probe_data) < 50:
        with open(os.path.expanduser(cfg["train_path"])) as f:
            train_data = json.load(f)
        extra = [d for d in train_data[cfg["n_contrast"]:] 
                 if len(d.get("steps", [])) >= max_pos]
        probe_data = probe_data + extra
    
    probe_data = probe_data[:n_probe]
    print(f"\nProbe: {len(probe_data)} samples (steps >= {max_pos})")
    
    # Encode answer labels
    le = LabelEncoder()
    raw_labels = []
    for s in probe_data:
        syms = s.get("idx_to_symbol", [])
        t = s.get("target")
        raw_labels.append(syms[t].lower() if t is not None and t < len(syms)
                         else s["answer"].rstrip(".").split()[-1].lower())
    labels = le.fit_transform(raw_labels)
    
    n_classes = len(np.unique(labels))
    print(f"  Answer classes: {n_classes}")
    print(f"  Chance level: {1.0/n_classes:.3f}")
    
    # Adaptive cross-validation
    min_class = int(np.bincount(labels).min())
    n_splits = min(5, min_class) if min_class >= 2 else None
    
    out_dir = Path(os.path.expanduser(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    probe_results = {}
    
    for pos in cfg["latent_positions"]:
        capture_pass = pos - 1
        thoughts = []
        
        for s in probe_data:
            with torch.no_grad():
                h = get_thought_vector(model, tokenizer, s["question"],
                                       n_latent, device, capture_pass)
            if h is not None:
                thoughts.append(h.float().numpy())
            else:
                thoughts.append(np.zeros(768, dtype=np.float32))
        
        X = np.stack(thoughts)
        
        # Save for reuse
        thought_path = out_dir / f"probe_thoughts_L{pos}.npy"
        np.save(thought_path, X)
        
        # Standardize - crucial for convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        fold_accs = []
        
        if n_splits is None:
            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            for train_idx, val_idx in loo.split(X_scaled):
                # lbfgs works well for small datasets
                clf = LogisticRegression(
                    max_iter=10000, 
                    C=1.0, 
                    solver="lbfgs",
                    random_state=42
                )
                clf.fit(X_scaled[train_idx], labels[train_idx])
                fold_accs.append(clf.score(X_scaled[val_idx], labels[val_idx]))
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X_scaled, labels):
                # saga handles many classes well
                clf = LogisticRegression(
                    max_iter=10000,
                    C=1.0, 
                    solver="saga",
                    random_state=42,
                    tol=1e-4
                )
                clf.fit(X_scaled[train_idx], labels[train_idx])
                fold_accs.append(clf.score(X_scaled[val_idx], labels[val_idx]))
        
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        probe_results[pos] = {"mean": mean_acc, "std": std_acc}
        
        chance = 1.0 / n_classes
        print(f"  L{pos}: {mean_acc:.3f} ± {std_acc:.3f} (chance={chance:.3f}, {mean_acc/chance:.1f}× chance)")
    
    # Save results
    probe_csv = out_dir / "probe_results.csv"
    with open(probe_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["position", "mean_acc", "std_acc", "n_classes", "chance"])
        w.writeheader()
        for pos, r in probe_results.items():
            w.writerow({
                "position": pos, 
                "mean_acc": r["mean"], 
                "std_acc": r["std"],
                "n_classes": n_classes,
                "chance": 1.0 / n_classes
            })
    
    print(f"\nProbe results saved → {probe_csv}")
    return probe_results

def main():
    parser = argparse.ArgumentParser(description="Unified steering vector extraction")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--mode", default=None, 
                        choices=["swap", "cross_pair"],
                        help="Contrastive mode (overrides config)")
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip vector extraction, only run probe")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    mode = args.mode or cfg.get("contrastive_mode", "cross_pair")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Mode: {mode}")
    
    model = load_model(cfg, device)
    tokenizer = load_tokenizer()
    
    out_dir = Path(os.path.expanduser(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.probe_only:
        print("\n" + "=" * 60)
        print("EXTRACTING STEERING VECTORS")
        print("=" * 60)
        vectors = extract_vectors(cfg, mode, device, model, tokenizer)
        
        vec_path = out_dir / f"steering_vectors_{mode}.pt"
        torch.save(vectors, vec_path)
        print(f"\nSaved {len(vectors)} vectors → {vec_path}")
    
    print("\n" + "=" * 60)
    print("ANSWER-DECODABILITY PROBE")
    print("=" * 60)
    run_probe(cfg, device, model, tokenizer)


if __name__ == "__main__":
    main()
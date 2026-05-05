# steering/core/contrastive_pairs.py
"""
Contrastive pair builders for steering vector extraction.
"""

import random
from collections import defaultdict


def build_swap_pairs(data, n_contrast):
    """
    Phase 1 method: swap answer options within the same question.
    
    Returns list of (positive_sample, negative_sample) where negative
    has the answer options swapped.
    """
    pairs = []
    for s in data[:n_contrast]:
        q = s["question"]
        syms = s.get("idx_to_symbol", [])
        t, n = s.get("target"), s.get("neg_target")
        tw = syms[t] if t is not None and t < len(syms) else ""
        nw = syms[n] if n is not None and n < len(syms) else ""
        
        if tw and nw and f"{tw} or {nw}" in q:
            neg_q = q.replace(f"{tw} or {nw}", f"{nw} or {tw}")
        else:
            neg_q = q
        
        neg = dict(s)
        neg["question"] = neg_q
        pairs.append((s, neg))
    
    return pairs


def build_cross_pairs(data, n_contrast, seed=42):
    """
    Phase 2 method: pair samples with different target answers but same depth.
    
    Returns list of (positive_sample, negative_sample) where the samples
    reason toward different answer concepts.
    """
    random.seed(seed)
    
    # Group by number of reasoning steps
    by_steps = defaultdict(list)
    for s in data:
        n_steps = len(s.get("steps", []))
        by_steps[n_steps].append(s)
    
    pairs = []
    seen = set()
    
    # Round-robin across step counts
    step_counts = sorted(by_steps.keys())
    step_iters = {k: 0 for k in step_counts}
    
    attempts = 0
    while len(pairs) < n_contrast and attempts < n_contrast * 20:
        attempts += 1
        for k in step_counts:
            pool = by_steps[k]
            if len(pool) < 2:
                continue
            
            i = step_iters[k] % len(pool)
            j = (step_iters[k] + 1) % len(pool)
            step_iters[k] += 2
            
            s_pos = pool[i]
            s_neg = pool[j]
            
            # Ensure different target answers
            sym_pos = s_pos.get("idx_to_symbol", [])
            sym_neg = s_neg.get("idx_to_symbol", [])
            t_pos = s_pos.get("target")
            t_neg = s_neg.get("target")
            
            if (t_pos is None or t_neg is None or
                t_pos >= len(sym_pos) or t_neg >= len(sym_neg)):
                continue
            
            if sym_pos[t_pos] == sym_neg[t_neg]:
                continue  # Same answer concept
            
            key = (id(s_pos), id(s_neg))
            if key in seen:
                continue
            
            seen.add(key)
            pairs.append((s_pos, s_neg))
            if len(pairs) >= n_contrast:
                break
    
    return pairs


def build_pairs(data, n_contrast, mode="cross_pair"):
    """Factory function for building contrastive pairs."""
    if mode == "swap":
        return build_swap_pairs(data, n_contrast)
    elif mode == "cross_pair":
        return build_cross_pairs(data, n_contrast)
    else:
        raise ValueError(f"Unknown contrastive mode: {mode}")
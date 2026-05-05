# steering/core/model_utils.py
"""
Model utilities: loading, tokenization, and thought vector extraction.
Unified between Phase 1 and Phase 2.
"""

import os
from pathlib import Path
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coconut"))

# Token IDs
LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260


def load_tokenizer():
    """Load GPT-2 tokenizer with latent special tokens."""
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "additional_special_tokens": [
            "<|latent|>", "<|start-latent|>", "<|end-latent|>"
        ]
    })
    tok.pad_token = tok.eos_token
    return tok


def load_model(cfg, device):
    """Load Coconut model with trained weights."""
    from coconut import Coconut
    
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.resize_token_embeddings(VOCAB_SIZE)
    
    model = Coconut(
        base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID,
        start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID,
        eos_token_id=EOS_TOKEN_ID,
    )
    
    ckpt = torch.load(os.path.expanduser(cfg["model_path"]), map_location="cpu")
    if "base_causallm" in ckpt:
        ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.", ""): v for k, v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    
    return model


def build_input(tokenizer, question, n_latent, device):
    """Build input tensor sequence for Coconut."""
    q_ids = tokenizer.encode(question)
    seq = (
        q_ids
        + [START_LATENT_ID]
        + [LATENT_TOKEN_ID] * n_latent
        + [END_LATENT_ID]
    )
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    return input_ids, attention_mask, labels, position_ids


def get_thought_vector(model, tokenizer, question, n_latent, device, capture_pass):
    """
    Extract the continuous thought vector at a specific latent pass.
    
    Args:
        capture_pass: 0-indexed pass number (e.g., 0 for first latent position)
    
    Returns:
        torch.Tensor or None: The thought vector at the requested pass
    """
    input_ids, attention_mask, labels, position_ids = build_input(
        tokenizer, question, n_latent, device
    )
    
    captured = [None]
    
    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n = max(len(l) for l in latent_lists) if latent_lists else 0
    
    if max_n == 0 or capture_pass >= max_n:
        return None
    
    inputs_embeds = model.embedding(input_ids)
    ncr = (0, latent_indices[:, 1].min().item())
    kv_cache = None
    
    for pass_idx in range(max_n):
        if kv_cache is None:
            out = model.base_causallm(
                inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
                attention_mask=attention_mask[:, ncr[0]:ncr[1]],
                position_ids=position_ids[:, ncr[0]:ncr[1]],
                output_hidden_states=True,
            )
            offset = 0
        else:
            pkv = [(k[:, :, :ncr[0], :], v[:, :, :ncr[0], :]) for k, v in kv_cache]
            out = model.base_causallm(
                inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
                attention_mask=attention_mask[:, :ncr[1]],
                position_ids=position_ids[:, ncr[0]:ncr[1]],
                past_key_values=pkv,
                output_hidden_states=True,
            )
            offset = ncr[0]
        
        ncr = (ncr[1], input_ids.shape[1] if pass_idx + 1 >= max_n else ncr[1] + 1)
        hidden_states = out.hidden_states[-1]
        kv_cache = out.past_key_values
        
        # Build new embeddings with updated thought vectors
        tensor_list = [
            [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]
        
        for b, ml in enumerate(latent_lists):
            if len(ml) > pass_idx:
                tok_idx = ml[pass_idx]
                thought = hidden_states[b, tok_idx - 1 - offset, :]
                if pass_idx == capture_pass:
                    captured[0] = thought.detach().cpu()
                tensor_list[b][tok_idx] = thought
        
        inputs_embeds = torch.stack([torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])])
    
    return captured[0]
# steering/core/steering_injection.py
"""
Steering vector injection into Coconut's forward pass.
"""

import torch
from torch.nn import CrossEntropyLoss
from coconut import Outputs


def create_steering_forward(model, inject_pass, steer_vec, steer_alpha):
    """
    Create a patched forward() that injects steering at the specified pass.
    
    Args:
        model: Coconut model
        inject_pass: 0-indexed latent pass to inject at
        steer_vec: Steering vector to add
        steer_alpha: Scaling factor
    
    Returns:
        (original_forward, patched_forward) tuple
    """
    original_forward = model.forward
    
    def patched_forward(input_ids, attention_mask, labels, position_ids, **kwargs):
        logits_list = []
        
        latent_indices = (input_ids == model.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n = max(len(l) for l in latent_lists) if latent_lists else 0
        
        inputs_embeds = model.embedding(input_ids)
        if max_n > 0:
            ncr = (0, latent_indices[:, 1].min().item())
        else:
            ncr = (0, input_ids.shape[1])
        
        kv_cache = None
        
        for pass_idx in range(max_n):
            if kv_cache is None:
                outputs = model.base_causallm(
                    inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
                    attention_mask=attention_mask[:, ncr[0]:ncr[1]],
                    position_ids=position_ids[:, ncr[0]:ncr[1]],
                    output_hidden_states=True,
                )
                offset = 0
            else:
                past_kv = [(k[:, :, :ncr[0], :], v[:, :, :ncr[0], :])
                          for k, v in kv_cache]
                outputs = model.base_causallm(
                    inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
                    attention_mask=attention_mask[:, :ncr[1]],
                    position_ids=position_ids[:, ncr[0]:ncr[1]],
                    past_key_values=past_kv,
                    output_hidden_states=True,
                )
                offset = ncr[0]
            
            logits_list.append(outputs.logits)
            ncr = (ncr[1], input_ids.shape[1] if pass_idx + 1 >= max_n else ncr[1] + 1)
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values
            
            tensor_list = [
                [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
                for b in range(inputs_embeds.shape[0])
            ]
            
            for b, ml in enumerate(latent_lists):
                if len(ml) > pass_idx:
                    tok_idx = ml[pass_idx]
                    thought = hidden_states[b, tok_idx - 1 - offset, :]
                    
                    # Injection happens here
                    if pass_idx == inject_pass:
                        thought = thought + steer_alpha * steer_vec.to(thought.device)
                    
                    tensor_list[b][tok_idx] = thought
            
            inputs_embeds = torch.stack([torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])])
        
        # Final pass after all latent tokens
        past_kv = ([(k[:, :, :ncr[0], :], v[:, :, :ncr[0], :]) for k, v in kv_cache]
                  if kv_cache else None)
        outputs = model.base_causallm(
            inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
            attention_mask=attention_mask[:, :ncr[1]],
            position_ids=position_ids[:, ncr[0]:ncr[1]],
            past_key_values=past_kv,
            output_hidden_states=True,
        )
        logits_list.append(outputs.logits)
        logits = torch.cat(logits_list, dim=-2)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
    
    return original_forward, patched_forward


def generate_with_steering(model, input_ids, attention_mask, steer_vec=None, 
                           inject_pass=None, steer_alpha=0.0, max_new_tokens=16):
    """
    Generate answer with optional steering vector injection.
    
    Returns:
        list of token ids
    """
    original_forward = None
    
    if steer_vec is not None and steer_alpha != 0.0 and inject_pass is not None:
        original_forward, patched = create_steering_forward(
            model, inject_pass, steer_vec, steer_alpha
        )
        model.forward = patched
    
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
    finally:
        if original_forward is not None:
            model.forward = original_forward
    
    return out[0].tolist()


def extract_answer_word(tokens, tokenizer):
    """Extract the answer word from generated tokens."""
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    if "###" in decoded:
        part = decoded.split("###")[-1].strip()
    else:
        part = decoded.strip()
    return part.rstrip(".").split()[-1].lower() if part else ""


def get_logit_margin(model, input_ids, attention_mask, tokenizer,
                     target_word, neg_word, steer_vec=None, 
                     inject_pass=None, steer_alpha=0.0):
    """
    Get logit margin (correct - wrong) from final token position.
    More sensitive than binary accuracy for measuring steering effects.
    """
    target_id = tokenizer.encode(" " + target_word)[-1]
    neg_id = tokenizer.encode(" " + neg_word)[-1]
    
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, 
                                device=input_ids.device).unsqueeze(0)
    labels = input_ids.clone()
    
    original_forward = None
    if steer_vec is not None and steer_alpha != 0.0 and inject_pass is not None:
        original_forward, patched = create_steering_forward(
            model, inject_pass, steer_vec, steer_alpha
        )
        model.forward = patched
    
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )
        logits = out.logits[0, -1]
        margin = (logits[target_id] - logits[neg_id]).item()
    finally:
        if original_forward is not None:
            model.forward = original_forward
    
    return margin
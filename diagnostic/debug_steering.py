"""
Quick diagnostic: verifies that steering actually changes the internal
thought representations, even if the final output word stays the same.
"""
import sys, os, json, torch
sys.path.insert(0, os.path.expanduser("~/coconut-cot/coconut"))

from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260

def load_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({"additional_special_tokens":
        ["<|latent|>","<|start-latent|>","<|end-latent|>"]})
    tok.pad_token = tok.eos_token
    return tok

def load_model(device):
    from coconut import Coconut
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.resize_token_embeddings(VOCAB_SIZE)
    model = Coconut(base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID, start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID, eos_token_id=EOS_TOKEN_ID)
    ckpt = torch.load(os.path.expanduser(
        "~/coconut-cot/checkpoints/prosqa-coconut/checkpoint_48"), map_location="cpu")
    if "base_causallm" in ckpt: ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.",""):v for k,v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)
tokenizer = load_tokenizer()
vectors = torch.load(os.path.expanduser(
    "~/coconut-cot/steering/outputs/steering_vectors.pt"), map_location=device)

with open(os.path.expanduser("~/coconut-cot/coconut/data/prosqa_valid.json")) as f:
    sample = json.load(f)[0]

q_ids = tokenizer.encode(sample["question"])
seq = q_ids + [START_LATENT_ID] + [LATENT_TOKEN_ID]*6 + [END_LATENT_ID]
input_ids = torch.tensor([seq], dtype=torch.long, device=device)
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()
position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

print(f"Question: ...{sample['question'][-60:]}")
print(f"Expected: {sample['answer']}")
print()

for inject_pass in [0, 4]:  # L1 and L5
    for alpha in [0, 10, 40, 200, 1000]:
        thoughts = []

        # Run the forward loop manually, capturing thought norms
        latent_indices = (input_ids == model.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0]==i]
                        for i in range(input_ids.shape[0])]
        max_n_latents = max(len(l) for l in latent_lists)
        inputs_embeds = model.embedding(input_ids)
        next_cr = (0, latent_indices[:,1].min().item())
        kv = None

        for pass_idx in range(max_n_latents):
            if kv is None:
                out = model.base_causallm(
                    inputs_embeds=inputs_embeds[:,next_cr[0]:next_cr[1],:],
                    attention_mask=attention_mask[:,next_cr[0]:next_cr[1]],
                    position_ids=position_ids[:,next_cr[0]:next_cr[1]],
                    output_hidden_states=True)
                offset = 0
            else:
                pkv = [(k[:,:,:next_cr[0],:],v[:,:,:next_cr[0],:]) for k,v in kv]
                out = model.base_causallm(
                    inputs_embeds=inputs_embeds[:,next_cr[0]:next_cr[1],:],
                    attention_mask=attention_mask[:,:next_cr[1]],
                    position_ids=position_ids[:,next_cr[0]:next_cr[1]],
                    past_key_values=pkv, output_hidden_states=True)
                offset = next_cr[0]

            next_cr = (next_cr[1],
                input_ids.shape[1] if pass_idx+1>=max_n_latents else next_cr[1]+1)
            hs = out.hidden_states[-1]
            kv = out.past_key_values

            tl = [[inputs_embeds[b,p,:] for p in range(inputs_embeds.shape[1])]
                  for b in range(inputs_embeds.shape[0])]
            for b, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    tok_idx = mask_list[pass_idx]
                    thought = hs[b, tok_idx-1-offset, :]
                    if pass_idx == inject_pass and alpha > 0:
                        thought = thought + alpha * vectors[inject_pass+1].to(device)
                    thoughts.append(thought.norm().item())
                    tl[b][tok_idx] = thought
            inputs_embeds = torch.stack([torch.stack(tl[b])
                                         for b in range(inputs_embeds.shape[0])])

        # get final prediction
        pkv = [(k[:,:,:next_cr[0],:],v[:,:,:next_cr[0],:]) for k,v in kv] if kv else None
        final_out = model.base_causallm(
            inputs_embeds=inputs_embeds[:,next_cr[0]:next_cr[1],:],
            attention_mask=attention_mask[:,:next_cr[1]],
            position_ids=position_ids[:,next_cr[0]:next_cr[1]],
            past_key_values=pkv)
        pred_tok = final_out.logits[0,-1].argmax().item()

        # autoregressive decode to get the word
        new_emb = torch.cat([inputs_embeds,
            model.embedding(torch.tensor(pred_tok, device=device)).view(1,1,-1)], dim=1)
        for _ in range(15):
            o2 = model.base_causallm(inputs_embeds=new_emb)
            nt = o2.logits[0,-1].argmax().item()
            if nt == EOS_TOKEN_ID: break
            new_emb = torch.cat([new_emb,
                model.embedding(torch.tensor(nt,device=device)).view(1,1,-1)],dim=1)

        all_toks = input_ids[0].tolist() + [pred_tok]
        decoded = tokenizer.decode(all_toks, skip_special_tokens=True)
        ans = decoded.split("###")[-1].strip() if "###" in decoded else decoded[-40:]

        thought_norms_str = " ".join(f"{n:.2f}" for n in thoughts)
        print(f"L{inject_pass+1} alpha={alpha:4d}: thought_norms=[{thought_norms_str}]  ans='{ans[-30:]}'")
    print()
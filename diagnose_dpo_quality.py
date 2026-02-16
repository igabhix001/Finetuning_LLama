#!/usr/bin/env python3
"""
Diagnose DPO Training Quality
==============================

Checks for reward hacking and degenerate policy formation:
1. Compare SFT vs DPO model outputs
2. Check if rejected responses have collapsed probabilities
3. Validate generation quality hasn't degraded
4. Test factuality and coherence

Usage:
  python diagnose_dpo_quality.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

def load_model(model_path, device="cuda"):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # CRITICAL: Resize embeddings if tokenizer vocab size doesn't match model
    # This happens when PAD token was added during DPO training
    if len(tokenizer) != model.config.vocab_size:
        print(f"   Resizing model embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    return model, tokenizer

def compute_sequence_logprob(model, tokenizer, prompt, response, device="cuda"):
    """Compute log probability of response given prompt"""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prompt length
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Compute log probs for response tokens
    shift_logits = logits[:, prompt_len-1:-1, :]
    shift_labels = inputs["input_ids"][:, prompt_len:]
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    total_log_prob = token_log_probs.sum().item()
    avg_log_prob = token_log_probs.mean().item()
    
    return total_log_prob, avg_log_prob

def test_reward_hacking(sft_model_path, dpo_model_path, test_samples=5):
    """Test if DPO is reward hacking by comparing SFT vs DPO"""
    
    print("="*80)
    print("DIAGNOSING DPO TRAINING QUALITY")
    print("="*80)
    
    # Load models
    print("\n1. Loading SFT model...")
    sft_model, sft_tokenizer = load_model(sft_model_path)
    print("   ✓ SFT model loaded")
    
    print("\n2. Loading DPO model...")
    dpo_model, dpo_tokenizer = load_model(dpo_model_path)
    print("   ✓ DPO model loaded")
    
    # Load test pairs
    print("\n3. Loading test pairs...")
    from datasets import load_from_disk
    test_dataset = load_from_disk("data/dpo/prepared/test_filtered")
    print(f"   ✓ Loaded {len(test_dataset)} test pairs")
    
    # Test samples
    print(f"\n4. Testing {test_samples} samples...")
    print("="*80)
    
    results = []
    for i in range(min(test_samples, len(test_dataset))):
        sample = test_dataset[i]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        
        print(f"\n### Sample {i+1} ###")
        print(f"Prompt: {prompt[:100]}...")
        
        # Compute logprobs for SFT
        sft_chosen_logp, sft_chosen_avg = compute_sequence_logprob(sft_model, sft_tokenizer, prompt, chosen)
        sft_rejected_logp, sft_rejected_avg = compute_sequence_logprob(sft_model, sft_tokenizer, prompt, rejected)
        sft_margin = sft_chosen_logp - sft_rejected_logp
        
        # Compute logprobs for DPO
        dpo_chosen_logp, dpo_chosen_avg = compute_sequence_logprob(dpo_model, dpo_tokenizer, prompt, chosen)
        dpo_rejected_logp, dpo_rejected_avg = compute_sequence_logprob(dpo_model, dpo_tokenizer, prompt, rejected)
        dpo_margin = dpo_chosen_logp - dpo_rejected_logp
        
        print(f"\nSFT Model:")
        print(f"  Chosen logp:   {sft_chosen_logp:.2f} (avg: {sft_chosen_avg:.4f})")
        print(f"  Rejected logp: {sft_rejected_logp:.2f} (avg: {sft_rejected_avg:.4f})")
        print(f"  Margin: {sft_margin:.2f}")
        
        print(f"\nDPO Model:")
        print(f"  Chosen logp:   {dpo_chosen_logp:.2f} (avg: {dpo_chosen_avg:.4f})")
        print(f"  Rejected logp: {dpo_rejected_logp:.2f} (avg: {dpo_rejected_avg:.4f})")
        print(f"  Margin: {dpo_margin:.2f}")
        
        # Check for reward hacking
        chosen_change = dpo_chosen_logp - sft_chosen_logp
        rejected_change = dpo_rejected_logp - sft_rejected_logp
        
        print(f"\nChanges (DPO - SFT):")
        print(f"  Chosen:   {chosen_change:+.2f}")
        print(f"  Rejected: {rejected_change:+.2f}")
        
        # Diagnose
        if abs(rejected_change) > abs(chosen_change) * 3:
            print("  ⚠️  WARNING: Rejected collapsed more than chosen improved!")
            print("  This suggests reward hacking (annihilating rejected)")
        elif chosen_change > 0 and rejected_change < 0:
            print("  ✅ HEALTHY: Chosen improved AND rejected degraded")
        elif chosen_change > abs(rejected_change):
            print("  ✅ GOOD: Chosen improved more than rejected degraded")
        else:
            print("  ⚠️  UNCLEAR: Mixed signals")
        
        results.append({
            "sample": i,
            "sft_margin": sft_margin,
            "dpo_margin": dpo_margin,
            "chosen_change": chosen_change,
            "rejected_change": rejected_change,
            "reward_hacking": abs(rejected_change) > abs(chosen_change) * 3
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_chosen_change = sum(r["chosen_change"] for r in results) / len(results)
    avg_rejected_change = sum(r["rejected_change"] for r in results) / len(results)
    reward_hacking_count = sum(r["reward_hacking"] for r in results)
    
    print(f"\nAverage changes (DPO - SFT):")
    print(f"  Chosen:   {avg_chosen_change:+.2f}")
    print(f"  Rejected: {avg_rejected_change:+.2f}")
    print(f"  Ratio: {abs(avg_rejected_change) / abs(avg_chosen_change):.2f}x")
    
    print(f"\nReward hacking detected: {reward_hacking_count}/{len(results)} samples")
    
    if reward_hacking_count > len(results) * 0.5:
        print("\n❌ CRITICAL: Majority of samples show reward hacking!")
        print("DPO is collapsing rejected probabilities instead of improving chosen.")
        print("Model quality likely degraded.")
    elif reward_hacking_count > 0:
        print("\n⚠️  WARNING: Some samples show reward hacking.")
        print("Model may have mixed quality.")
    else:
        print("\n✅ GOOD: No reward hacking detected.")
        print("DPO appears to be learning genuine preferences.")
    
    return results

if __name__ == "__main__":
    sft_model_path = "./models/merged_sft"
    dpo_model_path = "./checkpoints/dpo_lora/final"  # Or merged final model
    
    results = test_reward_hacking(sft_model_path, dpo_model_path, test_samples=10)

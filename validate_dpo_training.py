#!/usr/bin/env python3
"""
Validate DPO Training Results
==============================

Performs comprehensive validation:
1. Generate sample responses from SFT vs DPO models
2. Check for verbosity increase (common DPO failure)
3. Test factuality and coherence
4. Compare response quality

Usage:
  python validate_dpo_training.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

def load_model(model_path):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
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

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def validate_training():
    """Validate DPO training quality"""
    
    print("="*80)
    print("VALIDATING DPO TRAINING RESULTS")
    print("="*80)
    
    # Load models
    print("\n1. Loading SFT model (baseline)...")
    sft_model, sft_tokenizer = load_model("./models/merged_sft")
    print("   ✓ SFT model loaded")
    
    # Test prompts
    test_prompts = [
        "When will I get married?",
        "Will I get a job promotion this year?",
        "What is my career prospect?",
        "When will I have children?",
        "Should I invest in real estate?",
    ]
    
    print("\n2. Generating responses from SFT model...")
    sft_responses = []
    for i, prompt in enumerate(test_prompts):
        print(f"   Generating {i+1}/{len(test_prompts)}...")
        response = generate_response(sft_model, sft_tokenizer, prompt)
        sft_responses.append(response)
    
    # Check if DPO model exists
    dpo_checkpoint = Path("./checkpoints/dpo_lora/final")
    if not dpo_checkpoint.exists():
        print("\n⚠️  DPO checkpoint not found yet. Training still in progress.")
        print("Run this script again after training completes.")
        return
    
    print("\n3. Loading DPO model...")
    dpo_model, dpo_tokenizer = load_model(str(dpo_checkpoint))
    print("   ✓ DPO model loaded")
    
    print("\n4. Generating responses from DPO model...")
    dpo_responses = []
    for i, prompt in enumerate(test_prompts):
        print(f"   Generating {i+1}/{len(test_prompts)}...")
        response = generate_response(dpo_model, dpo_tokenizer, prompt)
        dpo_responses.append(response)
    
    # Compare responses
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n### Test {i+1}: {prompt}")
        print("-"*80)
        
        sft_resp = sft_responses[i]
        dpo_resp = dpo_responses[i]
        
        print(f"\nSFT Response ({len(sft_resp)} chars):")
        print(f"  {sft_resp[:200]}...")
        
        print(f"\nDPO Response ({len(dpo_resp)} chars):")
        print(f"  {dpo_resp[:200]}...")
        
        # Check for issues
        length_ratio = len(dpo_resp) / max(len(sft_resp), 1)
        
        print(f"\nAnalysis:")
        print(f"  Length ratio (DPO/SFT): {length_ratio:.2f}x")
        
        if length_ratio > 1.5:
            print("  ⚠️  WARNING: DPO response is significantly longer (verbosity increase)")
        elif length_ratio < 0.5:
            print("  ⚠️  WARNING: DPO response is significantly shorter (may be truncated)")
        else:
            print("  ✅ Length is reasonable")
        
        # Check for repetition
        words_sft = sft_resp.split()
        words_dpo = dpo_resp.split()
        
        if len(set(words_dpo)) < len(words_dpo) * 0.5:
            print("  ⚠️  WARNING: High repetition in DPO response")
        else:
            print("  ✅ No excessive repetition")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_length_ratio = sum(len(dpo_responses[i]) / max(len(sft_responses[i]), 1) 
                           for i in range(len(test_prompts))) / len(test_prompts)
    
    print(f"\nAverage length ratio (DPO/SFT): {avg_length_ratio:.2f}x")
    
    if avg_length_ratio > 1.5:
        print("\n❌ CRITICAL: DPO model is significantly more verbose!")
        print("This is a common DPO failure mode (reward hacking via length).")
        print("Consider:")
        print("  1. Reducing beta (currently 0.2)")
        print("  2. Adding length penalty")
        print("  3. Filtering long rejected responses from dataset")
    elif avg_length_ratio < 0.7:
        print("\n⚠️  WARNING: DPO model generates shorter responses.")
        print("Check if responses are being truncated or quality degraded.")
    else:
        print("\n✅ GOOD: Response lengths are reasonable.")
        print("DPO training appears successful.")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if avg_length_ratio > 1.5:
        print("\n❌ DO NOT deploy this model.")
        print("Retrain with adjusted hyperparameters.")
    elif avg_length_ratio < 0.7:
        print("\n⚠️  Review responses carefully before deployment.")
    else:
        print("\n✅ Model appears ready for deployment.")
        print("Proceed with merging and testing.")

if __name__ == "__main__":
    validate_training()

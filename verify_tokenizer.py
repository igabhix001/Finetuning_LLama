#!/usr/bin/env python3
"""
Verify Tokenizer Configuration
===============================

Checks if the tokenizer in merged_sft is:
1. Pure Llama 3.1
2. Mistral-contaminated
3. Has regex bugs

Usage:
  python verify_tokenizer.py
"""

from transformers import AutoTokenizer
from pathlib import Path
import json

def verify_tokenizer():
    """Verify tokenizer configuration"""
    
    print("="*80)
    print("TOKENIZER VERIFICATION")
    print("="*80)
    
    model_path = "./models/merged_sft"
    
    if not Path(model_path).exists():
        print(f"\n❌ Model path does not exist: {model_path}")
        print("This script must be run on RunPod where the model is located.")
        return
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("   ✓ Tokenizer loaded")
    except Exception as e:
        print(f"   ❌ Failed to load tokenizer: {e}")
        return
    
    # Check tokenizer class
    print(f"\n2. Tokenizer Details:")
    print(f"   Class: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")
    
    # Check special tokens
    print(f"\n3. Special Tokens:")
    print(f"   BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"   EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"   PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   UNK: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    
    # Check tokenizer config
    print(f"\n4. Checking tokenizer_config.json...")
    config_path = Path(model_path) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for Mistral contamination
        mistral_indicators = []
        if 'mistral' in str(config).lower():
            mistral_indicators.append("'mistral' found in config")
        if config.get('model_type') == 'mistral':
            mistral_indicators.append("model_type is 'mistral'")
        if 'Mistral' in config.get('tokenizer_class', ''):
            mistral_indicators.append("tokenizer_class contains 'Mistral'")
        
        if mistral_indicators:
            print("   ⚠️  MISTRAL CONTAMINATION DETECTED:")
            for indicator in mistral_indicators:
                print(f"      - {indicator}")
        else:
            print("   ✓ No Mistral contamination detected")
        
        # Print relevant config fields
        print(f"\n   Config fields:")
        print(f"     model_type: {config.get('model_type', 'N/A')}")
        print(f"     tokenizer_class: {config.get('tokenizer_class', 'N/A')}")
        print(f"     chat_template: {config.get('chat_template', 'N/A')[:100]}...")
    else:
        print("   ⚠️  tokenizer_config.json not found")
    
    # Expected values for Llama 3.1
    print(f"\n5. Comparison with Expected Llama 3.1 Values:")
    
    expected = {
        'vocab_size': 128256,
        'bos_token': '<|begin_of_text|>',
        'eos_token': '<|eot_id|>',  # Llama 3.1 Instruct uses <|eot_id|> for chat
        'model_max_length': 131072,
    }
    
    checks = []
    
    if len(tokenizer) == expected['vocab_size']:
        print(f"   ✓ Vocab size matches: {len(tokenizer)}")
        checks.append(True)
    else:
        print(f"   ❌ Vocab size mismatch: {len(tokenizer)} (expected {expected['vocab_size']})")
        checks.append(False)
    
    if tokenizer.bos_token == expected['bos_token']:
        print(f"   ✓ BOS token matches: {tokenizer.bos_token}")
        checks.append(True)
    else:
        print(f"   ❌ BOS token mismatch: {tokenizer.bos_token} (expected {expected['bos_token']})")
        checks.append(False)
    
    if tokenizer.eos_token == expected['eos_token']:
        print(f"   ✓ EOS token matches: {tokenizer.eos_token}")
        checks.append(True)
    else:
        print(f"   ❌ EOS token mismatch: {tokenizer.eos_token} (expected {expected['eos_token']})")
        checks.append(False)
    
    # Check PAD token
    if tokenizer.pad_token is None:
        print(f"   ⚠️  PAD token is None (will be set during training)")
    elif tokenizer.pad_token == tokenizer.eos_token:
        print(f"   ❌ PAD token == EOS token (will cause issues)")
        checks.append(False)
    else:
        print(f"   ✓ PAD token is separate: {tokenizer.pad_token}")
        checks.append(True)
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all(checks):
        print("\n✅ TOKENIZER IS PURE LLAMA 3.1")
        print("\nRecommendation:")
        print("  - Remove warning suppression in 15_train_dpo.py (lines 28-33)")
        print("  - The regex warning is likely a false positive")
        print("  - Proceed with training")
    elif len(tokenizer) == 128256:
        print("\n⚠️  TOKENIZER IS LLAMA 3.1 BUT HAS ISSUES")
        print("\nRecommendation:")
        print("  - Tokenizer is Llama 3.1 base")
        print("  - But may have config contamination or bugs")
        print("  - Try: fix_mistral_regex=True when loading")
        print("  - Or reload clean tokenizer from meta-llama/Llama-3.1-8B-Instruct")
    else:
        print("\n❌ TOKENIZER IS NOT LLAMA 3.1")
        print("\nRecommendation:")
        print("  - Reload clean Llama 3.1 tokenizer from HuggingFace")
        print("  - Use: meta-llama/Llama-3.1-8B-Instruct")
        print("  - Resize model embeddings to match")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    verify_tokenizer()

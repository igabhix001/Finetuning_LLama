#!/usr/bin/env python3
"""
Upload DPO-trained model to HuggingFace Hub
===========================================

Usage:
  python upload_to_huggingface.py --model ./models/llama-3.1-8b-dpo-final --repo YOUR_USERNAME/kp-astrology-dpo

Prerequisites:
  1. Install huggingface_hub: pip install huggingface_hub
  2. Login: huggingface-cli login
  3. Or set HF_TOKEN environment variable
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_model(model_path: str, repo_id: str, private: bool = False):
    """Upload model to HuggingFace Hub"""
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print("="*80)
    print("UPLOADING MODEL TO HUGGINGFACE HUB")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print("="*80)
    
    # Get HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n⚠️  HF_TOKEN not found in environment")
        print("Please login with: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    print(f"\n1. Creating repository: {repo_id}")
    try:
        create_repo(repo_id, private=private, token=hf_token, exist_ok=True)
        print(f"   ✓ Repository created/verified")
    except Exception as e:
        print(f"   ✗ Failed to create repository: {e}")
        return
    
    # Upload model files
    print(f"\n2. Uploading model files...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Upload DPO-trained KP Astrology model"
        )
        print(f"   ✓ Model uploaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to upload model: {e}")
        return
    
    print("\n" + "="*80)
    print("✅ UPLOAD COMPLETE")
    print("="*80)
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print("\nTo use the model:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to model directory (e.g., ./models/llama-3.1-8b-dpo-final)")
    parser.add_argument("--repo", type=str, required=True,
                       help="HuggingFace repo ID (e.g., YOUR_USERNAME/kp-astrology-dpo)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private (default: public)")
    
    args = parser.parse_args()
    
    upload_model(args.model, args.repo, args.private)

if __name__ == "__main__":
    main()

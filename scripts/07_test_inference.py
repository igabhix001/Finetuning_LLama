"""
Test model inference with sample KP astrology questions
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("="*80)
print("TESTING MODEL INFERENCE")
print("="*80)

# Try quantized model first, fallback to merged model
quantized_path = Path("./models/quantized_8bit/")
merged_path = Path("./models/merged/")

# Check which path has actual model files (not just empty dirs)
def has_model_files(p):
    return p.exists() and any(p.glob("*.safetensors")) or any(p.glob("model.pt"))

if has_model_files(quantized_path):
    model_path = quantized_path
    print(f"Using quantized model: {model_path}")
elif has_model_files(merged_path):
    model_path = merged_path
    print(f"Using merged model: {model_path}")
else:
    print("❌ No trained model found")
    print("Please run merge script first: python scripts/05_merge_adapters.py")
    sys.exit(1)

# Load model
print("\n1. Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"✓ Model loaded from: {model_path}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Create pipeline
print("\n2. Creating inference pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(f"✓ Pipeline ready")

# Test questions
test_questions = [
    "What does the 7th house sub-lord signify in marriage timing according to KP astrology?",
    "Explain the role of Venus in KP astrology for relationships and marriage.",
    "How do I calculate the ruling planets for a horary question in KP system?",
    "What is the significance of the 11th cusp sub-lord for financial gains?",
    "Describe the Mahadasha-Antardasha system in KP astrology."
]

print("\n3. Testing predictions...")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(test_questions)}")
    print(f"{'='*80}")
    print(f"Question: {question}")
    print(f"{'-'*80}")
    
    # Format prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate
    try:
        response = pipe(prompt, return_full_text=False)[0]['generated_text']
        
        # Clean up response
        response = response.split("<|eot_id|>")[0].strip()
        
        print(f"Answer:\n{response}")
        
        # Check for quality indicators
        has_rules = "rules_used:" in response.lower() or "rule" in response.lower()
        has_confidence = "confidence:" in response.lower()
        
        print(f"\n{'─'*80}")
        print(f"Quality Check:")
        print(f"  ✓ Rule citations: {'Yes' if has_rules else 'No'}")
        print(f"  ✓ Confidence level: {'Yes' if has_confidence else 'No'}")
        print(f"  ✓ Length: {len(response)} characters")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

print(f"\n{'='*80}")
print("TESTING COMPLETE")
print(f"{'='*80}")
print("\nQuality Assessment:")
print("  - Check if answers cite KP rules")
print("  - Verify proper KP terminology usage")
print("  - Ensure confidence levels are included")
print("  - Validate answer coherence and accuracy")
print("\nIf quality is good, model is ready for deployment!")
print(f"{'='*80}\n")

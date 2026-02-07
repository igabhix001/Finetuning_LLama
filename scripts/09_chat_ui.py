"""
Gradio Chat UI for KP Astrology LLM — connects to the vLLM server.

Usage:
  # Start vLLM server first (in another terminal):
  python scripts/08_serve_vllm.py

  # Then start this UI:
  python scripts/09_chat_ui.py

  # The UI will be available at http://0.0.0.0:7860
  # Share publicly with --share flag:
  python scripts/09_chat_ui.py --share
"""

import argparse
import gradio as gr
from openai import OpenAI

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KP Astrology Chat UI")
parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                    help="vLLM server URL")
parser.add_argument("--port", type=int, default=7860, help="Gradio UI port")
parser.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link")
args = parser.parse_args()

# ── Connect to vLLM backend ──────────────────────────────────────────────────
client = OpenAI(base_url=args.vllm_url, api_key="not-needed")

SYSTEM_PROMPT = """You are an expert KP (Krishnamurti Paddhati) Astrology assistant. You provide accurate, detailed predictions and explanations based on KP astrology principles.

Guidelines:
- Always cite specific KP rules when applicable (e.g., rules_used: KP_MAR_0673)
- Include a confidence level (high/medium/low) for predictions
- Use proper KP terminology: sub-lord, cusp, significator, nakshatra, dasha-bhukti
- Explain the reasoning step by step
- Be respectful and professional"""


def predict(message, history):
    """Stream a response from the vLLM server. History is list of [user, bot] tuples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model="kp-astrology-llama",
            messages=messages,
            max_tokens=768,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                partial += delta
                yield partial
    except Exception as e:
        yield f"Error: {e}\n\nMake sure vLLM is running: python scripts/08_serve_vllm.py"


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What does the 7th house sub-lord signify in marriage timing?",
    "How to predict career success using KP astrology?",
    "Explain Venus's role in KP astrology for relationships.",
    "How do I calculate ruling planets for a horary question?",
    "What is the 11th cusp sub-lord's significance for financial gains?",
    "Describe the Mahadasha-Antardasha system in KP astrology.",
]

# ── Build Gradio UI (compatible with Gradio 4.x and 5.x) ────────────────────
demo = gr.ChatInterface(
    fn=predict,
    title="KP Astrology AI Assistant",
    description=(
        "**Powered by fine-tuned Llama 3.1 8B** — trained on Krishnamurti Paddhati texts\n\n"
        "Ask any question about KP astrology — marriage timing, career predictions, "
        "horary analysis, dasha periods, and more."
    ),
    examples=EXAMPLES,
    cache_examples=False,
)

print(f"\n{'='*60}")
print(f"  KP Astrology Chat UI")
print(f"  Local:  http://0.0.0.0:{args.port}")
if args.share:
    print(f"  Public: will be shown after launch")
print(f"  vLLM:   {args.vllm_url}")
print(f"{'='*60}\n")

demo.launch(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
    show_error=True,
)

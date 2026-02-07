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
    """Stream a response from the vLLM server."""
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
            max_tokens=1024,
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
        yield f"Error connecting to vLLM server: {e}\n\nMake sure the server is running:\n  python scripts/08_serve_vllm.py"


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What does the 7th house sub-lord signify in marriage timing according to KP astrology?",
    "How to predict career success using KP astrology?",
    "Explain the role of Venus in KP astrology for relationships.",
    "How do I calculate the ruling planets for a horary question?",
    "What is the significance of the 11th cusp sub-lord for financial gains?",
    "Describe the Mahadasha-Antardasha system in KP astrology.",
]

# ── Build Gradio UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="KP Astrology AI Assistant",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="amber"),
    css="""
    .gradio-container { max-width: 900px !important; margin: auto; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown(
        """
        # 🔮 KP Astrology AI Assistant
        ### Powered by fine-tuned Llama 3.1 8B — trained on Krishnamurti Paddhati texts

        Ask any question about KP astrology — marriage timing, career predictions,
        horary analysis, dasha periods, and more.
        """
    )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/crystal-ball_1f52e.png"),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a KP astrology question...",
            show_label=False,
            scale=9,
            container=False,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
        retry_btn = gr.Button("🔄 Retry", size="sm")

    gr.Markdown("### 💡 Try these questions:")
    with gr.Row():
        for i in range(0, len(EXAMPLES), 2):
            with gr.Column():
                for ex in EXAMPLES[i:i+2]:
                    gr.Button(ex, size="sm").click(
                        fn=lambda e=ex: e, outputs=msg
                    ).then(
                        fn=predict,
                        inputs=[msg, chatbot],
                        outputs=chatbot,
                    ).then(
                        fn=lambda: "", outputs=msg
                    )

    # Wire up main chat
    def user_submit(message, history):
        return "", history + [[message, None]]

    def bot_respond(history):
        message = history[-1][0]
        history[-1][1] = ""
        for partial in predict(message, history[:-1]):
            history[-1][1] = partial
            yield history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    clear_btn.click(lambda: [], outputs=chatbot)

    def retry_last(history):
        if history and history[-1][0]:
            history[-1][1] = ""
            for partial in predict(history[-1][0], history[:-1]):
                history[-1][1] = partial
                yield history

    retry_btn.click(retry_last, chatbot, chatbot)

    gr.Markdown(
        """
        ---
        *Model: Llama 3.1 8B fine-tuned on KP astrology corpus (DAPT + SFT) •
        RAG: 1,207 KP rules indexed in Pinecone •
        Serving: vLLM on RunPod*
        """
    )

print(f"\n{'='*60}")
print(f"  KP Astrology Chat UI")
print(f"  Local:  http://0.0.0.0:{args.port}")
if args.share:
    print(f"  Public: will be shown after launch")
print(f"  vLLM:   {args.vllm_url}")
print(f"{'='*60}\n")

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=args.port,
    share=args.share,
    show_error=True,
)

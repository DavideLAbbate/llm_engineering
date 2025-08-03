import os
import logging
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
import gradio as gr
import anthropic
import openai

# --- ENV & LOGGING ---
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Reasoning Object ---
class Reasoning:
    def __init__(self, raw_markdown=""):
        self.raw_markdown = raw_markdown.strip()

    def as_dict(self, user_input=""):
        return {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "reasoning_steps": self.raw_markdown
        }

    def to_markdown(self):
        if not self.raw_markdown:
            return "### Piano d'azione\n_Richiesta semplice, nessuna scomposizione necessaria._"
        return (
            "### Piano d'azione\n"
            f"{self.raw_markdown}\n"
        )

# --- Reasoning Logging ---
class ReasoningLogger:
    def __init__(self, csv_path="reasoning_log.csv", jsonl_path="reasoning_log.jsonl"):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.csv_fields = ["timestamp", "user_input", "reasoning_steps"]
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                writer.writeheader()
        if not os.path.isfile(self.jsonl_path):
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                pass

    def log(self, reasoning: Reasoning, user_input: str):
        row = reasoning.as_dict(user_input)
        with open(self.csv_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

reasoning_logger = ReasoningLogger()

# --- SYSTEM PROMPT BASE ---
SYSTEM_RULES = [
    "Always respond in Markdown.",
    "Be concise, precise and focused.",
    "Break complex tasks into sub-steps.",
    "Validate the final answer against the original input.",
    "Pause and ask if input is ambiguous.",
    "Do not invent information.",
    "Explain your method before summarizing or interpreting content.",
    "Always refer to yourself as 'Sara' when speaking in first person."
]
SYSTEM_GOAL = (
    "Your name is Sara. You are a rigorous and professional AI assistant. "
    "Clarity and correctness over speed."
)
SYSTEM_MESSAGE_BASE = SYSTEM_GOAL + "\n\n" + "\n".join(f"- {rule}" for rule in SYSTEM_RULES)

# --- Reasoning Extraction ---
if anthropic_api_key:
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
else:
    claude_client = None
    logger.warning("ANTHROPIC_API_KEY non trovato. Reasoning pattern extraction non disponibile.")

def estrai_reasoning_pattern(prompt_utente, claude_client, model="claude-3-5-sonnet-20240620"):
    if not claude_client:
        return Reasoning(raw_markdown="")

    # Using the new prompt provided by the user
    claude_prompt = (
        "Agisci come un 'Reasoning Extractor AI'. Il tuo compito è ricevere un prompt da un utente umano e analizzarlo semanticamente, senza fornire risposte.\n"
        "Scomponi la richiesta in una lista ordinata di passaggi logici, operazioni, decisioni o chiarimenti che un assistente AI dovrebbe seguire per eseguire il task in modo corretto.\n\n"
        "❌ Non rispondere al prompt.\n"
        "❌ Non fornire soluzioni o spiegazioni.\n"
        "✅ Produci **solo** una lista di step in formato Markdown (con `1.`, `2.`, ecc).\n"
        "✅ Ogni step deve essere breve, chiaro, e mirato all’azione.\n\n"
        f"### Prompt utente:\n{prompt_utente.strip()}"
    )

    try:
        response = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            stream=False,
            temperature=0.0,  # Set to 0 for deterministic output
            messages=[{"role": "user", "content": claude_prompt}]
        )
        text = response.content[0].text.strip() if response.content else ""
        logger.info(f"Raw response from Claude for reasoning pattern:\n---\n{text}\n---")
        
        # The function now returns the raw text wrapped in the Reasoning object
        return Reasoning(raw_markdown=text)
    except Exception as e:
        logger.error(f"Errore nella chiamata a Claude: {e}")
        return Reasoning(raw_markdown="Errore durante l'estrazione del ragionamento.")

# --- SaraChatbot senza stato complesso ---
class SaraChatbot:
    def __init__(self, api_key, system_message, history=None, max_turns=10):
        self.api_key = api_key
        self.system_message = system_message
        self.max_turns = max_turns
        self.history = history if history is not None else []
        if not self.api_key:
            logger.error("OPENAI_API_KEY non trovato.")
            raise ValueError("OPENAI_API_KEY non trovato.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def trim_history(self):
        self.history = [msg for i, msg in enumerate(self.history) if not (msg.get("role") == "system" and i > 0)]
        if len(self.history) > 0 and self.history[0]["role"] == "system":
            keep = 1 + self.max_turns * 2
            self.history = self.history[:1] + self.history[-(keep-1):]
        else:
            self.history = self.history[-self.max_turns*2:]

    def build_prompt(self, user_input):
        if not self.history or self.history[0]["role"] != "system":
            self.history.insert(0, {"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": user_input})
        self.trim_history()
        logger.debug(f"Prompt costruito: {self.history}")
        return self.history

    def ask_stream(self, user_input, reasoning: Reasoning):
        self.system_message = SYSTEM_MESSAGE_BASE + "\n\n" + reasoning.to_markdown()
        self.history = [msg for msg in self.history if msg.get("role") != "system"]
        self.history.insert(0, {"role": "system", "content": self.system_message})
        self.build_prompt(user_input)
        self.history.append({"role": "assistant", "content": ""})
        return self.history

    def replace_last_assistant_message(self, new_content):
        for i in range(len(self.history)-1, -1, -1):
            if self.history[i]["role"] == "assistant":
                self.history[i]["content"] = new_content
                break

    def get_history_for_ui(self):
        pairs = []
        user_msg = None
        for msg in self.history:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg is not None:
                pairs.append((user_msg, msg["content"]))
                user_msg = None
        return pairs

# --- Streaming Chat Interface ---
def chat_stream(user_input, history):
    logger.info(f"User: {user_input}")

    # 1. Extract structured reasoning
    reasoning = estrai_reasoning_pattern(user_input, claude_client)
    
    # 2. Log the extracted reasoning pattern
    logger.info(f"Piano d'azione estratto:\n{reasoning.to_markdown()}")
    reasoning_logger.log(reasoning, user_input)

    # 3. Build chatbot with reasoning injected into the system prompt
    chatbot_obj = SaraChatbot(openai_api_key, SYSTEM_MESSAGE_BASE, history=history)
    _ = chatbot_obj.ask_stream(user_input, reasoning)
    chat_history = chatbot_obj.get_history_for_ui()
    yield chat_history, chatbot_obj.history, ""

    messages = chatbot_obj.history.copy()
    try:
        full_response = ""
        response = chatbot_obj.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                chatbot_obj.replace_last_assistant_message(full_response)
                chat_history = chatbot_obj.get_history_for_ui()
                yield chat_history, chatbot_obj.history, ""
        
        chatbot_obj.replace_last_assistant_message(full_response)
        # Log Sara's final response
        logger.info(f"Sara: {full_response}")
    except Exception as e:
        logger.error(f"Errore nello streaming OpenAI: {e}")
        error_message = "Si è verificato un errore nell'elaborazione della richiesta."
        chatbot_obj.replace_last_assistant_message(error_message)
        chat_history = chatbot_obj.get_history_for_ui()
        yield chat_history, chatbot_obj.history, ""

def reset_chat(history):
    return [], [], ""

# --- UI ---
def main():
    with gr.Blocks(theme=gr.themes.Soft(), css="body {background: #181A20;}") as demo:
        gr.Markdown(
            "<h1 style='color:#000;'>Sara - AI Chatbot Demo</h1>"
            "<p style='color:#bbb;'>Assistant professionale, rigorosa e precisa.</p>"
        )
        chatbot_state = gr.State([])

        with gr.Row():
            chatbot_ui = gr.Chatbot(
                label="Chat con Sara",
                elem_id="chatbot",
                height=900,
                bubble_full_width=True,
                avatar_images=("https://cdn-icons-png.flaticon.com/512/1946/1946429.png", None),
            )

        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Scrivi qui...",
                elem_id="input",
                container=False,
                scale=8
            )
            send_btn = gr.Button("Invia", elem_id="send", scale=1)
            reset_btn = gr.Button("Reset", elem_id="reset", scale=1)

        send_btn.click(
            fn=chat_stream,
            inputs=[user_input, chatbot_state],
            outputs=[chatbot_ui, chatbot_state, user_input],
            queue=True,
            show_progress=True,
            api_name="stream_chat"
        )
        user_input.submit(
            fn=chat_stream,
            inputs=[user_input, chatbot_state],
            outputs=[chatbot_ui, chatbot_state, user_input],
            queue=True,
            show_progress=True,
            api_name="stream_chat_submit"
        )
        reset_btn.click(
            fn=reset_chat,
            inputs=[chatbot_state],
            outputs=[chatbot_ui, chatbot_state, user_input],
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()

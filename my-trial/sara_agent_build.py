import os
import logging
from dotenv import load_dotenv
import openai
import anthropic
import gradio as gr

# ENVIRONMENT & LOGGING
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_api_key:
    raise Exception("OPENAI_API_KEY non presente nell'.env'")
if not anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY non presente nell'.env'")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/chatbot.log"),
        logging.StreamHandler()
    ]
)

# ================ Step 1: Reasoning extraction with Claude ================
def extract_reasoning_pattern_claude(user_prompt, anthropic_api_key):
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    prompt = (
        "Agisci come un 'Reasoning Extractor AI'. Il tuo compito è ricevere un prompt da un utente umano e analizzarlo semanticamente, senza fornire risposte.\n"
        "Scomponi la richiesta in una lista ordinata di passaggi logici, operazioni, decisioni o chiarimenti che un assistente AI dovrebbe seguire per eseguire il task in modo corretto.\n\n"
        "❌ Non rispondere al prompt.\n"
        "❌ Non fornire soluzioni o spiegazioni.\n"
        "✅ Produci **solo** una lista di step in formato Markdown (con `1.`, `2.`, ecc).\n"
        "✅ Ogni step deve essere breve, chiaro, e mirato all’azione.\n\n"
        f"### Prompt utente:\n{user_prompt.strip()}"
    )
    response = client.messages.create(
        model="claude-4-sonnet-20250514",  # Oppure "claude-3-sonnet-20240229" se disponibile
        temperature=0.2,
        max_tokens=10000,   # <-- da aggiungere (obbligatorio per Claude)
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text.strip()

# ================ Step 2: Risposta GPT che segue il reasoning pattern ================
BASE_SYSTEM_RULES = [
    "Always respond in Markdown.",
    "Be concise, precise and focused.",
    "Break complex tasks into sub-steps following the reasoning pattern provided.",
    "Validate the final answer against the original input.",
    "Pause and ask if input is ambiguous.",
    "Do not invent information.",
    "Explain your method before summarizing or interpreting content.",
    "Always refer to yourself as 'Sara' when speaking in first person."
]
SYSTEM_GOAL = "Your name is Sara. You are a rigorous and professional AI assistant. Clarity and correctness over speed."
BASE_SYSTEM_MESSAGE = SYSTEM_GOAL + "\n\n" + "\n".join(f"- {rule}" for rule in BASE_SYSTEM_RULES)

class ChatOrchestrator:
    def __init__(self, base_system_message, openai_api_key, anthropic_api_key):
        self.base_system_message = base_system_message
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gpt_client = openai.OpenAI(api_key=openai_api_key)
        self.history = []

    def ask(self, user_input):
        try:
            reasoning_pattern = extract_reasoning_pattern_claude(user_input, self.anthropic_api_key)
        except Exception as e:
            logging.error(f"Errore reasoning pattern Claude: {e}")
            return "Errore nella generazione dello schema analitico del task tramite Claude."
        
        system_message = (
            f"{self.base_system_message}\n\n"
            "## Reasoning pattern (segui scrupolosamente questa guida nel rispondere):\n"
            f"{reasoning_pattern}\n"
        )

        # Limita la history per evitare overflow di tokens
        if len(self.history) > 10:
            self.history = self.history[-10:]
        messages = [{"role": "system", "content": system_message}]
        for entry in self.history:
            messages.append(entry)
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0.6,
            )
            assistant_reply = response.choices[0].message.content
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_reply})
            logging.info(f"[Reasoning]\n{reasoning_pattern}")
            logging.info(f"> Sara: {assistant_reply}")
            return assistant_reply
        except Exception as e:
            logging.error(f"Errore OpenAI API: {e}")
            return "Errore nel generare la risposta. Riprova."

orchestrator = ChatOrchestrator(BASE_SYSTEM_MESSAGE, openai_api_key, anthropic_api_key)

def chat_interface(user_input, history):
    reply = orchestrator.ask(user_input)
    history = history or []
    history.append((user_input, reply))
    return history, history, ""

# --- UI demo minimale (puoi incollare una CSS più bella!) ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        "<div id='header'><h1>👋 Sara - AI Assistant</h1>"
        "<div><small>Reasoning by Claude, execution by GPT • Powered by OpenAI & Anthropic</small></div></div>"
    )
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot([], elem_id="chatbox", height=480)
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Scrivi qui un task AI: es. \"Confronta file excel X con Y e spiegami…\"",
                    label=None,
                    elem_id="user-input",
                    lines=3,
                    scale=8
                )
                submit_btn = gr.Button("Invia", elem_id="submit-btn", scale=1)

            def user_submit(message, chat_history):
                return chat_interface(message, chat_history)

            submit_btn.click(
                user_submit, 
                [user_input, chatbot], 
                [chatbot, chatbot, user_input]
            )
            user_input.submit(
                user_submit, 
                [user_input, chatbot], 
                [chatbot, chatbot, user_input]
            )

demo.launch()
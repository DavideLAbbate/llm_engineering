import logging
import json
import csv
import os
from datetime import datetime
import gradio as gr
import time

# --- Import from local modules ---
from config import SYSTEM_MESSAGE_BASE
from chatbot_state import task_state_manager
from chatbot_orchestrator import run_multi_response_pipeline
from chatbot_verifier import run_advanced_verification
from chatbot_utils import get_claude_response, get_openai_response

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("chatbot.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Legacy classes for logging ---
class Reasoning:
    def __init__(self, raw_markdown=""):
        self.raw_markdown = raw_markdown.strip()
    def to_markdown(self):
        if not self.raw_markdown: return ""
        return f"### Piano d'azione\n{self.raw_markdown}\n"
class ReasoningLogger:
    def __init__(self, path="reasoning_log.csv"):
        self.path = path
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline='', encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "user_input", "reasoning_steps"])
    def log(self, reasoning, user_input):
        with open(self.path, "a", newline='', encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), user_input, reasoning.raw_markdown])
reasoning_logger = ReasoningLogger()

# --- Core AI Functions ---
def analyze_request_requirements(user_prompt: str, history: list) -> dict:
    logger.info("ANALYZER: Analyzing request requirements...")
    with open("prompts/request_analyzer_prompt.txt", "r", encoding="utf-8") as f:
        analyzer_prompt = f.read()
    history_str = json.dumps(history)
    prompt = f"history: {history_str}\n\nuser_input: {user_prompt}"
    raw_response = get_claude_response(analyzer_prompt, prompt, is_json=True)
    try:
        analysis = json.loads(raw_response)
        logger.info(f"ANALYZER: Analysis complete: {analysis}")
        return analysis
    except json.JSONDecodeError:
        logger.error("ANALYZER: Failed to decode analysis JSON. Defaulting to simple task.")
        return {"outputRequireMultipleResponses": False, "numberOfResponse": 1}

def run_simple_workflow(user_prompt: str, history: list):
    yield "Pianificazione (semplice)..."
    with open("prompts/flowchart_generation_prompt.txt", "r", encoding="utf-8") as f:
        flowchart_prompt = f.read()
    flowchart_str = get_claude_response(flowchart_prompt, user_prompt)
    yield "Prioritizzazione (semplice)..."
    with open("prompts/prioritization_prompt.txt", "r", encoding="utf-8") as f:
        prioritizer_prompt = f.read()
    prioritized_flowchart_str = get_claude_response(prioritizer_prompt, f"Flowchart:\n{flowchart_str}")
    reasoning_logger.log(Reasoning(prioritized_flowchart_str), user_prompt)
    yield "Esecuzione (semplice)..."
    system_prompt = f"{SYSTEM_MESSAGE_BASE}\n\n{Reasoning(prioritized_flowchart_str).to_markdown()}"
    final_response = get_openai_response(system_prompt, user_prompt, history)
    yield final_response

# --- Main Chat Stream ---
def chat_stream(user_input: str, history: list):
    history = history or []
    yield history + [[user_input, None]], "Analisi della richiesta..."
    analysis = analyze_request_requirements(user_input, history)
    if analysis.get("numberOfResponse", 1) > 10:
        yield history + [[user_input, "Errore: la richiesta è troppo complessa. Chiedi aiuto a un amministratore."]], "Errore"
        return
    is_new_task = not task_state_manager.is_active or analysis.get('currentContextPart') == 1
    if analysis.get("outputRequireMultipleResponses"):
        if is_new_task:
            task_state_manager.reset()
            yield history + [[user_input, None]], "Pianificazione del flusso di lavoro complesso..."
            with open("prompts/flowchart_generation_prompt.txt", "r", encoding="utf-8") as f:
                flowchart_prompt = f.read()
            flowchart_str = get_claude_response(flowchart_prompt, user_input)
            with open("prompts/prioritization_prompt.txt", "r", encoding="utf-8") as f:
                prioritizer_prompt = f.read()
            prioritized_flowchart_str = get_claude_response(prioritizer_prompt, f"Flowchart:\n{flowchart_str}")
            task_state_manager.start_task(analysis, prioritized_flowchart_str)
            reasoning_logger.log(Reasoning(prioritized_flowchart_str), user_input)
        for status in run_multi_response_pipeline(user_input, history):
            yield history + [[user_input, None]], status
        verification_ok = False
        for status in run_advanced_verification(history, task_state_manager.flowchart):
            yield history + [[user_input, None]], status
            if "successo" in status:
                verification_ok = True
        if verification_ok:
            all_responses = task_state_manager.get_all_responses()
            full_response = "\n\n---\n\n".join([r['content'] for r in all_responses])
            history.append([user_input, ""])
            for i in range(0, len(full_response), 20):
                chunk = full_response[i:i+20]
                history[-1][1] += chunk
                yield history, "Streaming della risposta finale..."
                time.sleep(0.01)
        else:
            history.append([user_input, "Elaborazione fallita dopo i tentativi di correzione."])
            yield history, "Fallito"
        task_state_manager.reset()
    else:
        final_bot_response = ""
        for response in run_simple_workflow(user_input, history):
            final_bot_response = response
            yield history + [[user_input, None]], "Elaborazione richiesta semplice..."
        history.append([user_input, final_bot_response])
        yield history, "Completato"

# --- UI ---
def main():
    with gr.Blocks(theme=gr.themes.Soft(), css="body {background: #181A20;}") as demo:
        gr.Markdown("<h1 style='color:#fff;'>Sara - AI Chatbot v9</h1><p style='color:#bbb;'>Assistant con orchestrazione multi-risposta e verifica avanzata.</p>")
        chatbot = gr.Chatbot(label="Chat con Sara", elem_id="chatbot", height=600, bubble_full_width=True, avatar_images=("https://cdn-icons-png.flaticon.com/512/1946/1946429.png", None))
        status_display = gr.Textbox(label="Stato", interactive=False, placeholder="L'agente è in attesa...")
        
        with gr.Row():
            user_input = gr.Textbox(show_label=False, placeholder="Scrivi qui...", elem_id="input", container=False, scale=8)
            send_btn = gr.Button("Invia", elem_id="send", scale=1)

        # --- INIZIO DELLA CORREZIONE DEFINITIVA ---
        # Applichiamo la dependency injection di Gradio. Questo è il modo corretto
        # per dire a FastAPI di non validare l'oggetto Request come parte del body.
        def handle_submit(user_msg, chat_history, request: gr.Request):
            logger.info(f"HANDLE_SUBMIT: Richiesta ricevuta da {request.client.host}. La dependency injection ha funzionato.")
            current_history = chat_history or []
            for history_update, status_update in chat_stream(user_msg, current_history):
                yield history_update, status_update, ""
        # --- FINE DELLA CORREZIONE DEFINITIVA ---
        
        # La chiamata a .click() non cambia. Gradio gestisce l'iniezione automaticamente.
        send_btn.click(handle_submit, [user_input, chatbot], [chatbot, status_display, user_input])
        user_input.submit(handle_submit, [user_input, chatbot], [chatbot, status_display, user_input])

    demo.launch(share=True)

if __name__ == "__main__":
    main()

import os
import logging
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
import gradio as gr
import anthropic
import openai
import time
from openai import OpenAI

# --- UTF-8 console & robust logging handlers (injected) ---
import sys, logging
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True,  # reset any prior basicConfig
)
# ----------------------------------------------------------


def _short(s, maxlen=4000):
    try:
        s = str(s)
    except Exception:
        return s
    return s if len(s) <= maxlen else s[:maxlen] + " …[truncated]"



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

# --- CLIENTS ---
if not openai_api_key or not anthropic_api_key:
    raise ValueError("OPENAI_API_KEY and ANTHROPIC_API_KEY must be set.")
openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)


# --- REASONING & LOGGING CLASSES (Unchanged) ---
class Reasoning:
    def __init__(self, raw_markdown=""):
        self.raw_markdown = raw_markdown.strip()

    def as_dict(self, user_input=""):
        return {"timestamp": datetime.now().isoformat(), "user_input": user_input, "reasoning_steps": self.raw_markdown}

    def to_markdown(self):
        if not self.raw_markdown:
            return "### Piano d'azione\n_Richiesta semplice, nessuna scomposizione necessaria._"
        return f"### Piano d'azione\n{self.raw_markdown}\n"

class ReasoningLogger:
    def __init__(self, csv_path="reasoning_log.csv", jsonl_path="reasoning_log.jsonl"):
        self.csv_path, self.jsonl_path = csv_path, jsonl_path
        self.csv_fields = ["timestamp", "user_input", "reasoning_steps"]
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline='', encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()
        if not os.path.isfile(self.jsonl_path):
            open(self.jsonl_path, "w").close()

    def log(self, reasoning: Reasoning, user_input: str):
        row = reasoning.as_dict(user_input)
        with open(self.csv_path, "a", newline='', encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writerow(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

reasoning_logger = ReasoningLogger()

# --- SYSTEM PROMPT BASE ---
SYSTEM_RULES = ["Always respond in Markdown.", "Be concise, precise and focused.", "Break complex tasks into sub-steps.", "Validate the final answer against the original input.", "Pause and ask if input is ambiguous.", "Do not invent information.", "Explain your method before summarizing or interpreting content.", "Always refer to yourself as 'Sara' when speaking in first person."]
SYSTEM_GOAL = "Your name is Sara. You are a rigorous and professional AI assistant. Clarity and correctness over speed."
SYSTEM_MESSAGE_BASE = f"{SYSTEM_GOAL}\n\n" + "\n".join(f"- {rule}" for rule in SYSTEM_RULES)


# --- CORE AI FUNCTIONS ---

def is_complex_task(user_prompt: str, history: list) -> bool:
    """Determines if the task is complex using Claude."""
    logger.info("DETERMINING COMPLEXITY: Checking if the task is complex with Claude...")
    
    # Load the complex task determination prompt
    with open("prompts/complex_task_determination_prompt.txt", "r", encoding="utf-8") as f:
        complexity_prompt = f.read()
    
    # Include history in the prompt
    history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    claude_prompt = f"{complexity_prompt}\n\nConversation History:\n{history_text}\n\nUser Input:\n{user_prompt.strip()}"
    
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        text = response.content[0].text.strip().lower()
        logger.info(f"DETERMINING COMPLEXITY: Claude's response: {text}")
        return "yes" in text
    except Exception as e:
        logger.error(f"DETERMINING COMPLEXITY: Error calling Claude for complexity determination: {e}")
        return False

def estrai_piano_azione(user_prompt: str, history: list) -> Reasoning:
    """Generates an initial action plan using Claude."""
    logger.info("PLANNING: Generating initial action plan with Claude...")
    
    # Load the flowchart generation prompt
    with open("prompts/flowchart_generation_prompt.txt", "r", encoding="utf-8") as f:
        flowchart_prompt = f.read()
    
    # Include history in the prompt
    history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    claude_prompt = f"{flowchart_prompt}\n\nConversation History:\n{history_text}\n\nUser Input:\n{user_prompt.strip()}"

    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        text = response.content[0].text.strip()
        logger.info(f"PLANNING: Raw action plan from Claude:\n---\n{text}\n---")
        return Reasoning(raw_markdown=text)
    except Exception as e:
        logger.error(f"PLANNING: Error calling Claude for plan extraction: {e}")
        return Reasoning(raw_markdown="Errore durante la pianificazione.")

def prioritizza_piano_azione(user_prompt: str, piano_iniziale: Reasoning) -> Reasoning:
    """Prioritizes the action plan by marking required steps."""
    logger.info("PRIORITIZING: Identifying critical steps in the plan...")
    if not piano_iniziale.raw_markdown or "Errore" in piano_iniziale.raw_markdown:
        return piano_iniziale

    # Load the prioritization prompt
    with open("prompts/prioritization_prompt.txt", "r", encoding="utf-8") as f:
        prioritizer_prompt = f.read()

    prioritizer_prompt = (
        f"{prioritizer_prompt}\n\n"
        f"### Original User Prompt:\n{user_prompt}\n\n"
        f"### Action Plan to Prioritize:\n{piano_iniziale.raw_markdown}"
    )
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=8192, temperature=0.0,
            messages=[{"role": "user", "content": prioritizer_prompt}]
        )
        text = response.content[0].text.strip()
        logger.info(f"PRIORITIZING: Prioritized action plan\n---\n{_short(text)}\n---")
        return Reasoning(raw_markdown=text)
    except Exception as e:
        logger.error(f"PRIORITIZING: Error during prioritization: {e}")
        return piano_iniziale # Fallback to the original plan

def esegui_piano(user_prompt: str, piano_azione: Reasoning, history: list) -> str:
    logger.info("EXECUTING: Generating initial response with OpenAI...")
    system_message = f"{SYSTEM_MESSAGE_BASE}\n\n{piano_azione.to_markdown()}"
    
    messages = [{"role": "system", "content": system_message}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"EXECUTING: Error calling OpenAI: {e}")
        return "Errore durante l'esecuzione del piano."

def verifica_risposta(user_prompt: str, piano_azione: Reasoning, risposta: str) -> dict:
    logger.info("VERIFYING: Checking response against *required* steps in the action plan...")
    if "✅ required" not in piano_azione.raw_markdown:
        logger.info("VERIFYING: No required steps to verify.")
        return {"passed": True, "reason": "No required steps to verify."}

    verifier_prompt = (
        "You are a meticulous Quality Assurance AI. Your task is to verify if a generated response correctly and completely follows all steps marked as `✅ required` in a given action plan. **Ignore all steps that are not marked as `✅ required`**.\n"
        "Respond with a single word: 'Yes' if all `✅ required` points are met.\n"
        "If not, respond with 'No', followed by a newline and a Markdown list of the specific, numbered `✅ required` points from the action plan that were missed or incorrectly implemented.\n\n"
        f"### Original User Prompt:\n{user_prompt}\n\n"
        f"### Action Plan to Follow:\n{piano_azione.raw_markdown}\n\n"
        f"### Generated Response to Verify:\n\`\`\`\n{risposta}\n\`\`\`\n\n"
        "--- \n"
        "Does the 'Generated Response' fully and correctly implement **only** the steps from the 'Action Plan' that are marked as `✅ required`?"
    )
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=8192, temperature=0.0,
            messages=[{"role": "user", "content": verifier_prompt}]
        )
        verification_text = response.content[0].text.strip()
        logger.info(f"VERIFYING: Verifier response:\n---\n{verification_text}\n---")
        if verification_text.lower().startswith("yes"):
            return {"passed": True}
        else:
            return {"passed": False, "reason": verification_text.replace("No", "").strip()}
    except Exception as e:
        logger.error(f"VERIFYING: Error during verification: {e}")
        return {"passed": True, "reason": "Verification failed, assuming pass."}

def correggi_risposta(risposta_precedente: str, punti_mancanti: str) -> str:
    logger.info(f"CORRECTING: Refining response. Missing required points:\n{punti_mancanti}")
    corrector_system_prompt = (
        "You are a code refactoring and correction AI. You will receive a previous, incorrect response and a list of critical points that were missed from an original plan. "
        "Your task is to rewrite and complete the response to incorporate and fix these missing points. "
        "Do not apologize or explain your changes. Simply provide the full, corrected, final response that satisfies all the listed requirements."
    )
    corrector_user_prompt = (
        f"### Previous Incorrect Response:\n\`\`\`\n{risposta_precedente}\n\`\`\`\n\n"
        f"### The following required points from the original plan were missed or incorrect. Please correct the response to include them:\n{punti_mancanti}"
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[
                {"role": "system", "content": corrector_system_prompt},
                {"role": "user", "content": corrector_user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"CORRECTING: Error during correction: {e}")
        return risposta_precedente

def simple_task_response(user_prompt: str) -> str:
    """Generates a response for simple tasks using Claude."""
    logger.info("SIMPLE TASK: Generating response for a simple task with Claude...")
    
    # Load the simple task prompt
    with open("prompts/simple_task_prompt.txt", "r", encoding="utf-8") as f:
        simple_prompt = f.read()
    
    claude_prompt = f"{simple_prompt}\n\nUser Input:\n{user_prompt.strip()}"
    
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": claude_prompt}]
        )
        text = response.content[0].text.strip()
        logger.info(f"SIMPLE TASK: Claude's response: {text}")
        return text
    except Exception as e:
        logger.error(f"SIMPLE TASK: Error calling Claude for simple task: {e}")
        return "Errore durante l'elaborazione della richiesta."

# --- MAIN CHAT STREAM ---
def chat_stream(user_input: str, history: list):
    history = history or []
    history.append([user_input, ""])
    
    # 1. Determine Complexity
    yield history, "Analisi della complessit della richiesta..."
    is_complex = is_complex_task(user_input, history[:-1])
    logger.info(f"COMPLEXITY ANALYSIS: Task is complex: {is_complex}")

    if is_complex:
        # 2. Plan
        yield history, "Pianificazione..."
        piano_iniziale = estrai_piano_azione(user_input, history[:-1])
        
        # 3. Prioritize
        yield history, "Prioritizzazione del piano..."
        piano_prioritizzato = prioritizza_piano_azione(user_input, piano_iniziale)
        reasoning_logger.log(piano_prioritizzato, user_input)

        # 4. Execute
        yield history, "Esecuzione del piano..."
        risposta_corrente = esegui_piano(user_input, piano_prioritizzato, history[:-1])
        
        # 5. Verify & Correct Loop
        max_retries = 3
        last_failure_reason = None
        for attempt in range(max_retries):
            yield history, f"Verifica dei punti critici ({attempt + 1}/{max_retries})..."
            verification = verifica_risposta(user_input, piano_prioritizzato, risposta_corrente)
            
            if verification["passed"]:
                logger.info("VERIFICATION PASSED: Final response is ready.")
                break
            
            # --- INIZIO NUOVA LOGICA ---
            current_failure_reason = verification.get("reason", "")
            if current_failure_reason and current_failure_reason == last_failure_reason:
                logger.warning(f"REPEATED FAILURE DETECTED. Accepting current response to avoid loop. Reason: {current_failure_reason}")
                break # Esce dal ciclo di correzione
            last_failure_reason = current_failure_reason
            # --- FINE NUOVA LOGICA ---

            logger.warning(f"VERIFICATION FAILED (Attempt {attempt + 1}/{max_retries}). Reason: {verification['reason']}")
            yield history, f"Correzione della risposta (tentativo {attempt + 1})..."
            
            risposta_corrente = correggi_risposta(risposta_corrente, verification["reason"])
        else:
            logger.error("Max correction retries reached. Sending best-effort response.")

        # 6. Final Output Stream
        final_response = risposta_corrente
        history[-1][1] = ""
        yield history, "Verifica completata. Streaming della risposta finale."
        
        for i in range(0, len(final_response), 10):
            chunk = final_response[i:i+10]
            history[-1][1] += chunk
            yield history, "Verifica completata. Streaming della risposta finale."
            time.sleep(0.01)
        
        logger.info(f"Final Answer: {final_response}")
    else:
        # Simple Task Response
        yield history, "Generazione della risposta..."
        
        # Load the simple task prompt
        with open("prompts/simple_task_prompt.txt", "r", encoding="utf-8") as f:
            simple_prompt = f.read()
        
        system_message = f"{SYSTEM_MESSAGE_BASE}\n\n{simple_prompt}"
    
        messages = [{"role": "system", "content": system_message}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": user_input})

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.3
            )
            simple_response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"SIMPLE TASK: Error calling OpenAI: {e}")
            simple_response = "Errore durante l'elaborazione della richiesta."
        
        history[-1][1] = simple_response
        yield history, "Risposta generata."
        logger.info(f"Final Answer (Simple Task): {simple_response}")

# --- UI (Unchanged) ---
def main():
    with gr.Blocks(theme=gr.themes.Soft(), css="body {background: #181A20;}") as demo:
        gr.Markdown("<h1 style='color:#fff;'>Sara - AI Chatbot Demo</h1><p style='color:#bbb;'>Assistant con ciclo di auto-correzione e prioritizzazione.</p>")
        
        chatbot = gr.Chatbot(label="Chat con Sara", elem_id="chatbot", height=600, bubble_full_width=True, avatar_images=("https://cdn-icons-png.flaticon.com/512/1946/1946429.png", None))
        status_display = gr.Textbox(label="Stato", interactive=False, placeholder="L'agente  in attesa...")
        
        with gr.Row():
            user_input = gr.Textbox(show_label=False, placeholder="Scrivi qui...", elem_id="input", container=False, scale=8)
            send_btn = gr.Button("Invia", elem_id="send", scale=1)

        def handle_submit(user_msg, chat_history):
            yield chat_history + [[user_msg, None]], "...", "" 
            
            final_history = None
            final_status = "Errore"
            for history, status in chat_stream(user_msg, chat_history):
                final_history = history
                final_status = status
                yield final_history, final_status, ""
        
        send_btn.click(handle_submit, [user_input, chatbot], [chatbot, status_display, user_input])
        user_input.submit(handle_submit, [user_input, chatbot], [chatbot, status_display, user_input])

    demo.queue().launch(share=True)

if __name__ == "__main__":
    main()
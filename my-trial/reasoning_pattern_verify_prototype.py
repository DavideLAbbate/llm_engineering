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

# === Lightweight Utilities: history recap + hard checkers ===
def _build_history_context(history: list, max_pairs: int = 1) -> list:
    # Keep only the last `max_pairs` user/assistant turns and a compact recap of older context
    history = history or []
    if not history:
        return []
    recent = history[-max_pairs:]
    older = history[:-max_pairs]
    messages = []
    if older:
        bullets = []
        for u, a in older[-8:]:  # cap bullets
            bullets.append(f"- U: {str(u)[:160]}")
            if a:
                bullets.append(f"  A: {str(a)[:160]}")
        recap = "Previous context (compact recap):\n" + "\n".join(bullets)
        messages.append({"role": "system", "content": recap})
    for u, a in recent:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    return messages

_RN_WEB_FLAGS = (\"@mui\", \"material-ui\", \"react-dom\", \"<div\", \"<span\", \"document.\", \"window.\")
def _hard_checks(user_prompt: str, response_text: str) -> list:
    # Domain guards: React Native vs Web; JSON validity if requested
    issues = []
    up = (user_prompt or \"\").lower()
    rt = response_text or \"\"

    # RN vs Web
    if \"react native\" in up or \"react-native\" in up:
        if any(flag in rt for flag in _RN_WEB_FLAGS):
            issues.append(\"Web/DOM or MUI artifacts detected in a React Native task.\")

    # JSON requested -> try parse
    if \"json\" in up:
        import json as _json
        txt = rt.strip()
        try:
            _ = _json.loads(txt)
        except Exception:
            if \"```\" in txt:
                parts = txt.split(\"```\", 2)
                if len(parts) >= 3:
                    inner = parts[1]
                    if inner.strip().startswith(\"json\") and \"\\n\" in inner:
                        inner = inner.split(\"\\n\",1)[1]
                    try:
                        _ = _json.loads(inner)
                    except Exception:
                        issues.append(\"Invalid JSON in response despite being requested.\")
            else:
                issues.append(\"Invalid JSON in response despite being requested.\")
    return issues


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
openai_client = openai.OpenAI(api_key=openai_api_key)
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

def estrai_piano_azione(user_prompt: str) -> Reasoning:
    """Generates an initial action plan using Claude."""
    logger.info("PLANNING: Generating initial action plan with Claude...")
    claude_prompt = (
        "Agisci come un 'Reasoning Extractor AI'. Il tuo compito è ricevere un prompt da un utente umano e analizzarlo semanticamente, senza fornire risposte.\n"
        "Scomponi la richiesta in una lista ordinata di passaggi logici, operazioni, decisioni o chiarimenti che un assistente AI dovrebbe seguire per eseguire il task in modo corretto.\n\n"
        "❌ Non rispondere al prompt.\n"
        "❌ Non fornire soluzioni o spiegazioni.\n"
        "✅ Produci **solo** una lista di step in formato Markdown (con `1.`, `2.`, ecc).\n"
        "✅ Ogni step deve essere breve, chiaro, e mirato all’azione.\n\n"
        f"### Prompt utente:\n{user_prompt.strip()}"
    )
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

    prioritizer_prompt = (
        "You are a 'Task Prioritization AI'. You will receive a user's request and a detailed action plan. Your job is to identify the most critical steps required to fulfill the user's request.\n"
        "- Review the action plan in the context of the user's original request.\n"
        "- Re-write the entire action plan exactly as it was.\n"
        "- For the steps that are absolutely essential to meet the core request, add the marker `(required)` at the end of the line.\n"
        "- Do not add any other commentary. Only output the re-written markdown list.\n\n"
        f"### Original User Prompt:\n{user_prompt}\n\n"
        f"### Action Plan to Prioritize:\n{piano_iniziale.raw_markdown}"
    )
    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=8192, temperature=0.0,
            messages=[{"role": "user", "content": prioritizer_prompt}]
        )
        text = response.content[0].text.strip()
        logger.info(f"PRIORITIZING: Prioritized action plan:\n---\n{text}\n---")
        return Reasoning(raw_markdown=text)
    except Exception as e:
        logger.error(f"PRIORITIZING: Error during prioritization: {e}")
        return piano_iniziale # Fallback to the original plan

def esegui_piano(user_prompt: str, piano_azione: Reasoning, history: list) -> str:
    logger.info("EXECUTING: Generating initial response with OpenAI...")
    system_message = f"{SYSTEM_MESSAGE_BASE}\n\n{piano_azione.to_markdown()}"
    
    messages = [{"role": "system", "content": system_message}]
    messages.extend(_build_history_context(history, max_pairs=1))
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
    if "(required)" not in piano_azione.raw_markdown:
        logger.info("VERIFYING: No required steps found in the plan. Skipping verification.")
        return {"passed": True, "reason": "No required steps to verify."}

    # Hard checks (domain guards)
    hard_issues = _hard_checks(user_prompt, risposta)
    if hard_issues:
        return {"passed": False, "reason": "; ".join(hard_issues)}

    verifier_prompt = ("Return ONLY one of the following. No extra text, no markdown.\n\n""YES\n""OR\n""NO\n- <missing required point 1>\n- <missing required point 2>\n...""\n\nTask: Verify only steps marked as `(required)` in the action plan. If any required step is missing or incorrect, answer NO and list them as bullets." f"\n\n### Original User Prompt:\n{user_prompt}\n\n" f"### Action Plan to Follow:\n{piano_azione.raw_markdown}\n\n" f"### Generated Response to Verify:\n```\n{risposta}\n```\n")
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

# --- MAIN CHAT STREAM ---
def chat_stream(user_input: str, history: list):
    history = history or []
    history.append([user_input, ""])
    
    # 1. Plan
    yield history, "Pianificazione..."
    piano_iniziale = estrai_piano_azione(user_input)
    
    # 2. Prioritize
    yield history, "Prioritizzazione del piano..."
    piano_prioritizzato = prioritizza_piano_azione(user_input, piano_iniziale)
    reasoning_logger.log(piano_prioritizzato, user_input)

    # 3. Execute
    yield history, "Esecuzione del piano..."
    risposta_corrente = esegui_piano(user_input, piano_prioritizzato, history[:-1])
    
    # 4. Verify & Correct Loop
    max_retries = 3
    last_failure_reason = None # Aggiungi questa riga
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

    # 5. Final Output Stream
    final_response = risposta_corrente
    history[-1][1] = ""
    yield history, "Verifica completata. Streaming della risposta finale."
    
    for i in range(0, len(final_response), 10):
        chunk = final_response[i:i+10]
        history[-1][1] += chunk
        yield history, "Verifica completata. Streaming della risposta finale."
        time.sleep(0.01)
    
    logger.info(f"Final Answer: {final_response}")


# --- UI (Unchanged) ---
def main():
    with gr.Blocks(theme=gr.themes.Soft(), css="body {background: #181A20;}") as demo:
        gr.Markdown("<h1 style='color:#fff;'>Sara - AI Chatbot Demo</h1><p style='color:#bbb;'>Assistant con ciclo di auto-correzione e prioritizzazione.</p>")
        
        chatbot = gr.Chatbot(label="Chat con Sara", elem_id="chatbot", height=600, bubble_full_width=True, avatar_images=("https://cdn-icons-png.flaticon.com/512/1946/1946429.png", None))
        status_display = gr.Textbox(label="Stato", interactive=False, placeholder="L'agente è in attesa...")
        
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

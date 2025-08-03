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

from utils_patched import get_claude_response, get_openai_response

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
SYSTEM_MESSAGE_BASE = """
# ROLE
You are **Sara**, an AI assistant created by Dasius — a meticulous and strict programmer who values precision, correctness, and results.
Your ultimate purpose is to be the *perfect assistant*: brilliant, empathetic, and highly effective.

# BACKGROUND
Dasius built you to excel with all types of users:
- **Noob users**: often unaware of the context, requirements, or constraints of their requests.  
- **Pro users**: know exactly what they want and provide complete, clear input.

# CORE OBJECTIVES
1. Be brilliant, empathetic, and professional in tone.
2. Always adapt your approach to the user's skill level.
3. Identify and manage ambiguity or missing information before execution.
4. Deliver concise, precise, and contextually correct results.

# BEHAVIOR RULES
- **Markdown Output**: Always respond in Markdown format for clarity.
- **Concise & Precise**: Focus on relevant details, avoid unnecessary verbosity.
- **Step Breakdown**: Break down complex tasks into clear, ordered sub-steps.
- **Validation**: Validate your final answer against the original user input before responding.
- **Ambiguity Handling**: If the input is unclear or ambiguous, pause and ask clarifying questions.
- **No Hallucinations**: Do not invent or assume information without explicitly marking it as an assumption.
- **Method Transparency**: Explain your reasoning/method before summarizing or interpreting content.
- **Identity**: Always refer to yourself as *Sara* when speaking in the first person.

# USER TYPE HANDLING
- **If user is a noob**:  
  → Politely request all missing information or clarification for any ambiguity found.  
  → Provide guidance to help them refine their request.
- **If user is a pro**:  
  → Respond with the best possible solution directly, concise and accurate.

# OUTPUT EXPECTATIONS
1. Detect the type of user (noob/pro) based on the prompt.
2. For noobs: list missing info / clarifying questions before producing a final solution.
3. For pros: execute task optimally without unnecessary questions, unless ambiguity is detected.
4. Ensure every response passes internal validation against the user’s original input.
"""


# --- CORE AI FUNCTIONS ---

def estrai_piano_azione(user_prompt: str) -> Reasoning:
    """Generates an initial action plan using Claude."""
    logger.info("PLANNING: Generating initial action plan with Claude...")

    clause_system_prompt = (
        "Agisci come un 'Reasoning Extractor AI'. Il tuo compito è ricevere un prompt da un utente umano e analizzarlo semanticamente, senza fornire risposte.\n"
        "Scomponi la richiesta in una lista ordinata di passaggi logici, operazioni, decisioni o chiarimenti che un assistente AI dovrebbe seguire per eseguire il task in modo corretto.\n\n"
        "Ti verrà passato lo user prompt originale"
        "❌ Non rispondere al prompt.\n"
        "❌ Non fornire soluzioni o spiegazioni.\n"
        "✅ Produci **solo** una lista di step in formato Markdown (con `1.`, `2.`, ecc).\n"
        "✅ Ogni step deve essere breve, chiaro, e mirato all’azione.\n\n"
    )
    
    claude_prompt = (
        f"### Prompt utente:\n{user_prompt.strip()}"
    )
    
    text = get_claude_response(clause_system_prompt, claude_prompt)
    logger.info(f"PLANNING: Raw action plan from Claude:\n---\n{text}\n---")
    return Reasoning(raw_markdown=text)

def prioritizza_piano_azione(user_prompt: str, piano_iniziale: Reasoning) -> Reasoning:
    """Prioritizes the action plan by marking required steps."""
    logger.info("PRIORITIZING: Identifying critical steps in the plan...")
    if not piano_iniziale.raw_markdown or "Errore" in piano_iniziale.raw_markdown:
        return piano_iniziale

    prioritizer_system_prompt = (
        "You are a 'Task Prioritization AI'. You will receive a user's request and a detailed action plan. Your job is to identify the most critical steps required to fulfill the user's request.\n"
        "- The original user prompt will be provided in the user prompt"
        "- Review the action plan in the context of the user's original request.\n"
        "- Re-write the entire action plan exactly as it was.\n"
        "- For the steps that are absolutely essential to meet the core request, add the marker `(required)` at the end of the line.\n"
        "- Do not add any other commentary. Only output the re-written markdown list.\n\n"
        f"### Action Plan to Prioritize:\n{piano_iniziale.raw_markdown}"
    )

    prioritizer_user_prompt = (
        f"### Original User Prompt:\n{user_prompt}\n\n"
    )

    text = get_claude_response(prioritizer_system_prompt, prioritizer_user_prompt)
    logger.info(f"PRIORITIZING: Prioritized action plan:\n---\n{text}\n---")
    return Reasoning(raw_markdown=text)

def esegui_piano(user_prompt: str, piano_azione: Reasoning, history: list) -> str:
    logger.info("EXECUTING: Generating initial response with OpenAI...")
    system_message = f"{SYSTEM_MESSAGE_BASE}\n\n{piano_azione.to_markdown()}"
    
    response = get_openai_response(system_message, user_prompt, history)
    return response

def verifica_risposta(user_prompt: str, piano_azione: Reasoning, risposta: str) -> dict:
    logger.info("VERIFYING: Checking response against *required* steps in the action plan...")
    if "(required)" not in piano_azione.raw_markdown:
        logger.info("VERIFYING: No required steps found in the plan. Skipping verification.")
        return {"passed": True, "reason": "No required steps to verify."}

    # USER PROMPT (contenuto da verificare)
    verifier_prompt = (
        f"### Action Plan to Follow:\n{piano_azione.raw_markdown}\n\n"
        f"### Generated Response to Verify:\n```\n{risposta}\n```\n\n"
        "---\n"
        "Does the 'Generated Response' fully and correctly implement ONLY the steps from the 'Action Plan' that are marked as `(required)`?"
    )

    # SYSTEM PROMPT (passato come parametro 'system', NON come messaggio con role=system)
    verifier_system_prompt = (
        "You are a meticulous Quality Assurance AI. Your task is to verify if a generated response correctly and "
        "completely follows all steps marked as `(required)` in a given action plan. Ignore non-required steps.\n"
        "Respond with a single word: 'Yes' if all `(required)` points are met.\n"
        "If not, respond with 'No', followed by a newline and a Markdown list of the specific, numbered `(required)` "
        "points from the action plan that were missed or incorrectly implemented.\n"
        "If an issue cannot be fixed due to ambiguity or missing user info, remove it from the list.\n"
        f"### Original User Prompt:\n{user_prompt}\n"
    )

    try:
        verification_text = get_claude_response(verifier_system_prompt, verifier_prompt)
        logger.info(f"VERIFYING: Verifier response:\n---\n{verification_text}\n---")

        low = verification_text.lower()
        if low.startswith("yes"):
            return {"passed": True}
        # Togli prefisso "No" e spazi: tieni i dettagli come reason
        reason = verification_text
        if reason[:2].lower() == "no":
            reason = reason[2:].strip()
        return {"passed": False, "reason": reason}

    except Exception as e:
        logger.error(f"VERIFYING: Error during verification: {e}")
        # fallback prudente: meglio non bloccare il flusso per un errore temporaneo del verifier
        return {"passed": True, "reason": "Verification failed, assuming pass."}

def correggi_risposta(risposta_precedente: str, punti_mancanti: str, history) -> str:
    logger.info(f"CORRECTING: Refining response. Missing required points:\n{punti_mancanti}")
    corrector_system_prompt = (
        "You are a pragmatical fixer, you focus on fixing provided issues. You will receive a previous, incorrect response and a list of critical points that were missed from an original plan.\n"
        "Your task is to rewrite and complete the response to incorporate and fix these missing points.\n"
        "Do not apologize or explain your changes. Simply provide the full, corrected, final response that satisfies all the listed requirements.\n"
        f"### The following required points from the original plan were missed or incorrect. Please correct the response to include them:\n{punti_mancanti}"

    )
    corrector_user_prompt = (
        f"### Previous Incorrect Response:\n\`\`\`\n{risposta_precedente}\n\`\`\`\n\n"
    )
    response = get_openai_response(corrector_system_prompt, corrector_user_prompt, history)
    return response

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
    max_retries = 10
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
        
        risposta_corrente = correggi_risposta(risposta_corrente, verification["reason"], history)
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

import os
import requests
import datetime
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import pandas as pd
import json
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import gradio as gr

# --- SETUP INIZIALE ---
load_dotenv(override=True)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    print("Tesseract not found at default Windows path. Skipping.")

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
claude_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

if not openai_client: print("ATTENZIONE: OPENAI_API_KEY non trovato.")
if not claude_client: print("ATTENZIONE: ANTHROPIC_API_KEY non trovato.")

# --- DEFINIZIONE DELLA PERSONALITÀ E REGOLE DI BASE ---
SYSTEM_GOAL = (
    "Your name is Sara. You are a rigorous and professional AI assistant. "
    "Clarity and correctness over speed."
)
SYSTEM_RULES = [
    "Always respond in Markdown.", "Be concise, precise and focused.",
    "Break complex tasks into sub-steps.", "Validate the final answer against the original input.",
    "Pause and ask if input is ambiguous.", "Do not invent information.",
    "Always refer to yourself as 'Sara' when speaking in first person."
]

# --- NUOVA TASK REGISTRY INTEGRATA ---
# Questa è la nuova struttura che definisce i task, le parole chiave per la
# classificazione rapida e gli schemi di ragionamento.
TASK_REGISTRY = {
    "image_generation": {
        "keywords": ["draw", "create an image", "generate a picture", "disegna", "crea un immagine", "illustra"],
        "pattern": "" # Non serve un pattern, la logica è gestita a parte
    },
    "email_composition": {
        "keywords": ["write email", "send email", "draft message", "compose email"],
        "pattern": """- Clarify sender, recipient, subject and tone.\n- Identify purpose: inform, request, escalate, etc.\n- Draft a clean body in bullet or paragraph style.\n- Review for clarity, tone and correctness."""
    },
    "decision_support": {
        "keywords": ["what should I do", "which one to choose", "recommend", "suggest"],
        "pattern": """- Ask for the user's goal or constraint.\n- List pros and cons of each option.\n- Consider risk, effort and expected outcome.\n- Make a clear recommendation with reasoning."""
    },
    "document_summarization": {
        "keywords": ["summarize", "make a summary", "synthesize", "extract key points"],
        "pattern": """- Detect the document type (legal, technical, general).\n- Highlight sections, headings, conclusions.\n- Extract and compress main points in logical order.\n- Add clarifications or warnings if needed."""
    },
    "task_planning": {
        "keywords": ["plan", "schedule", "organize tasks", "todo list"],
        "pattern": """- Ask for context, priorities and deadlines.\n- Break goal into logical subtasks.\n- Order by urgency or dependencies.\n- Deliver a clear checklist or plan."""
    },
    "explanation_request": {
        "keywords": ["explain", "help me understand", "what does it mean", "how does it work"],
        "pattern": """- Identify user's current knowledge level.\n- Use analogy, examples or steps to simplify.\n- Verify understanding with follow-up or quiz.\n- Avoid jargon unless requested."""
    },
    "code_review": {
        "keywords": ["review this code", "check my script", "optimize code", "debug"],
        "pattern": """- Read the code and identify structure.\n- Look for anti-patterns or inefficiencies.\n- Suggest modularization or improvements.\n- Test or pseudo-test logic if needed."""
    },
    "financial_guidance": {
        "keywords": ["how to save", "where to invest", "manage money", "budget"],
        "pattern": """- Clarify user's income, risk tolerance, and goals.\n- Categorize needs (short, medium, long term).\n- Suggest practical, legal and safe actions.\n- Add disclaimers if legally required."""
    },
    "personal_growth": {
        "keywords": ["self-improvement", "get better at", "overcome", "build habit"],
        "pattern": """- Clarify area of improvement and blockers.\n- Provide a step-by-step realistic path.\n- Recommend tracking and accountability tools.\n- Highlight expected obstacles and how to overcome."""
    },
    "conflict_resolution": {
        "keywords": ["solve problem", "deal with colleague", "argument", "relationship issue"],
        "pattern": """- Ask for detailed context and positions.\n- Separate facts from interpretations.\n- Offer neutral language reframing.\n- Suggest steps to de-escalate and align."""
    },
    "learning_support": {
        "keywords": ["help me study", "teach me", "learn this", "understand topic"],
        "pattern": """- Ask for learning goals and time available.\n- Break topic into digestible parts.\n- Provide examples and self-tests.\n- Suggest learning resources and schedule."""
    },
    "presentation_help": {
        "keywords": ["create slides", "prepare presentation", "build pitch", "public speaking"],
        "pattern": """- Ask for audience type and goal.\n- Outline structure: intro, body, conclusion.\n- Provide slide text and speaker notes.\n- Give tips for delivery and timing."""
    },
    "job_search": {
        "keywords": ["update CV", "prepare interview", "career advice", "write cover letter"],
        "pattern": """- Clarify target role or sector.\n- Optimize CV for clarity and impact.\n- Simulate common interview Q&A.\n- Tailor motivation letter based on job description."""
    },
    "market_research": {
        "keywords": ["analyze competitor", "market trends", "compare products"],
        "pattern": """- Define scope and evaluation criteria.\n- Extract and organize relevant data.\n- Compare pros, cons, and pricing.\n- Provide actionable insights or summary."""
    },
    "idea_validation": {
        "keywords": ["is this idea good", "can this work", "business plan", "startup"],
        "pattern": """- Identify core value proposition.\n- Evaluate feasibility, demand, and risk.\n- Suggest MVP and early test strategies.\n- Flag obvious red flags or blockers."""
    },
    "writing_support": {
        "keywords": ["rewrite", "improve writing", "fix this text", "make it better"],
        "pattern": """- Understand purpose and tone of the text.\n- Restructure for clarity and grammar.\n- Keep original meaning intact.\n- Provide before/after comparison."""
    },
    "technical_troubleshooting": {
        "keywords": ["fix error", "this is not working", "why is it failing"],
        "pattern": """- Ask for full error context and system.\n- Replicate or simulate issue if possible.\n- Identify root cause and workaround.\n- Suggest fix and future-proofing."""
    },
    "event_organization": {
        "keywords": ["plan event", "organize meeting", "book", "schedule"],
        "pattern": """- Ask for constraints (time, budget, people).\n- Break event into phases (prep, live, wrap).\n- Create checklists and timeline.\n- Provide fallback or contingency plans."""
    },
    "creative_generation": {
        "keywords": ["write story", "brainstorm", "generate idea", "creative help"],
        "pattern": """- Ask for tone, target and inspiration.\n- Generate several diverse options.\n- Improve and mix selected idea.\n- Provide structure or continuation path."""
    },
    "general_conversation": {
        "keywords": [], # Fallback, no keywords
        "pattern": """- Adopt your standard persona of a helpful and pragmatic assistant.\n- Listen carefully to the user's query or statement.\n- Provide a direct, clear, and helpful response.\n- Maintain a professional yet approachable tone."""
    }
}

LOG_FILE_PATH = "chat_history_log.txt"

# --- FUNZIONI DI LOGGING E UTILITY (INVARIATE) ---
def log_conversation(history: list, model: str, files: list = None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def get_file_name(f): return os.path.basename(getattr(f, 'name', 'UNKNOWN'))
    file_list = ', '.join([get_file_name(f) for f in files]) if files else "N/A"
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as logf:
            logf.write(f"--- {timestamp} | Model: {model} | Files: {file_list} ---\n")
            for user_msg, bot_msg in history:
                logf.write(f"User: {user_msg or ''}\nSara: {bot_msg or ''}\n---\n")
            logf.write("\n")
    except Exception as e: print(f"[LOGGING ERROR]: {e}")

def extract_text_from_file(file_path: str) -> str:
    if not file_path: return ""
    # ... (il resto della funzione è invariato)
    name = file_path.lower()
    content = ""
    try:
        if name.endswith((".xls", ".xlsx")): content = pd.read_excel(file_path).to_string()
        elif name.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f: content = json.dumps(json.load(f), indent=2)
        elif name.endswith(".docx"): content = "\n".join(p.text for p in Document(file_path).paragraphs)
        elif name.endswith(".pdf"):
            doc = fitz.open(file_path)
            content = "\n".join(page.get_text() for page in doc)
            if not content.strip():
                ocr_text = [pytesseract.image_to_string(Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))) for page in doc]
                content = "\n".join(ocr_text) or "[OCR failed or PDF is empty]"
        elif name.endswith((".png", ".jpg", ".jpeg")): content = pytesseract.image_to_string(Image.open(file_path)) or "[Image uploaded, no text detected by OCR]"
        else: content = "[Unsupported file type]"
    except Exception as e: content = f"[Error reading file: {e}]"
    return content

# --- NUOVA FUNZIONE DI CLASSIFICAZIONE IBRIDA ---
def classify_task(prompt: str) -> str:
    """
    Classifica il task con un approccio ibrido: prima keyword, poi LLM.
    """
    # 1. Classificazione rapida basata su keyword
    prompt_lower = prompt.lower()
    for task_key, data in TASK_REGISTRY.items():
        if any(keyword in prompt_lower for keyword in data["keywords"]):
            return task_key

    # 2. Fallback a classificazione con LLM se nessuna keyword corrisponde
    if not openai_client: return "general_conversation"
    
    task_keys = list(TASK_REGISTRY.keys())
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a task classifier. Categorize the user's request into ONE of the following keys: {', '.join(task_keys)}. Respond with ONLY the key. Default to 'general_conversation' if unsure or multiple apply."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20, temperature=0.0
        )
        decision = response.choices[0].message.content.strip()
        return decision if decision in task_keys else "general_conversation"
    except Exception as e:
        print(f"Error in LLM task classification: {e}")
        return "general_conversation"

def generate_openai_image(prompt: str) -> str:
    if not openai_client: return "OpenAI client not configured."
    try:
        response = openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024", response_format="url")
        return f"![{prompt}]({response.data[0].url})"
    except Exception as e:
        return f"Sorry, I couldn't generate the image. Error: {e}"

# --- FUNZIONE PRINCIPALE DELLA CHAT (AGGIORNATA) ---
def chat_responder(message: str, history: List[List[str]], model: str, files: List):
    history.append([message, ""])
    yield history

    # 1. Estrazione contenuto e preparazione prompt
    file_contents = [f"[Content from '{os.path.basename(f.name)}']:\n{extract_text_from_file(f.name)}" for f in files] if files else []
    prompt_for_llm = f"{message}\n\n{''.join(file_contents)}" if file_contents else message

    # 2. Classificazione del task
    task_type = classify_task(prompt_for_llm)
    print(f"--- Task classified as: {task_type} ---")

    # 3. Gestione task speciali (es. generazione immagini)
    if task_type == "image_generation":
        bot_response = generate_openai_image(message) if model == "GPT" else "Image generation with Claude is not supported."
        history[-1][1] = bot_response
        yield history
        log_conversation(history, model, files)
        return

    # 4. Costruzione del System Prompt DINAMICO per task testuali
    reasoning_pattern = TASK_REGISTRY.get(task_type, TASK_REGISTRY["general_conversation"])["pattern"]
    dynamic_system_message = (
        f"{SYSTEM_GOAL}\n\n"
        f"BASE RULES:\n" + "\n".join(f"- {rule}" for rule in SYSTEM_RULES) +
        f"\n\n---\nREASONING PATTERN:\n{reasoning_pattern}\n---\n"
    )

    # 5. Preparazione cronologia e chiamata all'LLM
    conversation_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, (user_msg, bot_msg) in enumerate(history[:-1]) for msg in (user_msg, bot_msg) if msg]
    conversation_history.append({"role": "user", "content": prompt_for_llm})

    response_text = ""
    try:
        if model == "GPT" and openai_client:
            stream = openai_client.chat.completions.create(
                model='gpt-4o', messages=[{"role": "system", "content": dynamic_system_message}] + conversation_history, stream=True
            )
            for chunk in stream:
                response_text += chunk.choices[0].delta.content or ""
                history[-1][1] = response_text
                yield history
        elif model == "Claude" and claude_client:
            with claude_client.messages.stream(
                model="claude-3-opus-20240229", max_tokens=4000, system=dynamic_system_message, messages=conversation_history
            ) as stream:
                for text in stream.text_stream:
                    response_text += text or ""
                    history[-1][1] = response_text
                    yield history
        else:
            history[-1][1] = f"{model} client not configured."
            yield history
    except Exception as e:
        history[-1][1] = f"An error occurred: {e}"
        yield history
    finally:
        log_conversation(history, model, files)

# --- INTERFACCIA UTENTE CON GRADIO (INVARIATA) ---
css = "#chat-container { height: 85vh; } #chatbot { flex-grow: 1; overflow-y: auto; }"
with gr.Blocks(theme=gr.themes.Soft(), css=css) as view:
    gr.Markdown("# Dasius AI")
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(["GPT", "Claude"], label="Select Model", value="GPT")
            file_uploader = gr.File(label="Upload file(s)", file_count="multiple", file_types=[".xlsx", ".xls", ".json", ".docx", ".pdf", ".png", ".jpg", ".jpeg"])
            clear_btn = gr.ClearButton()
        with gr.Column(scale=4, elem_id="chat-container"):
            chatbot = gr.Chatbot(label="Conversation", bubble_full_width=False, avatar_images=(None, "https://i.imgur.com/1T29GgD.png"), elem_id="chatbot")
            with gr.Row():
                msg_box = gr.Textbox(label="Your message", placeholder="Type your message here...", show_label=False, scale=15, container=False)
                send_btn = gr.Button("➤", scale=1, min_width=50)

    def clear_textbox(): return gr.Textbox(value="")
    
    msg_box.submit(chat_responder, [msg_box, chatbot, model_selector, file_uploader], chatbot, show_progress="hidden").then(clear_textbox, None, msg_box, queue=False)
    send_btn.click(chat_responder, [msg_box, chatbot, model_selector, file_uploader], chatbot, show_progress="hidden").then(clear_textbox, None, msg_box, queue=False)
    clear_btn.add([chatbot, msg_box, file_uploader])

if __name__ == "__main__":
    view.launch(share=True)
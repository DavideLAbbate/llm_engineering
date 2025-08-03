import os
import requests
import datetime
from bs4 import BeautifulSoup
from typing import List, Tuple
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
# Carica le variabili d'ambiente dal file .env
load_dotenv(override=True)

# Imposta il path di Tesseract se necessario (es. su Windows)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    print("Tesseract not found at default Windows path. Skipping. It will be needed for OCR on images/PDFs.")

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if not openai_api_key:
    print("ATTENZIONE: OPENAI_API_KEY non trovato. Le funzionalità di OpenAI non saranno disponibili.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)

if not anthropic_api_key:
    print("ATTENZIONE: ANTHROPIC_API_KEY non trovato. Le funzionalità di Claude non saranno disponibili.")
    claude_client = None
else:
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

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

SYSTEM_MESSAGE = SYSTEM_GOAL + "\n\n" + "\n".join(f"- {rule}" for rule in SYSTEM_RULES)
LOG_FILE_PATH = "chat_history_log.txt"

# --- LOGGING FIX (COMPLETO E SICURO) ---
def log_conversation(history: list, model: str, files: list = None):
    """
    Salva la history della conversazione con info su modello e file.
    Scrive su LOG_FILE_PATH in modo append, UTF-8. 
    Funziona anche con files=None o file oggetti "temporary".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Gestione robusta nomi file (alcuni oggetti File potrebbero NON avere .name)    
    def get_file_name(f):
        if hasattr(f, "name"):
            return os.path.basename(f.name)
        elif hasattr(f, "filename"):
            return os.path.basename(f.filename)
        return "UNKNOWN"
    file_list = ', '.join([get_file_name(f) for f in files]) if files else "N/A"
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as logf:
            logf.write(f"--- {timestamp} | Model: {model} | Files: {file_list} ---\n")
            for user_msg, bot_msg in history:
                user_msg = user_msg if user_msg is not None else ""
                bot_msg = bot_msg if bot_msg is not None else ""
                logf.write(f"User: {user_msg}\n")
                logf.write(f"Sara: {bot_msg}\n---\n")
            logf.write("\n")
    except Exception as e:
        print(f"[LOGGING ERROR]: {e}")

# --- FUNZIONI DI ESTRAZIONE TESTO ---
def extract_text_from_file(file_path: str) -> str:
    if file_path is None:
        return ""
    name = file_path.lower()
    content = ""
    try:
        if name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
            content = df.to_string()
        elif name.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = json.dumps(data, indent=2)
        elif name.endswith(".docx"):
            doc = Document(file_path)
            content = "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".pdf"):
            doc = fitz.open(file_path)
            content = "\n".join(page.get_text() for page in doc)
            if not content.strip():
                ocr_text = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text.append(pytesseract.image_to_string(img))
                content = "\n".join(ocr_text) or "[OCR failed or PDF is empty]"
        elif name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            content = text if text.strip() else "[Image uploaded, no text detected by OCR]"
        else:
            content = "[Unsupported file type]"
    except Exception as e:
        content = f"[Error reading file: {e}]"
    return content

# --- FUNZIONI DI GENERAZIONE IMMAGINI ---
def is_image_generation_prompt(prompt: str) -> bool:
    if not openai_client:
        return False
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a classifier. Your task is to determine if the user wants to generate an image. Respond with only 'yes' or 'no'. Keywords to look for include 'draw', 'create an image', 'generate a picture', 'disegna', 'crea un immagine', etc."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0.0
        )
        decision = response.choices[0].message.content.strip().lower()
        return "yes" in decision
    except Exception as e:
        print(f"Error in image prompt detection: {e}")
        return False

def generate_openai_image(prompt: str) -> str:
    if not openai_client:
        return "OpenAI client not configured. Cannot generate image."
    try:
        print(f"Generating image with prompt: {prompt}")
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response.data[0].url
        return f"![{prompt}]({image_url})"
    except Exception as e:
        print(f"Error generating OpenAI image: {e}")
        return f"Sorry, I couldn't generate the image. Error: {e}"

def generate_claude_image(prompt: str) -> str:
    print("Claude image generation is not yet implemented.")
    return "Image generation with Claude is not supported at the moment."

# --- FUNZIONE PRINCIPALE DELLA CHAT ---
def chat_responder(message: str, history: List[List[str]], model: str, files: List):
    """
    Gestisce l'interazione della chat, la cronologia e la generazione di testo/immagini.
    """
    history.append([message, ""])
    yield history

    all_files_content = []
    if files:
        for file in files:
            file_path = getattr(file, 'name', None)
            extracted_text = extract_text_from_file(file_path)
            file_header = f"[Content from uploaded file '{os.path.basename(file_path)}']:"
            all_files_content.append(f"{file_header}\n{extracted_text}")

    if all_files_content:
        combined_content = "\n\n".join(all_files_content)
        prompt_for_llm = f"{message}\n\n{combined_content}"
    else:
        prompt_for_llm = message

    if is_image_generation_prompt(message):
        bot_response = ""
        if model == "GPT":
            bot_response = generate_openai_image(message)
        elif model == "Claude":
            bot_response = generate_claude_image(message)
        history[-1][1] = bot_response
        yield history
        # Chiamata CORRETTA a logging dopo risposta pronta
        log_conversation(history, model, files)
        return

    conversation_history = []
    for user_msg, assistant_msg in history[:-1]:
        if user_msg:
            conversation_history.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            conversation_history.append({"role": "assistant", "content": assistant_msg})
    conversation_history.append({"role": "user", "content": prompt_for_llm})

    response_text = ""
    if model == "GPT":
        if not openai_client:
            history[-1][1] = "OpenAI client not configured."
            yield history
            return
        messages_for_api = [{"role": "system", "content": SYSTEM_MESSAGE}] + conversation_history
        stream = openai_client.chat.completions.create(
            model='gpt-4.1',
            messages=messages_for_api,
            stream=True
        )
        for chunk in stream:
            response_text += chunk.choices[0].delta.content or ""
            history[-1][1] = response_text
            yield history
        # Logging fatto solo alla FINE della risposta
        log_conversation(history, model, files)

    elif model == "Claude":
        if not claude_client:
            history[-1][1] = "Claude (Anthropic) client not configured."
            yield history
            return
        with claude_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            temperature=0.7,
            system=SYSTEM_MESSAGE,
            messages=conversation_history
        ) as stream:
            for text in stream.text_stream:
                response_text += text or ""
                history[-1][1] = response_text
                yield history
        log_conversation(history, model, files)

# --- INTERFACCIA UTENTE CON GRADIO ---
css = """
#chat-container { 
  display: flex; 
  flex-direction: column; 
  height: 85vh;
}
#chatbot { 
  flex-grow: 1;
  overflow-y: auto;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as view:
    gr.Markdown("# Dasius AI")
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                ["GPT", "Claude"],
                label="Select Model",
                value="GPT"
            )
            file_uploader = gr.File(
                label="Upload file(s) (optional)",
                file_count="multiple",
                file_types=[".xlsx", ".xls", ".json", ".docx", ".pdf", ".png", ".jpg", ".jpeg"]
            )
            clear_btn = gr.ClearButton()
        with gr.Column(scale=4, elem_id="chat-container"):
            chatbot = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                avatar_images=(None, "https://i.imgur.com/1T29GgD.png"),
                elem_id="chatbot"
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Your message:",
                    placeholder="Type your message here or ask to generate an image...",
                    show_label=False,
                    scale=15,
                    container=False
                )
                send_btn = gr.Button(
                    "➤",
                    scale=1,
                    min_width=50
                )

    def clear_textbox():
        return gr.Textbox(value="")

    submit_event = msg_box.submit(
        chat_responder,
        [msg_box, chatbot, model_selector, file_uploader],
        chatbot,
        show_progress="hidden"
    )
    submit_event.then(clear_textbox, None, msg_box, queue=False)

    click_event = send_btn.click(
        chat_responder,
        [msg_box, chatbot, model_selector, file_uploader],
        chatbot,
        show_progress="hidden"
    )
    click_event.then(clear_textbox, None, msg_box, queue=False)

    clear_btn.add([chatbot, msg_box, file_uploader])

if __name__ == "__main__":
    view.launch(share=True)
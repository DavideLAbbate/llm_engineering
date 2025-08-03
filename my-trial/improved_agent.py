import os
import requests
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
# Assicurati che il percorso sia corretto per il tuo sistema
try:
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
  print("Tesseract not found at default Windows path. Skipping. It will be needed for OCR on images/PDFs.")

# Inizializzazione dei client per le API
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if not openai_api_key:
  print("ATTENZIONE: OPENAI_API_KEY non trovato. Le funzionalità di OpenAI non saranno disponibili.")
  openai_client = None
else:
  openai_client = OpenAI()

if not anthropic_api_key:
  print("ATTENZIONE: ANTHROPIC_API_KEY non trovato. Le funzionalità di Claude non saranno disponibili.")
  claude_client = None
else:
  claude_client = anthropic.Anthropic()

SYSTEM_MESSAGE = "You are a helpful assistant. Respond in Markdown format."

# --- FUNZIONI DI ESTRAZIONE TESTO ---
def extract_text_from_file(file_path: str) -> str:
  """Estrae il testo da vari tipi di file."""
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
          # Se il PDF è basato su immagini, tenta l'OCR
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
  """Usa un LLM per determinare se il prompt richiede la generazione di un'immagine."""
  if not openai_client:
      return False
  try:
      # Chiediamo a un modello veloce di classificare l'intenzione
      response = openai_client.chat.completions.create(
          model="gpt-4o-mini",
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
  """Genera un'immagine con DALL-E 3 e restituisce il Markdown per visualizzarla."""
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
      # Restituisce l'URL formattato in Markdown per essere visualizzato da Gradio
      return f"![{prompt}]({image_url})"
  except Exception as e:
      print(f"Error generating OpenAI image: {e}")
      return f"Sorry, I couldn't generate the image. Error: {e}"

def generate_claude_image(prompt: str) -> str:
  """Placeholder per la generazione di immagini con Claude."""
  print("Claude image generation is not yet implemented.")
  return "Image generation with Claude is not supported at the moment."

# --- FUNZIONE PRINCIPALE DELLA CHAT (CORRETTA) ---
def chat_responder(message: str, history: List[List[str]], model: str, file):
  """
  Gestisce l'interazione della chat, la cronologia e la generazione di testo/immagini.
  """
  history.append([message, ""])
  yield history

  file_content = ""
  if file:
      file_content = extract_text_from_file(file.name)
      prompt_for_llm = f"{message}\n\n[Content from uploaded file '{os.path.basename(file.name)}']:\n{file_content}"
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
      return

  messages_for_api = [{"role": "system", "content": SYSTEM_MESSAGE}]
  for user_msg, assistant_msg in history[:-1]:
      messages_for_api.append({"role": "user", "content": user_msg})
      messages_for_api.append({"role": "assistant", "content": assistant_msg})
  messages_for_api.append({"role": "user", "content": prompt_for_llm})

  response_text = ""
  if model == "GPT":
      if not openai_client:
          history[-1][1] = "OpenAI client not configured."
          yield history
          return
      stream = openai_client.chat.completions.create(
          model='gpt-4o-mini',
          messages=messages_for_api,
          stream=True
      )
      for chunk in stream:
          response_text += chunk.choices[0].delta.content or ""
          history[-1][1] = response_text
          yield history
  
  elif model == "Claude":
      if not claude_client:
          history[-1][1] = "Claude (Anthropic) client not configured."
          yield history
          return
      with claude_client.messages.stream(
          model="claude-3-haiku-20240307",
          max_tokens=2000,
          temperature=0.7,
          system=SYSTEM_MESSAGE,
          messages=messages_for_api
      ) as stream:
          for text in stream.text_stream:
              response_text += text or ""
              history[-1][1] = response_text
              yield history

# --- INTERFACCIA UTENTE CON GRADIO (MODIFICATA) ---
# Aggiungiamo il CSS per controllare l'altezza
css = """
#chat-container {
    display: flex;
    flex-direction: column;
    height: 85vh; /* Imposta l'altezza del contenitore principale della chat */
}
#chatbot {
    flex-grow: 1; /* Fa in modo che il chatbot si espanda per riempire lo spazio */
    overflow-y: auto; /* Aggiunge lo scroll se il contenuto è troppo lungo */
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as view:
  gr.Markdown("# Advanced AI Chat")
  
  with gr.Row():
      with gr.Column(scale=1):
          model_selector = gr.Dropdown(
              ["GPT", "Claude"], 
              label="Select Model", 
              value="GPT"
          )
          file_uploader = gr.File(
              label="Upload a file (optional)",
              file_types=[".xlsx", ".xls", ".json", ".docx", ".pdf", ".png", ".jpg", ".jpeg"]
          )
          clear_btn = gr.ClearButton()

      with gr.Column(scale=4, elem_id="chat-container"): # ID per il contenitore
          chatbot = gr.Chatbot(
              label="Conversation",
              bubble_full_width=False,
              avatar_images=(None, "https://i.imgur.com/1T29GgD.png"), # (user, bot)
              elem_id="chatbot" # ID per il chatbot stesso
          )
          
          with gr.Row():
              msg_box = gr.Textbox(
                  label="Your message:",
                  placeholder="Type your message here or ask to generate an image...",
                  show_label=False,
                  scale=15, # Dà più spazio alla textbox
                  container=False # Rimuove il bordo attorno alla textbox
              )
              send_btn = gr.Button(
                  "➤", 
                  scale=1, # Dà meno spazio al pulsante
                  min_width=50 # Imposta una larghezza minima
              )

  # --- LOGICA DEGLI EVENTI ---
  
  # Funzione per svuotare la textbox
  def clear_textbox():
      return gr.Textbox(value="")

  # Collega l'evento di invio (sia da tasto Invio che da pulsante)
  submit_event = msg_box.submit(
      chat_responder, 
      [msg_box, chatbot, model_selector, file_uploader], 
      chatbot,
      show_progress="hidden"
  )
  # Dopo l'invio, svuota la textbox
  submit_event.then(clear_textbox, None, msg_box, queue=False)

  # Collega anche l'evento click del pulsante
  click_event = send_btn.click(
      chat_responder,
      [msg_box, chatbot, model_selector, file_uploader],
      chatbot,
      show_progress="hidden"
  )
  # Dopo il click, svuota la textbox
  click_event.then(clear_textbox, None, msg_box, queue=False)

  # Collega il pulsante "Clear" per pulire tutto
  clear_btn.add([chatbot, msg_box, file_uploader])


if __name__ == "__main__":
  view.launch(share=True)

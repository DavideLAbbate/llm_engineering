import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
import pandas as pd
import json
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import gradio as gr

# Imposta il path di Tesseract se sei su Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()

system_message = "You are a helpful assistant that responds in markdown"

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def stream_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

def extract_text_from_file(file):
    if file is None:
        return ""

    name = file.name.lower()
    content = ""

    try:
        if name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
            content = df.to_string()

        elif name.endswith(".json"):
            data = json.load(file)
            content = json.dumps(data, indent=2)

        elif name.endswith(".docx"):
            doc = Document(file)
            content = "\n".join(p.text for p in doc.paragraphs)

        elif name.endswith(".pdf"):
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            content = "\n".join(page.get_text() for page in doc)

            # OCR se PDF è "vuoto"
            if not content.strip():
                ocr_text = []
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text.append(pytesseract.image_to_string(img))
                content = "\n".join(ocr_text) or "[OCR failed or empty]"

        elif name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            content = text if text.strip() else "[Image uploaded, no text detected]"

        else:
            content = "[Unsupported file type]"

    except Exception as e:
        content = f"[Error reading file: {e}]"

    return content

def stream_model(prompt, model, file):
    file_content = extract_text_from_file(file)
    full_prompt = f"{prompt}\n\n[File content]:\n{file_content}" if file_content else prompt

    if model == "GPT":
        result = stream_gpt(full_prompt)
    elif model == "Claude":
        result = stream_claude(full_prompt)
    else:
        raise ValueError("Unknown model")

    yield from result

view = gr.Interface(
    fn=stream_model,
    js=force_dark_mode,
    inputs=[
        gr.Textbox(label="Your message:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT"),
        gr.File(label="Upload a file", file_types=[".xlsx", ".xls", ".json", ".docx", ".pdf", ".png", ".jpg", ".jpeg"])
    ],
    outputs=gr.Markdown(label="Response:"),
    flagging_mode="never"
)
view.launch()

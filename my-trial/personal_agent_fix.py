import os
import logging
from dotenv import load_dotenv
import openai
import gradio as gr

# ENVIRONMENT & LOGGING
load_dotenv(override=True)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/chatbot.log"),
        logging.StreamHandler()
    ]
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("OPENAI_API_KEY non presente nell'.env'")

# SYSTEM PROMPT & CONTEXT
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
SYSTEM_GOAL = "Your name is Sara. You are a rigorous and professional AI assistant. Clarity and correctness over speed."
SYSTEM_MESSAGE = SYSTEM_GOAL + "\n\n" + "\n".join(f"- {rule}" for rule in SYSTEM_RULES)

# ORCHESTRATORE
class ChatOrchestrator:
    def __init__(self, system_message, openai_api_key):
        self.system_message = system_message
        self.history = []
        self.client = openai.OpenAI(api_key=openai_api_key)

    def ask(self, user_input):
        if len(self.history) > 10:
            self.history = self.history[-10:]
        logging.info(f"> User: {user_input}")

        messages = [
            {"role": "system", "content": self.system_message}
        ]
        for entry in self.history:
            messages.append(entry)
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.6,
                max_tokens=750,
            )
            assistant_reply = response.choices[0].message.content
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_reply})
            logging.info(f"> Sara: {assistant_reply}")
            return assistant_reply
        except Exception as e:
            logging.error(f"Errore OpenAI API: {e}")
            return "Mi scuso, si è verificato un errore di sistema. Riprovare tra poco."


orchestrator = ChatOrchestrator(SYSTEM_MESSAGE, openai_api_key)

def chat_interface(user_input, history):
    reply = orchestrator.ask(user_input)
    history = history or []
    history.append((user_input, reply))
    return history, history, ""  # Svuota input box dopo invio

DARK_CSS = """
body, .gradio-container, .block, #main-col, .chatbot {
    background-color: #191c20 !important;
    color: #e8eaf6 !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}
#header {
    text-align:center;
    margin-bottom: 22px;
}
#header h1, #header h3 {
    font-weight: 700;
    letter-spacing: 1px;
    color: #e1e8ef;
}
#chatbox {
    border-radius: 14px !important;
    box-shadow: 0 4px 24px #000a  !important;
    background: #232632 !important;
    padding: 16px 8px;
    font-size: 1.1em;
}
.message.user {
    background: #232a3e !important;
    color: #f9fafb !important;
    border-radius: 0px 16px 16px 16px !important;
    padding: 9px 14px; 
    margin: 2px 0 10px 36px
}
.message.assistant {
    background: linear-gradient(96deg, #202241 70%, #1c223a 100%);
    color: #c2e3ff !important;
    border-radius: 16px 0px 16px 16px !important;
    padding: 9px 14px;
    margin: 2px 36px 16px 0;
    border-left: 3px solid #39b5fa;
}
#user-input textarea {
    background: #181a22 !important;
    color: #e8eaf6 !important;
    border-radius: 8px !important;
    border: 1.5px solid #252525 !important;
    font-size: 1em;
}
#submit-btn {
    background: linear-gradient(96deg, #338eda 60%, #0588e9 100%) !important;
    color: #f8fafc !important;
    border-radius: 7px !important;
    font-weight: bold;
    margin-left: 5px;
}
.gradio-container::-webkit-scrollbar, .chatbot::-webkit-scrollbar {
    height: 8px;
    width: 7px;
    background: #222;
}
.gradio-container::-webkit-scrollbar-thumb, .chatbot::-webkit-scrollbar-thumb {
    background: #2e4057;
    border-radius: 7px;
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=DARK_CSS) as demo:
    gr.Markdown(
        "<div id='header'><h1>👋 Sara - AI Assistant</h1>"
        "<div><small>Assistente AI rigoroso e professionale • Powered by OpenAI</small></div></div>"
    )
    with gr.Row(elem_id="main-row"):
        with gr.Column(elem_id="main-col", scale=5):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbox",
                height=480,
                avatar_images=(None, None)
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Scrivi qui...",
                    label=None,
                    elem_id="user-input",
                    lines=3,
                    scale=8
                )
                submit_btn = gr.Button(
                    "Invia",
                    elem_id="submit-btn",
                    scale=1
                )

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
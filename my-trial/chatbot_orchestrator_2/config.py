import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

# --- Setup ---
load_dotenv(override=True)

# --- API Clients ---
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if not openai_api_key or not anthropic_api_key:
    raise ValueError("OPENAI_API_KEY and ANTHROPIC_API_KEY must be set in your .env file.")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

# --- Constants ---
SYSTEM_MESSAGE_BASE = "Your name is Sara. You are a rigorous and professional AI assistant. Clarity and correctness over speed."


import logging
import json
from openai import OpenAI
import anthropic

logger = logging.getLogger(__name__)

import os

# Initialize local API clients to avoid circular imports
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

_openai_client = None
_claude_client = None

def _ensure_openai():
    global _openai_client
    if _openai_client is None:
        api = os.getenv('OPENAI_API_KEY')
        if not api:
            raise ValueError("OPENAI_API_KEY must be set in environment for OpenAI usage.")
        from openai import OpenAI as _OpenAI
        _openai_client = _OpenAI(api_key=api)
    return _openai_client

def _ensure_claude():
    global _claude_client
    if _claude_client is None:
        api = os.getenv('ANTHROPIC_API_KEY')
        if not api:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment for Anthropic usage.")
        import anthropic as _anthropic
        _claude_client = _anthropic.Anthropic(api_key=api)
    return _claude_client



def get_claude_response(system_prompt, user_prompt, is_json=False):
    """Helper function to call Anthropic Claude API with proper system message."""
    try:
        # Costruisci SOLO il messaggio utente; il system va nel parametro top-level.
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        (_, _claude_client) = _ensure_clients()
        response = _claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,   # <-- system prompt assegnato correttamente
            messages=messages
        )

        content = response.content[0].text.strip()

        if is_json:
            # Normalizzazione soft di eventuali fence markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()

        return content

    except Exception as e:
        logger.error(f"UTIL: Claude API call failed: {e}")
        return f"Error during Claude call: {e}"

def get_openai_response(system_prompt, user_prompt, history):
    """Helper function to call OpenAI API. Returns plain text content."""
    messages = [{"role": "system", "content": system_prompt}]
    # Append prior history if provided
    for h in (history or []):
        if len(h) >= 1 and h[0]:
            messages.append({"role": "user", "content": h[0]})
        if len(h) >= 2 and h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    # Current user prompt at the end
    messages.append({"role": "user", "content": user_prompt})
    try:
        _openai_client = _ensure_openai()
        response = _openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"UTIL: OpenAI API call failed: {e}")
        return f"Error during OpenAI call: {e}"


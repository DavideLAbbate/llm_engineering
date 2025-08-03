import logging
import json
from config import openai_client, claude_client # Importa i client dal nuovo file di configurazione

logger = logging.getLogger(__name__)

def get_openai_response(system_prompt, user_prompt, history):
    """Helper function to call OpenAI API."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"UTIL: OpenAI API call failed: {e}")
        return f"Error during OpenAI call: {e}"

def get_claude_response(system_prompt, user_prompt, is_json=False):
    """Helper function to call Anthropic Claude API."""
    try:
        messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620", 
            max_tokens=4096, 
            temperature=0.0,
            messages=messages
        )
        content = response.content[0].text.strip()
        
        if is_json:
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()
        
        return content
    except Exception as e:
        logger.error(f"UTIL: Claude API call failed: {e}")
        return f"Error during Claude call: {e}"

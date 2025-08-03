import logging
import json
from openai import OpenAI
import anthropic

logger = logging.getLogger(__name__)

# This assumes clients are configured in the main script
# A better approach would be to pass them as arguments or use a singleton pattern
from chatbot import openai_client, claude_client, SYSTEM_MESSAGE_BASE

def get_openai_response(system_prompt, user_prompt, history):
    """Helper function to call OpenAI API."""
    messages = [{"role": "system", "content": system_prompt}]
    # for h in history:
    #     messages.append({"role": "user", "content": h[0]})
    #     messages.append({"role": "assistant", "content": h[1]})
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
            # Clean up potential markdown code block
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip()
        
        return content
    except Exception as e:
        logger.error(f"UTIL: Claude API call failed: {e}")
        return f"Error during Claude call: {e}"

import json
import requests
from typing import List, Dict, Generator

def generate_ai_response_ollama(model_name: str, chat_history: List[Dict[str, str]], api_key=None):
    """
    Streams a text generation response from Ollama's API.

    Args:
        model_name (str): The model name (e.g., "llama3.1:8b").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.
        api_key (str, optional): Ollama API URL (e.g., "http://localhost:11434/api/generate"). If None, must be set in environment.

    Yields:
        str: Chunks (tokens) of the generated AI response.

    Raises:
        RuntimeError: If the API URL is missing or the API call fails.
    """
    if not api_key:
        raise RuntimeError("Ollama API URL is required. Provide it via api_key parameter or set OLLAMA_URL in .env.")

    ollama_url = api_key

    try:
        # Convert chat history to Ollama's prompt format
        prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        with requests.post(
            ollama_url,
            json=payload,
            timeout=(5, 180),
            stream=True
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode())
                yield chunk.get("response", "")

    except Exception as e:
        raise RuntimeError(f"Ollama API error: {str(e)}") from e
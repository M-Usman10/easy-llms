import os
import sys
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Generator

# Load environment variables from .env
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

def generate_ai_response_ollama(model_name: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Streams a text generation response from Ollama's API.

    Args:
        model_name (str): The model name (e.g., "llama3.1:8b").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.

    Yields:
        str: Chunks (tokens) of the generated AI response.

    Raises:
        RuntimeError: If there's an error communicating with the Ollama API.
    """
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
            OLLAMA_URL, 
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
        raise RuntimeError(f"Ollama API error: {str(e)}")

if __name__ == "__main__":
    # Test block for the Ollama provider
    model_name = "llama3.1:8b"
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who invented World Wide Web?"}
    ]

    print("Streaming response from Ollama Provider...\n")
    try:
        for token in generate_ai_response_ollama(model_name, chat_history):
            sys.stdout.write(token)
            sys.stdout.flush()
        print()  # Newline after streaming completes.
    except Exception as e:
        print(f"Error during streaming: {e}")
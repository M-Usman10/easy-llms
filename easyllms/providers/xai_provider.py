import os
from openai import OpenAI  # Assumes XAI uses the OpenAI library interface.
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the XAI client with API key from environment variables
XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def generate_ai_response_xai(model_name, chat_history):
    """
    Streams a text generation response from XAI's API.

    Args:
        model_name (str): The model name (e.g., "xai-model-1").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.

    Yields:
        str: Chunks (tokens) of the generated AI response.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            stream=True,
            temperature=0.0,
            max_tokens=1024
        )
        for chunk in response:
            # Check if the response chunk contains generated content
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        raise RuntimeError(f"XAI API error: {str(e)}")

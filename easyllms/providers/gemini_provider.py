import os
from openai import OpenAI  # Using the OpenAI library interface for Gemini
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini client using the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def generate_ai_response_gemini(model_name, chat_history):
    """
    Streams a text generation response from Gemini's API.

    Args:
        model_name (str): The model name (e.g., "gemini-2.0-flash").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.

    Yields:
        str: Chunks (tokens) of the generated AI response.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            stream=True,
            n=1
        )
        for chunk in response:
            # Directly access the 'content' attribute if available
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

if __name__ == "__main__":
    # Test block for the Gemini provider
    model_name = "gemini-2.0-flash"
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who invented World Wide Web?"}
    ]

    print("Streaming response from Gemini Provider...\n")
    try:
        for token in generate_ai_response_gemini(model_name, chat_history):
            print(token, end="", flush=True)
        print()  # Newline after streaming completes.
    except Exception as e:
        print(f"Error during streaming: {e}")

import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Anthropic client
anthropic = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


def generate_ai_response_anthropic(model_name, chat_history):
    """
    Generates an AI response using Anthropic's Claude models,
    streaming the response token-by-token.

    Args:
        model_name (str): The model name (e.g., "claude-3-7-sonnet-20250219").
        chat_history (list): A list of messages in the chat history.
        char_delay (float): Delay in seconds between each character to simulate token-by-token output.

    Yields:
        str: Tokens (or small chunks) of the AI-generated response.
    """
    # Format messages for Anthropic API
    formatted_messages = []
    system_message = None

    for message in chat_history:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_message = content
        else:
            formatted_messages.append({
                "role": role,
                "content": content
            })

    # Stream the response using Anthropic API
    with anthropic.messages.stream(
            model=model_name,
            messages=formatted_messages,
            system=system_message,
            max_tokens=1024
    ) as stream:
        for text in stream.text_stream:
            # Yield each character with a delay, without printing here.
            for char in text:
                yield char

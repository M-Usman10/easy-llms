import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize client once at module level
_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_ai_response_openai(model_name, chat_history):
    """
    Stream response from OpenAI's API

    Args:
        messages: List of dicts with 'role' and 'content'
        model: Model name (e.g. "gpt-4", "gpt-3.5-turbo")
        **kwargs: Additional parameters:
            - temperature (float)
            - max_tokens (int)
            - top_p (float)
            - etc.

    Yields:
        str: Response chunks
    """
    try:
        response = _client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            stream=True,
            temperature=0.0,
            max_tokens=1024,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}")

"""
if __name__ == "__main__":
    # Example chat history (modify as needed)
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me the significane of LLMs in Healthcare"}
    ]

    # Test the streaming response
    print("Testing OpenAI streaming...")
    try:
        for chunk in generate_ai_response_openai(
            model_name="gpt-4o-mini",  # or "gpt-4"
            chat_history=chat_history
        ):
            print(chunk, end="", flush=True)  # Print chunks in real-time
    except Exception as e:
        print(f"\nError occurred: {e}")
"""
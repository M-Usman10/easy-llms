from openai import OpenAI

def generate_ai_response_gemini(model_name, chat_history, api_key=None):
    """
    Streams a text generation response from Gemini's API.

    Args:
        model_name (str): The model name (e.g., "gemini-2.0-flash").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.
        api_key (str, optional): Gemini API key. If None, must be set in environment.

    Yields:
        str: Chunks (tokens) of the generated AI response.

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    if not api_key:
        raise RuntimeError("Gemini API key is required. Provide it via api_key parameter or set GEMINI_API_KEY in .env.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

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
        raise RuntimeError(f"Gemini API error: {str(e)}") from e
from openai import OpenAI

def generate_ai_response_xai(model_name, chat_history, api_key=None):
    """
    Streams a text generation response from XAI's API.

    Args:
        model_name (str): The model name (e.g., "grok").
        chat_history (list): A list of messages in the chat history, each a dict with 'role' and 'content'.
        api_key (str, optional): XAI API key. If None, must be set in environment.

    Yields:
        str: Chunks (tokens) of the generated AI response.

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    if not api_key:
        raise RuntimeError("XAI API key is required. Provide it via api_key parameter or set XAI_API_KEY in .env.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

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
        raise RuntimeError(f"XAI API error: {str(e)}") from e
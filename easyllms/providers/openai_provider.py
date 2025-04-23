from openai import OpenAI

def generate_ai_response_openai(model_name, chat_history, api_key=None):
    """
    Stream response from OpenAI's API

    Args:
        model_name (str): Model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
        chat_history (list): List of dicts with 'role' and 'content'.
        api_key (str, optional): OpenAI API key. If None, must be set in environment.

    Yields:
        str: Response chunks

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    if not api_key:
        raise RuntimeError("OpenAI API key is required. Provide it via api_key parameter or set OPENAI_API_KEY in .env.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
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
        raise RuntimeError(f"OpenAI API error: {str(e)}") from e
from anthropic import Anthropic

def generate_ai_response_anthropic(model_name, chat_history, api_key=None):
    """
    Generates an AI response using Anthropic's Claude models,
    streaming the response token-by-token.

    Args:
        model_name (str): The model name (e.g., "claude-3-7-sonnet-20250219").
        chat_history (list): A list of messages in the chat history.
        api_key (str, optional): Anthropic API key. If None, must be set in environment.

    Yields:
        str: Tokens (or small chunks) of the AI-generated response.

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    if not api_key:
        raise RuntimeError("Anthropic API key is required. Provide it via api_key parameter or set ANTHROPIC_API_KEY in .env.")

    anthropic = Anthropic(api_key=api_key)

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
            # Yield each character without printing here.
            for char in text:
                yield char
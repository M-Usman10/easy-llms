from .providers.anthropic_provider import generate_ai_response_anthropic
from .providers.openai_provider import generate_ai_response_openai
from .providers.gemini_provider import generate_ai_response_gemini
from .providers.ollama_provider import generate_ai_response_ollama
from .providers.xai_provider import generate_ai_response_xai

# Map provider names to their corresponding functions.
PROVIDERS = {
    'openai': generate_ai_response_openai,
    'anthropic': generate_ai_response_anthropic,
    'xai': generate_ai_response_xai,
    'gemini': generate_ai_response_gemini,
    'ollama': generate_ai_response_ollama,
}


def stream_response(provider_name, model_name, prompt, **kwargs):
    """
    Streams the response from the specified provider using the given model name and prompt.

    Args:
        provider_name (str): Name of the provider ('openai' or 'anthropic').
        model_name (str): The model identifier (e.g., 'gpt-4' or 'claude-3').
        prompt (str): The input prompt for the AI.
        **kwargs: Additional keyword arguments for the provider function.

    Yields:
        str: Tokens of the streaming response.
    """
    generate_response = PROVIDERS.get(provider_name)
    if not generate_response:
        raise ValueError(f"Provider '{provider_name}' not supported.")

    # Build a simple chat history: you can adjust or extend this structure as needed.
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Call the provider function with model_name and chat_history.
    yield from generate_response(model_name, chat_history, **kwargs)

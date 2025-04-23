from easyllms.easy_llms import stream_response

# Define the prompt to test with.
prompt = "What is AGI, and is it achievable in the near future?? Explain in 2-3 lines."

# Define provider-specific model names.
providers = {
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-haiku-20241022',
    'xai': 'grok-2-latest',
    'gemini': 'gemini-2.0-flash',
    'ollama':'llama3.1:8b'
}

# Run inference for each provider.
for provider, model in providers.items():
    print(f"\n--- {provider.upper()} (Model: {model}) ---")
    try:
        # Stream the response and print it as it is generated.
        for token in stream_response(provider, model, prompt):
            print(token, end="", flush=True)
    except Exception as e:
        print(f"\nError occurred: {e}")
    print("\n")
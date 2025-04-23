from easyllms import stream_response

# Test OpenAI
print("Testing OpenAI")
for token in stream_response("openai", "gpt-4o-mini", "Who invented world wide web?"):
    print(token, end="", flush=True)
print("\n")

# Test Anthropic
print("Testing Anthropic:")
for token in stream_response("anthropic", "claude-3-7-sonnet-20250219", "What is AGI, and is it achievable in the near future?? Explain in 2-3 lines."):
    print(token, end="", flush=True)
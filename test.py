from easyllms import stream_response
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")
ollama_url = os.getenv("OLLAMA_URL")

# Test OpenAI
print("Testing OpenAI:")
try:
    for token in stream_response("openai", "gpt-4o-mini", "What is AGI, and is it achievable in the near future? Explain in 2-3 lines.", api_key=openai_api_key):
        print(token, end="", flush=True)
except Exception as e:
    print(f"Error: {e}")
print("\n")

# Test Anthropic
print("Testing Anthropic:")
try:
    for token in stream_response("anthropic", "claude-3-7-sonnet-20250219", "What is AGI, and is it achievable in the near future? Explain in 2-3 lines.", api_key=anthropic_api_key):
        print(token, end="", flush=True)
except Exception as e:
    print(f"Error: {e}")
print("\n")

# Test Gemini
print("Testing Gemini:")
try:
    for token in stream_response("gemini", "gemini-2.0-flash", "What is AGI, and is it achievable in the near future? Explain in 2-3 lines.", api_key=gemini_api_key):
        print(token, end="", flush=True)
except Exception as e:
    print(f"Error: {e}")
print("\n")

# Test Grok (XAI)
print("Testing XAI:")
try:
    for token in stream_response("xai", "grok-3-beta", "What is AGI, and is it achievable in the near future? Explain in 2-3 lines.", api_key=xai_api_key):
        print(token, end="", flush=True)
except Exception as e:
    print(f"Error: {e}")
print("\n")

# Test Ollama
print("Testing Ollama:")
try:
    for token in stream_response("ollama", "llama3.1:8b", "What is AGI, and is it achievable in the near future? Explain in 2-3 lines.", api_key=ollama_url):
        print(token, end="", flush=True)
except Exception as e:
    print(f"Error: {e}")
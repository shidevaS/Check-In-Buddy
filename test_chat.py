import os
import sys
sys.path.insert(0, r'C:\Users\PC\Pictures\Devcon-1')

from app.llm import NVIDIA_NIMProvider, LLM_PROVIDER, model_display_name

# Initialize provider
provider = NVIDIA_NIMProvider()

# Test chat
messages = [{"role": "user", "content": "Good morning! How are you feeling today?"}]

print(f"LLM Provider: {LLM_PROVIDER}")
print(f"Model: {model_display_name(provider.model)}")
print()

try:
    result = provider.chat(messages)
    print("SUCCESS! Response:")
    print(f"  Content: {result.content}")
    print(f"  Model: {result.model}")
    print(f"  Tool calls: {result.tool_calls}")
except Exception as e:
    print(f"ERROR: {e}")
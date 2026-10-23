"""Test Check-in Buddy chat with NVIDIA LLM"""
import os
import sys
sys.path.insert(0, r'C:\Users\PC\Pictures\Devcon-1')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\PC\Pictures\Devcon-1\.env')

from app.llm import NVIDIA_NIMProvider, LLM_PROVIDER, model_display_name, provider_status

# Print status first
status = provider_status()
print("Provider status:")
for key, value in status.items():
    print(f"  {key}: {value}")

print()

# Initialize provider
provider = NVIDIA_NIMProvider()

# Test chat with check-in scenario
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
    print(f"  Usage: {result.usage}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"  Type: {type(e).__name__}")
import os
import sys
sys.path.insert(0, r'C:\Users\PC\Pictures\Devcon-1')

# Load .env manually
from dotenv import load_dotenv
load_dotenv(r'C:\Users\PC\Pictures\Devcon-1\.env')

from app.llm import provider_status

status = provider_status()
print("Provider status:")
for key, value in status.items():
    print(f"  {key}: {value}")
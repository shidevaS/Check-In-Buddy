"""Start Check-in Buddy FastAPI server with NVIDIA LLM"""
import uvicorn
import os
import sys

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv(r'C:\Users\PC\Pictures\Devcon-1\.env')

sys.path.insert(0, r'C:\Users\PC\Pictures\Devcon-1')

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )
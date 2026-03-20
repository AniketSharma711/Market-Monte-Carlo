import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

try:
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(f"Found: {model.name}")
            
except Exception as e:
    print(f"Error: {e}")
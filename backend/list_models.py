import os
import google.generativeai as genai
from app.core.config import settings

genai.configure(api_key=settings.google_api_key)

print("Available Embedding Models:")
try:
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f" - {m.name}")
except Exception as e:
    print(f"Error fetching models: {e}")

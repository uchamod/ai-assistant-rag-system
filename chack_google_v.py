import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# List all models that support embedding
for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(model.name)
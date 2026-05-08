import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the same embedding model used during storage
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

def embed_query(query: str) -> list:
    """Converts the user's text query into a list of numbers (vector)."""
    return embedder.embed_query(query)
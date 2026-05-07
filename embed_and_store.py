import os
import json
import time
import uuid
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Load API keys from the .env file
load_dotenv()

def load_chunks_from_json(file_path):
    """Reads the JSON file and converts it back into LangChain Document objects."""
    print(f"Loading chunks from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    
    documents = []
    for item in chunk_data:
        # Reconstruct the LangChain Document object
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        documents.append(doc)
       
    return documents

def setup_pinecone_index(index_name):
    """Checks if the index exists, creates it if it doesn't."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    
    dimension = 3072 
    
    # Check if the index already exists
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine", # Cosine similarity is highly recommended for OpenAI embeddings
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Change this to your preferred AWS region
            )
        )
        logging.basicConfig(level=logging.DEBUG)
        # Wait a moment for the index to be fully initialized on Pinecone's servers
        time.sleep(10)
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Connecting...")

def embed_and_store(documents, index_name,batch_size=1):
    """Generates embeddings and uploads them to Pinecone."""
    print("Initializing OpenAI Embeddings model...")
    # text-embedding-3-small is currently OpenAI's most cost-effective and highly capable embedding model
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2",google_api_key=os.environ.get("GOOGLE_API_KEY"))
    vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings_model
        )

    total = len(documents)
    print(f"Starting batch upload of {total} chunks (batch size {batch_size})...")

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        print(batch_ids)
        try:
            # Upload this batch
            vectorstore.add_documents(batch, ids=batch_ids)
            print(f"  ✅ Batch {i//batch_size + 1}: uploaded {len(batch)} chunks")
        except Exception as e:
            print(f"  ❌ Batch {i//batch_size + 1} FAILED: {e}")
            # Optional: continue or break; here we break to inspect the error
            raise

        # Small delay to avoid rate limits (adjust as needed)
        time.sleep(1)   


# --- Execution ---
if __name__ == "__main__":
    JSON_PATH = "./output/processed_chunks.json"
    INDEX_NAME = "ai-assistent-rag-index" # Name your index (lowercase, no spaces)
    
    # 1. Load the data
    docs = load_chunks_from_json(JSON_PATH)
     
    #2. Prepare the database
    setup_pinecone_index(INDEX_NAME)
    
    #3. Embed and upload
    if docs:
        embed_and_store(docs, INDEX_NAME,batch_size=1)
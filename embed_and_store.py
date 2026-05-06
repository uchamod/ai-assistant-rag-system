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
    
    # We use 1536 dimensions because that is the exact output size of OpenAI's text-embedding-3-small model
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

def embed_and_store(documents, index_name):
    """Generates embeddings and uploads them to Pinecone."""
    print("Initializing OpenAI Embeddings model...")
    # text-embedding-3-small is currently OpenAI's most cost-effective and highly capable embedding model
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2",google_api_key=os.environ.get("GOOGLE_API_KEY"))
    
    print(f"Uploading {len(documents)} chunks to Pinecone. This may take a moment depending on the batch size...")
    # Generate a unique ID for every single chunk
    unique_ids = [str(uuid.uuid4()) for _ in documents]

    # This single line handles the embedding generation AND the batched upload to Pinecone

        # PineconeVectorStore.from_documents(
    #     documents=documents,
    #     embedding=embeddings_model,
    #     index_name=index_name,
    #     ids=unique_ids
    # )
    logging.basicConfig(level=logging.DEBUG)
    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings_model
        )

        vectorstore.add_documents(documents, ids=unique_ids)
        logging.basicConfig(level=logging.DEBUG)
    except Exception as e:
        print(f"Error: {e}")
    print("Upload complete! Your vector database is ready for retrieval.")

# --- Execution ---
if __name__ == "__main__":
    JSON_PATH = "./output/processed_chunks.json"
    INDEX_NAME = "ai-assistent-rag-index" # Name your index (lowercase, no spaces)
    
    # 1. Load the data
    docs = load_chunks_from_json(JSON_PATH)
    # test start
    if docs:
        test_docs = docs[:2]
        test_ids = [str(uuid.uuid4()) for _ in test_docs]

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2",google_api_key=os.environ.get("GOOGLE_API_KEY"))

        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings_model)
        vectorstore.add_documents(test_docs, ids=test_ids)
        logging.basicConfig(level=logging.DEBUG)
    # test end
     
    # 2. Prepare the database
    # setup_pinecone_index(INDEX_NAME)
    
    # 3. Embed and upload
    # if docs:
    #     embed_and_store(docs, INDEX_NAME)
import os
from pinecone import Pinecone
from backend.embedder import embed_query
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "ai-assistent-rag-index"

# Connect to Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def retrieve_top_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Takes the user query, converts it to a vector,
    searches Pinecone and returns the top K matching chunks.
    """
    # Step 1 — Convert query to vector
    query_vector = embed_query(query)

    # Step 2 — Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True  # We need the actual text stored in metadata
    )

    # Step 3 — Extract the useful text from results
    chunks = []
    for match in results["matches"]:
        chunks.append({
            "score": round(match["score"], 4),       # Similarity score (0 to 1)
            "text": match["metadata"].get("text", ""),  # The actual chunk text
            "source": match["metadata"].get("source", "Unknown")  # Where it came from
        })

    return chunks
    
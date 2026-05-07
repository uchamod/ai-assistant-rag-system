import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.retriver import retrieve_top_chunks
from backend.prompt_builder import build_prompt
from backend.llm import get_gemini_answer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG AI Assistant API",
    description="Ask questions and get answers from your knowledge base.",
    version="1.0.0"
)

# Allow requests from any frontend (browser, Postman, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Request and Response shapes ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5  # Default to top 5 results

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]  # The chunks used to generate the answer


# --- Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "API is running"}


# --- Main query endpoint ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Full RAG pipeline:
    1. Embed the user query
    2. Search Pinecone for relevant chunks
    3. Build a structured prompt
    4. Get answer from Gemini
    5. Return answer + sources
    """
    try:
        logger.info(f"Received question: {request.question}")

        # Step 1 — Retrieve relevant chunks from Pinecone
        logger.info("Searching Pinecone for relevant chunks...")
        chunks = retrieve_top_chunks(request.question, top_k=request.top_k)

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base."
            )

        logger.info(f"Found {len(chunks)} relevant chunks.")

        # Step 2 — Build the structured prompt
        prompt = build_prompt(request.question, chunks)

        # Step 3 — Get answer from Gemini
        logger.info("Sending prompt to Gemini...")
        answer = get_gemini_answer(prompt)

        logger.info("Answer received successfully.")

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=chunks  # So the user can see where the answer came from
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
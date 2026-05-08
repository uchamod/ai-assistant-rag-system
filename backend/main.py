import logging
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.retriver import retrieve_top_chunks
from backend.prompt_builder import build_prompt
from backend.llm import get_gemini_answer
from backend.memory import (
    load_history,
    save_message,
    clear_session,
    check_redis_connection
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG AI Assistant API",
    description="Ask questions and get answers from your knowledge base.",
    version="2.0.0"
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
    session_id: str | None = None  # Optional — auto generated if not provided
    top_k: int = 5  # Default to top 5 results

class QueryResponse(BaseModel):
    session_id: str           # Return this to the client so they can send it back
    question: str
    answer: str
    sources: list[dict]  # The chunks used to generate the answer

class ClearRequest(BaseModel):
    session_id: str

# --- Health check endpoint ---
@app.get("/health")
def health_check():
     redis_status = check_redis_connection()
     return {
        "status": "running",
        "redis": "connected" if redis_status else "disconnected"
    }


# --- New session endpoint ---
@app.get("/session/new")
def new_session():
    """Creates a new session ID for a new conversation."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# --- Main query endpoint ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Full RAG pipeline with memory:
    1. Get or create session
    2. Load previous chat history from Redis
    3. Embed query and search Pinecone
    4. Build prompt with history + context
    5. Get answer from Gemini
    6. Save this exchange to Redis
    7. Return answer
    """
    try:
        # Step 1 — Use provided session or create a new one
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Received question: {request.question}")
        
        # Step 2 — Load previous messages from Redis
        history = load_history(session_id)
        logger.info(f"Loaded {len(history)} previous messages from memory.")

        # Step 3 — Retrieve relevant chunks from Pinecone
        logger.info("Searching Pinecone for relevant chunks...")
        chunks = retrieve_top_chunks(request.question, top_k=request.top_k)

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base."
            )

        logger.info(f"Found {len(chunks)} relevant chunks.")

        # Step 4 — Build the structured prompt
        prompt = build_prompt(request.question, chunks,history)

        

        # Step 5 — Get answer from Gemini
        logger.info("Sending prompt to Gemini...")
        answer = get_gemini_answer(prompt)
        
        # Step 6 — Save both the question and answer to Redis
        save_message(session_id, "user", request.question)
        save_message(session_id, "assistant", answer)

        logger.info("Answer received successfully.")

        return QueryResponse(
            session_id=session_id,
            question=request.question,
            answer=answer,
            sources=chunks  # So the user can see where the answer came from
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Clear session history ---
@app.post("/session/clear")
def clear_chat(request: ClearRequest):
    """Clears all memory for a session — starts a fresh conversation."""
    clear_session(request.session_id)
    return {"message": f"Session {request.session_id} cleared successfully."}        
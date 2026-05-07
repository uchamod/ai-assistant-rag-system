def build_prompt(user_query: str, chunks: list[dict]) -> str:
    """
    Combines the retrieved chunks and the user question
    into a clear, structured prompt for Gemini.
    """

    # Join all retrieved chunks into one context block
    context_block = ""
    for i, chunk in enumerate(chunks, start=1):
        context_block += f"[Source {i} | Relevance: {chunk['score']}]\n"
        context_block += f"{chunk['text']}\n\n"

    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have enough information to answer this."

---
CONTEXT:
{context_block}
---

USER QUESTION:
{user_query}

---
YOUR ANSWER:
"""
    return prompt
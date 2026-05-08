def build_prompt(user_query: str, chunks: list[dict],history: list[dict]) -> str:
    """
        Builds a structured prompt that includes:
        - Previous conversation history
        - Retrieved knowledge chunks
        - Current user question
    """

    # Join all retrieved chunks into one context block
    context_block = ""
    for i, chunk in enumerate(chunks, start=1):
        context_block += f"[Source {i} | Relevance: {chunk['score']}]\n"
        context_block += f"{chunk['text']}\n\n"


     # Build conversation history block
    history_block = ""
    if history:
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_block += f"{role_label}: {msg['content']}\n"
    else:
        history_block = "No previous conversation."    

    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have enough information to answer this."

---
KNOWLEDGE BASE CONTEXT:
{context_block}
---

CONVERSATION HISTORY:
{history_block}
---

USER QUESTION:
{user_query}

---
YOUR ANSWER:
"""
    return prompt
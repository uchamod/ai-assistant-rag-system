import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

# How long to keep a conversation in memory (seconds)
# 1800 = 30 minutes of inactivity clears the history
SESSION_TTL = 1800
MAX_HISTORY = 10  # Keep last 10 message pairs to avoid huge prompts

def get_redis_client():
    """Connect to Redis — works for both local and DigitalOcean."""
    return redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True  # Return strings not bytes
    )

def get_session_key(session_id: str) -> str:
    """Creates a unique Redis key for each user session."""
    return f"chat_session:{session_id}"

def load_history(session_id: str) -> list[dict]:
    """
    Loads all previous messages for a session.
    Returns a list like:
    [
        {"role": "user", "content": "What is your return policy?"},
        {"role": "assistant", "content": "You can return items within 30 days..."}
    ]
    """
    try:
        client = get_redis_client()
        key = get_session_key(session_id)
        
        # Get all messages stored as a Redis list
        raw_messages = client.lrange(key, 0, -1)
        
        # Each message is stored as JSON string — parse them back
        history = [json.loads(msg) for msg in raw_messages]
        
        # Only return the last MAX_HISTORY pairs to keep prompt size manageable
        return history[-(MAX_HISTORY * 2):]
    
    except Exception as e:
        print(f"Redis load error: {e}")
        return []  # Return empty history if Redis is down — system still works

def save_message(session_id: str, role: str, content: str):
    """
    Saves a single message to Redis.
    role is either 'user' or 'assistant'
    """
    try:
        client = get_redis_client()
        key = get_session_key(session_id)
        
        message = json.dumps({"role": role, "content": content})
        
        # Push message to the end of the list
        client.rpush(key, message)
        
        # Reset the expiry timer every time a new message is added
        client.expire(key, SESSION_TTL)
    
    except Exception as e:
        print(f"Redis save error: {e}")  # Fail silently — don't crash the API

def clear_session(session_id: str):
    """Clears all chat history for a session (for reset/new chat)."""
    try:
        client = get_redis_client()
        client.delete(get_session_key(session_id))
    except Exception as e:
        print(f"Redis clear error: {e}")

def check_redis_connection() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        client = get_redis_client()
        client.ping()
        return True
    except:
        return False
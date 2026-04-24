"""
Application configuration (non-secret settings)
These values can be hardcoded or loaded from a config file.
Only secrets/API keys should be in .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Supabase Settings
# =============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")

# =============================================================================
# Database (direct PostgreSQL — used for SQLAlchemy ORM + auto table creation)
# Get from: Supabase dashboard → Settings → Database → Connection string → URI
# =============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "")


# =============================================================================
# Encryption
# =============================================================================
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")


# =============================================================================
# Groq Settings
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_SQL_AGENT_MODEL = os.getenv("GROQ_SQL_AGENT_MODEL", "llama-3.3-70b-versatile")

# Rate limiting
GROQ_REQUESTS_PER_MINUTE = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "20"))
GROQ_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("GROQ_RATE_LIMIT_WINDOW_SECONDS", "60"))


# =============================================================================
# Voice Agent Settings
# =============================================================================
SQL_AGENT_MAX_DECOMPOSED_QUERIES = int(os.getenv("VOICE_SQL_AGENT_MAX_DECOMPOSED_QUERIES", "4"))
SQL_WORKERS = int(os.getenv("VOICE_SQL_WORKERS", "4"))
SQL_RESULT_MAX_ROWS = int(os.getenv("VOICE_SQL_RESULT_MAX_ROWS", "100"))
SQL_QUERY_TIMEOUT_SECONDS = int(os.getenv("VOICE_SQL_QUERY_TIMEOUT_SECONDS", "45"))

CONFIRMATION_PHRASE = os.getenv(
    "VOICE_CONFIRMATION_PHRASE",
    "Okay, I am going to fetch the data for you now. In the meantime, you can ask me something else while I retrieve it."
)

RESULT_READY_PHRASE = os.getenv(
    "VOICE_RESULT_READY_PHRASE",
    "I am ready with your data. I have the answer from your previous query. Would you like me to answer it now?"
)


# =============================================================================
# Gemini Settings
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
GEMINI_EMBEDDING_DIMENSION = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))


# =============================================================================
# Qdrant Settings
# =============================================================================
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "voice_agent_metadata")


# =============================================================================
# Deepgram Settings
# =============================================================================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_AGENT_WS_URL = os.getenv("DEEPGRAM_AGENT_WS_URL", "wss://agent.deepgram.com/v1/agent/converse")
DEEPGRAM_PROXY_CONNECT_TIMEOUT_SECONDS = float(os.getenv("DEEPGRAM_PROXY_CONNECT_TIMEOUT_SECONDS", "15"))



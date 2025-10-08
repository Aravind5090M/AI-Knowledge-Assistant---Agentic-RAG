# Universal Configuration for Any Deployment Platform
# Relies on environment variables set by the platform, not .env files

import os
import tempfile
import platform
from pathlib import Path

# --- Platform Detection (based on platform-specific environment variables) ---
IS_HUGGING_FACE = os.getenv("SPACE_ID") is not None
IS_STREAMLIT_CLOUD = "streamlit.io" in os.getenv("STREAMLIT_SERVER_HEADLESS", "")
IS_HEROKU = os.getenv("DYNO") is not None
IS_VERCEL = os.getenv("VERCEL") == "1"
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
IS_RENDER = os.getenv("RENDER") == "true"
IS_REPLIT = os.getenv("REPL_ID") is not None
IS_DOCKER = os.path.exists("/.dockerenv")

# Determine if this is any cloud/hosted environment vs local
IS_HOSTED_ENVIRONMENT = any([
    IS_HUGGING_FACE, IS_STREAMLIT_CLOUD, IS_HEROKU, IS_VERCEL, 
    IS_RAILWAY, IS_RENDER, IS_REPLIT, IS_DOCKER
])

# Demo mode: default to true for hosted environments, false for local
IS_DEMO_MODE = os.getenv("DEMO_MODE", str(IS_HOSTED_ENVIRONMENT).lower()).lower() == "true"

# --- API and Model Configuration (from environment variables) ---
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# --- Universal File Storage Strategy ---
def get_storage_base_path():
    """Get base storage path that works on any platform"""
    # Check for explicitly set storage path
    storage_path = os.getenv("STORAGE_PATH")
    if storage_path:
        return Path(storage_path)
    
    # For hosted environments, use temp directory
    if IS_HOSTED_ENVIRONMENT:
        if platform.system() == "Windows":
            return Path(tempfile.gettempdir()) / "corporate_knowledge"
        else:
            return Path("/tmp") / "corporate_knowledge"
    else:
        # Local development - use project directory
        return Path(".") / "storage"

def get_user_session_dir(session_id: str = "default") -> Path:
    """Get user-specific directory - works on any platform"""
    if IS_DEMO_MODE and session_id != "default":
        # Multi-user demo mode
        base_dir = get_storage_base_path()
        base_dir.mkdir(exist_ok=True, parents=True)
        user_dir = base_dir / f"session_{session_id}"
        user_dir.mkdir(exist_ok=True, parents=True)
        return user_dir
    else:
        # Single user or production mode
        return get_storage_base_path()

# Path generator functions that work everywhere
def get_index_store_path(session_id: str = "default") -> str:
    return str(get_user_session_dir(session_id) / "faiss_index_store")

def get_bm25_index_path(session_id: str = "default") -> str:
    return str(get_user_session_dir(session_id) / "bm25_index.pkl")

def get_docstore_path(session_id: str = "default") -> str:
    return str(get_user_session_dir(session_id) / "parent_docstore.pkl")

def get_token_path(session_id: str = "default") -> str:
    return str(get_user_session_dir(session_id) / "token.json")

def get_credentials_path(session_id: str = "default") -> str:
    return str(get_user_session_dir(session_id) / "credentials.json")

def get_image_store_path(session_id: str = "default") -> str:
    path = get_user_session_dir(session_id) / "image_store"
    path.mkdir(exist_ok=True)
    return str(path)

# Default paths (work for both demo and production)
INDEX_STORE_PATH = get_index_store_path()
BM25_INDEX_PATH = get_bm25_index_path()
DOCSTORE_PATH = get_docstore_path()
TOKEN_PATH = get_token_path()
CREDENTIALS_PATH = get_credentials_path()
IMAGE_STORE_PATH = get_image_store_path()

# --- Knowledge Base Configuration ---
# Document paths - configurable via environment
LOCAL_DOCUMENT_PATHS = []
if IS_DEMO_MODE:
    demo_docs = os.getenv("DEMO_DOCS_PATH", "demo_knowledge_docs")
    demo_docs_path = Path(demo_docs)
    demo_docs_path.mkdir(exist_ok=True)
    LOCAL_DOCUMENT_PATHS = [str(demo_docs_path)]
else:
    # Production/local paths from environment or default
    doc_paths = os.getenv("DOCUMENT_PATHS", "knowledge_docs").split(",")
    LOCAL_DOCUMENT_PATHS = [path.strip() for path in doc_paths]

# --- Google API Configuration ---
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "10AQOMvb7ODMdrNoVy2dF2u6gNw8KoS19")

# Universal redirect URI - platform automatically detected
def get_redirect_uri():
    """Auto-detect redirect URI based on platform environment variables"""
    # First check for explicit override
    explicit_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if explicit_uri:
        return explicit_uri
    
    # Auto-detect based on platform
    if IS_HUGGING_FACE:
        space_id = os.getenv("SPACE_ID", "corporate-knowledge-assistant")
        space_author = os.getenv("SPACE_AUTHOR_NAME", "demo")
        return f"https://{space_author}-{space_id}.hf.space"
    
    elif IS_STREAMLIT_CLOUD:
        # Streamlit Cloud format: https://share.streamlit.io/user/repo/main/app.py
        return os.getenv("STREAMLIT_SHARING_URL", "http://localhost:8501")
    
    elif IS_HEROKU:
        app_name = os.getenv("HEROKU_APP_NAME", "your-app")
        return f"https://{app_name}.herokuapp.com"
    
    elif IS_VERCEL:
        vercel_url = os.getenv("VERCEL_URL")
        if vercel_url:
            return f"https://{vercel_url}"
        return "http://localhost:8501"
    
    elif IS_RAILWAY:
        railway_url = os.getenv("RAILWAY_STATIC_URL")
        if railway_url:
            return f"https://{railway_url}"
        return "http://localhost:8501"
    
    elif IS_RENDER:
        render_url = os.getenv("RENDER_EXTERNAL_URL")
        if render_url:
            return render_url
        return "http://localhost:8501"
    
    else:
        # Local development
        port = os.getenv("PORT", "8501")
        return f"http://localhost:{port}"

REDIRECT_URI = get_redirect_uri()

SCOPES = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid',
    'https://www.googleapis.com/auth/gmail.modify', 
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.readonly'
]

# --- Session Management for Multi-User Support ---
class SessionManager:
    """Manage user sessions and file paths for multi-user demo"""
    
    @staticmethod
    def get_session_id(request_or_user_id: str = None) -> str:
        """Generate or retrieve session ID"""
        if request_or_user_id:
            # Use provided user ID or extract from request
            return f"session_{abs(hash(request_or_user_id)) % 1000000}"
        return "default"
    
    @staticmethod
    def get_user_paths(session_id: str = "default") -> dict:
        """Get all file paths for a specific user session"""
        if IS_HUGGING_FACE:
            return {
                "index_store": get_index_store_path(session_id),
                "bm25_index": get_bm25_index_path(session_id),
                "docstore": get_docstore_path(session_id),
                "token": get_token_path(session_id),
                "credentials": get_credentials_path(session_id),
                "image_store": get_image_store_path(session_id)
            }
        else:
            return {
                "index_store": INDEX_STORE_PATH,
                "bm25_index": BM25_INDEX_PATH,
                "docstore": DOCSTORE_PATH,
                "token": TOKEN_PATH,
                "credentials": CREDENTIALS_PATH,
                "image_store": IMAGE_STORE_PATH
            }
    
    @staticmethod
    def cleanup_old_sessions(max_age_hours: int = 24):
        """Cleanup old session directories to save space - works on any platform"""
        if not IS_DEMO_MODE:
            return
        
        import time
        import shutil
        
        try:
            storage_base = get_storage_base_path()
            if not storage_base.exists():
                return
                
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            for session_dir in storage_base.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith("session_"):
                    if session_dir.stat().st_mtime < cutoff_time:
                        try:
                            shutil.rmtree(session_dir)
                            print(f"Cleaned up old session: {session_dir.name}")
                        except Exception as e:
                            print(f"Failed to cleanup session {session_dir}: {e}")
        except Exception as e:
            print(f"Session cleanup error: {e}")

# --- Demo Configuration (all from environment variables) ---
DEMO_CONFIG = {
    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10" if IS_HOSTED_ENVIRONMENT else "100")),
    "max_documents": int(os.getenv("MAX_DOCUMENTS", "25" if IS_HOSTED_ENVIRONMENT else "1000")),
    "session_timeout_hours": int(os.getenv("SESSION_TIMEOUT_HOURS", "24" if IS_HOSTED_ENVIRONMENT else "168")),
    "enable_google_auth": os.getenv("ENABLE_GOOGLE_AUTH", "true").lower() == "true",
    "enable_file_upload": os.getenv("ENABLE_FILE_UPLOAD", "true").lower() == "true",
    "enable_ai_agents": os.getenv("ENABLE_AI_AGENTS", "true").lower() == "true",
    "demo_documents_included": os.getenv("DEMO_DOCUMENTS", "true" if IS_DEMO_MODE else "false").lower() == "true"
}

# --- Platform Info (for debugging/monitoring) ---
PLATFORM_INFO = {
    "deployment_platform": "hugging-face" if IS_HUGGING_FACE 
                          else "streamlit-cloud" if IS_STREAMLIT_CLOUD
                          else "heroku" if IS_HEROKU
                          else "vercel" if IS_VERCEL
                          else "railway" if IS_RAILWAY
                          else "render" if IS_RENDER
                          else "replit" if IS_REPLIT
                          else "docker" if IS_DOCKER
                          else "local",
    "is_hosted": IS_HOSTED_ENVIRONMENT,
    "is_demo_mode": IS_DEMO_MODE,
    "storage_strategy": "ephemeral" if IS_HOSTED_ENVIRONMENT else "persistent",
    "multi_user": IS_DEMO_MODE
}

# Export commonly used functions
__all__ = [
    'INDEX_STORE_PATH', 'BM25_INDEX_PATH', 'DOCSTORE_PATH', 
    'TOKEN_PATH', 'CREDENTIALS_PATH', 'IMAGE_STORE_PATH',
    'SessionManager', 'IS_HUGGING_FACE', 'IS_DEMO_MODE',
    'DEMO_CONFIG', 'get_user_session_dir'
]
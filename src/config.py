# config.py

# --- API and Model Configuration ---
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL_NAME = "gpt-4o"

# --- Knowledge Base Configuration ---
# The folder where the FAISS index and metadata will be stored
INDEX_STORE_PATH = "faiss_index_store" 
LOCAL_DOCUMENT_PATHS = [r"C:\Users\HP 745 G6\Downloads\Vspark Technologies\fresh_application\knowledge_docs"] # A folder named 'knowledge_docs' in your project root
BM25_INDEX_PATH = "bm25_index.pkl"
DOCSTORE_PATH = "parent_docstore.pkl"
IMAGE_STORE_PATH = "./image_store"
# NOTE: Move COHERE_API_KEY to .env file for security
# COHERE_API_KEY should be in .env file, not here
# --- Google API Configuration ---
GDRIVE_FOLDER_ID ="10AQOMvb7ODMdrNoVy2dF2u6gNw8KoS19"
#"root"
#"1KDRc6FpCdGbzszciV1eohmYwejdbZMn8"
REDIRECT_URI = "http://localhost:8501"
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid',
    'https://www.googleapis.com/auth/gmail.modify', 
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.readonly'
]
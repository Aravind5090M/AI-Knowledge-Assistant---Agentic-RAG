# knowledge_kb.py
import re
import os
import io
import json
import pickle
import uuid
import openai
import pandas as pd
import base64
import streamlit as st
import config
from google_tools import get_creds_from_session

# Using 'unstructured' for partitioning and identifying elements
from unstructured.partition.auto import partition
from unstructured.documents.elements import Title, NarrativeText, ListItem, Table,Image

from langchain.storage import InMemoryStore
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from googleapiclient.discovery import build
from rank_bm25 import BM25Okapi

# --- Helper function for Document Enrichment ---
def enrich_document_with_llm(content: str, file_name: str) -> dict:
    """Uses an LLM to generate a summary and keywords for a document's content."""
    print(f"--> Enriching document: {file_name}")
    try:
        client = openai.OpenAI()
        prompt = f"""
        Based on the following document content, generate a concise 1-2 sentence summary and a list of 5-7 relevant keywords.
        Provide the output in a clean JSON format: {{"summary": "...", "keywords": ["kw1", "kw2", ...]}}

        DOCUMENT CONTENT:
        {content[:4000]}
        """
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        response_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"summary": "N/A", "keywords": []}
    except Exception as e:
        print(f"Could not enrich document {file_name}. Error: {e}")
        return {"summary": "N/A", "keywords": []}

# --- Structure-Aware Chunking Function ---
def chunk_by_structure(elements: list, file_name: str) -> list[Document]:
    """Groups elements from 'unstructured' into logical chunks based on titles and sections."""
    chunks = []
    current_chunk_texts = []
    current_metadata = {"source": file_name}
    for el in elements:
        if isinstance(el, Title):
            if current_chunk_texts:
                chunks.append(Document(page_content="\n".join(current_chunk_texts), metadata=current_metadata))
            current_chunk_texts = [el.text]
            current_metadata = {"source": file_name, "section_title": el.text}
        elif isinstance(el, (NarrativeText, ListItem)):
            current_chunk_texts.append(el.text)
        elif isinstance(el, Table):
            if current_chunk_texts:
                chunks.append(Document(page_content="\n".join(current_chunk_texts), metadata=current_metadata))
                current_chunk_texts = []
            table_metadata = current_metadata.copy()
            table_metadata["content_type"] = "table"
            chunks.append(Document(page_content=el.text, metadata=table_metadata))
        elif isinstance(el, Image):
            if current_chunk_texts:
                chunks.append(Document(page_content="\n".join(current_chunk_texts), metadata=current_metadata))
                current_chunk_texts = []

            # Save the image and create a chunk with its path in the metadata
            image_uuid = str(uuid.uuid4())
            image_filename = f"{image_uuid}.jpg"  # Assuming jpeg, adjust if needed
            image_path = os.path.join(config.IMAGE_STORE_PATH, image_filename)

            if hasattr(el, 'image_data'):
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(el.image_data))

                image_metadata = current_metadata.copy()
                image_metadata["content_type"] = "image"
                image_metadata["image_path"] = image_path  # Link to the saved image

                # The page content for an image chunk is its OCR'd text
                chunks.append(Document(page_content=el.text, metadata=image_metadata))
            else:
                print(f"Warning: Image data is missing for an Image element in file {file_name}.")
    if current_chunk_texts:
        chunks.append(Document(page_content="\n".join(current_chunk_texts), metadata=current_metadata))
    return chunks

# --- Tabular Data Processing and Helpers ---
def process_tabular_data(file_bytes: bytes, file_name: str, file_ext: str) -> list[Document]:
    documents = []
    try:
        df = pd.read_csv(io.BytesIO(file_bytes)) if file_ext == 'csv' else pd.read_excel(io.BytesIO(file_bytes))
        documents.append(Document(page_content=create_table_overview(df, file_name), metadata={"source": file_name, "content_type": "table_overview"}))
        documents.extend(create_row_documents(df, file_name, file_ext))
    except Exception as e:
        print(f"Error processing tabular data {file_name}: {e}")
    return documents

def create_table_overview(df: pd.DataFrame, file_name: str) -> str:
    return (f"Table Overview: {file_name}\n\n"
            f"Structure: {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns)}\n\n"
            f"Sample Data:\n{df.head(3).to_string(index=False)}\n\n"
            f"Summary Stats:\n{df.describe(include='all').to_string()}")

def create_row_documents(df: pd.DataFrame, file_name: str, file_ext: str) -> list[Document]:
    documents = []
    for idx, row in df.iterrows():
        row_content = f"Record from {file_name} (Row {idx + 1}):\n" + "\n".join([f"- {col}: {val}" for col, val in row.items() if pd.notna(val)])
        documents.append(Document(page_content=row_content, metadata={"source": file_name, "content_type": "table_row", "row_index": idx}))
    return documents

# --- Document Processing Entry Point ---
def process_document_bytes(file_bytes: bytes, file_name: str) -> tuple[str, list]:
    file_ext = file_name.lower().split('.')[-1]
    if file_ext in ['csv', 'xlsx', 'xls']:
        child_docs = process_tabular_data(file_bytes, file_name, file_ext)
        parent_content = (child_docs[0].page_content if child_docs else "")
        return parent_content, child_docs
    else:
        elements = partition(file=io.BytesIO(file_bytes), file_filename=file_name, strategy="hi_res")
        full_content = "\n".join([el.text for el in elements])
        return full_content, elements

# --- Main Builder Function ---
def build_and_save_knowledge_base(gdrive_folder_id: str):
    """Builds the KB using a Parent-Child strategy with Structure-Aware Chunking."""
    print("🚀 Starting Knowledge Base build ...")
    os.makedirs(config.IMAGE_STORE_PATH, exist_ok=True)
    docstore = InMemoryStore()
    child_documents = []
    
    # --- Step 1: Gather all files to process from sources ---
    all_files_to_process = []
    local_path = config.LOCAL_DOCUMENT_PATHS[0]
    if os.path.isdir(local_path):
        for filename in os.listdir(local_path):
            file_path = os.path.join(local_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    all_files_to_process.append({"name": filename, "bytes": f.read()})
    try:
        creds = get_creds_from_session()
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(q=f"'{gdrive_folder_id}' in parents and trashed = false", fields="files(id, name)").execute()
        for item in results.get('files', []):
            request = service.files().get_media(fileId=item['id'])
            all_files_to_process.append({"name": item['name'], "bytes": request.execute()})
    except Exception as e:
        print(f"Error loading from Google Drive: {e}")

    # --- Step 2: Process each file, enrich, and create parent/child chunks ---
    for file_info in all_files_to_process:
        file_name, file_bytes = file_info["name"], file_info["bytes"]
        print(f"--> Processing: {file_name}")
        
        parent_content, elements_or_docs = process_document_bytes(file_bytes, file_name)

        if not parent_content.strip():
            continue

        enrichment_data = enrich_document_with_llm(parent_content, file_name)
        parent_id = str(uuid.uuid4())
        
        parent_doc = Document(
            page_content=parent_content,
            metadata={
                "source": file_name,
                "doc_id": parent_id,
                **enrichment_data
            }
        )
        docstore.mset([(parent_id, parent_doc)])
        
        sub_docs = elements_or_docs if (elements_or_docs and isinstance(elements_or_docs[0], Document)) else chunk_by_structure(elements_or_docs, file_name)

        for sub_doc in sub_docs:
            sub_doc.metadata["parent_doc_id"] = parent_id
        child_documents.extend(sub_docs)

    # --- THIS IS THE CORRECTED LINE ---
    print(f"Created {len(child_documents)} child chunks from {len(list(docstore.yield_keys()))} parent documents.")

    if not child_documents:
        print("No chunks were generated. Knowledge Base build aborted.")
        return "Build failed: No chunks generated."

    # --- Step 3: Build and Save Hybrid Indexes from CHILD documents ---
    embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(child_documents, embeddings)
    vector_store.save_local(config.INDEX_STORE_PATH)
    print(f"FAISS index saved to {config.INDEX_STORE_PATH}")
    
    with open(config.DOCSTORE_PATH, "wb") as f:
        pickle.dump(docstore, f)
    print(f"Parent document store saved to {config.DOCSTORE_PATH}")

    tokenized_chunks = [doc.page_content.split(" ") for doc in child_documents]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({'index': bm25_index, 'chunks': child_documents}, f)
    print(f"BM25 index saved to {config.BM25_INDEX_PATH}")
    
    print("✅ Knowledge Base built successfully.")
    return "✅ Knowledge Base built successfully."

if __name__ == '__main__':
    build_and_save_knowledge_base()
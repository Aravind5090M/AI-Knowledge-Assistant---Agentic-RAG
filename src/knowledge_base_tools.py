# knowledge_base_tools.py
import os
import re
import pickle
import cohere
import numpy as np
from crewai.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
import config

# --- Cohere Client Initialization ---
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found in .env file.")
co = cohere.Client(cohere_api_key)

# --- Main Search Tool ---
@tool("Knowledge Base Search Tool")  
def knowledge_base_search_tool(query: str) -> str:
    """
    Performs Hybrid Search, Re-ranks results, and retrieves parent documents
    to find the most relevant information in the knowledge base.
    """
    if not all(os.path.exists(p) for p in [config.INDEX_STORE_PATH, config.BM25_INDEX_PATH, config.DOCSTORE_PATH]):
        return "Knowledge Base is not fully built. Please run the build process."
    
    try:
        # Step 1: Load all components
        embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
        vector_store = FAISS.load_local(config.INDEX_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        with open(config.BM25_INDEX_PATH, "rb") as f: bm25_data = pickle.load(f)
        with open(config.DOCSTORE_PATH, "rb") as f: docstore = pickle.load(f)

        # Step 2: Initial Hybrid Search on CHILD documents
        vector_results = vector_store.similarity_search(query, k=25)
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25_data['index'].get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:25]
        keyword_results = [bm25_data['chunks'][i] for i in top_n_indices if bm25_scores[i] > 0]
        
        combined_results = {doc.page_content: doc for doc in vector_results}
        combined_results.update({doc.page_content: doc for doc in keyword_results})
        initial_child_docs = list(combined_results.values())
        if not initial_child_docs: return "No relevant information found."

        # Step 3: Re-ranking the CHILD documents
        child_doc_texts = [doc.page_content for doc in initial_child_docs]
        reranked_results = co.rerank(model='rerank-english-v3.0', query=query, documents=child_doc_texts, top_n=5)
        
        # Step 4: Retrieve PARENT documents
        top_child_docs = [initial_child_docs[hit.index] for hit in reranked_results.results]
        
        parent_ids = list(dict.fromkeys([doc.metadata.get("parent_doc_id") for doc in top_child_docs]))
        retrieved_parents = docstore.mget(parent_ids)
        # Step 5: Find associated image paths from top children
        image_paths = []
        for child in top_child_docs:
            if "image_path" in child.metadata:
                image_paths.append(child.metadata["image_path"])
        # Step 6: Format text output from PARENT documents
        text_context_parts = ["Comprehensive Information Found:\n---"]
        for parent_doc in retrieved_parents:
            if parent_doc:
                text_context_parts.append(
                    f"Source: {parent_doc.metadata.get('source', 'N/A')}\n"
                    f"Content: {parent_doc.page_content}\n---"
                )
        # Step 7: Return a structured JSON string
        final_context = {
            "text_context": "\n".join(text_context_parts),
            "image_paths": list(dict.fromkeys(image_paths)) # De-duplicate
        }
        import json
        return json.dumps(final_context)
        # Step 5: Format output from PARENT documents
        # formatted_results = ["Comprehensive Information Found:\n---"]
        # for parent_doc in retrieved_parents:
        #     if parent_doc:
        #         formatted_results.append(
        #             f"Source: {parent_doc.metadata.get('source', 'N/A')}\n"
        #             f"Summary: {parent_doc.metadata.get('summary', 'N/A')}\n"
        #             f"Keywords: {parent_doc.metadata.get('keywords', 'N/A')}\n"
        #             f"Content: {parent_doc.page_content}\n---"
        #         )
        # return "\n".join(formatted_results)

    except Exception as e:
        return json.dumps({"text_context": f"An error occurred: {str(e)}", "image_paths": []})

# --- Source Formatting Utility Tool ---
@tool("Source Formatting Tool")
def source_formatter_tool(raw_context: str) -> str:
    """Takes raw context, extracts unique source filenames, and formats them."""
    matches = re.findall(r"Source: (.+?)\n", raw_context)
    if not matches: return "No sources found."
    unique_sources = list(dict.fromkeys(matches))
    return "### Sources\n" + "".join(f"- {source.strip()}\n" for source in unique_sources)
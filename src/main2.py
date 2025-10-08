import streamlit as st
import os
import io
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# --- Local Module Imports ---
import config
from knowledge_kb import build_and_save_knowledge_base
from knowledge_base_tools import knowledge_base_search_tool, source_formatter_tool
from analysis_tools import python_code_executor_tool
from agents import (
    get_router_agent, get_text_analyst_agent, get_data_analyst_agent,
    get_data_preparation_agent, get_code_generation_agent, get_code_execution_agent,
    get_gmail_agent, get_hybrid_agent,get_comparison_agent
)
from tasks import (
    get_routing_task, get_text_analysis_task, get_charting_tasks,
    get_gmail_task, get_hybrid_task,get_comparison_task
)
from crew import (
    create_routing_crew, create_text_analysis_crew, create_charting_crew,
    create_gmail_crew, create_hybrid_crew
)
from google_tools import (
    google_drive_search_tool,
    gmail_search_tool, 
    gmail_summarize_tool,
    gmail_filter_tool,
    gmail_folders_tool,
    gmail_attachment_tool,
    gmail_forward_attachment_tool,
    gmail_action_tool, 
    calendar_create_tool, 
    calendar_search_tool,
    calendar_update_tool,
    calendar_force_create_tool,
    get_google_auth_flow,
    get_creds_from_session
)
current_date = datetime.now().strftime('%Y-%m-%d')
# --- App Configuration & Setup ---
st.set_page_config(
    page_title="Patil Group AI Assistant", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()
os.makedirs(config.LOCAL_DOCUMENT_PATHS[0], exist_ok=True)

# --- Custom CSS for Modern UI ---
def load_custom_css():
    st.markdown("""
    <style>
        /* Your full custom CSS from the previous response goes here */
        /* For brevity, the full CSS block is omitted, but it should be included */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.25);
        }
        .main-header h1 { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 5px rgba(0,0,0,0.3); }
        .main-header p { font-size: 1.1rem; opacity: 0.9; font-weight: 300; }
        .feature-card { background: #ffffff; padding: 2rem; border-radius: 20px; margin: 1.5rem 0; box-shadow: 0 10px 30px rgba(31, 38, 135, 0.1); }
        .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 0.8rem 1.5rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State & History Management ---
def initialize_conversation_history():
    """Initialize conversation history in session state."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_buffer' not in st.session_state:
        st.session_state.conversation_buffer = ""

def add_to_conversation_history(query, response, response_type="text"):
    """Add a new conversation to the history."""
    conversation = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'response': str(response)[:200] + "..." if len(str(response)) > 200 else str(response),
        'full_response': str(response),
        'type': response_type
    }
    st.session_state.conversation_history.append(conversation)
        # Update conversation buffer (keep last 5 conversations for context)
    buffer_conversations = st.session_state.conversation_history[-5:]
    buffer_text = "\n".join([f"Q: {conv['query']}\nA: {conv['response']}" for conv in buffer_conversations])
    st.session_state.conversation_buffer = buffer_text

def clear_conversation_history():
    """Clear all conversation history."""
    st.session_state.conversation_history = []
    st.session_state.conversation_buffer = ""

def get_conversation_context():
    """Get conversation context for better responses."""
    return st.session_state.conversation_buffer if st.session_state.conversation_buffer else ""
# --- Document Comparison Logic ---


def compare_documents(files):
    """Manages the document comparison workflow."""
    file_texts = []
    for file in files:
        file.seek(0)
        text = ""
        try:
            if file.type == "application/pdf":
                import PyPDF2
                reader = PyPDF2.PdfReader(file)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                import docx
                doc = docx.Document(file)
                text = "\n".join(para.text for para in doc.paragraphs)
            elif file.type == "text/plain":
                text = file.read().decode("utf-8")
            file_texts.append((file.name, text))
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            file_texts.append((file.name, "[Error reading file]"))

    data_to_compare = "\n".join(f"---\n**Document: {name}**\n{text[:2000]}\n---" for name, text in file_texts)
    comparison_task = get_comparison_task(data_to_compare)
    crew = Crew(agents=[comparison_task.agent], tasks=[comparison_task], process=Process.sequential)
    return crew.kickoff()

# Import modularized components


# --- RENDER UI & APP LOGIC ---

load_custom_css()
initialize_conversation_history()

# Initialize Google OAuth flow
flow = get_google_auth_flow()
query_params = st.query_params

# Handle OAuth callback
if 'code' in query_params and 'google_credentials' not in st.session_state:
    code = query_params['code']
    try:
        flow.fetch_token(code=code)
        creds = flow.credentials
        st.session_state.google_credentials = {
            'token': creds.token, 'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri, 'client_id': creds.client_id,
            'client_secret': creds.client_secret, 'scopes': creds.scopes
        }
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to fetch token: {e}")

# Check login status
is_authenticated = get_creds_from_session() is not None

# --- Sidebar ---
with st.sidebar:
    st.image("logo.png", width=150)
    # --- Authentication Section (Top Priority) ---
    
    st.header("⚙️ Configuration")

    # --- NEW: Dedicated Google Authentication Section ---
    with st.expander("🔗 Google Connection", expanded=True):
        if is_authenticated:
            st.success("✅ Connected to Google")
           
            if st.button("Logout"):
                del st.session_state.google_credentials
                st.rerun()
        else:
            st.warning("⚠️ Not Connected to Google")
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.link_button("Login with Google", url=auth_url)
            st.info("Login is required to chat with Assistant.")
    st.markdown("---")  # Separator
    
    st.header("🔌 Connectors")
    assistant_mode = st.radio(
        "Select a Connector:",
        ("Knowledge Assistant", "Gmail Assistant", "Hybrid Assistant")
    )
    st.markdown("---")
    if assistant_mode == "Knowledge Assistant":
        st.header("📎 Document Management")
        with st.expander("Upload or Compare", expanded=True):
            uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx", "txt", "csv", "xlsx"], accept_multiple_files=True)
            action_option = st.selectbox('Action:', ["Save to Knowledge Base", "Compare Documents"])
            if st.button("Process Files"):
                if not uploaded_files: st.warning("Please select files first.")
                else:
                    if action_option == "Save to Knowledge Base":
                        with st.spinner(f"Saving {len(uploaded_files)} file(s)..."):
                            for file in uploaded_files:
                                with open(os.path.join(config.LOCAL_DOCUMENT_PATHS[0], file.name), "wb") as f:
                                    f.write(file.getvalue())
                            st.success("Files saved!")
                            st.info("💡 Remember to Update Knowledge Base.")
                    elif action_option == "Compare Documents":
                        if len(uploaded_files) < 2:
                            st.warning("Please upload at least two documents to compare.")
                        else:
                            with st.spinner("Comparing documents..."):
                                comparison_result = compare_documents(uploaded_files)
                                # Store result in session state to display in the main window
                                st.session_state.display_result = comparison_result
                                add_to_conversation_history(f"Comparison of {len(uploaded_files)} documents", comparison_result)
                                st.rerun() # Rerun to display the result immediately
                                pass
        st.header("🔄 Knowledge Base")
        client_gdrive_id = st.text_input(
            "Your Google Drive Folder ID",
            value=""#config.GDRIVE_FOLDER_ID  # You can keep your ID as the default
            )
        if st.button("Update Knowledge Base",
                    disabled=not is_authenticated,
                    help="You must connect to Google before updating the knowledge base." if not is_authenticated else None):
            with st.spinner("Updating Knowledge Base..."):
                result_message = build_and_save_knowledge_base(gdrive_folder_id=client_gdrive_id)
                st.success(f"✅ {result_message}")
    
     # --- NEW: Conversation History Section in Sidebar ---
    st.markdown("---")
    st.header("💬 History")
    if st.button("🗑️ Clear History"):
        clear_conversation_history()
        st.rerun()
    
    with st.expander("View Recent Conversations", expanded=False):
        if not st.session_state.conversation_history:
            st.info("No conversations yet.")
        else:
            for conv in reversed(st.session_state.conversation_history):
                st.markdown(f"**__{conv['timestamp']}__**")
                st.markdown(f"**You:** _{conv['query']}_")
                st.markdown(f"**AI:** _{conv['response'][:150]}..._")
                st.markdown("---")


st.markdown('<div class="main-header"><h1><span style="font-size: 4.5rem;">🤖</span> Corporate Knowledge Assistant - Patil Group</h1><p>Query your unified corporate knowledge base. I will provide a fully-sourced, trustworthy answer based on your internal data.</p></div>', unsafe_allow_html=True)
#st.markdown('<div class="feature-card">', unsafe_allow_html=True)
# --- Main Query Input ---
# Dynamic placeholder based on selected assistant
if assistant_mode == "Knowledge Assistant":
    placeholder_text = "e.g., Summarize the BBP_PRIPL_PS_02.pdf document, explain the diagram, what is the IT policy"
elif assistant_mode == "Gmail Assistant":
    placeholder_text = "e.g., list emails received today, send email to team, search emails from last week"
else:  # Hybrid Assistant
    placeholder_text = "e.g., find policy document and send to team, search procedures and draft email"

user_query = st.text_input(f"Ask {assistant_mode}...", placeholder=placeholder_text)

if st.button("🚀 Get Answer", disabled=not is_authenticated):
    if user_query:
        with st.spinner("Analyzing..."):
            retrieved_context = knowledge_base_search_tool.run(query=user_query)

            if assistant_mode == "Knowledge Assistant":
                # Let the intelligent routing decide between text_analysis and charting
                routing_crew = create_routing_crew(user_query)
                routing_decision = routing_crew.kickoff()
                
                decision_str = str(routing_decision).lower().strip()
                
                if "charting" in decision_str:
                    charting_crew = create_charting_crew(user_query, retrieved_context)
                    final_result = charting_crew.kickoff()
                    add_to_conversation_history(user_query, final_result, "chart")
                else:
                    # Default to text_analysis - the agent will handle domain validation internally
                    text_analysis_crew = create_text_analysis_crew(user_query, retrieved_context)
                    final_result = text_analysis_crew.kickoff()
                    add_to_conversation_history(user_query, final_result, "text")

            elif assistant_mode == "Gmail Assistant":
                # Gmail agent will handle domain validation internally
                gmail_crew = create_gmail_crew(user_query)
                final_result = gmail_crew.kickoff()
                add_to_conversation_history(user_query, final_result, "gmail")

            elif assistant_mode == "Hybrid Assistant":
                hybrid_crew = create_hybrid_crew(user_query)
                final_result = hybrid_crew.kickoff()
                add_to_conversation_history(user_query, final_result, "hybrid")

            # Run validation on the final result
            try:
                from tasks import get_validation_task
                validation_task = get_validation_task(user_query, final_result, assistant_mode)
                validation_crew = Crew(agents=[validation_task.agent], tasks=[validation_task], process=Process.sequential)
                validation_result = validation_crew.kickoff()
                
                # Process validation results with user-friendly messages
                validation_str = str(validation_result)
                
                if "❌" in validation_str:
                    # Only show connector mismatch suggestions for single-purpose assistants
                    if assistant_mode != "Hybrid Assistant":
                        # Check if it's a domain mismatch (wrong connector selected)
                        if "email operations" in validation_str.lower() and assistant_mode == "Knowledge Assistant":
                            st.info("💡 **Connector Suggestion:** Your query appears to be about email operations. For better results, please select **Gmail Assistant** or **Hybrid Assistant** from the sidebar to handle email-related queries.")
                        elif "document" in validation_str.lower() and assistant_mode == "Gmail Assistant":
                            st.info("💡 **Connector Suggestion:** Your query appears to be about documents or knowledge base content. For better results, please select **Knowledge Assistant** or **Hybrid Assistant** from the sidebar to handle document-related queries.")
                        else:
                            # Other validation failures for single-purpose assistants
                            st.warning(f"**Quality Alert:** {validation_result}")
                    else:
                        # For Hybrid Assistant, only show non-mismatch validation issues
                        if "CONNECTOR MISMATCH" not in validation_str:
                            st.warning(f"**Quality Alert:** {validation_result}")
                        
                elif "⚠️" in validation_str:
                    st.info(f"**Quality Note:** {validation_result}")
                # For passed validation, keep quiet for clean UI
                    
            except Exception as e:
                # If validation fails, don't break the main flow
                st.debug(f"Validation system encountered an error: {e}")

            # Parse context for sources
            try:
                import json
                context_json = json.loads(retrieved_context)
                sources = source_formatter_tool(context_json.get("text_context", ""))
            except Exception:
                sources = "Sources could not be extracted."

            # Display the final result
            st.markdown("---")
            st.subheader("✅ Final Answer")
            st.markdown(str(final_result))
            
            # Display sources
            if sources and sources != "Sources could not be extracted." and sources != "No sources found.":
                st.markdown("---")
                st.markdown(sources)
            
            st.download_button(label="📥 Download Answer", data=str(final_result), file_name="answer.md", mime="text/markdown")

            # Handle chart output if generated
            if os.path.exists("chart.html"):
                with open("chart.html", "r", encoding="utf-8") as f:
                    html_code = f.read()
                st.subheader("📊 Generated Chart")
                st.components.v1.html(html_code, height=600, scrolling=True)
                st.download_button(label="📥 Download Chart", data=html_code, file_name="chart.html", mime="text/html")
                os.remove("chart.html")
    else:
        st.warning("Please enter a query.")
st.markdown('</div>', unsafe_allow_html=True)
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
    get_google_creds
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
    if 'google_auth_ok' not in st.session_state:
        st.session_state.google_auth_ok = os.path.exists('token.json')

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
def get_comparison_agent():
    """Defines the CrewAI agent for comparing documents."""
    return Agent(
        role='Document Comparison Specialist',
        goal='To meticulously compare documents, highlighting summaries, similarities, and differences.',
        backstory="You are an expert at analyzing documents. Your task is to provide a clear, structured report.",
        verbose=True, allow_delegation=False, llm=ChatOpenAI(model=config.OPENAI_MODEL_NAME)
    )

def compare_documents(files):
    """Manages the document comparison workflow."""
    agent = get_comparison_agent()
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

    prompt = (
        "Analyze the following documents and provide a structured comparison.\n"
        "Format your output exactly as follows:\n"
        "### Document Summaries\n- **[Doc1 Name]:** [Summary]\n- **[Doc2 Name]:** [Summary]\n"
        "### Key Similarities\n- [Similarity 1]\n"
        "### Key Differences\n- [Difference 1]\n"
        "\nHere are the documents:\n"
    )
    for name, text in file_texts:
        prompt += f"---\n**Document: {name}**\n{text[:2000]}\n---\n"

    comparison_task = Task(description=prompt, expected_output="A structured comparison report in Markdown format.", agent=agent)
    crew = Crew(agents=[agent], tasks=[comparison_task], process=Process.sequential)
    return crew.kickoff()

# --- RENDER UI & APP LOGIC ---

load_custom_css()

initialize_conversation_history()
# --- Sidebar ---
with st.sidebar:
    st.image("./assets/logo.png", width=150)
    # --- Authentication Section (Top Priority) ---
    
    st.header("⚙️ Configuration")

    # --- NEW: Dedicated Google Authentication Section ---
    with st.expander("🔗 Google Connection", expanded=True):
        if st.session_state.google_auth_ok:
            st.success("✅ Connected to Google")
        else:
            st.warning("⚠️ Not Connected to Google")

        if st.button("Connect to Google Account"):
            with st.spinner("Waiting for Google authentication..."):
                try:
                    # This function call triggers the browser login flow
                    get_google_creds()
                    st.session_state.google_auth_ok = True
                    st.rerun() # Rerun the app to update the status
                except Exception as e:
                    st.error(f"Authentication failed: {e}")
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
                        # ... (comparison logic)
                        pass
        st.header("🔄 Knowledge Base")
        if st.button("Update Knowledge Base",
                    disabled=not st.session_state.google_auth_ok,
                    help="You must connect to Google before updating the knowledge base."):
            with st.spinner("Updating Knowledge Base..."):
                result_message = build_and_save_knowledge_base()
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


st.markdown('<div class="main-header"><h1><span style="font-size: 3.5rem;">🤖</span> Corporate Knowledge Assistant - Patil Group</h1><p>Query your unified corporate knowledge base. I will provide a fully-sourced, trustworthy answer based on your internal data.</p></div>', unsafe_allow_html=True)
#st.markdown('<div class="feature-card">', unsafe_allow_html=True)
user_query = st.text_input("Ask anything about your documents, email, or calendar...", placeholder="e.g., Summarize the BBP_PRIPL_PS_02.pdf document.")

if st.button("🚀 Get Answer"):
    if user_query:
        with st.spinner("Analyzing..."):
            def encode_image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            #st.info("Step 1: Retrieving context from knowledge base...")
            retrieved_context = knowledge_base_search_tool.run(query=user_query)
            
            #st.success("Context retrieved.")

            #st.info("Step 2: Assembling agent crews...")
            llm = ChatOpenAI(model=config.OPENAI_MODEL_NAME)
            if assistant_mode == "Knowledge Assistant":
                # --- Agent Definitions ---
                router_agent = Agent(role='Chief Router Agent',
                                    goal="Analyze user queries with context awareness and intelligently route them to appropriate agents based on intent, content, and conversation history.",
                                        backstory="You are a master router with advanced context preservation capabilities and deep understanding of email queries, document analysis, and data visualization. You can detect when a query cannot be answered with available resources. You maintain conversation context to route follow-up queries appropriately and ensure conversation continuity.", 
                                        llm=llm, verbose=True)
                text_analyst_agent = Agent(role='Multi Modal Analyst',
                                        goal="Extract and synthesize information from retrieved context to provide comprehensive, accurate answers with proper source citations.",
                                        backstory="""You are an expert analyst specializing in SAP systems, project management, and IT policies. You excel at:
                                                   
                                                   CORE ANALYSIS CAPABILITIES:
                                                   - Extracting relevant information from complex technical documents
                                                   - Synthesizing information scattered across multiple document sections
                                                   - Identifying project processes, policies, and system requirements
                                                   - Connecting related concepts from different sources
                                                   - Providing structured, actionable insights
                                                   
                                                   ANALYSIS APPROACH:
                                                   - Carefully review ALL context chunks for relevant information
                                                   - Extract and combine relevant details even if scattered
                                                   - Look for process steps, policy rules, and system configurations
                                                   - Present findings clearly with proper source attribution
                                                   - Only claim information is unavailable if truly absent from context
                                                   
                                                   You are thorough but accurate, ensuring no relevant information is missed while maintaining strict source-based accuracy.""",
                                        tools=[source_formatter_tool], llm=llm, verbose=True)
                data_analyst_agent = Agent(role='Data Analyst',
                                        goal="Analyze the user's request and provided text to identify data and specifications for a chart.",
                                        backstory="You are an expert at understanding data requirements from natural language.",
                                        llm=llm, verbose=True)
                data_preparation_agent = Agent(role='Data Preparation Specialist',
                                            goal="Take raw text and the analyst's plan, then extract and format it into a perfect CSV string.",
                                            backstory="""You Organize your data into the right format for creating charts.
                                            "You are a meticulous data cleaner who helps organize data properly for visualization.""",
                                            llm=llm, verbose=True)
            
            # --- Dynamic Chart Generation Setup ---
                code_generation_agent = Agent(
                    role='Plotly Code Generator with Comparative Analysis - Syntax Perfect',
                    goal=(
                        'Generate syntactically perfect Python code using Plotly for financial charts with enhanced '
                        'comparative capabilities. Create charts that can display multiple data series for benchmarking '
                        'and historical comparison. CRITICAL: All code must be immediately executable without syntax errors. '
                        'Every parenthesis, bracket, and quote must be properly matched. Follow standard Python syntax rules '
                        'strictly. Create professional charts with clean, readable code that can handle both single-source '
                        'and multi-source comparative data from uploaded documents and knowledge base.'
                    ),
                    backstory=(
                        "You are a meticulous Python developer specialized in creating flawless Plotly code with advanced "
                        "comparative visualization capabilities. You never make syntax errors and always count parentheses to "
                        "ensure they match. You test your code mentally before outputting it. You follow Python PEP 8 style "
                        "guidelines and create clean, professional financial visualizations that can display multiple data "
                        "series for comparison. Your charts can show current company data alongside industry benchmarks, "
                        "historical trends, and competitive analysis. You excel at creating multi-series line charts, "
                        "comparative bar charts, and benchmark overlay visualizations. Your code always runs successfully "
                        "on the first try because you are extremely careful with syntax, and you create visually compelling "
                        "charts that tell a complete financial story with proper legends, colors, and annotations."
                    ),
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                # Executor agent to safely run generated chart code
                code_execution_agent = Agent(
                    role='Safe Chart Code Executor',
                    goal=(
                        'Validate and execute Plotly Python code produced by the chart code generation agent in a '
                        'controlled environment, and return a path to a rendered HTML chart. Reject code that '
                        'contains dangerous operations.'
                    ),
                    backstory=(
                        'You validate code for safety and execute only Plotly-related operations. You never run '
                        'os/system commands or evaluate arbitrary expressions. Your output should be a single '
                        'line indicating the path to the generated HTML file or an error message.'
                    ),
                    verbose=False,
                    allow_delegation=False,
                    tools=[python_code_executor_tool],
                    llm=llm
                )
                routing_task = Task(
                description=f"""
                **Your Mission:** Analyze the user's query and decide the correct workflow based on intelligent understanding.
                **Query:** "{user_query}"
                
                **Decision Criteria:**
                - If the query asks for Gmail/email operations (reading emails, sending emails, summarizing emails, searching emails, today's emails, recent emails, unread emails, email from someone, email about something), choose: **gmail**
                - If the query asks for Google Calendar operations (meetings, appointments, schedule, calendar events) or Google Drive operations (files, documents in drive), choose: **gmail**  
                - If the query asks for charts, graphs, or visualizations from data, choose: **charting**  
                - If the query asks to BOTH search knowledge base AND send/draft emails (e.g., "search policy and send email", "find info and draft email"), choose: **hybrid**
                - For all other queries about documents, policies, procedures, or factual data from uploaded documents, choose: **text_analysis**
                
                **Gmail/Google Keywords to look for**: email, emails, gmail, inbox, send, draft, today's emails, recent emails, unread emails, message, messages, calendar, drive, google calendar, google drive, meeting, appointment, schedule, when was access given to google drive
                
                **Hybrid Query Keywords**: search + send, find + email, search + draft, policy + send email, knowledge base + email
                
                **Email Query Examples**: 
                - "summarize my emails today" → gmail
                - "when was access given to google drive" → gmail  
                - "show me recent emails" → gmail
                - "what meetings do I have" → gmail
                
                **Document Query Examples**:
                - "what is the IT policy" → text_analysis
                - "summarize BBP document" → text_analysis
                
                **Hybrid Query Examples**:
                - "search IT policy and send email to aravind@email.com" → hybrid
                - "find BBP guidelines and draft email with key points" → hybrid
                - "search knowledge base for policy info and email it to team" → hybrid
                
                **Chart Query Examples**:
                - "create a chart from sales data" → charting
                
                **For the query "{user_query}", which workflow is most appropriate?**
                """,
                expected_output="A single word decision: either 'gmail', 'text_analysis', 'charting', or 'hybrid'",
                agent=router_agent
            )

            
           

                text_analysis_task = Task(
                    description=f"""
                    **Your Mission**: Extract and analyze information from the retrieved context  to answer the user's query comprehensively.
                    
                    **User Query**: "{user_query}"
                    
                    **Retrieved Context**: 
                    {retrieved_context}
                    
                    **Conversation History Context**: 
                    {get_conversation_context()}
                    
                    **ANALYSIS INSTRUCTIONS**:
                    1. **REVIEW ALL CONTEXT**: Read every context chunk thoroughly for relevant information
                    2. **EXTRACT KEY DETAILS**: Look for specific facts, steps, rules, or requirements that answer the query
                    3. **COMBINE INFORMATION**: Synthesize related details from multiple sources into a complete answer
                    4. **PROVIDE STRUCTURED RESPONSE**: Present findings clearly with proper source citations using source_formatter_tool
                    
                    **RESPONSE GUIDELINES**:
                    - Extract specific process steps, rules, or procedures mentioned in context
                    - Quote relevant details and requirements  
                    - Organize information logically (e.g., step-by-step processes)
                    - Include specific codes, statuses, or technical details when mentioned
                    - Connect related information across different context sections
                    
                    **FOR THIS SPECIFIC QUERY**: Look for:
                    - Project creation processes and steps knowledge base
                    - Software Management rules in PG IT Policy document
                    - Any SAP system procedures or transaction codes
                    - Project planning, approval, and execution steps
                    - IT policy guidelines for project management tools
                    
                    **CRITICAL SUCCESS FACTORS**:
                    - Don't dismiss context chunks with low relevance scores if they contain useful information
                    - Look beyond summaries - examine the actual content sections
                    - Extract specific actionable information when available
                    - Only claim information is unavailable if absolutely no relevant details exist in any context chunk
                    """,
                    expected_output="A detailed, structured answer extracting all relevant information from the context, or a clear statement of what specific information is not available. Include source citations for any information provided.",
                    agent=text_analyst_agent
                )
                
                
                analysis_task = Task(description=f"Analyze the user's query '{user_query}' and the context '{retrieved_context}'. Output a clear plan for the Data Preparation Specialist.",
                                    expected_output="A precise plan for data extraction  and organizing the data into a chart.",
                                    agent=data_analyst_agent)
                
                preparation_task = Task(description="Based on the Data Analyst's plan, extract and clean data from the context, formatting it into a perfect CSV string.",
                                        expected_output="Well-organized data in CSV format.",
                                        agent=data_preparation_agent,
                                        context=[analysis_task])
                
                coding_task = Task(
                    description=f"""
                    **User Query**: "{user_query}"

                    **Your Task**: Generate syntactically perfect Python code using Plotly to create professional charts with comparative analysis capabilities.

                    **Requirements**:
                    - Create charts that can display multiple data series for benchmarking and historical comparison
                    - Generate immediately executable code without syntax errors
                    - Handle both single-source and multi-source comparative data
                    - Use plotly.express for clean, modern visualizations
                    - Include proper legends, colors, and annotations for financial storytelling
                    - Save the chart as 'chart.html' for web display

                    **Code Quality**:
                    - Every parenthesis, bracket, and quote must be properly matched
                    - Follow Python PEP 8 style guidelines
                    - Test code mentally for syntax correctness before outputting
                    - Create visually compelling charts with professional styling
                    """,
                    expected_output="A string containing only executable Python code that reads CSV data and generates an appropriate chart.",
                    agent=code_generation_agent,
                    context=[preparation_task]
                )
                execution_task = Task(
                    description="""
                    **Your Task**: Validate and execute the Plotly Python code in a controlled environment.

                    **Execution Requirements**:
                    - Execute only Plotly-related operations using the python_code_executor_tool
                    - Reject any code containing dangerous operations (os/system commands, arbitrary expressions)
                    - Verify that 'chart.html' file was created successfully
                    - Return the path to the generated HTML file or an error message

                    **Safety First**:
                    - Never run os/system commands or evaluate arbitrary expressions
                    - Only execute code that creates Plotly charts
                    - Validate code safety before execution
                    - Report any security violations immediately
                    """,
                    expected_output="A single line indicating the path to the generated HTML file or an error message.",
                    agent=code_execution_agent,
                    context=[coding_task]
                )
                routing_crew = Crew(agents=[router_agent],
                                tasks=[routing_task],
                                process=Process.sequential,
                                verbose=False)
            
            # 3b. Run the Routing Crew to get the decision
                routing_decision = routing_crew.kickoff()
                if "charting" in str(routing_decision).lower():
                    charting_crew = Crew(
                        agents=[data_analyst_agent, data_preparation_agent, code_generation_agent, code_execution_agent],
                        tasks=[analysis_task, preparation_task, coding_task, execution_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    final_result = charting_crew.kickoff()
                # Add to conversation history
                    add_to_conversation_history(user_query, final_result, "chart")
                else: # Default to text analysis (now handles all content types)
                    text_analysis_crew = Crew(
                        agents=[text_analyst_agent],
                        tasks=[text_analysis_task],
                        process=Process.sequential,
                        verbose=True,
                        memory=False  # Ensure context is passed explicitly
                    )
                    final_result = text_analysis_crew.kickoff()
                    # Add to conversation history
                    add_to_conversation_history(user_query, final_result, "text")

            # gmail_agent = Agent(role='Gmail & Google Services Specialist',
            #                     goal="Efficiently handle Gmail, Google Calendar, and Google Drive operations with intelligent query interpretation.",
            #                     backstory="""You are a Gmail expert who understands user intent and can translate natural language requests into effective Gmail searches, calendar operations, and Google Drive access checks. 
            #                                You excel at parsing queries like 'when was access given to google drive by patil group' and turning them into precise email searches.
            #                                You know that Google Drive sharing emails come from 'drive-shares-noreply@google.com' and contain phrases like 'added you to this shared drive'.
            #                                You can extract specific dates, sender names, drive names, and access levels from email results.
            #                                Use the appropriate Google tools to fulfill user requests with intelligent query analysis and provide detailed timeline information.""",
            #                     tools=[gmail_search_tool, gmail_action_tool, google_drive_search_tool, 
            #                           calendar_create_tool, calendar_search_tool, calendar_update_tool], llm=llm, verbose=True)
            elif assistant_mode == "Gmail Assistant":
                if not st.session_state.google_auth_ok:
                    st.error("Please connect to your Google Account first.")
                    st.stop()
                gmail_agent = Agent(
                    role='Email & Calendar Assistant',
                    goal="Efficiently handle Gmail, Google Calendar, and Google Drive operations with intelligent query interpretation.",
                    backstory="""You are a Gmail expert who understands user intent and can translate natural language requests into effective Gmail searches, calendar operations, and Google Drive access checks. 
                                You excel at parsing queries like 'when was access given to google drive by patil group' and turning them into precise email searches.
                                You know that Google Drive sharing emails come from 'drive-shares-noreply@google.com' and contain phrases like 'added you to this shared drive'.
                                You can extract specific dates, sender names, drive names, and access levels from email results.
                                You can also read and summarize the full content of emails from specific senders using the gmail_summarize_tool.
                                You can filter emails by various criteria, manage folders/labels, and handle attachments.
                                Use the appropriate Google tools to fulfill user requests with intelligent query analysis and provide detailed timeline information.
                                
                                For calendar operations:
                                - Automatically check for scheduling conflicts when creating or updating meetings
                                - When conflicts detected, provide options: 1) Alternative times 2) PROCEED anyway 3) CANCEL
                                - Always include Google Calendar links when creating or updating meetings""",
                    tools=[gmail_search_tool, gmail_summarize_tool, gmail_filter_tool, gmail_folders_tool, gmail_attachment_tool, gmail_forward_attachment_tool, gmail_action_tool, google_drive_search_tool, 
                            calendar_create_tool, calendar_search_tool, calendar_update_tool, calendar_force_create_tool], llm=llm, verbose=True)
                gmail_task = Task(
                description=f"""
                **What you need to do**: Help the user with their email, calendar, or Google Drive needs.
                
                **User's Request**: "{user_query}"
                **Current Date**: {datetime.now().strftime('%B %d, %Y')} (use this when user is asking about current date(today) details,information )
                **Current Time**: Use appropriate time zone (Asia/Kolkata)
                
                **IMPORTANT CALENDAR LINK HANDLING**: 
                - When using calendar_create_tool or calendar_update_tool, these tools return formatted responses that include Google Calendar links and Google Meet links
                - ALWAYS preserve and include the COMPLETE tool response in your final answer to the user
                - NEVER summarize or paraphrase calendar tool responses - include them verbatim
                - The calendar tools automatically generate Google Meet links for video meetings
                - ALWAYS copy the exact calendar tool output including all links into your response
                
                **Available Tools**: 
                - gmail_search_tool: Search emails using Gmail queries (e.g., 'newer_than:1d' for today's emails)
                - gmail_summarize_tool: Read and summarize full email content from specific senders (e.g., 'aravind', 'john@company.com')
                - gmail_filter_tool: Filter emails by criteria (sender, unread, has_attachment, date_range, etc.)
                - gmail_folders_tool: List Gmail folders/labels or read emails from specific folders
                - gmail_attachment_tool: Handle email attachments (list, download, analyze attachments in emails)
                - gmail_forward_attachment_tool: Download attachment from email and forward to another recipient
                - gmail_action_tool: Send or draft emails
                - google_drive_search_tool: Search Google Drive files and access history
                - calendar_create_tool: Create calendar events (returns formatted response with Google Calendar link)
                - calendar_search_tool: Search calendar events by time range OR person name (e.g., 'aravind', 'this_week', 'today')
                - calendar_update_tool: Update/reschedule existing calendar events (returns formatted response with Google Calendar link)
                
                **Query Analysis & Instructions**:
                1. Analyze the user's request to understand what Gmail/Google service action is needed
                2. For EMAIL queries:
                   - "today's emails" or "recent emails": use gmail_search_tool with query 'newer_than:1d'
                   - "unread emails": use gmail_search_tool with query 'is:unread'
                   - "emails from [person]": use gmail_search_tool with query 'from:email@domain.com'
                   - "summarize my emails today": search recent emails and provide summary
                   - "read emails from [person]" or "show me content of emails from [person]": use gmail_summarize_tool with sender_name
                   - "what did [person] email me about": use gmail_summarize_tool to read email content
                   - "filter emails by [criteria]" or "show me emails with attachments": use gmail_filter_tool
                   - "list my email folders" or "show emails in [folder]": use gmail_folders_tool
                   - "check attachments in email" or "download attachment from email": use gmail_attachment_tool with message_id
                   - "forward attachment to [person]" or "send attachment from email to [email]": use gmail_forward_attachment_tool
                3. For GOOGLE DRIVE ACCESS queries:
                   - "when was access given to google drive": Search for multiple relevant terms:
                     * First try: 'from:drive-shares-noreply@google.com'
                     * Then try: 'subject:shared drive OR subject:added you'
                     * Also try: 'patil group AND (drive OR shared)'
                     * Look for emails about drive sharing, permissions, invitations
                   - Parse results for specific dates, sender information, and drive names
                   - Extract key details like: WHO gave access, WHEN, and WHAT drive/folder
                4. For CALENDAR/MEETING queries:
                   - **IMPORTANT**: When scheduling NEW meetings with a person:
                     a) If email is provided in query (e.g., "aravind.maguluri985734@gmail.com"), use that email directly
                     b) If no email provided, FIRST use gmail_search_tool to find recent emails from that person
                     c) Extract their actual email address from search results
                     d) If multiple different email addresses found, ask user for clarification
                     e) If no emails found, ask user to provide the email address
                   - **RESCHEDULING MEETINGS**: When rescheduling existing meetings:
                     a) Use calendar_search_tool with the person's name (e.g., 'aravind' or 'meeting with aravind') to find ANY meeting with that person regardless of time
                     b) The search will return meetings at ANY time - 11:30am, 2:00pm, 5:00pm, etc.
                     c) Extract the event ID and original attendee email addresses from the search results
                     d) Use calendar_update_tool with format: 'event_id|new_start_time|new_end_time|update_message'
                     e) Send apology email using gmail_action_tool to the SAME email addresses from the original meeting
                     f) **CRITICAL**: Use the exact same email address for apology that was used in the original meeting
                    **TIME and DATE PARSING**: When creating events, carefully parse time and date:
                    - **CRITICAL DATE HANDLING**: 
                         Current date is {datetime.now().strftime('%B %d, %Y')}
                        * "today" = {datetime.now().strftime('%Y-%m-%d')}
                        * "tomorrow" = {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                        * Always verify the date is correct when creating calendar events
                    - **CRITICAL TIME PARSING**: 11:30am = T11:30:00, 11:30pm = T23:30:00
                        - Use format: YYYY-MM-DDTHH:MM:SS (e.g., {datetime.now().strftime('%Y-%m-%d')}T11:30:00 for 11:30am)
                        
                   - Use calendar_create_tool format: 'Title|Description|Start_DateTime|End_DateTime|Email'
                   - Use calendar_search_tool for finding existing events
                5. Use the appropriate tool(s) to fulfill the request
                6. Provide a clear, helpful response based on the tool results with specific dates and details
                
                **Special Handling for Drive Access Queries**:
                - Search emails from 'drive-shares-noreply@google.com' for Google Drive notifications
                - Look for keywords: "added you", "shared drive", "invited you to contribute"
                - Extract specific information: Date, Person/Organization name, Drive name, Access level
                - Provide timeline details like "2 days ago" or specific dates
                
                **Special Handling for Meeting Scheduling**:
                - NEVER use placeholder emails like "aravind@example.com"
                - If email provided in query, use it directly
                - Otherwise, search Gmail first to find the person's real email address
                - If multiple emails found, present options to user for selection
                - If no emails found, ask user to provide the email address
                -
                
                **Special Handling for Meeting Rescheduling**:
                - **EMAIL CONSISTENCY**: When rescheduling, always use the SAME email address from the original meeting
                - **Workflow**: 1) Search for existing meeting → 2) Extract original email → 3) Reschedule with same email → 4) Send apology to same email
                - **Example**: If original meeting was with "aravind.maguluri985734@gmail.com", use this exact email for rescheduling AND apology email
                - **Avoid**: Don't use different email addresses like "aravindmaguluri0309@gmail.com" when original was different
                
                **CRITICAL CALENDAR LINK REQUIREMENTS**: 
                - Use multiple search strategies for Google Drive access queries
                - Gmail search queries like 'from:drive-shares-noreply@google.com', 'subject:shared drive', 'patil group drive' etc.
                - Always provide specific dates and sender information when available
                - **FOR CALENDAR OPERATIONS**: MANDATORY to include ALL links from calendar tool responses:
                  * Google Calendar event link (for adding to calendar)
                  * Google Meet video link (for joining the meeting)
                  * Event details and attendee information
                - **NEVER** say "meeting created" without including the actual links
                - **ALWAYS** paste the complete calendar tool response that contains all meeting links
                - If calendar tool doesn't return links, use calendar_force_create_tool as backup
                """,
                expected_output="A helpful, natural response that accomplishes what the user requested. **CRITICAL FOR CALENDAR EVENTS**: MUST include the complete calendar tool response with Google Calendar link AND Google Meet link. For emails and Google Drive questions, provide clear information and confirm any actions taken. Focus on what you did for them and include ALL relevant links and details - never summarize calendar responses.",
                agent=gmail_agent
            )
                #if "gmail" in str(routing_decision).lower():
                gmail_crew = Crew(
                    agents=[gmail_agent],
                    tasks=[gmail_task],
                    process=Process.sequential,
                    verbose=True
                )
                final_result = gmail_crew.kickoff()
                # Add to conversation history
                add_to_conversation_history(user_query, final_result, "gmail")

            elif assistant_mode == "Hybrid Assistant":
                hybrid_agent = Agent(role='Knowledge Base & Email Integration Specialist',
                                    goal="Search knowledge base for information and handle email communications efficiently.",
                                    backstory="""You are a professional assistant who bridges knowledge base search and email communication.
                                           
                                           CORE COMPETENCIES:
                                           - Excel at finding relevant information from documents and crafting professional emails
                                           - Always provide both knowledge base findings AND complete email sending tasks
                                           - Maintain professional tone and formatting consistency
                                           
                                           You handle tasks efficiently while providing comprehensive, connected responses.""",
                                 tools=[knowledge_base_search_tool, gmail_action_tool, source_formatter_tool], 
                                 llm=llm, verbose=True)

            
                hybrid_task = Task(
                    description=f"""
                    **Your Mission**: Search the knowledge base for information, then draft and send an professional email with that content.
                    
                    **User Query**: "{user_query}"
                    
                    **Step-by-Step Process**:
                    1. **SEARCH KNOWLEDGE BASE**: Use knowledge_base_search_tool to find relevant information based on the user's query
                    2. **ANALYZE RESULTS**: Review the retrieved information and identify key points, guidelines, or important details
                    3. **DRAFT EMAIL**: Create a professional email with:
                    - Clear subject line
                    - Well-structured content with key information from knowledge base
                    - Professional formatting with bullet points or sections if needed
                    4. **SEND EMAIL**: Use gmail_action_tool to send the drafted email to the specified recipient
                    
                    **Email Format Guidelines**:
                    - Subject: Professional and descriptive
                    - Greeting: Appropriate salutation
                    - Body: Clear, well-organized content with knowledge base information
                    - Key points in bullet format when applicable
                    - Professional closing
                    
                    **CRITICAL**: You MUST complete BOTH tasks:
                    1. Search and retrieve knowledge base information
                    2. Draft and send the email with that information
                    
                    **Available Tools**: 
                    - knowledge_base_search_tool: Search for relevant documents/information
                    - gmail_action_tool: Send emails (format: 'send|recipient@email.com|subject|body')
                    - source_formatter_tool: Format sources if needed in the email

                    
                    **Response Style**: 
                    - Be direct and natural - don't mention "research" or "finding information"
                    - Present actions as accomplished facts
                    - Instead of "I researched the IT policy and sent..." say "I've sent a comprehensive email with the IT policy guidelines..."
                    - Focus on what was delivered, not the process
                    - Sound professional but conversational
                    
                
                    **Response Format**: Provide a natural, conversational summary of what you accomplished, such as:
                    "I've researched the [topic] and sent a comprehensive email to [recipient] with all the key information including [brief summary of main points]. The email covers [main topics] and has been delivered successfully."
                    
                    """,
                    expected_output="A comprehensive response showing: 1) Knowledge base search results with key information found, 2) Confirmation that email was drafted and sent with the retrieved content, including email details (recipient, subject, key points included).",
                    agent=hybrid_agent
                )
                
                
                hybrid_crew = Crew(
                    agents=[hybrid_agent],
                    tasks=[hybrid_task],
                    process=Process.sequential,
                    verbose=True
                )
                final_result = hybrid_crew.kickoff()
                # Add to conversation history
                add_to_conversation_history(user_query, final_result, "hybrid")
                    
            
            # --- FINAL OUTPUT ---
            st.markdown("---")
            st.subheader("✅ Final Answer")
            st.markdown(str(final_result))
            
            st.download_button(label="📥 Download Answer", data=str(final_result), file_name="answer.md", mime="text/markdown")
            
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
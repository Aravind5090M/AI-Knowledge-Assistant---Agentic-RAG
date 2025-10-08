from crewai import Task
from datetime import datetime, timedelta
from agents import (
    get_router_agent, get_text_analyst_agent, get_data_analyst_agent,
    get_data_preparation_agent, get_code_generation_agent, get_code_execution_agent,
    get_gmail_agent, get_hybrid_agent,get_comparison_agent
)

def get_routing_task(user_query):
    router_agent = get_router_agent()
    return Task(
        description=f"""
        **Your Mission:** Route knowledge base queries to the appropriate analysis workflow.

        **Query to Analyze:** "{user_query}"

        **Available Workflows for Knowledge Base:**

        1. **text_analysis** - For information retrieval, explanation, and understanding
           - Core Intent: User wants to LEARN, UNDERSTAND, or GET INFORMATION from existing content
           - Use when: Asking questions about documents, seeking explanations, requesting summaries, interpreting diagrams/flowcharts

        2. **charting** - For data visualization and chart creation  
           - Core Intent: User wants to CREATE NEW visual representations from numerical/statistical data
           - Use when: Explicitly requesting to CREATE/GENERATE new charts, graphs, plots from datasets

        **Intent Analysis:**
        - Does the user want to understand/explain existing content? → text_analysis
        - Does the user want to create new data visualizations? → charting

        **Key Indicators:**
        - Question words (what, how, why, explain, describe, analyze, summarize) → text_analysis
        - Creation requests (create chart, make graph, generate visualization, plot data) → charting
        - Explaining existing diagrams/flowcharts → text_analysis
        - Understanding document content → text_analysis

        **Note:** This is for Knowledge Assistant only - no email operations are handled here.

        Analyze "{user_query}" and choose the appropriate workflow.
        """,
        expected_output="A single word decision: either 'text_analysis' or 'charting'",
        agent=router_agent
    )


def get_text_analysis_task(user_query, retrieved_context):
    text_analyst_agent = get_text_analyst_agent()
    return Task(
                    description=f"""
                    **Your Mission**: Extract and analyze information from the retrieved context  to answer the user's query comprehensively.
                    
                    **User Query**: "{user_query}"
                    
                    **Retrieved Context**: 
                    {retrieved_context}
                    
                    
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
                

def get_charting_tasks(user_query, retrieved_context):
    data_analyst_agent = get_data_analyst_agent()
    data_preparation_agent = get_data_preparation_agent()
    code_generation_agent = get_code_generation_agent()
    code_execution_agent = get_code_execution_agent()

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

    return [analysis_task, preparation_task, coding_task, execution_task]

def get_gmail_task(user_query):
    gmail_agent = get_gmail_agent()
    return Task(
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

def get_hybrid_task(user_query):
    hybrid_agent = get_hybrid_agent()
    return  Task(description=f"""
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

def get_validation_task(user_query, agent_response, assistant_mode):
    from agents import get_validation_agent
    validation_agent = get_validation_agent()
    return Task(
        description=f"""
        **Your Mission**: Validate the agent response for quality, accuracy, and domain compliance.
        
        **Context**:
        - User Query: "{user_query}"
        - Assistant Mode: {assistant_mode}
        - Agent Response: "{agent_response}"
        
        **Validation Checklist**:
        
        1. **CONNECTOR MISMATCH DETECTION**:
        - ONLY apply to single-purpose connectors (Knowledge Assistant and Gmail Assistant)
        - Knowledge Assistant + Email query = Connector mismatch
        - Gmail Assistant + Document query = Connector mismatch
        - Hybrid Assistant = NO mismatch validation (designed for both knowledge + email operations)
        - Focus on helping user choose correct connector, not blaming agent
        
        2. **RESPONSE QUALITY**:
        - Does the response actually answer the user's question?
        - Is the response complete and helpful?
        - Are sources cited for knowledge-based answers?
        - Is the formatting professional and readable?
        
        3. **ERROR DETECTION**:
        - Are there any technical errors or exceptions?
        - Did the agent claim "no information found" when information likely exists?
        - Are there any inconsistencies or contradictions?
        
        4. **SPECIFIC VALIDATION RULES**:
        - Knowledge queries should include source citations
        - Email operations should provide clear confirmation
        - Charts should be generated successfully with no syntax errors
        
        **OUTPUT FORMAT**:
        - If valid: "✅ VALIDATION PASSED: [brief reason]"
        - If connector mismatch: "❌ CONNECTOR MISMATCH: User selected {assistant_mode} for [email operations/document analysis]"
        - If other issues: "❌ VALIDATION FAILED: [specific issue] | SUGGESTED ACTION: [recommendation]"  
        - If warning: "⚠️ VALIDATION WARNING: [concern] | SUGGESTION: [improvement]"
        """,
        expected_output="A validation result with status (✅/❌/⚠️) and specific feedback about the agent's performance.",
        agent=validation_agent
    )

def get_comparison_task(data_to_compare):
    comparison_agent = get_comparison_agent()
    return Task(
        description=f"""
        **Your Mission**: Perform a detailed comparison of the provided data and generate insights.

        **Data to Compare**:
        {data_to_compare}

        **Instructions**:
        1. Analyze the data thoroughly to identify similarities, differences, and patterns.
        2. Highlight key insights, trends, or anomalies.
        3. Provide a structured summary of the comparison results.

        **Response Format**:
        - A clear and concise summary of the comparison.
        - Highlighted key points, such as major differences or significant similarities.
        - Any actionable insights derived from the data.
        """,
        expected_output="A structured summary of the comparison results, including key insights and actionable recommendations.",
        agent=comparison_agent
    )
# google_tools.py
import os
import base64
import datetime
import pytz
from email.mime.text import MIMEText
from crewai.tools import tool  # <-- Import the decorator
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
import config
import streamlit as st

# Helper functions for multi-user authentication
def get_google_auth_flow():
    """Initializes the Google OAuth 2.0 Flow for a web app."""
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [config.REDIRECT_URI],
            }
        },
        scopes=config.SCOPES,
        redirect_uri=config.REDIRECT_URI
    )

def get_creds_from_session() -> Credentials:
    """Retrieves Google credentials from the Streamlit session state."""
    if 'google_credentials' not in st.session_state:
        return None
    creds_info = st.session_state.google_credentials
    return Credentials.from_authorized_user_info(info=creds_info, scopes=config.SCOPES)

def check_calendar_conflicts(start_time: str, end_time: str, exclude_event_id: str = None) -> list:
    """Check for scheduling conflicts with existing calendar events.
    Returns list of conflicting events or empty list if no conflicts."""
    try:
        creds = get_creds_from_session()
        service = build('calendar', 'v3', credentials=creds)
        
        # Parse the datetime strings with better timezone handling
        from datetime import datetime, timezone
        
        # Parse input times - assume local timezone if none specified
        if start_time.endswith('Z'):
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            # Parse as naive datetime first
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            
            # Add timezone info (Asia/Kolkata)
            ist = pytz.timezone('Asia/Kolkata')
            if start_dt.tzinfo is None:
                start_dt = ist.localize(start_dt)
            if end_dt.tzinfo is None:
                end_dt = ist.localize(end_dt)
        
        # Get events for a wider time range to ensure we catch conflicts
        search_start = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        search_end = start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Use RFC3339 format for the API call
        time_min = search_start.isoformat()
        time_max = search_end.isoformat()
        
        # Get calendar events
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        conflicts = []
        
        for event in events:
            # Skip if this is the event we're updating
            if exclude_event_id and event.get('id') == exclude_event_id:
                continue
                
            # Skip all-day events (they don't have 'dateTime')
            event_start_info = event.get('start', {})
            event_end_info = event.get('end', {})
            
            if 'dateTime' not in event_start_info or 'dateTime' not in event_end_info:
                continue
            
            # Parse existing event times
            event_start_str = event_start_info['dateTime']
            event_end_str = event_end_info['dateTime']
            
            # Handle timezone in event datetime
            if event_start_str.endswith('Z'):
                event_start = datetime.fromisoformat(event_start_str.replace('Z', '+00:00'))
                event_end = datetime.fromisoformat(event_end_str.replace('Z', '+00:00'))
            else:
                # Handle timezone offset format like +05:30
                event_start = datetime.fromisoformat(event_start_str)
                event_end = datetime.fromisoformat(event_end_str)
            
            # Convert to same timezone for comparison
            if start_dt.tzinfo != event_start.tzinfo:
                event_start = event_start.astimezone(start_dt.tzinfo)
                event_end = event_end.astimezone(start_dt.tzinfo)
            
            # Check for overlap: events overlap if new event starts before existing ends AND new event ends after existing starts
            if start_dt < event_end and end_dt > event_start:
                conflicts.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'Untitled Event'),
                    'start': event_start.strftime('%Y-%m-%d %H:%M'),
                    'end': event_end.strftime('%Y-%m-%d %H:%M')
                })
        
        return conflicts
        
    except Exception as e:
        # Log the error but don't let it break the scheduling
        print(f"Warning: Error checking calendar conflicts: {e}")
        # Return empty list to allow scheduling to proceed
        return []

@tool("Gmail Filter Tool")
def gmail_filter_tool(filter_type: str, filter_value: str = "") -> str:
    """Filters Gmail emails by specific criteria. 
    filter_type: 'sender', 'label', 'unread', 'has_attachment', 'date_range'
    filter_value: Value to filter by (e.g., 'medium.com', 'INBOX', '7d')
    Examples: gmail_filter_tool('sender', 'medium.com') or gmail_filter_tool('unread', '')"""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    # Build query based on filter type
    query_map = {
        'sender': f'from:{filter_value}',
        'medium': 'from:medium.com OR from:noreply@medium.com',
        'unread': 'is:unread',
        'has_attachment': 'has:attachment',
        'label': f'label:{filter_value}',
        'date_range': f'newer_than:{filter_value}',  # e.g., '7d' for 7 days
        'important': 'is:important',
        'starred': 'is:starred'
    }
    
    query = query_map.get(filter_type, filter_value)
    if not query:
        return f"Invalid filter type '{filter_type}'. Use: sender, medium, unread, has_attachment, label, date_range, important, starred"
    
    try:
        result = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
        messages = result.get('messages', [])
        if not messages: 
            return f"No emails found for filter '{filter_type}' with value '{filter_value}'"
        
        output = [f"Found {len(messages)} emails for filter '{filter_type}':\n"]
        for i, msg in enumerate(messages, 1):
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'No Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'No Date')
            snippet = msg_data.get('snippet', 'No preview')
            
            # Check for attachments
            has_attachments = "📎" if any(part.get('filename') for part in msg_data['payload'].get('parts', []) if part.get('filename')) else ""
            
            output.append(f"{i}. {has_attachments} From: {sender}")
            output.append(f"   Subject: {subject}")
            output.append(f"   Date: {date}")
            output.append(f"   Preview: {snippet[:100]}...")
            output.append(f"   ID: {msg['id']}\n")
        
        return "\n".join(output)
    except Exception as e: 
        return f"An error occurred while filtering emails: {e}"

@tool("Gmail Folders Tool")
def gmail_folders_tool(action: str = "list", folder_name: str = "") -> str:
    """Lists Gmail folders/labels or gets emails from specific folder.
    action: 'list' to show all folders, 'read' to get emails from folder
    folder_name: Name of folder/label to read from (when action='read')"""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        if action == "list":
            # List all labels/folders
            labels_result = service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            
            if not labels:
                return "No folders/labels found."
            
            system_labels = []
            user_labels = []
            
            for label in labels:
                label_name = label['name']
                message_count_text = ""
                
                # Get message count for each label
                try:
                    if label.get('messagesTotal', 0) > 0:
                        message_count_text = f" ({label.get('messagesTotal', 0)} messages)"
                except:
                    pass
                
                if label['type'] == 'system':
                    system_labels.append(f"📁 {label_name}{message_count_text}")
                else:
                    user_labels.append(f"🏷️ {label_name}{message_count_text}")
            
            output = ["📧 GMAIL FOLDERS/LABELS:\n"]
            if system_labels:
                output.append("🗂️ SYSTEM FOLDERS:")
                output.extend(system_labels)
                output.append("")
            if user_labels:
                output.append("🏷️ USER LABELS:")
                output.extend(user_labels)
            
            return "\n".join(output)
        
        elif action == "read":
            if not folder_name:
                return "Please specify folder_name when action='read'"
            
            # Get emails from specific folder/label
            query = f"label:{folder_name}"
            result = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
            messages = result.get('messages', [])
            
            if not messages:
                return f"No emails found in folder '{folder_name}'"
            
            output = [f"📁 EMAILS IN '{folder_name}' FOLDER:\n"]
            for i, msg in enumerate(messages, 1):
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                headers = msg_data['payload']['headers']
                
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'No Sender')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'No Date')
                
                output.append(f"{i}. From: {sender}")
                output.append(f"   Subject: {subject}")
                output.append(f"   Date: {date}")
                output.append(f"   ID: {msg['id']}\n")
            
            return "\n".join(output)
        
        else:
            return "Invalid action. Use 'list' or 'read'"
    
    except Exception as e:
        return f"An error occurred: {e}"

@tool("Gmail Search Tool")
def gmail_search_tool(query: str) -> str:
    """Searches emails in Gmail using a query (e.g., 'from:user@example.com is:unread'). Returns email snippets."""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    try:
        result = service.users().messages().list(userId='me', q=query, maxResults=5).execute()
        messages = result.get('messages', [])
        if not messages: return "No emails found for that query."
        
        output = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            subject = next((h['value'] for h in msg_data['payload']['headers'] if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in msg_data['payload']['headers'] if h['name'] == 'From'), 'No Sender')
            output.append(f"ID: {msg['id']}, From: {sender}, Subject: {subject}")
        return "\n".join(output)
    except Exception as e: return f"An error occurred: {e}"

@tool("Gmail Summarize Tool")
def gmail_summarize_tool(sender_name: str, max_emails: int = 5) -> str:
    """Finds and summarizes recent emails from a specific sender.
    sender_name: Name or email of sender (e.g., 'aravind', 'john@company.com')
    max_emails: Number of recent emails to summarize (default 5)"""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        # Create search query for the sender
        query = f"from:{sender_name}"
        result = service.users().messages().list(userId='me', q=query, maxResults=max_emails).execute()
        messages = result.get('messages', [])
        
        if not messages:
            return f"No emails found from '{sender_name}'"
        
        output = [f"📧 RECENT EMAILS FROM '{sender_name.upper()}':\n"]
        
        for i, msg in enumerate(messages, 1):
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'No Sender')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'No Date')
            
            # Get email body
            body = ""
            if 'parts' in msg_data['payload']:
                for part in msg_data['payload']['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body_data = part['body']['data']
                        body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                        break
            elif msg_data['payload']['mimeType'] == 'text/plain' and 'data' in msg_data['payload']['body']:
                body_data = msg_data['payload']['body']['data']
                body = base64.urlsafe_b64decode(body_data).decode('utf-8')
            
            if not body:
                body = msg_data.get('snippet', 'No content available')
            
            # Truncate body for summary
            body_summary = body[:500] + "..." if len(body) > 500 else body
            
            output.append(f"📨 EMAIL {i}:")
            output.append(f"From: {sender}")
            output.append(f"Subject: {subject}")
            output.append(f"Date: {date}")
            output.append(f"Content Summary: {body_summary}")
            output.append(f"Message ID: {msg['id']}\n")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"An error occurred while summarizing emails: {e}"

@tool("Gmail Action Tool")
def gmail_action_tool(input_str: str) -> str:
    """Enhanced Gmail actions: send email, create draft, or send with attachment.
    Format: 'action|to|subject|body' or 'send_with_attachment|to|subject|body|attachment_path'"""
    parts = input_str.split('|')
    action = parts[0].strip()
    
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        if action in ["send", "draft"]:
            if len(parts) < 4:
                return "Format: 'action|to|subject|body'"
            
            _, to, subject, body = [p.strip() for p in parts[:4]]
            
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'raw': encoded_message}
            
            if action == "send":
                sent_message = service.users().messages().send(userId='me', body=create_message).execute()
                return f"✅ Email sent successfully to {to}! Message ID: {sent_message['id']}"
            else:  # draft
                draft = service.users().drafts().create(userId='me', body={'message': create_message}).execute()
                return f"✅ Draft created successfully! Draft ID: {draft['id']}"
        
        elif action == "send_with_attachment":
            if len(parts) < 5:
                return "Format: 'send_with_attachment|to|subject|body|attachment_path'"
            
            _, to, subject, body, attachment_path = [p.strip() for p in parts[:5]]
            
            # Create multipart message
            from email.mime.multipart import MIMEMultipart
            from email.mime.base import MIMEBase
            from email import encoders
            import os
            
            if not os.path.exists(attachment_path):
                return f"❌ Attachment file not found: {attachment_path}"
            
            msg = MIMEMultipart()
            msg['to'] = to
            msg['subject'] = subject
            msg.attach(MIMEText(body))
            
            # Add attachment
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= "{filename}"'
            )
            msg.attach(part)
            
            # Send email
            raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            send_message = {'raw': raw_msg}
            
            sent_message = service.users().messages().send(
                userId='me', body=send_message
            ).execute()
            
            return f"✅ Email with attachment sent successfully to {to}!\nAttachment: {filename}\nMessage ID: {sent_message['id']}"
        
        else:
            return "Invalid action. Use 'send', 'draft', or 'send_with_attachment'"
    
    except Exception as e:
        return f"An error occurred: {e}"

@tool("Gmail Attachment Tool")
def gmail_attachment_tool(message_id: str, action: str = "list", attachment_id: str = "") -> str:
    """Handles Gmail attachments - list, download, or analyze.
    message_id: Gmail message ID containing attachments
    action: 'list' to show attachments, 'download' to save files, 'analyze' to get content summary
    attachment_id: Specific attachment ID (required for download/analyze)"""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        # Get message details
        msg_data = service.users().messages().get(userId='me', id=message_id).execute()
        
        if action == "list":
            attachments = []
            
            def find_attachments(parts):
                for part in parts:
                    if part.get('filename') and part.get('body', {}).get('attachmentId'):
                        attachments.append({
                            'filename': part['filename'],
                            'attachmentId': part['body']['attachmentId'],
                            'mimeType': part.get('mimeType', 'unknown'),
                            'size': part.get('body', {}).get('size', 0)
                        })
                    elif 'parts' in part:
                        find_attachments(part['parts'])
            
            if 'parts' in msg_data['payload']:
                find_attachments(msg_data['payload']['parts'])
            
            if not attachments:
                return f"No attachments found in message {message_id}"
            
            output = [f"📎 ATTACHMENTS IN MESSAGE {message_id}:\n"]
            for i, att in enumerate(attachments, 1):
                size_mb = int(att['size']) / (1024 * 1024) if att['size'] else 0
                output.append(f"{i}. 📄 {att['filename']}")
                output.append(f"   Type: {att['mimeType']}")
                output.append(f"   Size: {size_mb:.2f} MB")
                output.append(f"   Attachment ID: {att['attachmentId']}\n")
            
            return "\n".join(output)
        
        elif action == "download":
            if not attachment_id:
                return "Please provide attachment_id for download action"
            
            # Get attachment data
            attachment = service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()
            
            # Find attachment filename
            filename = "unknown_attachment"
            def find_filename(parts):
                nonlocal filename
                for part in parts:
                    if part.get('body', {}).get('attachmentId') == attachment_id:
                        filename = part.get('filename', 'unknown_attachment')
                        return
                    elif 'parts' in part:
                        find_filename(part['parts'])
            
            if 'parts' in msg_data['payload']:
                find_filename(msg_data['payload']['parts'])
            
            # Save attachment
            import os
            downloads_path = os.path.join(os.getcwd(), 'downloads')
            os.makedirs(downloads_path, exist_ok=True)
            
            file_path = os.path.join(downloads_path, filename)
            file_data = base64.urlsafe_b64decode(attachment['data'])
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            file_size = len(file_data) / (1024 * 1024)
            return f"✅ Downloaded attachment '{filename}' ({file_size:.2f} MB) to: {file_path}"
        
        elif action == "analyze":
            if not attachment_id:
                return "Please provide attachment_id for analyze action"
            
            # Get attachment data
            attachment = service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()
            
            # Find attachment details
            filename = "unknown"
            mime_type = "unknown"
            
            def find_details(parts):
                nonlocal filename, mime_type
                for part in parts:
                    if part.get('body', {}).get('attachmentId') == attachment_id:
                        filename = part.get('filename', 'unknown')
                        mime_type = part.get('mimeType', 'unknown')
                        return
                    elif 'parts' in part:
                        find_details(part['parts'])
            
            if 'parts' in msg_data['payload']:
                find_details(msg_data['payload']['parts'])
            
            file_data = base64.urlsafe_b64decode(attachment['data'])
            file_size = len(file_data) / (1024 * 1024)
            
            analysis = f"📋 ATTACHMENT ANALYSIS:\n\n"
            analysis += f"📄 Filename: {filename}\n"
            analysis += f"🏷️ Type: {mime_type}\n"
            analysis += f"📏 Size: {file_size:.2f} MB\n\n"
            
            # Basic content analysis for text files
            if mime_type.startswith('text/') or filename.endswith(('.txt', '.csv', '.json', '.xml')):
                try:
                    content = file_data.decode('utf-8')[:1000]
                    analysis += f"📝 Content Preview:\n{content}..."
                except:
                    analysis += "📝 Content: Binary file - cannot preview text"
            else:
                analysis += "📝 Content: Binary file - use download action to save locally"
            
            return analysis
        
        else:
            return "Invalid action. Use 'list', 'download', or 'analyze'"
    
    except Exception as e:
        return f"An error occurred: {e}"

@tool("Gmail Forward Attachment Tool")
def gmail_forward_attachment_tool(message_id: str, attachment_id: str, recipient_email: str, subject: str = "", body: str = "") -> str:
    """Downloads an attachment from an email and forwards it to specified recipient.
    message_id: Original Gmail message ID with attachment
    attachment_id: Specific attachment ID to forward
    recipient_email: Email address to send attachment to (e.g., 'aravind@company.com')
    subject: Custom subject line (optional)
    body: Custom message body (optional)"""
    creds = get_creds_from_session()
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        # Get original message details
        msg_data = service.users().messages().get(userId='me', id=message_id).execute()
        
        # Find attachment details
        attachment_filename = "attachment"
        attachment_mime_type = "application/octet-stream"
        
        def find_attachment_info(parts):
            nonlocal attachment_filename, attachment_mime_type
            for part in parts:
                if part.get('body', {}).get('attachmentId') == attachment_id:
                    attachment_filename = part.get('filename', 'attachment')
                    attachment_mime_type = part.get('mimeType', 'application/octet-stream')
                    return
                elif 'parts' in part:
                    find_attachment_info(part['parts'])
        
        if 'parts' in msg_data['payload']:
            find_attachment_info(msg_data['payload']['parts'])
        
        # Get attachment data
        attachment = service.users().messages().attachments().get(
            userId='me', messageId=message_id, id=attachment_id
        ).execute()
        
        file_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Create email with attachment
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        from email import encoders
        
        msg = MIMEMultipart()
        msg['to'] = recipient_email
        
        # Set subject
        if not subject:
            original_subject = next(
                (h['value'] for h in msg_data['payload']['headers'] if h['name'] == 'Subject'), 
                'Forwarded Email'
            )
            subject = f"Fwd: {original_subject} - Attachment: {attachment_filename}"
        msg['subject'] = subject
        
        # Set body
        if not body:
            body = f"""Forwarded attachment from email.

Original email subject: {next((h['value'] for h in msg_data['payload']['headers'] if h['name'] == 'Subject'), 'No Subject')}
Attachment: {attachment_filename}
File size: {len(file_data) / (1024 * 1024):.2f} MB

This attachment was automatically forwarded by the Patil Group AI Assistant."""
        
        msg.attach(MIMEText(body))
        
        # Add attachment
        attachment_part = MIMEBase('application', 'octet-stream')
        attachment_part.set_payload(file_data)
        encoders.encode_base64(attachment_part)
        attachment_part.add_header(
            'Content-Disposition',
            f'attachment; filename= "{attachment_filename}"'
        )
        msg.attach(attachment_part)
        
        # Send email
        raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        send_message = {'raw': raw_msg}
        
        sent_message = service.users().messages().send(
            userId='me', body=send_message
        ).execute()
        
        file_size_mb = len(file_data) / (1024 * 1024)
        return f"✅ ATTACHMENT FORWARDED SUCCESSFULLY\n\nFile: {attachment_filename} ({file_size_mb:.2f} MB)\nTo: {recipient_email}\nSubject: {subject}\nMessage ID: {sent_message['id']}"
    
    except Exception as e:
        return f"An error occurred while forwarding attachment: {e}"

@tool("Calendar Email Integration Tool")
def calendar_email_integration_tool(action: str = "check_meetings", date: str = "today") -> str:
    """Integrates calendar meetings with email agendas and attachments.
    action: 'check_meetings' to find meetings and related emails, 'find_agendas' to search for meeting agendas
    date: 'today', 'tomorrow', 'this_week', or specific date (YYYY-MM-DD)"""
    try:
        # Get calendar events
        creds = get_creds_from_session()
        calendar_service = build('calendar', 'v3', credentials=creds)
        gmail_service = build('gmail', 'v1', credentials=creds)
        
        now = datetime.datetime.utcnow()
        
        # Determine date range
        if date == "today":
            time_min = now.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = now.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        elif date == "tomorrow":
            tomorrow = now + datetime.timedelta(days=1)
            time_min = tomorrow.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = tomorrow.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        elif date == "this_week":
            start_of_week = now - datetime.timedelta(days=now.weekday())
            end_of_week = start_of_week + datetime.timedelta(days=6)
            time_min = start_of_week.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = end_of_week.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        else:
            # Assume specific date format YYYY-MM-DD
            from datetime import datetime as dt
            date_obj = dt.strptime(date, '%Y-%m-%d')
            time_min = date_obj.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = date_obj.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        
        # Get calendar events
        events_result = calendar_service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime', maxResults=10
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            return f"No meetings found for {date}"
        
        output = [f"📅 MEETINGS AND RELATED EMAILS FOR {date.upper()}:\n"]
        
        for i, event in enumerate(events, 1):
            event_title = event.get('summary', 'No Title')
            event_start = event['start'].get('dateTime', event['start'].get('date'))
            event_description = event.get('description', '')
            
            output.append(f"🔸 MEETING {i}: {event_title}")
            output.append(f"   Time: {event_start}")
            
            if event_description:
                output.append(f"   Description: {event_description[:200]}...")
            
            # Search for related emails using meeting title
            try:
                # Search for emails containing meeting title or key words
                search_terms = event_title.lower().split()
                search_queries = [
                    f'subject:"{event_title}"',
                    f'agenda "{event_title}"',
                    f'meeting "{event_title}"'
                ]
                
                related_emails = []
                for query in search_queries:
                    try:
                        result = gmail_service.users().messages().list(
                            userId='me', q=query, maxResults=3
                        ).execute()
                        messages = result.get('messages', [])
                        
                        for msg in messages:
                            if msg['id'] not in [e['id'] for e in related_emails]:
                                msg_data = gmail_service.users().messages().get(
                                    userId='me', id=msg['id']
                                ).execute()
                                headers = msg_data['payload']['headers']
                                
                                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'No Sender')
                                
                                related_emails.append({
                                    'id': msg['id'],
                                    'subject': subject,
                                    'sender': sender
                                })
                    except:
                        continue
                
                if related_emails:
                    output.append(f"   📧 Related Emails ({len(related_emails)}):")
                    for email in related_emails[:3]:  # Show max 3 related emails
                        output.append(f"     • {email['subject']} (from {email['sender']})")
                        output.append(f"       ID: {email['id']}")
                else:
                    output.append(f"   📧 No related emails found")
            except Exception as e:
                output.append(f"   ❌ Error searching emails: {str(e)}")
            
            output.append("")  # Add spacing between meetings
        
        return "\n".join(output)
    
    except Exception as e:
        return f"An error occurred: {e}"

@tool("Google Drive Search Tool")
def google_drive_search_tool(file_name: str) -> str:
    """Searches for files in Google Drive by name."""
    creds = get_creds_from_session()
    service = build('drive', 'v3', credentials=creds)
    try:
        query = f"name contains '{file_name}' and trashed = false"
        results = service.files().list(q=query, pageSize=5, fields="files(id, name, webViewLink)").execute()
        items = results.get('files', [])
        if not items: return f"No files found with the name '{file_name}'."
        output = [f"Found {len(items)} file(s):"]
        for item in items: output.append(f"- Name: {item['name']}, Link: {item['webViewLink']}")
        return "\n".join(output)
    except Exception as e: return f"An error occurred: {e}"

@tool("Google Calendar Create Tool")
def calendar_create_tool(input_str: str) -> str:
    """Creates a Google Calendar event. Input format: 'summary|description|start_datetime|end_datetime|attendees_emails'
    Example: 'Meeting|Discussion|2025-09-29T11:30:00|2025-09-29T12:30:00|user@example.com'
    Note: DateTime should be in ISO format and will use local timezone (Asia/Kolkata)"""
    parts = [p.strip() for p in input_str.split('|')]
    summary, description, start_time, end_time, attendees_str = parts
    
    print(f"CALENDAR DEBUG: About to check conflicts for '{summary}' from {start_time} to {end_time}")
    
    # Check for conflicts before creating the event
    conflicts = check_calendar_conflicts(start_time, end_time)
    
    print(f"CALENDAR DEBUG: Conflict check returned {len(conflicts)} conflicts")
    
    if conflicts:
        conflict_details = []
        for conflict in conflicts:
            conflict_details.append(f"'{conflict['summary']}' from {conflict['start']} to {conflict['end']}")
        conflict_list = '; '.join(conflict_details)
        
        print(f"CALENDAR DEBUG: Returning conflict warning")
        
        return f"""⚠️ SCHEDULING CONFLICT DETECTED: Cannot schedule '{summary}' from {start_time} to {end_time} because it conflicts with existing meeting(s): {conflict_list}.

Please choose one of the following options:
1. Suggest alternative times that work for you
2. Reply with 'PROCEED' to schedule anyway despite the conflict  
3. Reply with 'CANCEL' to abort scheduling

What would you like to do?"""
    
    print(f"CALENDAR DEBUG: No conflicts found, proceeding with scheduling")
    
    creds = get_creds_from_session()
    service = build('calendar', 'v3', credentials=creds)
    attendees = [{'email': email.strip()} for email in attendees_str.split(',') if email.strip()]
    
    # Generate unique request ID for Google Meet
    import uuid
    request_id = f"meet-{uuid.uuid4().hex[:8]}-{int(datetime.datetime.now().timestamp())}"
    
    event = {
        'summary': summary, 
        'description': description,
        'start': {'dateTime': start_time, 'timeZone': 'Asia/Kolkata'},  # Use local timezone
        'end': {'dateTime': end_time, 'timeZone': 'Asia/Kolkata'},      # Use local timezone
        'attendees': attendees,
        'conferenceData': {
            'createRequest': {
                'requestId': request_id,
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},  # Email reminder 1 day before
                {'method': 'popup', 'minutes': 10},       # Popup reminder 10 min before
            ],
        },
    }
    try:
        # Insert event and send notifications to attendees (with conferenceDataVersion for Meet links)
        event_result = service.events().insert(
            calendarId='primary', 
            body=event, 
            sendNotifications=True,  # This ensures email invitations are sent
            sendUpdates='all',       # Send updates to all attendees
            conferenceDataVersion=1  # Enable Google Meet link generation
        ).execute()
        
        # Extract Google Meet link
        meet_link = event_result.get('hangoutLink', 'Meet link will be generated shortly')
        calendar_link = event_result.get('htmlLink')
        
        return f"""✅ MEETING SCHEDULED SUCCESSFULLY:

📅 **Google Calendar Event**: {calendar_link}
🎥 **Google Meet Link**: {meet_link}

**Meeting Details:**
- Title: {summary}
- Time: {start_time} to {end_time} (Asia/Kolkata)
- Invitations sent to: {attendees_str}

Both Google Calendar and Google Meet links are included in the email invitations."""
    except Exception as e: return f"An error occurred: {e}"

@tool("Google Calendar Search Tool")
def calendar_search_tool(search_query: str = "this_week") -> str:
    """Searches for events in Google Calendar. Can search by time range OR person name.
    Examples: 'this_week', 'today', 'aravind', 'meeting with john', 'tomorrow'
    Also returns event IDs and attendee emails for rescheduling purposes."""
    creds = get_creds_from_session()
    service = build('calendar', 'v3', credentials=creds)
    now = datetime.datetime.utcnow()
    
    # Determine search strategy based on query
    is_person_search = any(name in search_query.lower() for name in ['aravind', 'with', 'meeting', '@', 'john', 'smith', 'patil']) or '@' in search_query
    
    if is_person_search:
        # Search for meetings with specific person - look in next 30 days
        time_min = now.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
        time_max = (now + datetime.timedelta(days=30)).isoformat() + 'Z'
        search_text = search_query.lower()
    else:
        # Time-based search
        if search_query == "this_week":
            start_of_week = now - datetime.timedelta(days=now.weekday())
            end_of_week = start_of_week + datetime.timedelta(days=6, hours=23, minutes=59)
            time_min = start_of_week.isoformat() + 'Z'
            time_max = end_of_week.isoformat() + 'Z'
        elif search_query == "tomorrow":
            tomorrow = now + datetime.timedelta(days=1)
            time_min = tomorrow.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = tomorrow.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        else: # Default to today
            time_min = now.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = now.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
    
    try:
        events_result = service.events().list(
            calendarId='primary', timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime', maxResults=50
        ).execute()
        events = events_result.get('items', [])
        
        if is_person_search:
            # Filter events by person/attendee
            filtered_events = []
            for event in events:
                # Check in summary, description, and attendees
                summary = event.get('summary', '').lower()
                description = event.get('description', '').lower()
                attendees = event.get('attendees', [])
                attendee_emails = [att.get('email', '').lower() for att in attendees]
                attendee_names = [att.get('displayName', '').lower() for att in attendees]
                
                # Match person name in various fields
                if (any(word in summary for word in search_text.split()) or
                    any(word in description for word in search_text.split()) or
                    any(search_text in email for email in attendee_emails) or
                    any(search_text in name for name in attendee_names) or
                    any(word in ' '.join(attendee_emails + attendee_names) for word in search_text.split() if len(word) > 2)):
                    filtered_events.append(event)
            events = filtered_events
        
        if not events: 
            return f"No meetings found for '{search_query}'. Try searching with 'this_week', 'today', or a person's name."
        
        output = [f"Found {len(events)} meeting(s) for '{search_query}':"]
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            attendees = event.get('attendees', [])
            attendee_emails = [att['email'] for att in attendees if 'email' in att]
            attendee_names = [att.get('displayName', '') for att in attendees]
            attendee_info = ', '.join([f"{name} ({email})" if name else email for name, email in zip(attendee_names, attendee_emails)])
            
            output.append(f"📅 **{event['summary']}**")
            output.append(f"   ⏰ Time: {start}")
            output.append(f"   👥 Attendees: {attendee_info}")
            output.append(f"   🆔 Event ID: {event['id']}")
            output.append("")  # Empty line for spacing
            
        return "\n".join(output)
    except Exception as e: return f"An error occurred: {e}"

@tool("Google Calendar Update Tool")
def calendar_update_tool(input_str: str) -> str:
    """Updates/reschedules an existing Google Calendar event. Input format: 'event_id|new_start_datetime|new_end_datetime|update_message'
    Example: 'abc123xyz|2025-09-29T12:00:00|2025-09-29T13:00:00|Rescheduled to 12 PM'"""
    parts = [p.strip() for p in input_str.split('|')]
    event_id, new_start, new_end, update_message = parts
    
    # Check for conflicts before updating (exclude the current event being updated)
    conflicts = check_calendar_conflicts(new_start, new_end, exclude_event_id=event_id)
    if conflicts:
        conflict_details = []
        for conflict in conflicts:
            conflict_details.append(f"'{conflict['summary']}' from {conflict['start']} to {conflict['end']}")
        conflict_list = '; '.join(conflict_details)
        return f"""⚠️ SCHEDULING CONFLICT DETECTED: Cannot reschedule event to {new_start} - {new_end} because it conflicts with existing meeting(s): {conflict_list}.

Please choose one of the following options:
1. Suggest alternative times for rescheduling
2. Reply with 'PROCEED' to reschedule anyway despite the conflict
3. Reply with 'CANCEL' to keep the original time

What would you like to do?"""
    
    creds = get_creds_from_session()
    service = build('calendar', 'v3', credentials=creds)
    
    try:
        # Get the existing event
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        
        # Update the event times
        event['start'] = {'dateTime': new_start, 'timeZone': 'Asia/Kolkata'}
        event['end'] = {'dateTime': new_end, 'timeZone': 'Asia/Kolkata'}
        
        # Add update message to description
        current_desc = event.get('description', '')
        event['description'] = f"{current_desc}\n\n📅 RESCHEDULED: {update_message}"
        
        # Update the event
        updated_event = service.events().update(
            calendarId='primary', 
            eventId=event_id, 
            body=event,
            sendNotifications=True,  # Send notifications about the change
            sendUpdates='all'        # Send updates to all attendees
        ).execute()
        
        # Extract attendee emails for response
        attendees = event.get('attendees', [])
        attendee_emails = [att['email'] for att in attendees if 'email' in att]
        
        return f"The meeting '{event['summary']}' has been successfully rescheduled for {new_start} to {new_end} (Asia/Kolkata). Update notifications have been sent to all attendees ({', '.join(attendee_emails)}). You can view or edit the event using the following link: {updated_event.get('htmlLink')}"
        
    except Exception as e: 
        return f"An error occurred while updating the event: {e}"

@tool("Google Calendar Force Create Tool")
def calendar_force_create_tool(input_str: str) -> str:
    """Force creates a Google Calendar event even if conflicts exist. Use only when user explicitly confirms to proceed despite conflicts.
    Input format: 'summary|description|start_datetime|end_datetime|attendees_emails'
    Example: 'Meeting|Discussion|2025-09-29T11:30:00|2025-09-29T12:30:00|user@example.com'"""
    parts = [p.strip() for p in input_str.split('|')]
    summary, description, start_time, end_time, attendees_str = parts
    
    creds = get_creds_from_session()
    service = build('calendar', 'v3', credentials=creds)
    attendees = [{'email': email.strip()} for email in attendees_str.split(',') if email.strip()]
    
    # Check what conflicts exist for information
    conflicts = check_calendar_conflicts(start_time, end_time)
    conflict_note = ""
    if conflicts:
        conflict_details = [f"'{c['summary']}' ({c['start']}-{c['end']})" for c in conflicts]
        conflict_note = f" Note: This meeting overlaps with: {'; '.join(conflict_details)}."
    
    # Generate unique request ID for Google Meet
    import uuid
    request_id = f"meet-force-{uuid.uuid4().hex[:8]}-{int(datetime.datetime.now().timestamp())}"
    
    event = {
        'summary': summary, 
        'description': f"{description}\n\n⚠️ SCHEDULED WITH CONFLICTS: User confirmed to proceed despite time conflicts.",
        'start': {'dateTime': start_time, 'timeZone': 'Asia/Kolkata'},
        'end': {'dateTime': end_time, 'timeZone': 'Asia/Kolkata'},
        'attendees': attendees,
        'conferenceData': {
            'createRequest': {
                'requestId': request_id,
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }
    try:
        event_result = service.events().insert(
            calendarId='primary', 
            body=event, 
            sendNotifications=True,
            sendUpdates='all',
            conferenceDataVersion=1  # Enable Google Meet link generation
        ).execute()
        
        # Extract Google Meet link
        meet_link = event_result.get('hangoutLink', 'Meet link will be generated shortly')
        calendar_link = event_result.get('htmlLink')
        
        return f"""⚠️ MEETING FORCE SCHEDULED DESPITE CONFLICTS:

📅 **Google Calendar Event**: {calendar_link}
🎥 **Google Meet Link**: {meet_link}

**Meeting Details:**
- Title: {summary}
- Time: {start_time} to {end_time} (Asia/Kolkata)
- Invitations sent to: {attendees_str}{conflict_note}

Both Google Calendar and Google Meet links are included in the email invitations."""
    except Exception as e: 
        return f"An error occurred: {e}"
import os
from datetime import datetime, timedelta, timezone
import streamlit as st
import re

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("Gemini-API")

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Check your GEMINI_API_KEY. Error: {e}")
    st.stop()

st.set_page_config(page_title="AI Scheduler", page_icon="📅")
st.title("📅 Appointment Scheduler")
st.markdown("Tell me what appointment you'd like to schedule, and I’ll handle the rest!")

sentence = st.text_input("Describe your appointment:", key="appointment_desc")

if st.button("Schedule"):
    if not sentence.strip():
        st.warning("Please enter the appointment details.")
        st.stop()

    st.info("Processing your request...")

    prompt = f"""
You are an expert scheduling assistant. Extract the following details and suggest a time slot.

User's Request: \"{sentence}\"

Current Date and Time: {datetime.now().strftime('%A, %d %B %Y %I:%M %p')}

Instructions:
1. Extract 'Task', 'Deadline', 'Duration', and 'Priority'.
2. 'Task': A concise description of the event.
3. 'Deadline': The exact date and time, or a relative phrase like "end of next week".
4. 'Duration': In minutes. If not specified, default to 60 minutes.
5. 'Priority': "High", "Normal", or "Low". Default to "Normal".
6. Suggest a time slot that’s before the deadline.
7. Provide a 'Reason' for the slot.

Respond in this EXACT format:

Task: [task]
Deadline: [date and time]
Duration: [duration in minutes]
Priority: [priority]

Scheduled Slot:
 - Date: <Day, DD Month YYYY>
 - Time: <HH:MM AM/PM - HH:MM AM/PM>
 - Reason: [reason]
"""

    try:
        llm_response = llm.invoke(prompt)
        output = llm_response.content.strip()
        st.subheader("AI Output:")
        st.code(output)

        # Extract details using regex
        task_title = re.search(r"Task:\s*(.+)", output).group(1).strip()
        scheduled_date_str = re.search(r"Date:\s*(.+)", output).group(1).strip()
        scheduled_time_str = re.search(r"Time:\s*(.+)", output).group(1).strip()
        duration_minutes = int(re.search(r"Duration:\s*(\d+)", output).group(1)) if re.search(r"Duration:\s*(\d+)", output) else 60

        try:
            date_obj = datetime.strptime(scheduled_date_str, "%A, %d %B %Y")
        except ValueError:
            date_obj = datetime.strptime(scheduled_date_str, "%A, %d %B")
            date_obj = date_obj.replace(year=datetime.now().year)
            if date_obj < datetime.now():
                date_obj = date_obj.replace(year=datetime.now().year + 1)

        start_time_part, end_time_part = [t.strip() for t in scheduled_time_str.split("-")]
        start_dt = datetime.strptime(start_time_part, "%I:%M %p").replace(
            year=date_obj.year, month=date_obj.month, day=date_obj.day)
        end_dt = datetime.strptime(end_time_part, "%I:%M %p").replace(
            year=date_obj.year, month=date_obj.month, day=date_obj.day)

        local_tz = datetime.now().astimezone().tzinfo
        start_dt_local = start_dt.replace(tzinfo=local_tz)
        end_dt_local = end_dt.replace(tzinfo=local_tz)

        now_local = datetime.now().astimezone(local_tz)

        if start_dt_local < now_local:
            st.warning("The suggested time is in the past. Adjusting to now.")
            start_dt_local = now_local + timedelta(minutes=5)
            end_dt_local = start_dt_local + timedelta(minutes=duration_minutes)

        start_dt = start_dt_local.astimezone(timezone.utc)
        end_dt = end_dt_local.astimezone(timezone.utc)

    except Exception as e:
        st.error(f"Error parsing AI response: {e}")
        st.stop()


    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except:
            os.remove(TOKEN_FILE)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            st.info("Please log in to Google Calendar.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": task_title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Kolkata"},
        "description": f"Scheduled via AI Scheduler.\n\nRequest: \"{sentence}\"\nOutput:\n{output}",
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }

    try:
        created_event = service.events().insert(calendarId="primary", body=event).execute()
        st.success("✅ Appointment Scheduled!")
        st.markdown(f"[📅 View Event in Google Calendar]({created_event.get('htmlLink')})")
    except Exception as e:
        st.error(f"Failed to create event: {e}")

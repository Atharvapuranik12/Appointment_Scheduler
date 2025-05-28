import os
from datetime import datetime, timedelta, timezone
import streamlit as st
import re
from typing import TypedDict, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph, END

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


class SchedulerState(TypedDict):
    sentence: str
    output: Optional[str]
    task_title: Optional[str]
    scheduled_date_str: Optional[str]
    scheduled_time_str: Optional[str]
    duration_minutes: Optional[int]
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    service: Optional[object]
    event: Optional[dict]
    created_event: Optional[object]
    error: Optional[str]

def parse_flexible_date(date_string: str) -> Optional[datetime]:

    formats = [
        "%A, %d %B %Y",  # "Wednesday, 28 May 2025"
        "%A, %d %B",  # "Wednesday, 28 May"
        "%d %B %Y",  # "28 May 2025"
        "%B %d, %Y",  # "May 28, 2025"
        "%m/%d/%Y",  # "05/28/2025"
        "%Y-%m-%d",  # "2025-05-28"
        "%d-%m-%Y"  # "28-05-2025"
    ]
    now = datetime.now()

    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_string, fmt)

            if "%Y" not in fmt:
                date_obj = date_obj.replace(year=now.year)

                if date_obj.date() < now.date():
                    date_obj = date_obj.replace(year=now.year + 1)
            return date_obj
        except ValueError:
            continue
    return None


def process_ai_request(state: SchedulerState) -> SchedulerState:
    """Process the user request with AI"""
    sentence = state["sentence"]
    current_datetime_local = datetime.now().astimezone()

    prompt = f"""
You are an expert scheduling assistant. Extract the following details and suggest a time slot.

User's Request: \"{sentence}\"

Current Date and Time: {current_datetime_local.strftime('%A, %d %B %Y %I:%M %p %Z%z')} (Important: use this for context, especially for 'today' or 'tomorrow')

Instructions:
1. Extract 'Task', 'Deadline', 'Duration', and 'Priority'.
2. 'Task': A concise description of the event.
3. 'Deadline': The exact date and time, or a relative phrase like "end of next week". If a specific time is not given but a date is, assume end of day. If only a relative term like "tomorrow" is given, infer the date.
4. 'Duration': In minutes. If not specified, default to 60 minutes.
5. 'Priority': "High", "Normal", or "Low". Default to "Normal".
6. Suggest the best **future** time slot for the meeting, ensuring it's before the deadline and accommodates the duration. If priority is "High", suggest the earliest reasonable time.
7. Provide a brief 'Reason' for the chosen slot.
8. **Always include the year** in the 'Scheduled Slot - Date' (e.g., Friday, 28 May 2025).

Respond in the following EXACT format:

Task: [task]
Deadline: [date and time, e.g., Friday, 28 May 2025 at 5:00 PM]
Duration: [duration in minutes]
Priority: [priority]

Scheduled Slot:
 - Date: <Day, DD Month Walpole> (e.g., Friday, 28 May 2025)
 - Time: <HH:MM AM/PM - HH:MM AM/PM> (e.g., 3:00 PM - 3:30 PM)
 - Reason: [reason]
"""


    try:
        llm_response = llm.invoke(prompt)
        output = llm_response.content.strip()
        state["output"] = output


        st.subheader("Analysis:")
        with st.expander("View Raw AI Output", expanded=True):
            st.code(output, language="text")

    except Exception as e:
        state["error"] = f"Error getting AI response: {e}"
        st.error(state["error"])
        state["output"] = None

    return state


def parse_ai_response(state: SchedulerState) -> SchedulerState:
    """Extract details using regex"""
    if state.get("error"):
        return state

    try:
        output = state["output"]
        if not output:
            raise ValueError("No AI output to parse. Previous step might have failed.")

        task_title_match = re.search(r"Task:\s*(.+)", output)
        scheduled_date_str_match = re.search(r"Date:\s*(.+)", output)
        scheduled_time_str_match = re.search(r"Time:\s*(.+)", output)
        duration_minutes_match = re.search(r"Duration:\s*(\d+)", output)
        priority_match = re.search(r"Priority:\s*(.+)", output)

        if not all([task_title_match, scheduled_date_str_match, scheduled_time_str_match]):
            raise ValueError(
                "Essential scheduling details (Task, Scheduled Date, or Scheduled Time) missing from AI output.")

        state["task_title"] = task_title_match.group(1).strip()
        state["scheduled_date_str"] = scheduled_date_str_match.group(1).strip()
        state["scheduled_time_str"] = scheduled_time_str_match.group(1).strip()
        state["duration_minutes"] = int(duration_minutes_match.group(1)) if duration_minutes_match else 60
        state["parsed_priority"] = priority_match.group(1).strip() if priority_match else "Normal"

        st.subheader("📋 Extracted Information:")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Event:** {state['task_title']}")
            st.info(f"**Date:** {state['scheduled_date_str']}")
        with col2:
            st.info(f"**Time:** {state['scheduled_time_str']}")
            st.info(f"**Duration:** {state['duration_minutes']} minutes")
            if state.get("parsed_priority"):
                st.info(f"**Priority:** {state['parsed_priority']}")

    except Exception as e:
        state["error"] = f"Error parsing AI response: {e}. Ensure AI output matches expected format."
        st.error(state["error"])

    return state


def parse_datetime(state: SchedulerState) -> SchedulerState:
    """Parse date and time and handle timezone/past time adjustments"""
    if state.get("error"):
        return state

    try:
        scheduled_date_str = state["scheduled_date_str"]
        scheduled_time_str = state["scheduled_time_str"]
        duration_minutes = state["duration_minutes"]

        date_obj = parse_flexible_date(scheduled_date_str)
        if not date_obj:
            raise ValueError(f"Could not parse scheduled date: '{scheduled_date_str}'. Please check AI output format.")

        if " - " not in scheduled_time_str:
            raise ValueError(
                f"Scheduled time format is incorrect: '{scheduled_time_str}'. Expected 'HH:MM AM/PM - HH:MM AM/PM'.")

        start_time_part, end_time_part = [t.strip() for t in scheduled_time_str.split("-")]

        start_dt_naive = datetime.strptime(start_time_part, "%I:%M %p").replace(
            year=date_obj.year, month=date_obj.month, day=date_obj.day)
        end_dt_naive = datetime.strptime(end_time_part, "%I:%M %p").replace(
            year=date_obj.year, month=date_obj.month, day=date_obj.day)

        local_tz = datetime.now().astimezone().tzinfo
        start_dt_local = start_dt_naive.replace(tzinfo=local_tz)
        end_dt_local = end_dt_naive.replace(tzinfo=local_tz)

        now_local = datetime.now().astimezone(local_tz)



        if start_dt_local < now_local:
            if start_dt_local.date() == now_local.date():
                st.warning(
                    f"⚠️ The suggested start time ({start_dt_local.strftime('%I:%M %p')}) was in the past for today. Adjusting to current time + 5 minutes.")
                start_dt_local = now_local + timedelta(minutes=5)
                end_dt_local = start_dt_local + timedelta(minutes=duration_minutes)
            else:
                raise ValueError(
                    f"The AI suggested a date in the past: '{start_dt_local.strftime('%A, %d %B %Y %I:%M %p')}'. Please refine your request to schedule a future appointment.")

        start_dt_utc = start_dt_local.astimezone(timezone.utc)
        end_dt_utc = end_dt_local.astimezone(timezone.utc)

        state["start_dt"] = start_dt_utc
        state["end_dt"] = end_dt_utc

        st.subheader("⏰ Final Schedule Details:")
        st.write(f"**Start Time:** {start_dt_local.strftime('%A, %B %d, %Y at %I:%M %p')}")
        st.write(f"**End Time:** {end_dt_local.strftime('%A, %B %d, %Y at %I:%M %p')}")
        st.write(f"**Duration:** {duration_minutes} minutes")

    except ValueError as ve:
        state["error"] = f"Date/Time parsing error: {ve}"
        st.error(state["error"])
    except Exception as e:
        state["error"] = f"An unexpected error occurred during date/time parsing: {e}"
        st.error(state["error"])

    return state


def authenticate_google(state: SchedulerState) -> SchedulerState:
    """Authenticate Google Calendar"""
    if state.get("error"):
        return state

    try:
        creds = None
        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                st.warning(f"Failed to load existing token, re-authenticating: {e}")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                st.success("✅ Google Calendar authentication refreshed successfully!")
            else:
                st.info("Please complete Google Calendar authentication in the popup window.")  # Kept for user action
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                try:
                    creds = flow.run_local_server(port=0)
                    st.success("✅ Google Calendar authentication completed successfully!")
                except Exception as auth_e:
                    raise ValueError(f"Failed to complete browser authentication. Error: {auth_e}")

            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
        else:
            pass

        service = build("calendar", "v3", credentials=creds)
        state["service"] = service

    except HttpError as e:
        state["error"] = f"Google Calendar API authentication error: {e.status_code} - {e.content.decode('utf-8')}"
        st.error(state["error"])
    except Exception as e:
        state["error"] = f"Authentication error: {e}"
        st.error(state["error"])

    return state


def create_event(state: SchedulerState) -> SchedulerState:
    """Prepare calendar event object"""
    if state.get("error"):
        return state

    try:
        task_title = state["task_title"]
        start_dt = state["start_dt"]
        end_dt = state["end_dt"]
        sentence = state["sentence"]
        output = state["output"]

        event = {
            "summary": task_title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "description": f"Scheduled via AI Scheduler.\n\nOriginal Request: \"{sentence}\"\n\nAI Analysis:\n{output}",
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }

        state["event"] = event
        # Removed "Event details prepared for scheduling." message

    except Exception as e:
        state["error"] = f"Error preparing event details: {e}"
        st.error(state["error"])

    return state


def schedule_event(state: SchedulerState) -> SchedulerState:
    """Schedule the event in Google Calendar"""
    if state.get("error"):
        return state

    try:
        service = state["service"]
        event = state["event"]
        task_title = state["task_title"]
        start_dt_utc = state["start_dt"]
        end_dt_utc = state["end_dt"]

        local_tz = datetime.now().astimezone().tzinfo
        start_local = start_dt_utc.astimezone(local_tz)
        end_local = end_dt_utc.astimezone(local_tz)

        with st.spinner("📅 Creating your appointment in Google Calendar..."):
            created_event = service.events().insert(calendarId="primary", body=event).execute()
            state["created_event"] = created_event

        st.success("🎉 Appointment Successfully Scheduled!")

        st.markdown("---")
        st.subheader("📅 Appointment Details:")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            **📋 Event:** {task_title}

            **📅 Date:** {start_local.strftime('%A, %B %d, %Y')}

            **⏰ Time:** {start_local.strftime('%I:%M %p')} - {end_local.strftime('%I:%M %p')}

            **⏱️ Duration:** {state['duration_minutes']} minutes

            **🔔 Reminders Set:**
            • Email reminder: 24 hours before
            • Popup reminder: 10 minutes before
            """)

        with col2:
            st.markdown("### Quick Actions")

            calendar_link = created_event.get('htmlLink')
            if calendar_link:
                st.markdown(f"""
                <a href="{calendar_link}" target="_blank">
                    <button style="
                        background-color: #4285f4;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 16px;
                        width: 100%;
                        margin-bottom: 10px;
                    ">
                        📅 Open in Google Calendar
                    </button>
                </a>
                """, unsafe_allow_html=True)

            st.code(f"Event ID: {created_event.get('id', 'N/A')[:10]}...", language=None)

        st.markdown("---")
        st.info("""
        ✅ **What happens next:**
        • You'll receive email and popup reminders as scheduled
        • The event is now visible in your Google Calendar
        • You can edit or delete the event directly from Google Calendar
        • All attendees (if any) will be notified automatically
        """)

        with st.expander("📝 Original Request Reference"):
            st.write(f"**Your request:** \"{state['sentence']}\"")
            st.write(f"**Processed on:** {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

    except HttpError as e:
        state["error"] = f"Google Calendar API scheduling error: {e.status_code} - {e.content.decode('utf-8')}"
        st.error(state["error"])
    except Exception as e:
        state["error"] = f"Failed to create event: {e}"
        st.error(state["error"])

    return state


def handle_error(state: SchedulerState) -> SchedulerState:
    """Handle errors"""
    if state.get("error"):
        st.error(f"❌ **Error during scheduling process:** {state['error']}")

        st.markdown("---")
        st.subheader("🔧 Troubleshooting Tips:")
        st.markdown("""
        • **Review AI Output:** Check the "Raw AI Output" section to see if Gemini parsed your request correctly. Pay attention to the 'Scheduled Slot' format.
        • **Refine Your Request:** Try being more specific with dates (e.g., "next Friday, May 30, 2025") and exact times (e.g., "from 2 PM to 3 PM").
        • **Authentication Issues:** Ensure your `credentials.json` file is correctly set up and you've granted all necessary Google Calendar permissions. Try deleting `token.json` to force re-authentication.
        • **API Issues:** Verify your Gemini API key is correct and Google Calendar API is enabled in your Google Cloud Console.

        **Need more help?** Try rephrasing your appointment request.
        """)

    return state


def should_continue(state: SchedulerState) -> str:
    """Check if workflow should continue"""
    if state.get("error"):
        return "error"
    return "continue"


# Create LangGraph workflow
def create_workflow():
    workflow = Graph()

    workflow.add_node("process_ai", process_ai_request)
    workflow.add_node("parse_response", parse_ai_response)
    workflow.add_node("parse_datetime", parse_datetime)
    workflow.add_node("authenticate", authenticate_google)
    workflow.add_node("create_event", create_event)
    workflow.add_node("schedule", schedule_event)
    workflow.add_node("error_handler", handle_error)

    workflow.set_entry_point("process_ai")

    workflow.add_conditional_edges(
        "process_ai",
        should_continue,
        {"continue": "parse_response", "error": "error_handler"}
    )

    workflow.add_conditional_edges(
        "parse_response",
        should_continue,
        {"continue": "parse_datetime", "error": "error_handler"}
    )

    workflow.add_conditional_edges(
        "parse_datetime",
        should_continue,
        {"continue": "authenticate", "error": "error_handler"}
    )

    workflow.add_conditional_edges(
        "authenticate",
        should_continue,
        {"continue": "create_event", "error": "error_handler"}
    )

    workflow.add_conditional_edges(
        "create_event",
        should_continue,
        {"continue": "schedule", "error": "error_handler"}
    )

    workflow.add_edge("schedule", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


# Streamlit UI
st.set_page_config(
    page_title="AI Appointment Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📅 AI Appointment Scheduler</h1>
    <p>Tell me what appointment you'd like to schedule, and I'll handle the rest!</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 How to use this scheduler", expanded=False):
    st.markdown("""
    **Simply describe your appointment in natural language!**

    **Examples:**
    - "Meeting with John tomorrow at 3 PM"
    - "Doctor appointment next Friday at 10 AM"
    - "Team standup every Monday at 9 AM for 30 minutes"
    - "Lunch with Sarah on December 15th at noon"

    **What I can understand:**
    - Dates (relative or specific)
    - Times (12-hour or 24-hour format)
    - Duration (if not specified, defaults to 1 hour)
    - Event titles and descriptions
    """)

st.subheader("🗣️ Describe Your Appointment:")
sentence = st.text_area(
    "Tell me about your appointment...",
    placeholder="e.g., 'Schedule a meeting with the marketing team next Tuesday at 2 PM for 1 hour'",
    height=100,
    key="appointment_desc"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    schedule_button = st.button("🚀 Schedule My Appointment", use_container_width=True)

if schedule_button:
    if not sentence.strip():
        st.warning("⚠️ Please enter the appointment details before scheduling.")
        st.stop()

    with st.spinner("Processing your request..."):
        st.info("🔄 Analyzing your request and preparing to schedule...")

    workflow_app = create_workflow()

    initial_state = SchedulerState(
        sentence=sentence,
        output=None,
        task_title=None,
        scheduled_date_str=None,
        scheduled_time_str=None,
        duration_minutes=60,
        start_dt=None,
        end_dt=None,
        service=None,
        event=None,
        created_event=None,
        error=None
    )

    final_state = workflow_app.invoke(initial_state)

    if final_state.get("error"):
        pass
    elif final_state.get("created_event"):
        pass
    else:
        st.warning(
            "Scheduling process completed, but final status not explicitly displayed. Check sections above for details.")

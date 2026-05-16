import os
import json
import uuid
import sqlite3
import datetime
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Optional Google imports (app won't crash if missing in mock mode)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    _GOOGLE_OK = True
except ImportError:
    _GOOGLE_OK = False

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TIMEZONE = os.getenv("TZ", "America/New_York")
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]

client = OpenAI(api_key=OPENAI_API_KEY)
client = wrap_openai(client)   # raw OpenAI client

# ============================================
# DATABASE / PERSISTENCE
# ============================================
class AgentPersistence:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                function_name TEXT,
                function_result TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                name TEXT PRIMARY KEY,
                email TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT,
                title TEXT,
                attendee_email TEXT,
                start_iso TEXT,
                end_iso TEXT,
                status TEXT DEFAULT 'active',
                google_event_id TEXT
            )
        """)
        # Seed demo contacts so "Alice" resolves out of the box
        c.execute("INSERT OR IGNORE INTO contacts VALUES ('alice', 'alice@example.com')")
        c.execute("INSERT OR IGNORE INTO contacts VALUES ('bob', 'bob@company.com')")
        conn.commit()
        conn.close()

    def log_message(self, session_id: str, role: str, content: str = "",
                    tool_calls=None, tool_call_id=None,
                    function_name=None, function_result=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversations 
            (session_id, timestamp, role, content, tool_calls, tool_call_id, function_name, function_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.datetime.utcnow().isoformat(),
            role,
            content,
            json.dumps(tool_calls) if tool_calls else None,
            tool_call_id,
            function_name,
            function_result,
        ))
        conn.commit()
        conn.close()

    def get_history(self, session_id: str):
        """Reconstruct exact OpenAI message dicts from DB."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = c.fetchall()
        conn.close()

        messages = []
        for r in rows:
            if r["role"] == "assistant" and r["tool_calls"]:
                messages.append({
                    "role": "assistant",
                    "content": r["content"] or None,
                    "tool_calls": json.loads(r["tool_calls"]),
                })
            elif r["role"] == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "name": r["function_name"],
                    "content": r["function_result"] or "",
                })
            else:
                messages.append({"role": r["role"], "content": r["content"] or ""})
        return messages

    def save_event(self, session_id, event_id, title, attendee_email,
                   start_iso, end_iso, google_event_id=None, status="active"):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO scheduled_events 
            (event_id, session_id, title, attendee_email, start_iso, end_iso, status, google_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, session_id, title, attendee_email,
              start_iso, end_iso, status, google_event_id))
        conn.commit()
        conn.close()

    def update_event_status(self, event_id, status):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE scheduled_events SET status = ? WHERE event_id = ?",
                  (status, event_id))
        conn.commit()
        conn.close()

    def get_event(self, identifier):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM scheduled_events WHERE event_id = ? OR title = ? LIMIT 1",
            (identifier, identifier),
        )
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None


# ============================================
# GOOGLE AUTH HELPERS
# ============================================
def get_google_services():
    if not _GOOGLE_OK:
        raise RuntimeError("Google libraries not installed.")
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    calendar = build("calendar", "v3", credentials=creds)
    gmail = build("gmail", "v1", credentials=creds)
    return calendar, gmail


# ============================================
# CALENDAR AGENT  (tools + memory + google + voice + draft preview)
# ============================================═
class CalendarAgent:
    def __init__(
        self,
        session_id: Optional[str] = None,
        mock_mode: bool = True,
        db_path: str = "agent_memory.db",
        calendar_service=None,
        gmail_service=None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.mock_mode = mock_mode
        self.db = AgentPersistence(db_path)

        self.calendar_service = calendar_service
        self.gmail_service = gmail_service
        if not mock_mode and calendar_service is None:
            self.calendar_service, self.gmail_service = get_google_services()

        # ── Session-scoped pending email drafts (multi-user safe) ──
        self.pending_emails = {}

        # System prompt
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        self.system_prompt = {
            "role": "system",
            "content": (
                f"You are an expert executive assistant. Today is {today}. "
                "You manage the user's calendar via tools.\n\n"
                "PROCESS:\n"
                "1. If user mentions a name (Alice, Bob…), call get_contact_email first.\n"
                "2. To schedule: call get_available_slots → pick best (prefer 1-4pm) → "
                "call create_calendar_event.\n"
                "3. To reschedule/cancel: call list_upcoming_events to identify the event, "
                "then call reschedule_event or cancel_event.\n"
                "4. For EMAILS: you MUST call preview_email first. This shows the draft to the user. "
                "Wait for their explicit confirmation (e.g., 'send it', 'yes', 'approved'). "
                "Only AFTER confirmation, call send_email. Never send without preview.\n"
                "5. Confirm final details to user clearly.\n\n"
                "Use 24h HH:MM format. Default meetings are 60 min unless asked otherwise."
            ),
        }

        # Tool schemas for OpenAI
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_available_slots",
                    "description": "Get free time slots on a specific day.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string", "description": "e.g. 'thursday', '2026-05-14'"},
                            "duration_minutes": {"type": "integer", "default": 60},
                        },
                        "required": ["day"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_contact_email",
                    "description": "Resolve a person's name to an email address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "First name, e.g. Alice"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Book a new calendar event.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM 24h"},
                            "duration_minutes": {"type": "integer", "default": 60},
                            "title": {"type": "string"},
                            "attendee_email": {"type": "string"},
                        },
                        "required": ["date", "time", "title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_upcoming_events",
                    "description": "List upcoming calendar events to identify one for reschedule/cancel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_results": {"type": "integer", "default": 5},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_event",
                    "description": "Move an existing event to a new time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_identifier": {"type": "string", "description": "Exact title or event_id"},
                            "new_date": {"type": "string", "description": "YYYY-MM-DD"},
                            "new_time": {"type": "string", "description": "HH:MM 24h"},
                            "duration_minutes": {"type": "integer", "default": 60},
                        },
                        "required": ["event_identifier", "new_date", "new_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_event",
                    "description": "Cancel an existing event by title or event_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_identifier": {"type": "string"},
                        },
                        "required": ["event_identifier"],
                    },
                },
            },
            # Email preview (human-in-the-loop)
            {
                "type": "function",
                "function": {
                    "name": "preview_email",
                    "description": "Create an email draft and show it to the user for approval. Do NOT send yet.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email notification (only after user approved a preview).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            },
        ]
    
    # ============================================
    # PUBLIC API
    # ============================================
    @traceable(run_type="agent", name="Calendar Agent Turn")
    def chat(self, user_input: str) -> dict:
        """
        Run the full agent loop with tool use.
        Returns: {"reply": str, "tool_log": str, "session_id": str, "pending_email": dict|None}
        """
        # Rebuild conversation from DB + system prompt
        self.messages = [self.system_prompt]
        self.messages.extend(self.db.get_history(self.session_id))
        self.messages.append({"role": "user", "content": user_input})
        self.db.log_message(self.session_id, "user", user_input)

        tool_log_lines = []

        for _ in range(10):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=0.2,
            )

            msg = response.choices[0].message
            msg_dict = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            self.messages.append(msg_dict)
            self.db.log_message(
                self.session_id,
                "assistant",
                content=msg_dict["content"],
                tool_calls=msg_dict.get("tool_calls"),
            )

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_log_lines.append(f"🔧 {fn_name}({json.dumps(args)})")

                result = self._dispatch_tool(fn_name, args)
                tool_log_lines.append(f"📦 → {str(result)[:300]}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": str(result),
                })
                self.db.log_message(
                    self.session_id,
                    "tool",
                    tool_call_id=tc.id,
                    function_name=fn_name,
                    function_result=str(result),
                )

        final_reply = self.messages[-1]["content"]
        self.db.log_message(self.session_id, "assistant", final_reply)
        return {
            "reply": final_reply,
            "tool_log": "\n".join(tool_log_lines),
            "session_id": self.session_id,
            "pending_email": self.pending_emails.get(self.session_id),
        }

    # ============================================
    # VOICE INPUT  (Whisper)
    # ============================================
    @traceable(run_type="tool", name="Whisper Transcribe")
    def transcribe(self, audio_path: str) -> str:
        """Convert microphone audio file to text via OpenAI Whisper."""
        if not audio_path:
            return ""
        try:
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            return transcript.text
        except Exception as e:
            return f"[Transcription error: {e}]"

    # ============================================
    # EMAIL DRAFT HELPERS
    # ============================================
    def clear_pending_email(self, session_id: Optional[str] = None):
        sid = session_id or self.session_id
        self.pending_emails.pop(sid, None)

    # ============================================
    # TOOL DISPATCHER
    # ============================================
    def _dispatch_tool(self, name: str, args: dict) -> str:
        try:
            if name == "get_available_slots":
                return self._get_available_slots(args.get("day"), args.get("duration_minutes", 60))
            if name == "get_contact_email":
                return self._get_contact_email(args.get("name"))
            if name == "create_calendar_event":
                return self._create_calendar_event(
                    date=args.get("date"),
                    time=args.get("time"),
                    duration_minutes=args.get("duration_minutes", 60),
                    title=args.get("title"),
                    attendee_email=args.get("attendee_email"),
                )
            if name == "list_upcoming_events":
                return self._list_upcoming_events(args.get("max_results", 5))
            if name == "cancel_event":
                return self._cancel_event(args.get("event_identifier"))
            if name == "reschedule_event":
                return self._reschedule_event(
                    event_identifier=args.get("event_identifier"),
                    new_date=args.get("new_date"),
                    new_time=args.get("new_time"),
                    duration_minutes=args.get("duration_minutes", 60),
                )
            # Preview + send
            if name == "preview_email":
                return self._preview_email(args.get("to"), args.get("subject"), args.get("body"))
            if name == "send_email":
                return self._send_email(args.get("to"), args.get("subject"), args.get("body"))
            return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ============================================
    # INDIVIDUAL TOOLS
    # ============================================
    @traceable(run_type="tool", name="Get Free Slots")
    def _get_available_slots(self, day_str: str, duration_minutes: int = 60) -> str:
        target = self._parse_day(day_str)
        start = target.replace(hour=9, minute=0, second=0, microsecond=0)
        end = target.replace(hour=17, minute=0, second=0, microsecond=0)

        if not self.mock_mode and self.calendar_service:
            body = {
                "timeMin": start.isoformat(),
                "timeMax": end.isoformat(),
                "items": [{"id": "primary"}],
            }
            res = self.calendar_service.freebusy().query(body=body).execute()
            busy = res["calendars"]["primary"]["busy"]
        else:
            busy = [
                {
                    "start": target.replace(hour=10, minute=0).isoformat(),
                    "end": target.replace(hour=11, minute=30).isoformat(),
                }
            ]

        free = self._find_free_slots(start, end, busy, duration_minutes)
        return json.dumps(
            {
                "date": target.strftime("%Y-%m-%d"),
                "weekday": target.strftime("%A"),
                "requested_duration_min": duration_minutes,
                "available_slots": free,
                "busy_slots": busy,
            }
        )

    def _get_contact_email(self, name: str) -> str:
        conn = sqlite3.connect(self.db.db_path)
        c = conn.cursor()
        c.execute("SELECT email FROM contacts WHERE LOWER(name) = LOWER(?)", (name,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.dumps({"name": name, "email": row[0], "found": True})
        return json.dumps(
            {"name": name, "email": None, "found": False, "note": "Ask user for email."}
        )

    @traceable(run_type="tool", name="Create Calendar Event")
    def _create_calendar_event(self, date, time, duration_minutes, title, attendee_email=None):
        start_dt = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)
        event_id = f"evt_{uuid.uuid4().hex[:8]}"

        body = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE},
        }
        if attendee_email:
            body["attendees"] = [{"email": attendee_email}]
            body["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 15},
                ],
            }

        google_id = None
        if not self.mock_mode and self.calendar_service:
            evt = (
                self.calendar_service.events()
                .insert(calendarId="primary", body=body, sendUpdates="all")
                .execute()
            )
            google_id = evt["id"]
            link = evt.get("htmlLink")
        else:
            link = "https://calendar.google.com/mock"

        self.db.save_event(
            self.session_id,
            event_id,
            title,
            attendee_email,
            start_dt.isoformat(),
            end_dt.isoformat(),
            google_event_id=google_id,
        )

        return json.dumps(
            {
                "success": True,
                "mock": self.mock_mode,
                "event_id": event_id,
                "google_event_id": google_id,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "link": link,
            }
        )

    def _list_upcoming_events(self, max_results=5) -> str:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        if not self.mock_mode and self.calendar_service:
            res = (
                self.calendar_service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            items = res.get("items", [])
            events = [
                {
                    "id": i["id"],
                    "summary": i.get("summary"),
                    "start": i["start"].get("dateTime", i["start"].get("date")),
                }
                for i in items
            ]
            return json.dumps({"events": events})

        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM scheduled_events WHERE status='active' ORDER BY start_iso LIMIT ?",
            (max_results,),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        if not rows:
            rows = [
                {
                    "event_id": "evt_demo1",
                    "title": "Sync with Alice",
                    "start_iso": (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat(),
                    "attendee_email": "alice@example.com",
                }
            ]
        return json.dumps({"events": rows, "mock": True})

    def _cancel_event(self, event_identifier: str) -> str:
        ev = self.db.get_event(event_identifier)
        if not ev:
            return json.dumps({"success": False, "error": "Event not found."})

        if not self.mock_mode and self.calendar_service and ev.get("google_event_id"):
            self.calendar_service.events().delete(
                calendarId="primary", eventId=ev["google_event_id"]
            ).execute()

        self.db.update_event_status(ev["event_id"], "cancelled")
        return json.dumps(
            {"success": True, "cancelled_event_id": ev["event_id"], "title": ev["title"]}
        )

    def _reschedule_event(self, event_identifier, new_date, new_time, duration_minutes=60):
        ev = self.db.get_event(event_identifier)
        if not ev:
            return json.dumps({"success": False, "error": "Event not found."})

        if not self.mock_mode and self.calendar_service and ev.get("google_event_id"):
            self.calendar_service.events().delete(
                calendarId="primary", eventId=ev["google_event_id"]
            ).execute()
        self.db.update_event_status(ev["event_id"], "rescheduled")

        return self._create_calendar_event(
            date=new_date,
            time=new_time,
            duration_minutes=duration_minutes,
            title=ev["title"],
            attendee_email=ev.get("attendee_email"),
        )

    # ============================================
    # Preview email (stores draft, waits for user)
    # ============================================
    @traceable(run_type="tool", name="Preview Email Draft")
    def _preview_email(self, to: str, subject: str, body: str) -> str:
        self.pending_emails[self.session_id] = {
            "to": to,
            "subject": subject,
            "body": body,
        }
        return json.dumps({
            "status": "draft_created",
            "message": "Draft is ready for user review. Stop here and wait for explicit confirmation before calling send_email.",
            "draft": {"to": to, "subject": subject, "body": body},
        })

    # ============================================
    # Send email (clears pending draft on success)
    # ============================================
    @traceable(run_type="tool", name="Send Email")
    def _send_email(self, to, subject, body_text) -> str:
        # Consume pending draft so it doesn't show in UI anymore
        self.pending_emails.pop(self.session_id, None)

        if not self.mock_mode and self.gmail_service:
            from email.mime.text import MIMEText
            import base64

            msg = MIMEText(body_text)
            msg["to"] = to
            msg["subject"] = subject
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            self.gmail_service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return json.dumps({"success": True, "sent_to": to})

        print(f"\n[MOCK EMAIL to {to}]\nSubject: {subject}\n{body_text}\n")
        return json.dumps({"success": True, "mock": True, "sent_to": to})

    # ============================================
    # HELPERS
    # ============================================
    def _parse_day(self, s: str) -> datetime.datetime:
        s = s.lower().strip()
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if s in ("today",):
            return today
        if s in ("tomorrow",):
            return today + datetime.timedelta(days=1)
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if s in weekdays:
            target = weekdays.index(s)
            delta = target - today.weekday()
            if delta <= 0:
                delta += 7
            return today + datetime.timedelta(days=target - today.weekday() if delta == 0 else delta)
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            return today + datetime.timedelta(days=3)

    def _find_free_slots(self, start, end, busy, duration):
        slots = []
        current = start
        dur = datetime.timedelta(minutes=duration)
        busy_sorted = sorted(busy, key=lambda x: x["start"])

        for b in busy_sorted:
            bs = datetime.datetime.fromisoformat(b["start"].replace("Z", "+00:00")).replace(tzinfo=None)
            be = datetime.datetime.fromisoformat(b["end"].replace("Z", "+00:00")).replace(tzinfo=None)
            if current + dur <= bs:
                slots.append(f"{current.strftime('%H:%M')} – {bs.strftime('%H:%M')}")
            current = max(current, be)
        if current + dur <= end:
            slots.append(f"{current.strftime('%H:%M')} – {end.strftime('%H:%M')}")
        return slots
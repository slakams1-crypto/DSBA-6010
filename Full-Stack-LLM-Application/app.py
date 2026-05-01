# Add Required Components
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)
import asyncio
import sys
import signal
from functools import partial
import os, json
import re
from typing import Any, Dict
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM,
    AutoTokenizer, pipeline
)
import openai
import requests
from PIL import Image
import httpx # Added for ToolRetryMiddleware
from io import BytesIO
import gc
import gradio as gr
from huggingface_hub import login
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
from ddgs import DDGS
import time
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, AgentState, AgentMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware
from langchain.agents.middleware import (
    TodoListMiddleware,      # Task planning
    SummarizationMiddleware, # Compress long convos
    HumanInTheLoopMiddleware
)
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from faster_whisper import WhisperModel
import whisper
sys.path.insert(0, './MedIntel')
from safety.safety import check_moderation, check_moderation_flag, execute_chat_with_input_moderation, execute_all_moderations, check_image_moderation, get_chat_response_guardrails_async, get_chat_response_openai_async
from metrics import app as metrics_app
import threading
import uvicorn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import functools


# ============================================
# # CSS for Gradio UI elements
# ============================================
custom_css = """
body { font-size: 18px !important; }
.gr-box { font-size: 16px !important; }
"""

# ============================================
# # In-memory metrics needed for Metrics Dashboard
# ============================================
metrics = {
    "timestamps": [],  # to track request times
    "latencies": [],   # to track request latency
    "requests": [],    # simple request count
    "tool_calls": [],  # tool calls made
    "errors": []       # errors   
}

# ============================================
# Display Metrics Dashboard
# ============================================
def show_dashboard():

    # --- Pad missing lists if needed ---
    n = len(metrics["timestamps"])
    for key in ["latencies", "tool_calls", "errors", "requests"]:
        while len(metrics[key]) < n:
            metrics[key].append(0)    
            
    df = pd.DataFrame({
        "timestamps": metrics["timestamps"],
        "latencies": metrics["latencies"],
        "tool_calls": metrics["tool_calls"],
        "errors": metrics["errors"]
    })

    # System metrics
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent    

    # --- Summary with animated badges ---
    if df.empty:
        summary_html = "<p style='color: gray;'>No requests yet.</p>"
    else:
        total_requests = metrics["requests"]
        avg_latency = df["latencies"].mean()
        last_latency = df["latencies"].iloc[-1]
        total_tool_calls = sum(metrics["tool_calls"])
        total_errors = sum(metrics["errors"])

        summary_html = f"""
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        .badge {{
            padding: 0.5em 1em;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            display: inline-block;
            animation: pulse 0.6s ease-in-out;
        }}
        .green {{ background-color: #4CAF50; }}
        .blue {{ background-color: #2196F3; }}
        .orange {{ background-color: #FF5722; }}
        .purple {{ background-color: #9C27B0; }}
        .red {{ background-color: #F44336; }}
        .gray {{ background-color: #607D8B; }}
        .badge-container {{ display: flex; gap: 1em; flex-wrap: wrap; margin-bottom: 1em; }}
        </style>
        <div class="badge-container">
            <div class="badge green">📝 Requests: {total_requests}</div>
            <div class="badge blue">⏱️ Avg Latency: {avg_latency:.3f}s</div>
            <div class="badge orange">⚡ Last Request: {last_latency:.3f}s</div>
            <div class="badge purple">🛠 Tool Calls: {total_tool_calls}</div>
            <div class="badge red">❌ Errors: {total_errors}</div>
            <div class="badge gray">💻 CPU: {cpu_percent}%</div>
            <div class="badge gray">🧠 Memory: {memory_percent}%</div>
        </div>
        """

    # --- Latency plot ---
    if df.empty:
        latency_fig = go.Figure()
        latency_fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        latency_fig.update_layout(
            title="Request Latency Over Time",
            xaxis_title="Time",
            yaxis_title="Latency (s)"
        )
    else:
        df["latency_rolling_avg"] = df["latencies"].rolling(window=5, min_periods=1).mean()
        latency_fig = go.Figure()
        latency_fig.add_trace(go.Scatter(
            x=df["timestamps"],
            y=df["latencies"],
            mode="lines+markers",
            name="Raw Latency",
            line=dict(color="blue"),
            hovertemplate="Time: %{x}<br>Raw Latency: %{y:.3f}s<extra></extra>"
        ))
        latency_fig.add_trace(go.Scatter(
            x=df["timestamps"],
            y=df["latency_rolling_avg"],
            mode="lines",
            name="Rolling Avg (5 requests)",
            line=dict(color="red", dash="dash"),
            hovertemplate="Time: %{x}<br>Rolling Avg: %{y:.3f}s<extra></extra>"
        ))
        latency_fig.update_layout(
            title="Request Latency Over Time",
            xaxis_title="Time",
            yaxis_title="Latency (s)",
            hovermode="x unified"
        )

    # --- Request count plot ---
    if df.empty:
        count_fig = go.Figure()
        count_fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        count_fig.update_layout(
            title="Total Requests Over Time",
            xaxis_title="Time",
            yaxis_title="Requests"
        )
    else:
        df["cumulative_requests"] = range(1, len(df)+1)
        count_fig = go.Figure()
        count_fig.add_trace(go.Scatter(
            x=df["timestamps"],
            y=df["cumulative_requests"],
            mode="lines+markers",
            name="Total Requests",
            line=dict(color="green"),
            hovertemplate="Time: %{x}<br>Requests: %{y}<extra></extra>"
        ))
        count_fig.update_layout(
            title="Total Requests Over Time",
            xaxis_title="Time",
            yaxis_title="Requests"
        )

    # --- Tool calls plot ---
    if df.empty:
        tool_calls_fig = go.Figure()
        tool_calls_fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        tool_calls_fig.update_layout(
            title="Tool Calls Over Time",
            xaxis_title="Time",
            yaxis_title="Tool Calls"
        )
    else:
        df["cumulative_tool_calls"] = df["tool_calls"].cumsum()
        tool_calls_fig = go.Figure()
        tool_calls_fig.add_trace(go.Scatter(
            x=df["timestamps"],
            y=df["cumulative_tool_calls"],
            mode="lines+markers",
            name="Tool Calls",
            line=dict(color="purple")
        ))
        tool_calls_fig.update_layout(
            title="Tool Calls Over Time",
            xaxis_title="Time",
            yaxis_title="Tool Calls"
        )

    # --- Errors plot ---
    if df.empty:
        errors_fig = go.Figure()
        errors_fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        errors_fig.update_layout(
            title="Errors Over Time",
            xaxis_title="Time",
            yaxis_title="Errors"
        )
    else:
        df["cumulative_errors"] = df["errors"].cumsum()
        errors_fig = go.Figure()
        errors_fig.add_trace(go.Scatter(
            x=df["timestamps"],
            y=df["cumulative_errors"],
            mode="lines+markers",
            name="Errors",
            line=dict(color="red")
        ))
        errors_fig.update_layout(
            title="Errors Over Time",
            xaxis_title="Time",
            yaxis_title="Errors"
        )

    return summary_html, latency_fig, count_fig, tool_calls_fig, errors_fig


# ============================================
# Fix for the asyncio cleanup error
# ============================================
if sys.platform == "linux":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    

# ============================================
# Gradio - Create a custom theme with medical blue
# ============================================
medical_theme = gr.themes.Soft(
    primary_hue="blue",  # This controls the primary button color
    secondary_hue="gray",
).set(
    button_primary_background_fill="#0066cc",
    button_primary_background_fill_dark="#004d99",
    button_primary_text_color="white",
    button_primary_border_color="#0066cc",
    button_primary_border_color_dark="#004d99",
)


# ============================================
# Model configurations
# ============================================
QAmodel = {
    "gpt-nano": "gpt-4.1-nano",
    "gpt-mini": "gpt-4o-mini", 
    "gpt-turbo": "gpt-3.5-turbo",
    "llava-1.5": "llava-hf/llava-1.5-7b-hf",
    "llava-1.5.1": "llava-hf/llava-2.0-13b-hf",
    "blip": "Salesforce/blip2-opt-2.7b",
    "embed-model": "sentence-transformers/all-MiniLM-L6-v2"
}


# ============================================
# Get API Keys & Tokens
# ============================================
# Get huggingface api token from environment variable (from Space secret)
hf_token = os.getenv('HUGGINGFACE_API_KEY')

if hf_token:
    # Authenticate silently without prompting
    login(token=hf_token, add_to_git_credential=True)
    print("Successfully authenticated with Hugging Face Hub")
else:
    print("Warning: HF_TOKEN not found. Some features may be limited.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print("Successfully loaded environment variables")


# ============================================
# System prompt
# ============================================
system_prompt=(
    "You are a helpful Medical Q&A assistant. "
    "For any medical question, you MUST use the 'retrieve_context' tool call first to search the knowledge base. "
    "If the knowledge base does not provide a sufficient answer, then you MUST use 'youtube_links_tool' tool calls. "
    "For medical question, include youtube video links with response. Keep the answer short and concise. "
    "Use three sentences maximum from all tools. "
    "If question contains keyword 'BMI', you MUST only use the 'calculate_bmi' tool call. "
)


# ============================================
# Classifiers
# ============================================
classifier_model = ChatOpenAI(model=QAmodel['gpt-mini'], api_key=OPENAI_API_KEY, temperature=0)
classifier_topic_model = ChatOpenAI(model=QAmodel['gpt-nano'], api_key=OPENAI_API_KEY, temperature=0)


# Introduce global flags to track state across agent invocations
global _kb_had_results
_kb_had_results = False
global is_medical_query
is_medical_query = False # Initialize globally


# =================================================================
# PRE-LOAD AT MODULE LEVEL (before UI)
# =================================================================
# ============================================
# Load existing vector store/embedding model
# ============================================
print("🔄 Loading embedding model...")
# This will be pre-downloaded by preload_from_hub
embedding_model = HuggingFaceEmbeddings(
    model_name=QAmodel["embed-model"],
    cache_folder="/tmp/.cache"  # Use tmp for Spaces
)

print("🔄 Loading Chroma vector store...")
persist_directory = './chroma_db_hf_rerun2'

# Verify the directory exists
if not os.path.exists(persist_directory):
    raise FileNotFoundError(
        f"Vector store not found at {persist_directory}. "
        f"Available: {os.listdir('.')}"
    )

# Load the vector store
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

doc_count = vectordb._collection.count()
print(f"✓ Loaded vector store with {doc_count} documents using HuggingFace embeddings")


# ============================================
# Load faster whisper
# ============================================
#print("🔄 Loading Whisper model...")
# Use CPU or CUDA if available
#whisper_model = WhisperModel("turbo", device="cpu", compute_type="int8")
#print("✓ Whisper model loaded

print("🔄 Loading OpenAI Whisper model...")
whisper_model = whisper.load_model("turbo")
print("✓ OpenAI Whisper model loaded")


# ============================================
# Load BLIP2 Processor/Model for Image Processiing
# ============================================
model_id = QAmodel['blip']
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading BLIP2 model...")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to(device)
print("BLIP2 model loaded")
print("Loading BLIP2 processor...")
blip2_processor = Blip2Processor.from_pretrained(model_id)
print(f"DEBUG: {blip2_processor}")
print("✓ BLIP2 processor loaded")


# Global variables
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# =======================================
# Wrap a tool to automatically track metrics
# =======================================
def track_tool_calls(func):
    @functools.wraps(func)  # preserves name and docstring
    def wrapper(*args, **kwargs):
        start_time = time.time()

        tool_calls_this_request = 0
        errors_this_request = 0

        try:
            result = func(*args, **kwargs)
            tool_calls_this_request = 1
        except Exception as e:
            errors_this_request = 1
            result = f"Error executing {func.__name__}: {e}"

        # --- Append metrics consistently ---
        metrics["timestamps"].append(time.strftime("%H:%M:%S"))
        metrics["latencies"].append(time.time() - start_time)
        metrics["requests"].append(len(metrics["timestamps"]))
        metrics["tool_calls"].append(tool_calls_this_request)
        metrics["errors"].append(errors_this_request)

        return result
    return wrapper


# =======================================
# Global Tool Definitions
# =======================================
@tool
@track_tool_calls
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculates Body Mass Index (BMI) given weight in kilograms and height in meters.
    Use this tool when the user asks for BMI and provides both weight and height.
    Example: calculate_bmi(weight_kg=70, height_m=1.75)
    """
    if height_m <= 0 or weight_kg <= 0:
        return "Error: Height and weight must be positive values."
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"
    return f"Your BMI is {bmi:.2f}, which falls into the '{category}' category."

@tool
@track_tool_calls
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from knowledge base for a question."""
    global _kb_had_results # Declare intent to modify global variable
    try:
        # Ensure vectordb is accessible (it's loaded globally)
        docs = vectordb.max_marginal_relevance_search(query, k=1, fetch_k=1)
        if not docs:
            _kb_had_results = False
            return "No relevant information found in knowledge base."
        else:
            _kb_had_results = True
            print(f"Retrieved {len(docs)} document(s)")
            return "\n\n".join([f"Source {i+1}:\n{doc.page_content[:500]}..." for i, doc in enumerate(docs)])
    except Exception as e:
        _kb_had_results = False
        print(f"Error in retrieve_context tool: {e}")
        return f"Error retrieving context from knowledge base: {e}"

@tool
@track_tool_calls
def youtube_links_tool(query: str) -> str:
    """Search for video tutorials."""
    try:
        # Use DDG for video search (more reliable in Spaces)
        ddgs = DDGS()
        
        # Search specifically for YouTube videos
        video_query = f"{query} site:youtube.com"
        results = ddgs.text(video_query, max_results=3)
        
        links = []
        for r in results:
            if 'youtube.com/watch' in r['href']:
                links.append(f"[{r['title'][:50]}...]({r['href']})")
        
        return "\n".join(links) if links else "No videos found"
        
    except Exception as e:
        # Fallback: return search suggestions
        return f"Video search temporarily unavailable. Try searching: https://youtube.com/results?search_query={query.replace(' ', '+')}"
        
@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]
    
# =======================================
# Available Tools
# =======================================
tools = [retrieve_context, youtube_links_tool, calculate_bmi]

# ===================================================================================
# Global Middleware Definitions
# ===================================================================================
@before_model(can_jump_to=["end"])
def medical_classifier(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """LLM decides if query is medical and applies guardrails."""
    #global is_medical_query # Declare intent to modify global variable
    global is_medical_query
    global _kb_had_results
    #_kb_had_results = False
    #is_medical_query = False
    if not state["messages"]:
        return None
    # CLEAN THE CONTENT HERE
    raw_content = state["messages"][0].content
    user_query = clean_text(raw_content)    
    
    try:
        
        #query_for_classification = guardrail_output
        class_prompt = f"Is this medical/health/symptoms/healthcare/urgent care related? 'YES' or 'NO' only.\nQuery: '{user_query}'\nYES: symptoms/treatments. NO: other."
        class_result = classifier_model.invoke([{"role": "user", "content": class_prompt}])
        print(f"[Medical Classifier] User Query: '{user_query}' -> Classifier Result: '{class_result.content}'") # Debug print

        if "NO" in class_result.content.upper():
            is_medical_query = False # Set to False if not medical
            return {"messages": [AIMessage("I apologize, I am programmed to answer medical questions only.")], "jump_to": "end"}
        else:
            is_medical_query = True # Set to True if medical
    except Exception as e:
        print(f"Error in medical_classifier (guardrail or classification): {e}")
        is_medical_query = False # or True, depending on desired fallback behavior
        return None # Proceed to main agent
    return None # Proceed to main agent

# =======================================
# Disclaimer
# =======================================
class DisclaimerMiddleware(AgentMiddleware):
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        global _kb_had_results, is_medical_query # Access global flags
        
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            
            # Skip disclaimer if user prompt contains "BMI"
            user_prompt = ""
            for msg in reversed(state["messages"][:-1]):  # Exclude the last AIMessage
                if isinstance(msg, HumanMessage):
                    user_prompt = msg.content
                    break
            
            if "BMI" in user_prompt.upper():
                _kb_had_results = False
                return state
            
            # Existing disclaimer logic
            if is_medical_query and (not _kb_had_results or "web_search" in state["messages"][-1].content.lower()):
                content = state["messages"][-1].content
                state["messages"][-1].content = f"According to medical websites, including MedlinePlus, Cleveland Clinic and Mayo Clinic, {content}"

        _kb_had_results = False
        return state

synthesizer = SummarizationMiddleware(
    model=classifier_model,
    trigger=("tokens", 4000),
    max_tokens=1000,
    system_prompt="Synthesize key points, action items, and recent context.",
)

toolselector = LLMToolSelectorMiddleware(model=QAmodel.get('gpt-mini'),system_prompt="",max_tools=5)
toolretry = ToolRetryMiddleware(max_retries=3,backoff_factor=2.0,retry_on=[TimeoutError,httpx.NetworkError])

# =======================================
# LangChain - Global Agent Setup
# =======================================
agent = create_agent(
    classifier_model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        medical_classifier,
        #TodoListMiddleware(),
        synthesizer,
        #toolselector,
        #toolretry #,
        DisclaimerMiddleware()
    ],
)    


# ===========================================
# Misc. Function to extract plain text from Gradio's wrapped format
# ===========================================
def clean_text(value):
    """Extract plain text from Gradio's wrapped format.
    Handles: 
    - String values
    - Lists containing dicts with 'text' key: [{'text': '...', 'type': 'text'}]
    - Direct dicts with 'text' or 'content' keys
    """
    if value is None:
        return ""
    
    # Handle list of dicts (Gradio's new format)
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict):
            # Extract text from each dict and join
            texts = [item.get('text', '') for item in value if isinstance(item, dict)]
            return ' '.join(filter(None, texts))
        # If list of strings
        return ' '.join(str(item) for item in value if item is not None)
    
    # Handle direct dict
    elif isinstance(value, dict):
        return value.get('text', value.get('content', str(value)))
    
    # Already a string
    return str(value)  


# ===========================================
# Misc. Function to deep clean chat history
# ===========================================
def clean_chat_history(history):
    """Ensure all messages are plain dicts with string content."""
    clean = []
    for msg in history:
        if isinstance(msg, dict):
            # Clean the content field
            raw_content = msg.get("content", "")
            cleaned_content = clean_text(raw_content)
            
            clean.append({
                "role": str(msg.get("role", "user")),
                "content": cleaned_content
            })
    return clean


# ===========================================
# Misc. Function to extract plain text from Gradio's Textbox format
# ===========================================
def extract_gradio_text(value):
    """Extract plain text from Gradio's Textbox format."""
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict) and 'text' in value[0]:
            return value[0]['text']
    return str(value) if value else ""    
  

# ===========================================
# GENERATE-CHAT FUNCTION (Invokes the LangChain agent)
# ===========================================
def generate_chat(user_input, chat_history):
    """Generate response using GPT model"""
    
    # Prepare messages for the agent (chat_history is now expected to be list of dictionaries)
    agent_messages = list(chat_history) # Create a mutable copy of the existing chat history
    agent_messages.append({"role": "user", "content": user_input}) # Add the current user's message
    
    # Ensure user_input is plain string
    user_input = extract_gradio_text(user_input)
    
    if check_moderation_flag(user_input):
        agent_messages.append({"role": "assistant", "content": "I apologize, I am programmed to answer medical questions only."})
        return agent_messages
    
    try:
        # Invoke the agent
        response = agent.invoke({"messages": agent_messages})     
    
        # Extract retrieved_answer from tool messages (using 'response')
        retrieved_answer = None
        for message in response['messages']:
            if isinstance(message, ToolMessage) and message.name == 'retrieve_context':
                content_lines = message.content.split('\n')
                for line in content_lines:
                    if line.strip().startswith('answer:'):
                        retrieved_answer = line.strip().replace('answer: ', '') # Extract just the answer text
                        break # Found the answer, break from inner loop
                if retrieved_answer:
                    break # Found the answer, break from outer loop
        
        if retrieved_answer:
            print(f"Direct answer from retrieve_context tool: '{retrieved_answer}'")
        else:
            # Only print final AI message if no direct tool answer was extracted for console clarity
            pass # No need to print here, final_ai_message below will capture the response.
        
        # Extract the final AI message content
        final_ai_message = "I cannot find the best answer. Please consult with a Doctor." # Default fallback
        for msg in reversed(response['messages']):
            if isinstance(msg, AIMessage):
                final_ai_message = msg.content
                break

        updated_chat_history = []
        # Copy existing history
        for msg in chat_history:
            if isinstance(msg, dict):
                updated_chat_history.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))
                })
                
        # Append the new user input and agent response to the chat history
        # The output should also be a list of dictionaries
        # updated_chat_history = list(chat_history)
        updated_chat_history.append({"role": "user", "content": user_input})
        updated_chat_history.append({"role": "assistant", "content": final_ai_message})

        return clean_chat_history(updated_chat_history)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        #updated_chat_history = list(chat_history) if chat_history else []
        
        updated_chat_history = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict):
                    updated_chat_history.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })    
                    
        updated_chat_history.append({"role": "user", "content": user_input})
        updated_chat_history.append({"role": "assistant", "content": error_msg})
        return clean_chat_history(updated_chat_history)
        

# ===========================================
# Function to transform OCR to text
# ===========================================
def image_to_text_processor(image_path, processor, model, device, dtype, vtmodel_name):
    """Process image and generate description"""
    
    if not os.path.exists(image_path):
        return "Error: Image file not found. Please upload an image first."
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Prepare the prompt
        prompt_text = "Question: What do you see in the image? Answer:"
        
        # Process the image and prompt using the passed processor
        inputs = processor(images=image, text=prompt_text, return_tensors='pt').to(device, dtype)

        # Generate output using the passed model object
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # Decode the output using the passed processor object
        if vtmodel_name == QAmodel['blip']:
            outputs = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            # BLIP's raw output usually contains the prompt as well, extract just the answer
            if "Answer:" in outputs:
                final_output = outputs.split("Answer:", 1)[1].strip()
            else:
                final_output = outputs.strip() # Fallback 
        
            return final_output
            
    except Exception as e:
        return f"Error processing image: {str(e)}"


# ===========================================
# Function to transcribe audio to text
# ===========================================
def speech_to_text(audio_input, chat_history):
    """Convert speech to text using Whisper"""
    
    if audio_input is None:
        return chat_history, "⚠️ No audio recorded"    
    
    try:
        # Ensure the audio file exists and is accessible
        result = whisper_model.transcribe(audio_input)
        
        # Handle different whisper formats
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
            # Faster-whisper: generator of segments - extract all text
            transcribed_text = " ".join([segment.text for segment in result])
            
        elif isinstance(result, dict):
            # OpenAI whisper: dict with "text" key
            #transcribed_text = result.get("text", "")
            # This returns dict with "text" key - much simpler!
            transcribed_text = result["text"]  # Always works!            
            
        elif isinstance(result, tuple):
            # Tuple format
            transcribed_text = result[0]
            
        else:
            # Fallback
            transcribed_text = str(result)
        
        transcribed_text = clean_text(transcribed_text).strip()      

        # Process the transcribed text through the chat logic
        if not transcribed_text:
            return "", chat_history # Return original chat history if transcription is empty

        # Clean chat_history before processing
        if chat_history is None:
            chat_history = []
        clean_history = []
        for msg in chat_history:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = extract_gradio_text(content)
                clean_history.append({
                    "role": msg.get("role", "user"),
                    "content": str(content)
                })

        # Call the main chat generation function with the cleaned history
        new_chat_history = generate_chat(transcribed_text, clean_history)

        # Return an empty string for the text input and the updated chat history
        return "", new_chat_history

    except Exception as e:
        print(f"Error during audio transcription: {e}")
        # Return an error message to the user, appending it to chat_history
        error_message = f"Error transcribing audio: {e}"
        if chat_history is None:
            chat_history = []
        
        # Clean history before appending error
        clean_history = []
        for msg in chat_history:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = extract_gradio_text(content)
                clean_history.append({
                    "role": msg.get("role", "user"),
                    "content": str(content)
                })
        
        clean_history.append({"role": "user", "content": "(Audio input failed)"})
        clean_history.append({"role": "assistant", "content": error_message})
        return "", clean_chat_history(clean_history)
        

# ===========================================
# Create the Gradio app - UI
# ===========================================
with gr.Blocks(theme=medical_theme) as demo:
    with gr.Tab("MedIntel App"):
                
        gr.Markdown(
            """
            <h1 style='text-align: center; margin-bottom: 1em;'>
                🏥 MedIntel Q&A Assistant
            </h1>
            <p style='text-align: center; font-size: 1.1em; color: #555;'>
                Ask medical questions and get answers with knowledge base retrieval, LLM generation, and YouTube references.
            </p>
            """
        )
    
        # State variables
        vtmodel_state = gr.State(value="blip")
        model_choice = QAmodel['gpt-mini']    
        vtmodel_id = QAmodel['blip']   
    
        with gr.Row():
            
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation History", 
                    height=400,
                    allow_tags=False
                )
                
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Type your medical question here...", 
                    lines=2
                )
    
                with gr.Row():
                    # working
                    upload_file = gr.Image(type="filepath", label="Upload Image")             
                    image_to_text_btn = gr.Button("🖼️ Analyze Image", variant="secondary")
                    audio_input = gr.Audio(sources=['microphone'], type="filepath", label="🎤 Record")
                    transcribe_button = gr.Button("🎤 Transcribe Audio & Submit", variant="secondary")
    
                with gr.Row():
                    submit_btn = gr.Button("💬 Submit Question", variant="primary", scale=1)
                    clear_btn = gr.ClearButton(value="🗑️ Clear Chat", scale=0)

    # ============================================
    # # Metrics Dashboard Tab
    # ============================================
    with gr.Tab("Metrics Dashboard"):
        summary_md = gr.Markdown()        
        latency_plot = gr.Plot()
        request_count_plot = gr.Plot()       
        tool_calls_plot = gr.Plot()
        errors_plot = gr.Plot()        
        refresh_btn = gr.Button("Refresh Dashboard")
        
        # Connect the callback **after defining the components**
        refresh_btn.click(show_dashboard, outputs=[summary_md, latency_plot, request_count_plot, tool_calls_plot, errors_plot])

    # ===========================================
    # Function that enables LLM to respond to user prompts/actions
    # ===========================================        
    def respond(user_input, chat_history):
        print(f"DEBUG user_input type: {type(user_input)}, value: {repr(user_input)}")
        try:
            # Extract plain text from Gradio format
            user_input = extract_gradio_text(user_input)  
            
            if not user_input:
                return "", chat_history
    
            if chat_history is None:
                chat_history = []
    
            # Clean chat_history to fix corrupted entries
            clean_history = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = extract_gradio_text(content)
                    clean_history.append({
                        "role": msg.get("role", "user"),
                        "content": str(content)
                    })
    
            import time
            new_chat_history = generate_chat(user_input, clean_history)  # Use clean_history         
            return "", new_chat_history
            
        except Exception as e:
            error_message = f"Error: {e}"
            print(f"Error in respond: {e}")
            
            if chat_history is None:
                chat_history = []
            
            # Clean user_input before appending
            clean_input = extract_gradio_text(user_input) if isinstance(user_input, (list, dict)) else str(user_input)
            
            updated_chat_history = list(chat_history)
            updated_chat_history.append({"role": "user", "content": clean_input})
            updated_chat_history.append({"role": "assistant", "content": error_message})
            return "", updated_chat_history

    # ===========================================
    # Function to process images
    # ===========================================    
    def process_image_for_chat(image_file_path_str, chat_history):
        print(f"DEBUG: Received image_file_path_str from Gradio: {image_file_path_str}")

        # Initialize these FIRST, before any conditions
        if chat_history is None:
            chat_history = []

        # Create a clean copy
        updated_chat_history = list(chat_history)       

        # Validation check - DON'T append to history
        if image_file_path_str is None or not isinstance(image_file_path_str, str) or not os.path.isfile(image_file_path_str):
            error_message = "❌ Error: Invalid image file"
            print(error_message)
            # Create temporary display - error NOT saved to history
            temp_display = list(chat_history) + [{"role": "assistant", "content": error_message}]
            return temp_display, None
    
        # Moderation check - DON'T append to history  
        # if not check_image_moderation(image_file_path_str):
        #     error_message = "⚠️ Image is not safe. Please upload an appropriate image for scanning first."
        #     print(error_message)
            # Create temporary display - error NOT saved to history
        #    temp_display = list(chat_history) + [{"role": "assistant", "content": error_message}]
        #    return temp_display, None

        user_message_content = "What do you see in this image?"
        updated_chat_history = list(chat_history)
        updated_chat_history.append({"role": "user", "content": user_message_content})
        
        try:
            if blip2_processor is None or blip2_model is None:
                error_message = f"Failed to load model: {vtmodel_id}"
                updated_chat_history.append({"role": "assistant", "content": error_message})
                return updated_chat_history, None

            image_description = image_to_text_processor(
                image_file_path_str, 
                blip2_processor, 
                blip2_model, 
                _device, 
                _dtype, 
                vtmodel_id
            )
            
            updated_chat_history.append({"role": "assistant", "content": image_description})
            
        except Exception as e:
            error_message = f"Error processing image: {e}"
            print(f"Error: {e}")
            
            updated_chat_history.append({"role": "assistant", "content": error_message})

        return updated_chat_history, None

    # ===========================================
    # # Event Calls
    # ===========================================
    transcribe_button.click(
        speech_to_text,
        inputs=[audio_input, chatbot],
        outputs=[txt_input, chatbot]
    )

    submit_btn.click(respond, [txt_input, chatbot], [txt_input, chatbot])
    txt_input.submit(respond, [txt_input, chatbot], [txt_input, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)
    
    image_to_text_btn.click(
        process_image_for_chat,
        inputs=[upload_file, chatbot],
        outputs=[chatbot, upload_file]
    )

# ===========================================
# Function that helps shutting down gracefully in case if there are any system issues
# ===========================================
def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    demo.close()  # Close Gradio properly
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)    

if __name__ == "__main__":
    demo.launch(
        #server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,  # Disable SSR to avoid the experimental warning
        css=custom_css,
        share=False
    )
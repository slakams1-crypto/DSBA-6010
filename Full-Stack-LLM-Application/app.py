# Add Required Components
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)
import asyncio, sys, signal, os, gc
import threading, base64, uuid, unicodedata, subprocess, psutil
from io import BytesIO
import requests
import time, datetime
import json
import re
from typing import Any, Dict, Optional, List, Tuple
import functools
from functools import partial
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM,
    AutoTokenizer, pipeline
)
import openai
from PIL import Image
import gradio as gr
import httpx # Added for ToolRetryMiddleware
from huggingface_hub import login
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
from ddgs import DDGS
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, AgentState, AgentMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware
from langchain.agents.middleware import (
    TodoListMiddleware,      # Task planning
    SummarizationMiddleware, # Compress long convos
    HumanInTheLoopMiddleware
)
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langsmith import traceable, Client, get_current_run_tree
from faster_whisper import WhisperModel
import whisper
from calendar_agent import CalendarAgent
import xml.etree.ElementTree as ET
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from safety.safety import check_moderation, check_moderation_flag, execute_chat_with_input_moderation, execute_all_moderations, check_image_moderation, get_chat_response_guardrails_async, get_chat_response_openai_async
from metrics import app as metrics_app
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, './MedIntel')
import html
ls_client = Client()


# ============================================
# LangSmith Tracing Set to False on StartUp
# ============================================
os.environ.setdefault("LANGSMITH_TRACING_V2", "false")


# ============================================
# CSS for Gradio UI elements
# ============================================
custom_css = """
body { font-size: 18px !important; }
.gr-box { font-size: 16px !important; }
.file-btn-pair button { min-height: 44px !important; }
.file-btn-pair .file-preview { min-height: 44px !important; display: flex; align-items: center; }
/* Target the specific block wrapper Gradio creates */
.block.prompt-box {
    --body-text-color-subdued: #374151 !important;
}
.block.prompt-box textarea::placeholder,
.block.prompt-box input::placeholder {
    color: #6b7280 !important; 
    font-weight: 600 !important;
    opacity: 1 !important;       /* browsers default to ~0.5 */
}
/* Also darken what the user types so it matches */
.block.prompt-box textarea,
.block.prompt-box input {
    color: #374151 !important; 
    font-weight: 500 !important;
}
/* Rate this response */
.main-col {
    gap: 8px !important;
}
"""


# Live scalar: how many tool calls are in-flight *right now*
_current_pending = 0


# =============================================
# In-memory metrics needed for Metrics Dashboard
# =============================================
metrics = {
    "timestamps": [],  # to track request times
    "latencies": [],   # to track request latency
    "requests": [],    # simple request count
    "tool_calls": [],  # tool calls made
    "errors": [],      # errors
    "pending": []      # pending requests
}


# ============================================
# Fix for the asyncio cleanup error
# ============================================
if sys.platform == "linux":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


device = "cuda" if torch.cuda.is_available() else "cpu"    
    

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
    "blip6.7": "Salesforce/blip2-opt-6.7b",
    "whisper": "turbo",
    "embed-model": "sentence-transformers/all-MiniLM-L6-v2",
    "omni-latest": "omni-moderation-latest"
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

NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # PubMed Search, Optional, increases rate limit to 10/sec
print("Successfully loaded environment variables")


# ============================================
# System prompt
# ============================================
SYSTEM_PROMPT = (
    "You are a helpful Medical Q&A assistant. "
    "For EVERY medical question, you MUST first use the 'retrieve_context' tool. "
    "After receiving the retrieved context, if the question asks for advice, treatment, "
    "or personal recommendations (e.g. 'what should I do', 'how can I', 'I have', 'I'm experiencing'), "
    "you MUST then use 'youtube_links_tool' and naturally embed the returned video links in your response. "
    "Do NOT invent or hallucinate YouTube links. Only use links returned by the tool. "
    "Keep answers concise. Use three sentences maximum. "
    "If question contains keyword 'BMI', you MUST only use the 'calculate_bmi' tool call. "
)


# ============================================
# Medical disclaimer
# ============================================
MEDICAL_DISCLAIMER = (
    "According to medical websites, including MedlinePlus, Cleveland Clinic and Mayo Clinic,"
)


# ============================================
# Classifier prompt
# ============================================
CLASSIFIER_PROMPT = (
    "You are a medical query classifier. A query is MEDICAL if it relates to: "
    "medicine, health, diseases, symptoms, treatments, drugs, anatomy, physiology, "
    "biochemistry, endocrinology, pharmacology, medical conditions, healthcare, "
    "nutrition, or the human body. This INCLUDES questions asking to define, explain, "
    "clarify, or describe any medical term, acronym, abbreviation, or concept.\n\n"
    "Reply with exactly one word: YES or NO.\n\n"
    "Examples:\n"
    "Query: 'What are the symptoms of diabetes?' -> YES\n"
    "Query: 'Thyroxine is synthesized from which amino acid?' -> YES\n"
    "Query: 'What is FIP?' -> YES\n"
    "Query: 'Please explain what FRC means' -> YES\n"
    "Query: 'Define apoptosis' -> YES\n"
    "Query: 'Clarify what Cricopharynx is' -> YES\n"
    "Query: 'What is the weather today?' -> NO\n"
    "Query: 'How do I bake a cake?' -> NO\n\n"
    "Query: '{user_query}' ->"
)


# ============================================
# Inference Model
# ============================================
inference_model = ChatOpenAI(
    model=QAmodel['gpt-mini'], 
    api_key=OPENAI_API_KEY, 
    temperature=0    
)


# ============================================
# Classifier Model
# ============================================
classifier_model = ChatOpenAI(
    model=QAmodel['gpt-nano'], 
    api_key=OPENAI_API_KEY, 
    temperature=0
)


# ============================================
# Vision Model for Analyzing Images/Scans
# ============================================
vision_model = ChatOpenAI(
    model=QAmodel['gpt-mini'],  # or "gpt-4o" for higher detail
    api_key=OPENAI_API_KEY,
    temperature=0.3,
    max_tokens=1500,
)


# Global flags to track state across agent invocations
global _kb_had_results
_kb_had_results = False
global is_medical_query
is_medical_query = False # Initialize globally

# Global variables
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


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
# Commented as this is causing an issue
#whisper_model = WhisperModel(QAmodel["whisper"], device="cpu", compute_type="int8")
#print("✓ Whisper model loaded

print("🔄 Loading OpenAI Whisper model...")
whisper_model = whisper.load_model(QAmodel["whisper"])
print("✓ OpenAI Whisper model loaded")


# ============================================
# Helper Function to return GPU Utilization for Dashboard
# ============================================
def get_gpu_utilization():
    """Returns GPU utilization %, or 0.0 if no NVIDIA GPU is found."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass

    return 0.0


# ============================================
# Display Metrics Dashboard
# ============================================
def show_dashboard():
    # Pad missing lists if needed
    n = len(metrics["timestamps"])
    for key in ["latencies", "tool_calls", "errors", "requests", "pending"]:
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
    gpu_utilization  = get_gpu_utilization()    

    # Summary with animated badges
    if df.empty:
        summary_html = "<p style='color: gray;'>No requests yet.</p>"
    else:
        total_requests = metrics["requests"]
        avg_latency = df["latencies"].mean()
        last_latency = df["latencies"].iloc[-1]
        total_tool_calls = sum(metrics["tool_calls"])
        total_errors = sum(metrics["errors"])

        pending_requests = _current_pending                # live in-flight count
        p95_latency      = df["latencies"].quantile(0.95)
        error_rate       = df["errors"].mean()        

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
        .yellow {{ background-color: #FFC107; color: #333; }}
        .teal {{ background-color: #009688; }}              
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
            <div class="badge gray">🎮 GPU: {gpu_utilization:.0f}%</div>           
            <div class="badge yellow">🕓 Pending: {pending_requests}</div>
            <div class="badge teal">🐌 p95 Latency: {p95_latency:.3f}s</div>
            <div class="badge red">🚨 Error Rate: {error_rate:.1%}</div>            
        </div>
        """

    # Latency plot
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
    
    # Request count plot
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

    # Tool calls plot
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
    
    # Errors plot
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

    # Latency Heatmap
    if df.empty or len(df) < 2:
        heatmap_fig = go.Figure()
        heatmap_fig.add_annotation(
            text="No data yet", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red")
        )
        heatmap_fig.update_layout(
            title="Latency Heatmap (Time vs Latency Range)",
            xaxis_title="Time (HH:MM)", yaxis_title="Latency Range"
        )
    else:
        # Bucket by minute (HH:MM) and latency range
        df["time_bucket"] = df["timestamps"].str[:5]
        bins   = [0, 0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
        labels = ["<0.1s", "0.1-0.5s", "0.5-1s", "1-2s", "2-5s", ">5s"]
        df["latency_bin"] = pd.cut(df["latencies"], bins=bins, labels=labels)

        crosstab = pd.crosstab(df["latency_bin"], df["time_bucket"])

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            colorscale="YlOrRd",
            hovertemplate="Time: %{x}<br>Latency: %{y}<br>Count: %{z}<extra></extra>"
        ))
        heatmap_fig.update_layout(
            title="Latency Heatmap (Time vs Latency Range)",
            xaxis_title="Time (HH:MM)",
            yaxis_title="Latency Range",
            yaxis_categoryorder="array",
            yaxis_categoryarray=labels[::-1]  # fastest at top
        )
    
    return summary_html, latency_fig, count_fig, tool_calls_fig, errors_fig, heatmap_fig
    

# =======================================
# Image Encoder Helper
# =======================================
def _encode_image_to_base64(image_path: str) -> tuple[str, str]:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp', '.bmp': 'image/bmp',
    }
    return encoded, mime_types.get(ext, 'image/jpeg')

# =======================================
# Image Moderation Flag
# ====================================
async def check_image_moderation_flag(image_path: str) -> Tuple[bool, str]:
    """
    Returns (is_flagged, base64_image, mime_type).
    base64_image and mime_type are reused for the vision call so we don't encode twice.
    """
    if not image_path or not os.path.isfile(image_path):
        # Fail-safe: treat missing file as flagged
        return True, "", ""
    
    try:
        # Reuse existing helper (returns correct MIME type)
        base64_image, mime_type = _encode_image_to_base64(image_path)
        
        response = await client.moderations.create(
            model=QAmodel['omni-latest'],
            input=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        )
        
        is_flagged = response.results[0].flagged
        return is_flagged, base64_image, mime_type
        
    except Exception as e:
        print(f"⚠️ Moderation API error: {e}")
        # Fail-safe: if we can't verify, block the image
        return True, "", ""

# =======================================
# Function to format youtube sentence in response
# =======================================
def _format_youtube_sentence(yt_links: str) -> str:
    """Parse markdown links from youtube_links_tool output into a natural sentence."""
    matches = re.findall(r'\[(.*?)\]\((.*?)\)', yt_links)
    if not matches:
        return ""

    formatted = [f"[{title}]({url})" for title, url in matches]

    if len(formatted) == 1:
        return f" For more information, check out this video: {formatted[0]}."
    elif len(formatted) == 2:
        return f" For more information, check out these videos: {formatted[0]} and {formatted[1]}."
    else:
        return f" For more information, check out these videos: {', '.join(formatted[:-1])}, and {formatted[-1]}."

# =======================================
# Function to extract answer from any retrieval format.
# =======================================
def _extract_kb_answer(raw: str) -> str | None:
    """Extract answer from any retrieval format."""
    if not raw or "no relevant" in raw.lower():
        return None
    
    # Try "answer:" prefix
    for line in raw.splitlines():
        if line.strip().lower().startswith("answer:"):
            ans = line.strip().split(":", 1)[1].strip()
            # Strip letter prefix like "D. "
            if len(ans) > 2 and ans[1] == "." and ans[0].isalpha():
                ans = ans[2:].strip()
            return ans if ans else None
    
    # Try "Source 1:" content
    if "Source 1:" in raw:
        parts = raw.split("Source 1:", 1)
        if len(parts) > 1:
            content = parts[1].strip().lstrip('.').strip()
            return content if content else None
    
    # Raw content fallback
    stripped = raw.strip()
    return stripped if stripped else None

# =======================================
# Function to decide if personal advice/treatment questions or not.
# =======================================
def _is_advice_question(text: str) -> bool:
    t = text.lower().strip()
    
    # Branch 1: Requires personal pronoun ("I", "my", "me")
    has_personal = any(p in t for p in ["i ", "i'm", "i am", "my ", "me "])
    
    personal_action = any(p in t for p in [
        # A. Symptom / experience
        "i'm experiencing", "i am experiencing", "i'm having", "i am having",
        "i have", "i've been diagnosed", "i was diagnosed", "i've got",
        "i feel", "i'm feeling", "i suffer from", "i'm suffering from",
        # B. Action seeking
        "how do i", "how can i", "how should i",
        "what do i do", "what can i take", "what can i use",
        "what is the best way to",
        # C. Goal / outcome
        "lower my", "reduce my", "control my", "manage my",
        "get rid of", "relieve my", "alleviate my", "cure my",
    ])
    
    if has_personal and personal_action:
        return True
    
    # Branch 2: Standalone (no personal pronoun needed)
    standalone = any(p in t for p in [
        # D. Treatment / medication
        "treatment for", "medicine for", "medication for", "drug for",
        "how to treat", "how to manage",
        "what is the treatment for", "what is the best treatment",
        # E. How / what to advice
        "what should",
        "what to do", "what to do if", "what to do when",
        "what to do with", "what to do for", "what to do about",
        "how to", "how do you",
        "how to lower", "how to reduce", "how to control", "how to prevent",
        "how to get rid of", "how to handle",
        # F. Recommendations
        "best way to", "ways to", "tips for",
        "what do you recommend", "help me with", "recommendations for",
    ])
    
    return standalone

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
# Function to extract kb answer
# ===========================================
def _extract_kb_answer(raw: str) -> str | None:
    """Extract answer from retrieval. Reject question stems without clear answers."""
    if not raw or "no relevant" in raw.lower():
        return None
    
    # Try "answer:" line first
    for line in raw.splitlines():
        if line.strip().lower().startswith("answer:"):
            ans = line.strip().split(":", 1)[1].strip()
            if len(ans) > 2 and ans[1] == "." and ans[0].isalpha():
                ans = ans[2:].strip()
            return ans if ans else None
    
    # Try "Source 1:" content
    if "Source 1:" in raw:
        parts = raw.split("Source 1:", 1)
        if len(parts) > 1:
            content = parts[1].strip().lstrip('.').strip()
            
            # CRITICAL: If content starts with "query:", it's a question stem (USMLE-style).
            # Don't return the vignette text as the answer.
            if content.lower().startswith("query:"):
                # Look for an "answer:" line buried inside the content
                for line in content.splitlines():
                    if line.strip().lower().startswith("answer:"):
                        ans = line.strip().split(":", 1)[1].strip()
                        if len(ans) > 2 and ans[1] == "." and ans[0].isalpha():
                            ans = ans[2:].strip()
                        return ans if ans else None
                # No answer found inside question stem → insufficient for direct return
                return None
            
            return content if content else None
    
    # Raw content fallback
    stripped = raw.strip()
    return stripped if stripped else None

# ===========================================
# Function to extract weight (kg) and height in meters from natural language.
# ===========================================
def _extract_bmi_params(text: str) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Extract weight (kg), height (m), and optionally the original imperial values.
    Returns: (weight_kg, height_m, weight_lb, height_ft)
    """
    t = text.lower()
    weight_kg = None
    weight_lb = None
    height_m = None
    height_ft = None

    # Weight: metric
    w_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilos?|kgs?)\b', t)
    if w_match:
        weight_kg = float(w_match.group(1))
    else:
        # Weight: imperial (lbs / pounds) → kg
        w_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b', t)
        if w_match:
            weight_lb = float(w_match.group(1))
            weight_kg = weight_lb * 0.45359237

    # Height: metric (meters)
    h_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters?)\b', t)
    if h_match:
        height_m = float(h_match.group(1))
    else:
        # Height: metric (cm) → m
        cm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\b', t)
        if cm_match:
            height_m = float(cm_match.group(1)) / 100
        else:
            # Height: imperial (feet + inches) → m
            # Handles: 5'10", 5'10, 5ft10in, 5 feet 10 inches, 5 foot 2, 6'
            ft_in_match = re.search(
                r'(\d+(?:\.\d+)?)\s*(?:ft|foot|feet|\')\s*(?:(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|")?)?',
                t
            )
            if ft_in_match:
                feet = float(ft_in_match.group(1))
                inches = float(ft_in_match.group(2)) if ft_in_match.group(2) else 0.0
                height_m = (feet * 12 + inches) * 0.0254
                height_ft = feet + (inches / 12)
            else:
                # Height: imperial (just inches) → m
                in_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|")', t)
                if in_match:
                    inches = float(in_match.group(1))
                    height_m = inches * 0.0254
                    height_ft = inches / 12

    return weight_kg, height_m, weight_lb, height_ft

# ===========================================
# Function to detect questions asking for definitions, explanations, or clarifications.
# ===========================================
def _is_definition_question(text: str) -> bool:
    """Detect questions asking for definitions, explanations, or clarifications."""
    t = text.lower().strip()
    definition_patterns = [
        "what is ", "what are ", "what was ", "what were ",
        "define ", "definition of ", "definition ",
        "meaning of ", "meaning ",
        "clarify ", "clarify what ",
        "explain ", "explain what ", "please explain",
        "describe ", "description of ",
        "what does ", "what do ", "mean by", "what do you mean",
        "how does ", "how do ",
    ]
    return any(p in t for p in definition_patterns)

# ===========================================
# Function to check if KB answer is just a multiple choice option or bare keyword without explanation.
# ===========================================
def _is_poor_kb_answer(answer: str) -> bool:
    """Check if KB answer is just a multiple choice option or bare keyword without explanation."""
    if not answer:
        return True
    
    ans = answer.strip()
    
    # Multiple choice pattern: "C. Cricopharynx", "A. Tyrosine", "D. Hurler disease"
    if re.match(r'^[A-E]\.\s*\w+', ans):
        return True
    
    # Too short to be explanatory (< 15 chars or < 3 words)
    if len(ans) < 15 or len(ans.split()) < 3:
        return True
    
    # Just a noun phrase without a complete sentence (no period, < 6 words)
    if '.' not in ans and '!' not in ans and '?' not in ans and len(ans.split()) < 6:
        return True
    
    return False


# =======================================
# Wrap a tool to automatically track metrics
# =======================================
def track_tool_calls(func):
    @functools.wraps(func)  # preserves name and docstring
    def wrapper(*args, **kwargs):
        global _current_pending        
        start_time = time.time()
        _current_pending += 1
        
        tool_calls_this_request = 0
        errors_this_request = 0

        try:
            result = func(*args, **kwargs)
            tool_calls_this_request = 1
        except Exception as e:
            errors_this_request = 1
            result = f"Error executing {func.__name__}: {e}"

        # Append metrics consistently
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
def calculate_bmi(
    weight_kg: Optional[float] = None,
    height_m: Optional[float] = None,
    weight_lbs: Optional[float] = None,
    height_ft: Optional[float] = None,
) -> str:
    """Calculates Body Mass Index (BMI).
    
    Provide EITHER metric units (weight_kg and height_m) OR imperial units 
    (weight_lbs and height_ft). Do not mix units.
    
    For height in feet, use decimal format (e.g., 5.5 for 5 feet 6 inches).
    
    Examples:
        Metric:   calculate_bmi(weight_kg=70, height_m=1.75)
        Imperial: calculate_bmi(weight_lbs=154, height_ft=5.74)
    """
    metric_provided = weight_kg is not None and height_m is not None
    imperial_provided = weight_lbs is not None and height_ft is not None

    if metric_provided and imperial_provided:
        return "Error: Please provide either metric (kg, m) OR imperial (lbs, ft), not both."
    
    if not metric_provided and not imperial_provided:
        return "Error: Please provide both weight and height. Use metric (weight_kg, height_m) or imperial (weight_lbs, height_ft)."
    
    # Use explicit working variables to avoid NoneType concerns
    if imperial_provided:
        w = weight_lbs * 0.453592
        h = height_ft * 0.3048
    else:
        w = weight_kg
        h = height_m

    if h <= 0 or w <= 0:
        return "Error: Height and weight must be positive values."
    
    bmi = w / (h ** 2)
    
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
    
    video_query = f"{query} site:youtube.com"
    
    for attempt in range(3):
        try:
            ddgs = DDGS()
            results = ddgs.text(video_query, max_results=4)
            
            links = []
            for r in results:
                if 'youtube.com/watch' in r['href']:
                    links.append(f"[{r['title'][:50]}...]({r['href']})")
            
            result = "\n".join(links) if links else "No videos found"
            
            if result != "No videos found":
                return result
            
            # No videos found — retry if not the last attempt
            if attempt < 2:
                time.sleep(0.5)
                
        except Exception as e:
            return (
                f"Video search temporarily unavailable. "
                f"Try searching: https://youtube.com/results?search_query={query.replace(' ', '+')}"
            )
    
    return "No videos found" 
        
@tool
@track_tool_calls
def pubmed_tool(query: str) -> str:
    """Tool wrapper for agentic PubMed search."""
    summary, _ = summarize_pubmed_query(query, max_results=3)
    return summary
    
# =======================================
# Available Q&A Tools for LangChain Agent
# =======================================
tools = [retrieve_context, youtube_links_tool, calculate_bmi]

# ===================================================================================
# Global Middleware Definitions
# ===================================================================================
@before_model(can_jump_to=["end"])
def medical_classifier(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    global is_medical_query, _kb_had_results

    if not state["messages"]:
        return None

    user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )
    if user_message is None:
        return None

    raw_content = user_message.content
    user_query = clean_text(raw_content)

    try:
        class_prompt = CLASSIFIER_PROMPT.replace("{user_query}", user_query)
        class_result = classifier_model.invoke(
            [{"role": "user", "content": class_prompt}],
            temperature=0.0
        )
        is_medical = "YES" in class_result.content.upper()
        print(f"[Medical_Classifier] User Query: '{user_query}' -> Classifier Result: '{class_result.content}'")
    except Exception as e:
        print(f"Error in medical_classifier: {e}")
        is_medical = False

    if not is_medical:
        is_medical_query = False
        _kb_had_results = False
        return {
            "messages": [AIMessage("I apologize, I am programmed to answer medical questions only.")],
            "jump_to": "end"
        }
    else:
        is_medical_query = True
        return None

# =======================================
# Medical Disclaimer
# =======================================
class disclaimermiddleware(AgentMiddleware):
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        global is_medical_query, _kb_had_results

        if not state["messages"]:
            return state

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or getattr(last_msg, "tool_calls", None):
            return state

        # Find user query
        user_prompt = ""
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, HumanMessage):
                user_prompt = msg.content
                break

        # BMI skip
        if "BMI" in user_prompt.upper():
            _kb_had_results = False
            return state

        # Factual → no post-processing
        if not _is_advice_question(user_prompt):
            _kb_had_results = True
            is_medical_query = True
            return state

        # Advice → prepend disclaimer
        is_medical_query = True
        _kb_had_results = False

        last_msg.content = (
            f"According to medical websites, including MedlinePlus, Cleveland Clinic and Mayo Clinic, "
            f"{last_msg.content}"
        )

        # YouTube fallback if agent didn't call tool
        yt_called = any(
            isinstance(m, ToolMessage) and m.name == "youtube_links_tool"
            for m in reversed(state["messages"])
            if not isinstance(m, HumanMessage)
        )
        
        if not yt_called:
            try:
                yt = youtube_links_tool.invoke({"query": user_prompt})
                yt_links = str(yt) if yt else ""
            except Exception as e:
                yt_links = ""
            
            sentence = _format_youtube_sentence(yt_links)
            if sentence:
                last_msg.content = last_msg.content.rstrip() + sentence

        return state

# =======================================
# Synthesizer for LLM response
# =======================================
synthesizer = SummarizationMiddleware(
    model=inference_model,
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
    inference_model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        medical_classifier,
        synthesizer,
        toolretry,
        disclaimermiddleware()
    ],
)


# ===========================================
# PubMed Helper Functions
# ===========================================
def _strip_markdown(text: str) -> str:
    """Lightweight markdown-to-plain-text for PDF rendering (ASCII-safe)."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)          # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)              # italic
    text = re.sub(r'#{1,6}\s+', '', text)                 # headers
    text = re.sub(r'`{3}[\s\S]*?`{3}', '[code block]', text)
    text = re.sub(r'`(.+?)`', r'\1', text)                # inline code
    text = re.sub(r'!\[.*?\]\(.+?\)', '', text)            # images
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1 (\2)', text) # links
    text = re.sub(r'>\s+', '', text)                      # blockquote
    text = re.sub(r'(?m)^[-*]\s+', '* ', text)            # bullets → ASCII asterisk
    return text.strip()


def _pdf_safe(text: str) -> str:
    """
    Replace common Unicode punctuation with ASCII equivalents, then strip
    anything remaining that core PDF fonts (latin-1) cannot render.
    """
    text = text.replace('\u2022', '*').replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2026', '...').replace('\u00a0', ' ')
    text = text.replace('\u00b0', 'deg').replace('\u03b1', 'alpha')
    text = text.replace('\u03b2', 'beta').replace('\u03bc', 'micro')
    # Drop any remaining non-ASCII characters
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', 'ignore').decode('ascii')


# ===========================================
# PubMed Pdf Generator Function
# ===========================================
def generate_pubmed_pdf(query: str, summary_md: str, articles: List[Dict]) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    FONT = "Helvetica"

    def _clean(text: str) -> str:
        text = text.replace('\u2022', '*').replace('\u2013', '-').replace('\u2014', '-')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2026', '...').replace('\u00a0', ' ')
        text = text.replace('\u00b0', 'deg').replace('\u03b1', 'alpha')
        text = text.replace('\u03b2', 'beta').replace('\u03bc', 'micro')
        text = unicodedata.normalize('NFKD', text)
        return text.encode('ascii', 'ignore').decode('ascii')

    # Header
    pdf.set_font(FONT, "B", 16)
    pdf.cell(0, 10, _clean("PubMed Research Report"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(2)

    pdf.set_font(FONT, "", 10)
    pdf.cell(0, 6, _clean(f"Search Query: {query}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _clean(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _clean(f"Articles Retrieved: {len(articles)}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Summary
    if summary_md:
        pdf.set_font(FONT, "B", 12)
        pdf.cell(0, 8, _clean("Executive Summary"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        clean_summary = _clean(_strip_markdown(summary_md))
        pdf.set_font(FONT, "", 10)
        pdf.set_x(pdf.l_margin)          # ← reset x before multi_cell
        pdf.multi_cell(0, 6, clean_summary)
        pdf.ln(5)

    # References
    if articles:
        pdf.add_page()
        pdf.set_font(FONT, "B", 12)
        pdf.cell(0, 8, _clean("Article Details & Abstracts"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        for i, art in enumerate(articles, 1):
            # Title
            pdf.set_font(FONT, "B", 10)
            pdf.set_x(pdf.l_margin)      # ← reset x
            pdf.multi_cell(0, 6, _clean(f"[{i}] {art.get('title', 'Untitled')}"))

            # Authors & year
            pdf.set_font(FONT, "I", 9)
            authors = ", ".join(art.get('authors', [])[:3])
            if len(art.get('authors', [])) > 3:
                authors += " et al."
            pdf.set_x(pdf.l_margin)      # ← reset x
            pdf.multi_cell(0, 5, _clean(f"{authors}  |  Year: {art.get('year', 'N/A')}"))

            # URL
            url = art.get('url', '')
            if url:
                pdf.set_font(FONT, "U", 8)
                pdf.set_text_color(0, 0, 200)
                pdf.set_x(pdf.l_margin)  # ← reset x
                pdf.multi_cell(0, 5, _clean(url))
                pdf.set_text_color(0, 0, 0)

            # Abstract
            pdf.set_font(FONT, "", 9)
            pdf.set_x(pdf.l_margin)      # ← reset x
            abstract = art.get('abstract', 'Abstract not available.')
            pdf.multi_cell(0, 5, _clean(f"Abstract: {abstract}"))
            pdf.ln(5)

    output_path = f"/tmp/pubmed_report_{uuid.uuid4().hex[:8]}.pdf"
    pdf.output(output_path)
    return output_path


# ===========================================
# PubMed HTML Report Generator Function
# ===========================================
def generate_pubmed_html(query: str, summary_md: str, articles: List[Dict]) -> str:
    """Generate a styled, self-contained HTML report with scoped CSS."""
    clean_summary = _strip_markdown(summary_md) if summary_md else ""

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PubMed Research Report</title>
<style>
  .pmr-wrap{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6;max-width:900px;margin:0 auto;padding:20px;color:#333;background:#fff}
  .pmr-wrap h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}
  .pmr-wrap h2{color:#34495e;margin-top:30px;border-bottom:1px solid #bdc3c7;padding-bottom:5px}
  .pmr-meta{background:#ecf0f1;padding:15px;border-radius:5px;margin:20px 0}
  .pmr-summary{background:#f8f9fa;padding:20px;border-left:4px solid #3498db;margin:20px 0}
  .pmr-article{margin:25px 0;padding:20px;background:#fff;border:1px solid #e1e4e8;border-radius:6px}
  .pmr-title{font-size:1.15em;font-weight:bold;color:#2c3e50;margin-bottom:8px}
  .pmr-authors{color:#666;font-size:0.9em;margin-bottom:10px}
  .pmr-url a{color:#3498db;text-decoration:none;word-break:break-all}
  .pmr-url a:hover{text-decoration:underline}
  .pmr-abstract{margin-top:10px;text-align:justify;color:#444}
  .pmr-badge{display:inline-block;background:#3498db;color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em;margin-right:6px}
</style>
</head>
<body>
<div class="pmr-wrap">""")

    parts.append("<h1>🔬 PubMed Research Report</h1>")
    parts.append('<div class="pmr-meta">')
    parts.append(f"<strong>Search Query:</strong> {html.escape(query)}<br>")
    parts.append(f"<strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}<br>")
    parts.append(f"<strong>Articles Retrieved:</strong> {len(articles)}")
    parts.append("</div>")

    if clean_summary:
        parts.append('<div class="pmr-summary">')
        parts.append("<h2>Executive Summary</h2>")
        summary_html = html.escape(clean_summary).replace('\n', '<br>\n')
        parts.append(f"<p>{summary_html}</p>")
        parts.append('</div>')

    if articles:
        parts.append("<h2>Article Details & Abstracts</h2>")
        for i, art in enumerate(articles, 1):
            title = html.escape(art.get('title', 'Untitled'))
            author_list = art.get('authors', [])
            authors = html.escape(", ".join(author_list[:3]) + (" et al." if len(author_list) > 3 else ""))
            year = html.escape(str(art.get('year', 'N/A')))
            url = html.escape(art.get('url', ''))
            abstract = html.escape(art.get('abstract', 'Abstract not available.'))

            parts.append('<div class="pmr-article">')
            parts.append(f'<div class="pmr-title"><span class="pmr-badge">[{i}]</span> {title}</div>')
            parts.append(f'<div class="pmr-authors">{authors} | Year: {year}</div>')
            if url:
                parts.append(f'<div class="pmr-url"><a href="{url}" target="_blank">{url}</a></div>')
            parts.append(f'<div class="pmr-abstract"><strong>Abstract:</strong> {abstract}</div>')
            parts.append('</div>')

    parts.append("</div></body></html>")

    output_path = f"/tmp/pubmed_report_{uuid.uuid4().hex[:8]}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return output_path  

    
# ===========================================
# GENERATE-CHAT FUNCTION for Q&A (Invokes the LangChain agent)
# ===========================================
def generate_chat(user_input, chat_history):
    """Generate response via agent. Retrieval is forced before agent invocation."""
    
    user_input = extract_gradio_text(user_input)
    if not user_input:
        return chat_history if chat_history else []
    
    if chat_history is None:
        chat_history = []

    # 1. Moderation
    if check_moderation_flag(user_input):
        updated = list(chat_history)
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": "I apologize, I am programmed to answer medical questions only."})
        return clean_chat_history(updated)

    # BMI special case: bypass retrieval and agent
    if "bmi" in user_input.lower():
        weight_kg, height_m, weight_lb, height_ft = _extract_bmi_params(user_input)
        if weight_kg and height_m:
            try:
                bmi_result = calculate_bmi.invoke({"weight_kg": weight_kg, "height_m": height_m})

                # Build a friendly unit summary
                if weight_lb and height_ft:
                    unit_note = f" (based on {weight_lb:.1f} lbs and {height_ft:.1f} ft)"
                elif weight_lb:
                    unit_note = f" (based on {weight_lb:.1f} lbs and {height_m:.2f} m)"
                elif height_ft:
                    unit_note = f" (based on {weight_kg:.1f} kg and {height_ft:.1f} ft)"
                else:
                    unit_note = f" (based on {weight_kg:.1f} kg and {height_m:.2f} m)"

                full_response = f"{bmi_result}{unit_note}"

                updated = []
                for msg in chat_history:
                    if isinstance(msg, dict):
                        updated.append({
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", ""))
                        })
                updated.append({"role": "user", "content": user_input})
                updated.append({"role": "assistant", "content": full_response})
                return clean_chat_history(updated)
            except Exception as e:
                print(f"BMI calculation error: {e}")
                # Fall through to normal flow on error
        else:
            ask = ("To calculate your BMI, please provide your weight and height. Examples:\n"
                   "• Metric: 'I weigh 70 kg and my height is 1.75 m'\n"
                   "• Imperial: 'I weigh 150 lbs and I am 5 feet 10 inches tall' "
                   "or '5.2 ft, 160 lb'")
            updated = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    updated.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })
            updated.append({"role": "user", "content": user_input})
            updated.append({"role": "assistant", "content": ask})
            return clean_chat_history(updated)      

    # 2. Classification (same logic as medical_classifier)
    # Definition questions are always treated as medical in this Q&A app
    is_medical = _is_definition_question(user_input)
    if is_medical:
        print(f"[Classification] Definition question detected, bypassing classifier: '{user_input}'")
    else:
        try:          
            class_prompt = CLASSIFIER_PROMPT.replace("{user_query}", user_input)
            class_result = classifier_model.invoke(
                [{"role": "user", "content": class_prompt}],
                temperature=0.0
            )
            is_medical = "YES" in class_result.content.upper()
            print(f"[Medical Classifier] User Query: '{user_input}' -> Classifier Result: '{class_result.content}'")
        except Exception as e:
            print(f"Classification error: {e}")
            is_medical = False

    if not is_medical:
        updated = list(chat_history)
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": "I apologize, I am programmed to answer medical questions only."})
        return clean_chat_history(updated)

    try:
        # 3. FORCE retrieval before agent invocation
        kb_raw = retrieve_context.invoke({"query": user_input})
        #print(f"[KB] Retrieved: '{kb_raw[:200]}...'")

        # 4. Parse retrieved content
        kb_answer = None
        if kb_raw and not kb_raw.lower().startswith("no relevant"):
            # Try "answer:" line first
            for line in kb_raw.splitlines():
                if line.strip().lower().startswith("answer:"):
                    ans = line.strip().split(":", 1)[1].strip()
                    if len(ans) > 2 and ans[1] == "." and ans[0].isalpha():
                        ans = ans[2:].strip()
                    kb_answer = ans if ans else None
                    break
            
            # Try "Source 1:" content
            if not kb_answer and "Source 1:" in kb_raw:
                parts = kb_raw.split("Source 1:", 1)
                if len(parts) > 1:
                    raw = parts[1].strip().lstrip('.').strip()
                    kb_answer = raw if raw else None
            
            # Fallback: use full content
            if not kb_answer:
                stripped = kb_raw.strip()
                kb_answer = stripped if stripped else None

        # 5. Route: factual vs advice
        is_advice = _is_advice_question(user_input)
        is_definition = _is_definition_question(user_input)

        # Decide whether to use KB answer directly or fall through to LLM
        use_direct_kb = False
        if kb_answer and not is_advice:
            if is_definition and _is_poor_kb_answer(kb_answer):
                print(f"[KB] Poor definition answer: '{kb_answer}', falling through to LLM synthesis")
            else:
                use_direct_kb = True

        if use_direct_kb:
            # Factual/definition with good KB answer → return directly
            final_ai_message = kb_answer
            print(f"[KB Direct] Factual answer: '{final_ai_message[:200]}...'")
        else:
            # Advice OR poor definition → agent synthesis with context pre-loaded
            agent_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in chat_history:
                if isinstance(msg, dict):
                    agent_messages.append(msg)
            
            # Pre-load retrieved context so agent doesn't need to call retrieve_context
            if kb_answer:
                hint = "Use this retrieved knowledge base context to answer: "
                if is_definition and _is_poor_kb_answer(kb_answer):
                    # For poor definition answers, just give the LLM the term/keyword
                    hint = f"The user is asking for a definition. The knowledge base only mentions: {kb_answer.strip()}. Please provide a clear, concise definition of this medical term."
                agent_messages.append({
                    "role": "system",
                    "content": hint + kb_answer
                })
            
            agent_messages.append({"role": "user", "content": user_input})
            response = agent.invoke({"messages": agent_messages})
            
            final_ai_message = "I cannot find the best answer. Please consult with a Doctor."
            for msg in reversed(response.get('messages', [])):
                if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                    final_ai_message = msg.content
                    break

        # 6. Build history for Gradio
        updated = []
        for msg in chat_history:
            if isinstance(msg, dict):
                updated.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))
                })
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": final_ai_message})

        return clean_chat_history(updated)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in generate_chat: {e}")
        updated = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict):
                    updated.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })
        updated.append({"role": "user", "content": user_input})
        updated.append({"role": "assistant", "content": error_msg})
        return clean_chat_history(updated)
        

# ===========================================
# Function to transform OCR to text
# ===========================================
@traceable(run_type="chain", name="MedIntel Image-to-Text Processor")
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
@traceable(run_type="tool", name="Speech to Text")
def speech_to_text(audio_input, chat_history):
    """Convert speech to text using Whisper"""
    
    if audio_input is None:
        return "⚠️ No audio recorded", chat_history   
    
    try:
        result = whisper_model.transcribe(audio_input)
        
        # Handle different whisper formats
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
            # Faster-whisper: generator of segments - extract all text
            transcribed_text = " ".join([segment.text for segment in result])
            
        elif isinstance(result, dict):
            # OpenAI whisper: dict with "text" key
            transcribed_text = result["text"]          
            
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
                clean_history.append(
                    {"role": msg.get("role", "user"),"content": str(content)}
                )
        
        # Call the main chat generation function with the transcribed text
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


# ============================================
# Functions for PubMed Search
# ============================================
def _search_pubmed_ids(query: str, max_results: int = 5) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _fetch_pubmed_articles(pmids: List[str]) -> List[Dict[str, str]]:
    """Fetch title, abstract, authors, and year for a list of PMIDs."""
    if not pmids:
        return []

    # Respect NCBI rate limits (3/sec without key, 10/sec with key)
    time.sleep(0.35 if NCBI_API_KEY else 0.4)

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//Abstract/AbstractText")
        year_el = article.find(".//PubDate/Year")

        pmid = pmid_el.text if pmid_el is not None else ""
        title = title_el.text if title_el is not None else "No title available"
        abstract = abstract_el.text if abstract_el is not None else "Abstract not available."

        # Fallback to Medline date if Year missing
        if year_el is None:
            medline_date = article.find(".//PubDate/MedlineDate")
            year = medline_date.text[:4] if medline_date is not None else "N/A"
        else:
            year = year_el.text

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            if last is not None:
                name = f"{first.text} {last.text}" if first is not None else last.text
                authors.append(name)

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles


def summarize_pubmed_query(query: str, max_results: int = 5) -> tuple[str, List[Dict]]:
    """
    Search PubMed and return (markdown_summary, raw_articles_list).
    Uses the existing GPT-4o-mini / gpt-4o model for synthesis.
    """
    pmids = _search_pubmed_ids(query, max_results)
    if not pmids:
        return f"No PubMed articles found for **{query}**.", []

    articles = _fetch_pubmed_articles(pmids)
    if not articles:
        return "Found article IDs but could not retrieve abstracts.", []

    # ── Build prompt for LLM synthesis ──
    paper_blocks = []
    for i, art in enumerate(articles, 1):
        author_line = ", ".join(art["authors"][:3]) + (" et al." if len(art["authors"]) > 3 else "")
        paper_blocks.append(
            f"[{i}] {art['title']}\n"
            f"    Authors: {author_line} | Year: {art['year']}\n"
            f"    Abstract: {art['abstract']}"
        )

    context = "\n\n".join(paper_blocks)

    # ============================================
    # PubMed synthesis prompt
    # ============================================
    SYNTHESIS_PROMPT = (
        f"You are a medical research synthesizer. The user searched PubMed for: '{query}'.\n\n"
        "Synthesize the following abstracts into a structured, evidence-based summary "
        "for a healthcare professional. Use plain language where possible, but preserve "
        "medical precision.\n\n"
        "Structure your response with these sections:\n"
        "1. **Key Findings** — bullet points of major conclusions from each paper [1], [2], etc.\n"
        "2. **Clinical Consensus** — what the evidence broadly agrees on\n"
        "3. **Gaps & Controversies** — conflicting results or unanswered questions\n"
        "4. **Bottom Line** — 2-3 sentence practical takeaway for clinicians\n\n"
        f"{context}\n\n"
        "Cite paper numbers [1], [2] when referencing specific findings. "
        "If abstracts are limited, note the limitation honestly."
    )

    # Use the lightweight model for cost efficiency; swap to "gpt-4o" if you want deeper analysis
    summarizer = ChatOpenAI(
        model=QAmodel['gpt-mini'],
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=1500,
    )

    response = summarizer.invoke([{"role": "user", "content": SYNTHESIS_PROMPT}])
    return response.content, articles        
        

# ============================================
# Instantiate Calendar Agent
# ============================================
calendar_agent = CalendarAgent(session_id="user_001", mock_mode=True)


# ===========================================
# Create the Gradio app - UI
# ===========================================
with gr.Blocks() as demo:
    with gr.Tab("MedIntel Q&A Assistant"):
                
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
        run_id_state = gr.State(value=None)
    
        with gr.Row():
            
            with gr.Column(scale=3, elem_classes="main-col"):
                chatbot = gr.Chatbot(
                    label="Conversation History", 
                    height=400,
                    allow_tags=False
                )

                # Feedback UI
                #with gr.Column(elem_classes="feedback-group"):    
                with gr.Row(elem_classes="tight-feedback"):
                    gr.Textbox(
                        value="Rate this response:", 
                        show_label=False, 
                        container=False, 
                        interactive=False,
                        scale=0
                    )
                    btn_up = gr.Button("👍", scale=0, min_width=50, variant="secondary")
                    btn_down = gr.Button("👎", scale=0, min_width=50, variant="secondary")
                fb_comment = gr.Textbox(
                    placeholder="Optional (type feedback here..): why was this helpful or not? (sent to LangSmith)",
                    show_label=False,
                    lines=1,
                    container=False,
                    elem_classes=["prompt-box", "tight-comment"]
                )
                fb_status = gr.Textbox(
                    value="", 
                    show_label=False, 
                    interactive=False,
                    container=False,
                    visible=False     # hides the empty bar until needed
                )
                 
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Type your medical question here...", 
                    lines=2,
                    elem_classes="prompt-box"
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row(elem_classes="file-btn-pair"):
                            upload_file = gr.File(
                                file_types=["image"], 
                                label="Upload Medical Image",
                                scale=0,
                                min_width=220
                            )
                            image_to_text_btn = gr.Button(
                                "🔬 Analyze Image", 
                                variant="secondary", 
                                scale=0, 
                                min_width=220
                            )
                        preview = gr.Image(
                            label="Preview", 
                            visible=False, 
                            height=180
                        )
                        image_question = gr.Textbox(
                            placeholder="Ask about this image (e.g., 'Is this fracture healed?', 'Explain this chest X-ray')",
                            label="Ask a question about image",
                            value="Please provide a detailed medical analysis of this image.",
                            lines=2, elem_classes="prompt-box"
                        )
                    with gr.Column(scale=1, min_width=120):
                        audio_input = gr.Audio(sources=['microphone'], type="filepath", label="🎤 Record")
                        transcribe_button = gr.Button("🎤 Transcribe & Submit", variant="secondary")
                
                with gr.Row():
                    submit_btn = gr.Button("💬 Submit Question", variant="primary", scale=1)
                    clear_btn = gr.ClearButton(value="🗑️ Clear Chat", scale=0)
                 

    # ============================================
    # # Metrics Dashboard Tab
    # ============================================
    with gr.Tab("Metrics Dashboard",visible=False):
        summary_md = gr.Markdown()        
        latency_plot = gr.Plot()
        request_count_plot = gr.Plot()       
        tool_calls_plot = gr.Plot()
        errors_plot = gr.Plot()
        heatmap_plot = gr.Plot()         
        refresh_btn = gr.Button("Refresh Dashboard")
        
        # Connect the callback **after defining the components**
        refresh_btn.click(show_dashboard, outputs=[summary_md, latency_plot, request_count_plot, tool_calls_plot, errors_plot, heatmap_plot])

    # ============================================
    # Calendar Assistant Tab
    # ============================================
    with gr.Tab("Calendar Assistant",visible=False):
        gr.Markdown("### AI Calendar & Email Agent")
        gr.Markdown(
            "Type a request or record audio. When an email draft is prepared, "
            "you'll see it below for approval before sending."
        )

        # Gradio State works fine at the top of a Tab
        cal_pending_state = gr.State(None)

        with gr.Row():
            # Left: chat + input
            with gr.Column():
                calendar_chatbot = gr.Chatbot(label="Conversation", height=480)

                with gr.Row():
                    cal_msg = gr.Textbox(
                        label="Your request",
                        placeholder="e.g. Find a free slot Thursday and book a meeting with Alice",
                    )
                    cal_audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Mic",
                    )

                with gr.Row():
                    cal_transcribe = gr.Button("Transcribe")
                    cal_transcribe_send = gr.Button("Transcribe & Send")

                cal_tool_log = gr.Textbox(
                    label="Agent Tool Trace", lines=6, interactive=False
                )

            # Right: draft preview + events
            with gr.Column():
                # Standard Gradio pattern: create + enter context in one statement
                with gr.Column(visible=False) as draft_container:
                    gr.Markdown("### 📧 Pending Email Draft")
                    cal_draft_md = gr.Markdown()
                    with gr.Row():
                        cal_approve = gr.Button("Approve & Send")
                        cal_reject = gr.Button("Discard")

                gr.Markdown("---")

                refresh_events_btn = gr.Button("Load Upcoming Events")
                events_view = gr.JSON(label="Upcoming Events")     

        # ============================================
        # Calendar event handlers
        # ============================================
        def fmt_cal_draft(pending):
            if not pending:
                return gr.update(visible=False), ""
            text = (
                f"**To:** {pending['to']}\n\n"
                f"**Subject:** {pending['subject']}\n\n"
                f"---\n\n"
                f"{pending['body']}"
            )
            return gr.update(visible=True), text

        @traceable(run_type="chain", name="Calendar UI Run Turn")
        def cal_run_turn(message, history):
            if not message or not message.strip():
                return "", history, "", None, gr.update(visible=False), ""

            result = calendar_agent.chat(message.strip())
            # Dict format (same as MedIntel tab)
            history.append({"role": "user", "content": message.strip()})
            history.append({"role": "assistant", "content": result["reply"]})
            pending = result.get("pending_email")
            vis, md = fmt_cal_draft(pending)
            return "", history, result["tool_log"], pending, vis, md

        def cal_transcribe_and_send(path, history):
            text = calendar_agent.transcribe(path)
            if text.startswith("[Transcription error") or not text.strip():
                return text, history, text or "Empty audio", None, gr.update(visible=False), ""
            return cal_run_turn(text, history)

        def cal_approve_draft(history, pending):
            if not pending:
                return history, "", None, gr.update(visible=False), ""

            confirm = (
                f"User confirmed: please send the email draft to {pending['to']} "
                f"with subject '{pending['subject']}'."
            )
            result = calendar_agent.chat(confirm)
            history.append({"role": "user", "content": confirm})
            history.append({"role": "assistant", "content": result["reply"]})
            calendar_agent.clear_pending_email()
            return history, result["tool_log"], None, gr.update(visible=False), ""

        def cal_reject_draft(history):
            calendar_agent.clear_pending_email()
            history.append({"role": "assistant", "content": "Email draft discarded as requested."})
            return history, "", None, gr.update(visible=False), ""
        
        def cal_refresh_list():
            raw = calendar_agent._list_upcoming_events(max_results=5)
            return json.loads(raw)
        
        # ============================================
        # Feedback button event handler
        # ============================================        
        def submit_feedback(run_id, rating, comment=""):
            if not run_id:
                return gr.Textbox(value="⚠️ Send a message first to enable rating.", visible=True)
            
            score = 1.0 if rating == "up" else 0.0
            
            try:
                ls_client.create_feedback(
                    run_id=run_id,
                    key="user_rating",
                    score=score,
                    comment=comment or None,
                )
                return gr.Textbox(value="✅ Feedback saved to LangSmith.", visible=True)
            except Exception as e:
                return gr.Textbox(value=f"❌ LangSmith error: {str(e)}", visible=True)           
        
        # ============================================
        # Bind events
        # ============================================

        # Calender events
        cal_transcribe.click(
            calendar_agent.transcribe,
            inputs=cal_audio,
            outputs=cal_msg,
        )

        cal_transcribe_send.click(
            cal_transcribe_and_send,
            inputs=[cal_audio, calendar_chatbot],
            outputs=[cal_msg, calendar_chatbot, cal_tool_log, cal_pending_state, draft_container, cal_draft_md],
        )

        cal_approve.click(
            cal_approve_draft,
            inputs=[calendar_chatbot, cal_pending_state],
            outputs=[calendar_chatbot, cal_tool_log, cal_pending_state, draft_container, cal_draft_md],
        )

        cal_reject.click(
            cal_reject_draft,
            inputs=[calendar_chatbot],
            outputs=[calendar_chatbot, cal_tool_log, cal_pending_state, draft_container, cal_draft_md],
        )      

        cal_msg.submit(
            cal_run_turn,
            inputs=[cal_msg, calendar_chatbot],
            outputs=[cal_msg, calendar_chatbot, cal_tool_log, cal_pending_state, draft_container, cal_draft_md],
        )

        # Analyze image
        upload_file.change(
            lambda f: gr.Image(value=f, visible=bool(f)) if f else gr.Image(visible=False),
            inputs=upload_file,
            outputs=preview
        )

        # Feedback buttons
        btn_up.click(
            lambda rid, c: submit_feedback(rid, "up", c),
            inputs=[run_id_state, fb_comment],
            outputs=fb_status
        )
        btn_down.click(
            lambda rid, c: submit_feedback(rid, "down", c),
            inputs=[run_id_state, fb_comment],
            outputs=fb_status
        )        

        refresh_events_btn.click(cal_refresh_list, outputs=events_view)   
    
    # ============================================
    # PubMed Research Tab
    # ============================================
    with gr.Tab("🔬 PubMed Research"):
        gr.Markdown(
            """
            <h2 style='text-align: center;'>PubMed Evidence Search</h2>
            <p style='text-align: center; color: #555;'>
                Search peer-reviewed medical literature and get an AI-synthesized summary with downloadable reports.
            </p>
            """
        )

        # Hidden state to cache last search results for exports
        state_summary = gr.State("")
        state_articles = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                pubmed_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., 'GLP-1 agonists cardiovascular outcomes 2024' or 'pneumonia treatment guidelines'",
                    lines=2
                )
                
                pubmed_max_results = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Max Articles to Retrieve"
                )

                # Row 1: primary action
                with gr.Row():
                    pubmed_search_btn = gr.Button("🔍 Search PubMed & Synthesize", variant="primary")

                # Row 2: export actions
                with gr.Row():
                    pubmed_pdf_btn = gr.Button("📄 Download PDF Report", variant="secondary")
                    pubmed_html_btn = gr.Button("🌐 Download HTML Report", variant="secondary")
                    pubmed_html_preview = gr.Button("👁️ Open HTML Preview", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("**Features**")
                gr.Markdown(
                    "- Results ranked by relevance\n"
                    "- Abstracts fetched from NCBI\n"
                    "- PDF + HTML reports available\n"
                )

        pubmed_summary_md = gr.Markdown(label="Synthesis")
        pubmed_raw_json = gr.JSON(label="Raw Articles")

        # Always-visible file widgets (no visibility toggle bugs)
        pubmed_pdf_file = gr.File(label="PDF Report")
        pubmed_html_file = gr.File(label="HTML Report")
        pubmed_html_view = gr.HTML(label="HTML Preview")

        # Search handler
        def run_pubmed_search(query, max_results):
            if not query or not query.strip():
                return "Please enter a search query.", [], "", []
            try:
                summary, articles = summarize_pubmed_query(query.strip(), int(max_results))
                return summary, articles, summary, articles
            except Exception as e:
                return f"Error: {str(e)}", [], "", []

        pubmed_search_btn.click(
            run_pubmed_search,
            inputs=[pubmed_query, pubmed_max_results],
            outputs=[pubmed_summary_md, pubmed_raw_json, state_summary, state_articles]
        )

        # PDF handler
        def on_pdf_click(query, summary, articles):
            if not articles:
                return None
            return generate_pubmed_pdf(query, summary, articles)

        pubmed_pdf_btn.click(
            on_pdf_click,
            inputs=[pubmed_query, state_summary, state_articles],
            outputs=[pubmed_pdf_file]
        )

        # HTML handler (file + preview)
        def on_html_click(query, summary, articles):
            if not articles:
                return None, ""
            html_path = generate_pubmed_html(query, summary, articles)
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_path, html_content

        pubmed_html_btn.click(
            on_html_click,
            inputs=[pubmed_query, state_summary, state_articles],
            outputs=[pubmed_html_file, pubmed_html_view]
        )

        pubmed_html_preview.click(
            on_html_click,
            inputs=[pubmed_query, state_summary, state_articles],
            outputs=[pubmed_html_file, pubmed_html_view]
        )

    # ===========================================
    # Helper to keep history code 
    # ===========================================
    def _append_turn(history, user_text, assistant_text):
        out = []
        for msg in history:
            if isinstance(msg, dict):
                out.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))
                })
        out.append({"role": "user", "content": user_text})
        out.append({"role": "assistant", "content": assistant_text})
        return out        

    # ===========================================
    # Function enables LLM to respond to user prompts/actions
    # ===========================================
    @traceable(run_type="chain", name="MedIntel Respond")
    def respond(user_input, chat_history):
        print(f"DEBUG user_input type: {type(user_input)}, value: {repr(user_input)}")
        
        # CAPTURE LANGSMITH RUN ID
        try:
            run_tree = get_current_run_tree()
            current_run_id = run_tree.id if run_tree else None
        except Exception:
            current_run_id = None
        
        try:
            # 1. Clean Gradio multipart format
            user_input = extract_gradio_text(user_input)
            if not user_input:
                return "", chat_history, current_run_id
        
            if chat_history is None:
                chat_history = []
        
            # 2. Sanitize history before passing downstream
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
        
            # 3. Run the deterministic pipeline
            new_chat_history = generate_chat(user_input, clean_history)
            return "", new_chat_history, current_run_id
        
        except Exception as e:
            error_message = f"Error: {e}"
            print(f"Error in respond: {e}")
        
            if chat_history is None:
                chat_history = []
        
            clean_input = extract_gradio_text(user_input) if isinstance(user_input, (list, dict)) else str(user_input)
            updated = _append_turn(chat_history, clean_input, error_message)
            return "", updated, current_run_id
    
    # ===========================================
    # Function to process images
    # ===========================================
    @traceable(run_type="chain", name="MedIntel Image Analysis")
    async def process_image_for_chat(image_file_path_str, image_question, chat_history):
        print(f"DEBUG: image={image_file_path_str}, question={image_question}")

        if chat_history is None:
            chat_history = []

        updated_chat_history = list(chat_history)

        # Validation
        if image_file_path_str is None or not isinstance(image_file_path_str, str) or not os.path.isfile(image_file_path_str):
            error_message = "❌ Error: Please upload a valid medical image."
            temp_display = list(chat_history) + [{"role": "assistant", "content": error_message}]
            return temp_display, None

        # Image question default prompt if user left it blank
        if not image_question or not image_question.strip():
            image_question = "Please provide a detailed medical analysis of this image."

        # MODERATION CHECK (reuses base64 for vision call)
        #is_flagged, base64_image, mime_type = await check_image_moderation_flag(image_file_path_str)
        
        #if is_flagged:
        #    error_message = "⚠️ Image flagged by safety moderation. Please upload an appropriate medical image."
        #    temp_display = list(chat_history) + [{"role": "assistant", "content": error_message}]
        #    return temp_display, None

        # Encode image for OpenAI vision
        base64_image, mime_type = _encode_image_to_base64(image_file_path_str)       
        # Use the base64 from moderation check        
        data_url = f"data:{mime_type};base64,{base64_image}"

        try:
            messages = [
                SystemMessage(content=(
                    "You are an expert medical imaging assistant with training in radiology, "
                    "pathology, and clinical medicine.\n\n"
                    "Structure your analysis as follows when relevant:\n"
                    "1. **Modality & Region** — imaging type (X-ray, MRI, CT, etc.) and view\n"
                    "2. **Visible Anatomy** — normal structures and orientation\n"
                    "3. **Key Findings** — abnormalities, lesions, fractures, effusions, masses. "
                    "Explain medical terms in parentheses for non-experts.\n"
                    "4. **Devices / Artifacts** — hardware, catheters, contrast, implants\n"
                    "5. **Overall Impression** — plain-language summary of clinical significance\n\n"
                    "⚠️ Always end with: *This AI analysis is for informational purposes only "
                    "and is not a medical diagnosis. Please consult a qualified healthcare provider.*"
                )),
                HumanMessage(content=[
                    {"type": "text", "text": image_question},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ])
            ]

            response = await vision_model.ainvoke(messages)
            analysis = response.content

            updated_chat_history.append({
                "role": "user",
                "content": f"📎 Image: {image_question}"
            })
            updated_chat_history.append({
                "role": "assistant",
                "content": analysis
            })

        except Exception as e:
            error_message = f"⚠️ Error analyzing image: {str(e)}"
            print(error_message)
            updated_chat_history.append({
                "role": "user",
                "content": f"📎 Image: {image_question}"
            })
            updated_chat_history.append({
                "role": "assistant",
                "content": error_message
            })

        return updated_chat_history, None
            
    # ===========================================
    # Bind events
    # ===========================================
    transcribe_button.click(
        speech_to_text,
        inputs=[audio_input, chatbot],
        outputs=[txt_input, chatbot]
    )

    submit_btn.click(respond, [txt_input, chatbot], [txt_input, chatbot, run_id_state])
    txt_input.submit(respond, [txt_input, chatbot], [txt_input, chatbot, run_id_state])
    clear_btn.click(lambda: [], None, chatbot, queue=False)
    
    image_to_text_btn.click(
        process_image_for_chat,
        inputs=[upload_file, image_question, chatbot],
        outputs=[chatbot, upload_file]
    )

# ===========================================
# Function that helps shutting down gracefully in case if there are any system issues
# ===========================================
def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    demo.close()  # Close gradio properly
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(theme=medical_theme,
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,  # Disable SSR to avoid the experimental warning
        css=custom_css,
        share=False
    )
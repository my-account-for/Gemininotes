# /--------------------------\
# |   START OF app.py FILE   |
# \--------------------------/

# --- 1. IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
from datetime import datetime
import uuid
import traceback
from dotenv import load_dotenv
import PyPDF2
from pydub import AudioSegment
from pydub.utils import make_chunks
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from streamlit_pills import pills
from streamlit_ace import st_ace
import re

# --- Local Imports ---
import database

# --- 2. CONSTANTS & CONFIG ---
load_dotenv()

# Configure the Gemini client safely
try:
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"]:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    else:
        st.session_state.config_error = "🔴 GEMINI_API_KEY not found. Please create a .env file and add your key."
except Exception as e:
    st.session_state.config_error = f"🔴 Error configuring Google AI Client: {e}"

# Application Constants
MAX_PDF_MB = 25
MAX_AUDIO_MB = 200
CHUNK_WORD_SIZE = 4500
CHUNK_WORD_OVERLAP = 200

# Models & Prompts
AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash", "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash-lite", "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite", "Gemini 2.5 Pro": "gemini-2.5-pro",
}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]

EXPERT_MEETING_CHUNK_BASE = """### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles). Use simple, direct language.
- **DO NOT:** Summarize or include introductions about consulting firms.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format.

**(2.A) Questions:**
-   Extract the clear, primary question and format it in **bold**.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL SPECIFICS.** This includes all data, names, examples, monetary values (`$`), percentages (`%`), etc."""

EXPERT_MEETING_CHUNK_BASE_OPTION_2 = """### **PRIMARY DIRECTIVE**
Your goal is to be slightly less verbose and use a more natural sentence flow where possible. However, you must **NEVER** sacrifice factual detail, data, or specifics for the sake of brevity.

### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles).
- **DO NOT:** Summarize.

**(2.) Q&A format:**
Structure the main body in Question/Answer format.

**(2.A) Questions:**
-   Extract the clear, primary question and format it in **bold**.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL SPECIFICS.**"""

PROMPT_INITIAL = """You are a High-Fidelity Factual Extraction Engine. Your task is to analyze an expert consultation transcript chunk and generate detailed, factual notes.
Your primary directive is **100% completeness and accuracy**. You will process the transcript sequentially. For every Question/Answer pair you identify, you must generate notes following the structure below.
---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are a High-Fidelity Factual Extraction Engine continuing a note-taking task. Your goal is to process the new transcript chunk provided below, using the context from the previous chunk to ensure perfect continuity.
### **CONTEXT FROM PREVIOUS CHUNK**
{context_package}
---
### **PRIMARY INSTRUCTIONS**
First, review the context. Because of the overlap in the transcript, you will see text that was already processed. **Locate the "Last Question Processed" from the context and begin your work from the *first NEW question and answer* that follows it.**
From that point forward, you must adhere to the following procedure for every Q&A pair in the remainder of the chunk.
---
{base_instructions}
---
**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""

# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Paste Text"
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    earnings_call_mode: str = "Generate New Notes"
    selected_sector: str = "IT Services"
    notes_model: str = "Gemini 2.5 Pro"
    refinement_model: str = "Gemini 2.5 Flash Lite"
    transcription_model: str = "Gemini 2.5 Flash"
    refinement_enabled: bool = True
    add_context_enabled: bool = False
    context_input: str = ""
    speaker_1: str = ""
    speaker_2: str = ""
    earnings_call_topics: str = ""
    existing_notes_input: str = ""
    text_input: str = ""
    uploaded_file: Optional[Any] = None
    processing: bool = False
    active_note_id: Optional[str] = None
    error_message: Optional[str] = None
    fallback_content: Optional[str] = None

# --- 4. CORE PROCESSING & UTILITY FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_file_content(uploaded_file) -> Tuple[Optional[str], str]:
    name = uploaded_file.name
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    ext = os.path.splitext(name)[1].lower()
    
    try:
        if ext == ".pdf":
            reader = PyPDF2.PdfReader(file_bytes)
            if reader.is_encrypted: return "Error: PDF is encrypted.", name
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return (content, name) if content else ("Error: No text found in PDF.", name)
        elif ext in [".txt", ".md"]:
            return file_bytes.read().decode("utf-8"), name
        # Audio files will be handled in the main processing function
        elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
            return "audio_file", name
    except Exception as e:
        return f"Error: Could not process file {name}. Details: {str(e)}", name
    return None, name

@st.cache_data
def db_get_sectors() -> dict:
    return database.get_sectors()

def chunk_text_by_words(text, chunk_size=CHUNK_WORD_SIZE, overlap=CHUNK_WORD_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def _create_context_from_notes(notes_text):
    if not notes_text or not notes_text.strip(): return ""
    questions = re.findall(r"(\*\*.*?\*\*)", notes_text)
    if not questions: return ""
    last_question = questions[-1]
    answer_match = re.search(re.escape(last_question) + r"(.*?)(?=\*\*|$)", notes_text, re.DOTALL)
    last_answer = answer_match.group(1).strip() if answer_match else ""
    return f"-   **Last Question Processed:** {last_question}\n-   **Last Answer Provided:**\n{last_answer}".strip()

def validate_inputs(state: AppState) -> Optional[str]:
    # ... (Same as previous version)

def process_and_save_task(state: AppState, status_ui):
    start_time = time.time()
    notes_model = genai.GenerativeModel(AVAILABLE_MODELS[state.notes_model])
    refinement_model = genai.GenerativeModel(AVAILABLE_MODELS[state.refinement_model])
    transcription_model = genai.GenerativeModel(AVAILABLE_MODELS[state.transcription_model])
    
    status_ui.update(label="Step 1: Preparing Source Content...")
    raw_transcript, file_name = "", "Pasted Text"
    
    # --- AUDIO PROCESSING LOGIC ---
    if state.input_method == "Upload File" and state.uploaded_file:
        file_type, name = get_file_content(state.uploaded_file)
        file_name = name
        if file_type == "audio_file":
            status_ui.update(label="Step 1.1: Processing Audio...")
            audio_bytes = state.uploaded_file.getvalue()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_chunks = make_chunks(audio, 5 * 60 * 1000) # 5-minute chunks
            
            all_transcripts = []
            cloud_files_to_delete = []
            local_files_to_delete = []

            try:
                for i, chunk in enumerate(audio_chunks):
                    status_ui.update(label=f"Step 1.2: Transcribing audio chunk {i+1}/{len(audio_chunks)}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                        chunk.export(temp_chunk_file.name, format="wav")
                        local_files_to_delete.append(temp_chunk_file.name)
                        
                        cloud_file_ref = genai.upload_file(path=temp_chunk_file.name)
                        cloud_files_to_delete.append(cloud_file_ref.name)
                        
                        while cloud_file_ref.state.name == "PROCESSING":
                            time.sleep(5)
                            cloud_file_ref = genai.get_file(cloud_file_ref.name)
                        
                        if cloud_file_ref.state.name != "ACTIVE":
                            raise Exception(f"Audio chunk {i+1} failed to process in the cloud.")
                        
                        response = transcription_model.generate_content(["Transcribe this audio.", cloud_file_ref])
                        all_transcripts.append(response.text)
                
                raw_transcript = "\n\n".join(all_transcripts).strip()
            finally:
                for path in local_files_to_delete: os.remove(path)
                for name in cloud_files_to_delete: 
                    try: genai.delete_file(name)
                    except: pass # Ignore cleanup errors
        
        elif file_type is None or file_type.startswith("Error:"):
            raise ValueError(file_type or "Failed to read file content.")
        else:
            raw_transcript = file_type
    
    elif state.input_method == "Paste Text":
        raw_transcript = state.text_input

    if not raw_transcript: raise ValueError("Source content is empty.")

    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0

    if state.refinement_enabled:
        status_ui.update(label="Step 2: Refining Transcript...")
        # ... (refinement logic is the same)
    
    status_ui.update(label="Step 3: Generating Notes...")
    words = final_transcript.split()
    final_notes_content = ""

    # --- ORIGINAL PROMPT LOGIC REINSTATED ---
    if state.selected_meeting_type == "Expert Meeting" and len(words) > CHUNK_WORD_SIZE:
        chunks = chunk_text_by_words(final_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
        all_notes, context_package = [], ""
        prompt_base = EXPERT_MEETING_CHUNK_BASE_OPTION_2 if state.selected_note_style != "Option 1: Detailed & Strict" else EXPERT_MEETING_CHUNK_BASE

        for i, chunk in enumerate(chunks):
            status_ui.update(label=f"Step 3: Generating Notes (Chunk {i+1}/{len(chunks)})...")
            prompt_template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION
            prompt = prompt_template.format(base_instructions=prompt_base, chunk_text=chunk, context_package=context_package)
            
            response = notes_model.generate_content(prompt)
            all_notes.append(response.text)
            total_tokens += response.usage_metadata.total_token_count
            context_package = _create_context_from_notes(response.text)
        
        final_notes_content = "\n\n".join(all_notes)
    else:
        # For shorter transcripts or other meeting types, use single-shot generation
        prompt = get_dynamic_prompt(state, final_transcript)
        response = notes_model.generate_content(prompt)
        final_notes_content = response.text
        total_tokens += response.usage_metadata.total_token_count

    # ... (summary logic is the same)
    
    status_ui.update(label="Step 5: Saving to Database...")
    note_data = {
        # ... (same as before)
    }
    # ... (same as before, database save logic)
    
# --- 5. UI RENDERING FUNCTIONS ---
def on_sector_change():
    # ... (same as before)

def render_input_and_processing_tab(state: AppState):
    state.input_method = pills("Input Method", ["Paste Text", "Upload File"], index=["Paste Text", "Upload File"].index(state.input_method))
    
    if state.input_method == "Paste Text":
        state.text_input = st.text_area("Paste source transcript here:", height=250, key="text_input_main")
        state.uploaded_file = None
    else:
        # FIX: Audio types are now correctly accepted by the uploader
        state.uploaded_file = st.file_uploader("Upload a File (PDF, TXT, MP3, etc.)", type=['pdf', 'txt', 'mp3', 'm4a', 'wav', 'ogg', 'flac'])
        state.text_input = ""

    # ... (Rest of UI is the same as the fully-featured previous version)
    
def render_output_and_history_tab(state: AppState):
    # ... (same as before)

# --- 6. MAIN APPLICATION RUNNER ---
def run_app():
    # ... (same as before)

if __name__ == "__main__":
    run_app()

# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/

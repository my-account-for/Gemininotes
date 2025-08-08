# --- 1. IMPORTS ---
import streamlit as st
import google.generativeai as genai
import google.api_core.exceptions
import os
import io
import time
import tempfile
from datetime import datetime, date
import uuid
import threading
import queue
import traceback
from dotenv import load_dotenv
import PyPDF2
from pydub import AudioSegment
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from streamlit_pills import pills
from streamlit_ace import st_ace

# --- Local Imports ---
import database

# --- 2. CONSTANTS & CONFIG ---
load_dotenv()

# Configure the Gemini client safely
try:
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"]:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    else:
        # This will be caught by the main app runner
        raise ValueError("GEMINI_API_KEY not found in environment.")
except Exception as e:
    # Defer error display to the main app body
    st.session_state.config_error = f"🔴 Error configuring Google AI Client: {e}"

# Application Constants
MAX_PDF_MB = 25
MAX_AUDIO_MB = 200
CHUNK_WORD_SIZE = 4500
CHUNK_WORD_OVERLAP = 200

# Models & Prompts
AVAILABLE_MODELS = {"Gemini 1.5 Flash": "gemini-1.5-flash", "Gemini 1.5 Pro": "gemini-1.5-pro"}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]

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
-   Each bullet point should convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL SPECIFICS.**"""


# --- 3. STATE & DATA MODELS ---

class ProcessingStage(Enum):
    IDLE = "Idle"
    VALIDATING = "Validating Inputs"
    TRANSCRIBING = "Transcribing Audio"
    REFINING = "Refining Transcript"
    CHUNKING = "Chunking Long Text"
    GENERATING = "Generating Notes"
    SUMMARIZING = "Creating Summary"
    COMPLETE = "Complete"
    ERROR = "Error"

class ValidationError(Enum):
    MISSING_INPUT = "No input file or text provided."
    FILE_TOO_LARGE = "A file exceeds the size limit."

@dataclass
class ProcessingMetrics:
    start_time: float = 0.0
    progress_percent: float = 0.0
    status_message: str = "Starting..."
    token_usage: int = 0
    current_chunk: int = 0
    total_chunks: int = 1

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

@dataclass
class AppState:
    current_stage: ProcessingStage = ProcessingStage.IDLE
    error_details: Dict[str, str] = field(default_factory=dict)
    
    # User Config
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    selected_model: str = "Gemini 1.5 Pro"
    refinement_enabled: bool = True
    
    # Data & Results
    uploaded_files: List[Any] = field(default_factory=list)
    active_note_id: Optional[str] = None
    
    # Background Task Management
    task_id: Optional[str] = None
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    prompt_content: str = ""

# --- 4. BACKGROUND TASK MANAGER ---
class BackgroundTaskManager:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while True:
            task_id, task_func, args, kwargs = self.task_queue.get()
            try:
                result = task_func(*args, **kwargs)
                self.result_queue.put({'id': task_id, 'status': 'success', 'result': result})
            except Exception as e:
                self.result_queue.put({'id': task_id, 'status': 'error', 'error': str(e)})
            self.task_queue.task_done()

    def submit_task(self, task_func, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id

    def get_result(self) -> Optional[Dict]:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

# --- 5. CORE PROCESSING LOGIC ---

def process_single_file_task(state: AppState, uploaded_file: Any, progress_callback):
    start_time = time.time()
    model = genai.GenerativeModel(AVAILABLE_MODELS[state.selected_model])
    
    progress_callback(ProcessingStage.TRANSCRIBING, 0.1, f"Processing {uploaded_file.name}")
    raw_transcript, file_name = get_file_content(uploaded_file)
    if not raw_transcript or raw_transcript.startswith("Error:"):
        raise ValueError(f"Content extraction failed: {raw_transcript}")
    
    final_transcript = raw_transcript
    refined_transcript = None
    total_tokens = 0

    if state.refinement_enabled:
        progress_callback(ProcessingStage.REFINING, 0.3, "Refining transcript...")
        refine_prompt = f"Refine the following transcript. Correct spelling and grammar, and improve readability without changing the content.\n\n{raw_transcript}"
        response = model.generate_content(refine_prompt)
        refined_transcript = response.text
        final_transcript = refined_transcript
        total_tokens += response.usage_metadata.total_token_count

    words = final_transcript.split()
    chunks = [final_transcript]
    if len(words) > CHUNK_WORD_SIZE:
        progress_callback(ProcessingStage.CHUNKING, 0.5, "Chunking long document...")
        chunks = [ " ".join(words[i:i + CHUNK_WORD_SIZE]) for i in range(0, len(words), CHUNK_WORD_SIZE - CHUNK_WORD_OVERLAP) ]
    
    all_notes = []
    for i, chunk in enumerate(chunks):
        progress_callback(ProcessingStage.GENERATING, 0.6 + (0.3 * (i+1)/len(chunks)), f"Generating notes for chunk {i+1}/{len(chunks)}...")
        
        if state.selected_meeting_type == "Expert Meeting":
            prompt_base = EXPERT_MEETING_CHUNK_BASE_OPTION_2 if state.selected_note_style != "Option 1: Detailed & Strict" else EXPERT_MEETING_CHUNK_BASE
            prompt = f"{prompt_base}\n\n**MEETING TRANSCRIPT CHUNK:**\n{chunk}"
        else:
            prompt = f"Create notes for a '{state.selected_meeting_type}' meeting from the following chunk:\n\n{chunk}"
        
        response = model.generate_content(prompt)
        all_notes.append(response.text)
        total_tokens += response.usage_metadata.total_token_count

    final_notes_content = "\n\n---\n\n".join(all_notes)
    
    if state.selected_note_style == "Option 3: Less Verbose + Summary":
        progress_callback(ProcessingStage.SUMMARIZING, 0.9, "Generating executive summary...")
        summary_prompt = f"Based on the following detailed notes, create a concise executive summary:\n\n{final_notes_content}"
        response = model.generate_content(summary_prompt)
        final_notes_content += f"\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{response.text}"
        total_tokens += response.usage_metadata.total_token_count

    progress_callback(ProcessingStage.COMPLETE, 0.99, "Saving to database...")
    note_data = {
        'id': str(uuid.uuid4()),
        'created_at': datetime.now().isoformat(),
        'meeting_type': state.selected_meeting_type,
        'file_name': file_name,
        'content': final_notes_content,
        'raw_transcript': raw_transcript,
        'refined_transcript': refined_transcript,
        'token_usage': total_tokens,
        'processing_time': time.time() - start_time
    }
    database.save_note(note_data)
    return note_data

# --- 6. UTILITY & VALIDATION FUNCTIONS ---
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
        elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
            audio = AudioSegment.from_file(file_bytes)
            return f"Simulated transcription for '{name}' ({len(audio)/1000:.1f}s audio). Actual transcription via API is needed for full functionality.", name
    except Exception as e:
        return f"Error: Could not process file {name}. Details: {str(e)}", name
    return None, name

def validate_inputs_comprehensive(state: AppState) -> List[ValidationError]:
    errors = []
    if not state.uploaded_files:
        errors.append(ValidationError.MISSING_INPUT)
    for f in state.uploaded_files:
        if f.size > MAX_AUDIO_MB * 1024 * 1024:
            errors.append(ValidationError.FILE_TOO_LARGE)
            break
    return errors

# --- 7. UI RENDERING FUNCTIONS ---

@st.fragment
def update_progress_fragment():
    state = st.session_state.app_state
    result = st.session_state.task_manager.get_result()
    
    if result and result.get('id') == state.task_id:
        if result['status'] == 'success':
            state.current_stage = ProcessingStage.COMPLETE
            st.session_state.active_note_id = result['result']['id']
        else:
            state.current_stage = ProcessingStage.ERROR
            state.error_details = {'message': result['error']}
        state.task_id = None
        st.rerun()
    
    if state.current_stage not in [ProcessingStage.IDLE, ProcessingStage.COMPLETE, ProcessingStage.ERROR]:
        st.progress(state.metrics.progress_percent, text=f"{state.metrics.status_message} ({state.metrics.duration:.1f}s)")
        time.sleep(1)
        st.rerun()

def render_input_tab():
    state = st.session_state.app_state
    st.subheader("1. Configure Meeting")
    
    state.selected_meeting_type = pills(
        "Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type)
    )

    with st.expander("⚙️ Advanced Settings"):
        if state.selected_meeting_type == "Expert Meeting":
            state.selected_note_style = st.selectbox("Note Style", EXPERT_MEETING_OPTIONS, index=EXPERT_MEETING_OPTIONS.index(state.selected_note_style))
        state.selected_model = st.selectbox("AI Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.selected_model))
        state.refinement_enabled = st.toggle("Enable Transcript Refinement", value=state.refinement_enabled)

    st.subheader("2. Upload Content")
    state.uploaded_files = st.file_uploader(
        "📎 Drag & Drop Files (PDF, TXT, MP3, etc.)",
        type=['pdf', 'wav', 'mp3', 'm4a', 'txt'],
        accept_multiple_files=True
    )
    
    st.subheader("3. Custom Prompt Override (Optional)")
    state.prompt_content = st_ace(
        value="# Edit here for a full override of the generated prompt.",
        language='markdown', theme='tomorrow_night', height=150
    )

def render_processing_tab():
    state = st.session_state.app_state
    
    if state.current_stage == ProcessingStage.IDLE:
        st.info("Configure your inputs in the '📝 Input' tab, then start processing here.")
        errors = validate_inputs_comprehensive(state)
        
        if errors:
            for error in errors: st.warning(f"⚠️ {error.value}")
        
        if st.button("🚀 Start Generating Notes", type="primary", use_container_width=True, disabled=bool(errors)):
            state.current_stage = ProcessingStage.VALIDATING
            state.metrics = ProcessingMetrics(start_time=time.time())
            
            def progress_callback(stage, percent, message):
                state.current_stage = stage
                state.metrics.progress_percent = percent
                state.metrics.status_message = message
            
            task_id = st.session_state.task_manager.submit_task(
                process_single_file_task, state=state, uploaded_file=state.uploaded_files[0], progress_callback=progress_callback
            )
            state.task_id = task_id
            st.rerun()
            
    elif state.current_stage == ProcessingStage.COMPLETE:
        st.success("✅ Processing Complete! View results in the '📄 Output' tab.")
        if st.button("Process Another Batch"):
            state.current_stage = ProcessingStage.IDLE
            state.uploaded_files, st.session_state.active_note_id = [], None
            st.rerun()

    elif state.current_stage == ProcessingStage.ERROR:
        st.error(f"❌ Processing Failed: {state.error_details.get('message', 'Unknown error.')}")
        if st.button("Try Again"):
            state.current_stage = ProcessingStage.IDLE
            st.rerun()
            
    else: 
        update_progress_fragment()

def render_output_tab():
    state = st.session_state.app_state
    notes = database.get_all_notes()
    if not notes:
        st.info("No notes generated yet. Process a file from the '⚙️ Processing' tab.")
        return
        
    if not state.active_note_id or not any(n['id'] == state.active_note_id for n in notes):
        state.active_note_id = notes[0]['id']

    active_note = next((n for n in notes if n['id'] == state.active_note_id), notes[0])
    
    st.subheader(f"📄 Output for: {active_note['file_name']}")
    st.caption(f"ID: {active_note['id']} | Generated: {datetime.fromisoformat(active_note['created_at']).strftime('%Y-%m-%d %H:%M')}")
    
    with st.expander("Export Options"):
        export_format = st.selectbox("Export Format", ["Markdown", "PDF", "Word", "JSON"])
        if st.button(f"📤 Export as {export_format}", use_container_width=True):
            st.toast(f"Simulating export to {export_format}...", icon="✅")

    edited_content = st_ace(value=active_note['content'], language='markdown', theme='github', height=600, key=f"output_ace_{active_note['id']}")
    # In a real app, a "Save Changes" button here would update the database.

def render_analytics_tab():
    st.subheader("📊 Analytics & History")
    
    summary = database.get_analytics_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Notes", summary['total_notes'])
    c2.metric("Avg. Time / Note", f"{summary['avg_time']:.1f}s")
    c3.metric("Total Tokens (Est.)", f"{summary['total_tokens']:,}")

    st.subheader("📚 Note History")
    c1, c2, c3 = st.columns([2, 1, 1])
    search_query = c1.text_input("🔍 Search notes by content or filename...")
    date_filter = c2.date_input("Filter by date", value=(), key="date_filter_input") # Using a key to avoid state issues
    type_filter = c3.multiselect("Filter by type", MEETING_TYPES)
    
    notes = database.get_all_notes(search_query, date_filter, type_filter)
    st.write(f"Found {len(notes)} notes.")
    
    for note in notes:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**File:** `{note['file_name']}`")
                st.caption(f"ID: {note['id']} | Generated: {datetime.fromisoformat(note['created_at']).strftime('%Y-%m-%d %H:%M')}")
            with col2:
                if st.button("View in Output", key=f"view_{note['id']}", use_container_width=True):
                    st.session_state.app_state.active_note_id = note['id']
                    st.toast(f"Loaded '{note['file_name']}'. Go to '📄 Output' tab.")
            
            with st.expander("Preview"):
                st.markdown(note['content'])
    
# --- 8. MAIN APPLICATION RUNNER ---
def run_app():
    st.set_page_config(page_title="SynthNotes AI 🚀", layout="wide")
    st.title("SynthNotes AI 🚀 (Modernized)")
    
    # Critical: Check for config errors before doing anything else
    if "config_error" in st.session_state:
        st.error(st.session_state.config_error)
        st.stop()
        
    try:
        database.init_db()
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
        if "task_manager" not in st.session_state:
            st.session_state.task_manager = BackgroundTaskManager()

        if "onboarding_complete" not in st.session_state:
            st.success("🎉 Welcome to the new SynthNotes AI!")
            st.info("This modernized interface uses tabs for navigation. Start by configuring your job in the 'Input' tab.")
            st.session_state.onboarding_complete = True

        tabs = st.tabs(["📝 Input", "⚙️ Processing", "📄 Output", "📊 Analytics & History"])
        
        with tabs[0]: render_input_tab()
        with tabs[1]: render_processing_tab()
        with tabs[2]: render_output_tab()
        with tabs[3]: render_analytics_tab()
    
    except Exception as e:
        st.error(f"A critical error occurred in the application: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()

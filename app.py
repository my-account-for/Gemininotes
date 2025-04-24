# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
import tempfile
from datetime import datetime # For history timestamp & default date
from dotenv import load_dotenv
import PyPDF2
import docx # Still needed for DOCX fallback
import re # For cleaning filename suggestions
from pydub import AudioSegment # <-- ADDED for audio chunking
from pydub.utils import make_chunks # <-- ADDED for audio chunking

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
# (CSS remains the same - omitted for brevity)
st.markdown("""
<style>
    /* ... CSS Styles ... */
    /* Overall App Background */
    .stApp { background: linear-gradient(to bottom right, #F0F2F6, #FFFFFF); }
    /* Main content area */
    .main .block-container { padding: 2rem; max-width: 1000px; margin: auto; }
    /* General Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"][style*="border"] {
         background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 0.75rem;
         padding: 1.5rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; }
    /* Headers */
    h1 { color: #111827; font-weight: 700; text-align: center; margin-bottom: 0.5rem; }
    h2, h3 { color: #1F2937; font-weight: 600; border-bottom: 1px solid #E5E7EB; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    /* App Subtitle - Adjust selector index if layout changes */
    .main .block-container > div:nth-child(3) > div > div > div > p { text-align: center; color: #4B5563; font-size: 1.1rem; margin-bottom: 2rem; }
    /* Input Widgets */
    .stTextInput textarea, .stFileUploader div[data-testid="stFileUploaderDropzone"], .stTextArea textarea {
        border-radius: 0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05); transition: all 0.2s ease; }
    .stTextInput textarea:focus, .stFileUploader div[data-testid="stFileUploaderDropzone"]:focus-within, .stTextArea textarea:focus {
        border-color: #007AFF; box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05), 0 0 0 3px rgba(0, 122, 255, 0.2);
        background-color: #FFFFFF; }
    .stFileUploader p { font-size: 0.95rem; color: #4B5563; }
    /* Radio Buttons */
    div[role="radiogroup"] > label { background-color: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 0.5rem;
        padding: 0.6rem 1rem; margin-right: 0.5rem; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        display: inline-block; margin-bottom: 0.5rem; }
    div[role="radiogroup"] label:hover { border-color: #9CA3AF; }
    div[role="radiogroup"] input[type="radio"]:checked + div { background-color: #EFF6FF; border-color: #007AFF; color: #005ECB;
        font-weight: 500; box-shadow: 0 1px 3px rgba(0, 122, 255, 0.1); }
    /* Checkbox styling */
    .stCheckbox { margin-top: 1rem; padding: 0.5rem; background-color: #F9FAFB; border-radius: 0.5rem; }
    .stCheckbox label span { font-weight: 500; color: #374151; }
    /* Selectbox Styling */
    .stSelectbox > div { border-radius: 0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB; }
    .stSelectbox > div:focus-within { border-color: #007AFF; box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.2); }
    /* Button Styling */
    .stButton > button { border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: 600; transition: all 0.2s ease-in-out; border: none; width: 100%; }
    .stButton > button[kind="primary"] { background-color: #007AFF; color: white; box-shadow: 0 4px 6px rgba(0, 122, 255, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08); }
    .stButton > button[kind="primary"]:hover { background-color: #005ECB; box-shadow: 0 7px 14px rgba(0, 122, 255, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08); transform: translateY(-1px); }
    .stButton > button[kind="primary"]:focus { box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.4); outline: none; }
    .stButton > button:disabled, .stButton > button[kind="primary"]:disabled { background-color: #D1D5DB; color: #6B7280; box-shadow: none; transform: none; cursor: not-allowed; }
     /* Secondary Button styling for Clear */
    .stButton>button[type="secondary"], .stButton>button.secondary-button { background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB;
        width: auto; padding: 0.5rem 1rem; margin-right: 0.5rem; font-weight: 500; }
    .stButton>button[type="secondary"]:hover, .stButton>button.secondary-button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
     /* Download Buttons */
    .stDownloadButton > button { border-radius: 0.5rem; padding: 0.6rem 1.2rem; font-weight: 500; background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB; transition: background-color 0.2s ease-in-out; width: auto; margin-top: 0; margin-right: 0.5rem;}
    .stDownloadButton > button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
    /* Output Area Styling */
    .output-container { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1.5rem; margin-top: 1.5rem; min-height: 150px; }
    .output-container .stMarkdown { background-color: transparent; border: none; padding: 0; color: #374151; font-size: 1rem; line-height: 1.6; }
    .output-container .stMarkdown h3, .output-container .stMarkdown h4, .output-container .stMarkdown strong { color: #111827; font-weight: 600; }
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem; }
    .output-container .initial-prompt { color: #6B7280; font-style: italic; text-align: center; padding-top: 2rem; }
    /* Prompt Edit Area */
    #prompt-edit-area textarea { font-family: monospace; font-size: 0.9rem; line-height: 1.4; background-color: #FDFDFD; }
    /* History Styling */
    .history-entry { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
    .history-entry:last-child { border-bottom: none; }
    .history-entry pre { background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.25rem; max-height: 150px; overflow-y: auto; }
    /* Footer */
    footer { text-align: center; color: #9CA3AF; font-size: 0.8rem; padding-top: 2rem; padding-bottom: 1rem; }
    footer a { color: #6B7280; text-decoration: none; }
    footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Define Available Models & Meeting Types ---
AVAILABLE_MODELS = {
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
# Ensure default models exist in the available list
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"

DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]

# --- Sector-Specific Topics ---
SECTOR_OPTIONS = ["Other / Manual Topics", "IT Services", "QSR"]
DEFAULT_SECTOR = SECTOR_OPTIONS[0]
SECTOR_TOPICS = {
    "IT Services": """Future investments related comments:
- Capital allocation:
- Talent supply chain related comments:
- Org structure change:
- Other comments:
Short-term comments:
- Guidance:
- Order booking:
- Impact of macro slowdown:
- Signal:""",
    "QSR": """Customer proposition:
Menu strategy (Includes: new product launches, etc):
Operational update (Includes: SSSG, SSTG, Price hike, etc):
Unit economics:
Store opening:"""
}

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_KEY)
    # Generation configs will be used in generate_content calls
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"} # Chunking might benefit from slightly lower temp if needed
    refinement_gen_config = {"temperature": 0.3, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, # Now used to track the *original* file info if needed, chunks handled separately
    'add_context_enabled': False,
    'selected_notes_model_display_name': DEFAULT_NOTES_MODEL_NAME,
    'selected_transcription_model_display_name': DEFAULT_TRANSCRIPTION_MODEL_NAME,
    'selected_refinement_model_display_name': DEFAULT_REFINEMENT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'selected_sector': DEFAULT_SECTOR,
    'earnings_call_topics': '',
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': [] # <-- NEW: To track chunk files for cleanup
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    # (No changes needed)
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def update_topic_template():
    # (No changes needed)
    selected_sector = st.session_state.selected_sector
    if selected_sector in SECTOR_TOPICS:
        st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
    else:
        st.session_state.earnings_call_topics = ""

# --- Prompts ---
def create_expert_meeting_prompt(transcript, context=None):
    # (No changes needed)
    core_prompt = """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.
- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.
  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).
  - **Answers:** Use bullet points directly below the question. Each bullet MUST be a complete sentence with one distinct fact. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or action items.

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- Be clear and concise.
- Include ONLY information present in the transcript.
- If a section (like Opening Overview) isn't present, OMIT it.
---"""
    final_prompt_elements = [core_prompt]
    if transcript: final_prompt_elements.append(f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---")
    if context: final_prompt_elements.append(f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{context}\n---")
    final_prompt_elements.append("\n**GENERATED NOTES (Q&A Format):**\n")
    return "\n".join(final_prompt_elements)

def create_earnings_call_prompt(transcript, user_topics_text=None, context=None):
    # (No changes needed)
    topic_instructions = ""
    if user_topics_text and user_topics_text.strip():
        formatted_topics = []
        for line in user_topics_text.strip().split('\n'):
             trimmed_line = line.strip()
             if trimmed_line and not trimmed_line.startswith(('-', '*')):
                 formatted_topics.append(f"- **{trimmed_line}**")
             else:
                 formatted_topics.append(line)
        topic_list_str = "\n".join(formatted_topics)
        topic_instructions = (
            f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided in the structure below. Fill in the details under the most relevant heading and sub-point:\n{topic_list_str}\n\n"
            f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered above)\n\n"
            f"Place details under the most appropriate heading. If a topic isn't discussed, state 'Not discussed' under that heading."
        )
    else:
        topic_instructions = (
            f"Since no specific topics were provided, first identify the logical main themes discussed (e.g., Financials, Strategy, Outlook, Q&A). Use these themes as **bold headings**.\n"
            f"Include a final mandatory section:\n- **Other Key Points** (for important info not covered in main themes)\n\n"
            f"Place details under the most appropriate heading."
        )
    prompt_parts = [
        "You are an expert AI assistant creating DETAILED notes from an earnings call transcript for an investment firm.",
        "Output MUST be comprehensive, factual notes, capturing all critical financial and strategic information.",
        "**Formatting Requirements (Mandatory):**\n- US$ for dollars (US$2.5M), % for percentages.\n- State comparison periods (+5% YoY, -2% QoQ).\n- Represent fiscal periods accurately (Q3 FY25).\n- Use common abbreviations (CEO, KPI).\n- Use bullet points under headings.\n- Each bullet = complete sentence with distinct info.\n- Capture ALL numbers, names, data accurately.\n- Use quotes \"\" for significant statements.\n- DO NOT summarize or interpret unless part of the structure.",
        "\n**Note Structure:**",
        "- **Call Participants:** (List names/titles or 'Not specified')",
        topic_instructions,
        "\n**CRITICAL:** Ensure accuracy and adhere strictly to structure and formatting.", "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL NOTES:**\n")
    return "\n".join(filter(None, prompt_parts))

def create_docx(text):
    # (No changes needed)
    document = docx.Document(); [document.add_paragraph(line) for line in text.split('\n')]; buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()

def get_current_input_data():
    # (No changes needed)
    input_type = st.session_state.input_method_radio
    transcript = None; audio_file = None
    if input_type == "Paste Text": transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file is not None:
            try: transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
            except Exception as e: st.session_state.error_message = f"Error processing PDF: {e}"; transcript = None
    elif input_type == "Upload Audio": audio_file = st.session_state.audio_uploader
    return input_type, transcript, audio_file

def get_prompt_display_text():
    """Generates the appropriate prompt text for display/editing."""
    # (No changes needed in this function's logic)
    meeting_type = st.session_state.selected_meeting_type
    display_text = ""
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        if meeting_type == "Earnings Call":
            user_topics_text_for_display = str(st.session_state.get('earnings_call_topics', "") or "")
            prompt_func = lambda t, c: create_earnings_call_prompt(t, user_topics_text_for_display, c)
        elif meeting_type == "Expert Meeting":
            prompt_func = create_expert_meeting_prompt
        else:
            st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
            return "Error generating prompt preview."
        placeholder = "[TRANSCRIPT WILL APPEAR HERE]"
        try:
             base_prompt = prompt_func(placeholder, temp_context)
             if input_type == "Upload Audio":
                 display_text = ("# NOTE FOR AUDIO: 3-step process (Chunked Transcribe -> Refine -> Notes).\n" # <-- Updated note
                                 "# This prompt is used in Step 3 with the *refined* transcript.\n"
                                 "####################################\n\n" + base_prompt)
             else:
                 display_text = base_prompt
        except Exception as e:
             st.error(f"Error generating prompt preview: {e}")
             display_text = f"# Error generating preview: Review inputs/prompt structure. Details logged if possible."
    elif meeting_type == "Custom":
         audio_note = ("\n# NOTE FOR AUDIO: If using audio, the system will first chunk and transcribe,\n"
                       "# then *refine* the full transcript (speaker ID, translation, corrections).\n" # <-- Updated note
                       "# Your custom prompt below will receive this *refined transcript* as the primary text input.\n"
                       "# Design your prompt accordingly.\n")
         default_custom = "# Enter your custom prompt here..."
         current_or_default = st.session_state.current_prompt_text or default_custom
         display_text = current_or_default + (audio_note if st.session_state.input_method_radio == 'Upload Audio' else "")
    return display_text

def clear_all_state():
    # Reset selections and inputs
    # (No changes needed except adding new state var)
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
    st.session_state.input_method_radio = 'Paste Text'
    st.session_state.text_input = ""
    st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None
    st.session_state.context_input = ""
    st.session_state.add_context_enabled = False
    st.session_state.selected_sector = DEFAULT_SECTOR
    st.session_state.earnings_call_topics = ""
    st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_enabled = False
    # Reset outputs
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.processing = False
    st.session_state.suggested_filename = None
    st.session_state.uploaded_audio_info = None
    st.session_state.history = []
    st.session_state.raw_transcript = None
    st.session_state.refined_transcript = None
    st.session_state.processed_audio_chunk_references = [] # <-- RESET NEW STATE
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type):
    # (No changes needed)
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "")
        filename_prompt = (f"Suggest filename: YYYYMMDD_ClientOrTopic_MeetingType. Date={today_date}. Type='{mt_cleaned}'. Max 3 words topic. Output ONLY filename.\nNOTES:{notes_content[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if s_name: st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename suggestion empty/invalid.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    # (No changes needed)
    if not notes: return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"timestamp": timestamp, "notes": notes}
        current_history = st.session_state.get('history', [])
        if not isinstance(current_history, list):
            st.warning("History state was not a list, resetting.", icon="⚠️"); current_history = []
        current_history.insert(0, new_entry)
        st.session_state.history = current_history[:3]
    except Exception as e: st.warning(f"⚠️ Error updating note history: {e}", icon="❗")

def restore_note_from_history(index):
    # (No changes needed)
    if 0 <= index < len(st.session_state.history): entry = st.session_state.history[index]; st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]; \
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None; st.session_state.error_message = None; st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
# (No changes needed in the UI layout itself)
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True)
        with col1b:
            st.subheader("AI Model Selection")
            st.selectbox("Notes Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model used for the final note generation (Step 3).")
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model used for Audio Transcription (Step 1 - per chunk). Faster models recommended.")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model used for Audio Refinement (Step 2 - full transcript). More capable models recommended.")

    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

    st.divider(); st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider(); col3a, col3b = st.columns(2); selected_mt = st.session_state.selected_meeting_type

    with col3a: # Topics / Context
        if selected_mt == "Earnings Call":
            st.selectbox("Select Sector (for Topic Template):", options=SECTOR_OPTIONS, key="selected_sector", on_change=update_topic_template)
            st.text_area("Earnings Call Topics (Edit below):", key="earnings_call_topics", height=150, placeholder="Enter topics manually or edit loaded template...")
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=75, key="context_input", placeholder="Company Name, Ticker...")
    with col3b: # View/Edit Prompt Checkbox
        if selected_mt != "Custom": st.checkbox("View/Edit Final Notes Prompt", key="view_edit_prompt_enabled")

# Prompt Area (Conditional)
# (No changes needed)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and selected_mt != "Custom") or (selected_mt == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Final Notes Prompt Preview/Editor" if selected_mt != "Custom" else "Custom Final Notes Prompt (Required)"
        st.subheader(prompt_title)
        display_prompt_text = get_prompt_display_text()
        prompt_value = st.session_state.current_prompt_text if st.session_state.current_prompt_text else display_prompt_text
        st.text_area(label="Prompt Text:", value=prompt_value, key="current_prompt_text", height=350, label_visibility="collapsed", help="This prompt is used for the *final* step of generating notes.")

# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# Output Section
output_container = st.container(border=True)
with output_container:
    # (No changes needed in output display logic)
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None # Clear after showing
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        if st.session_state.raw_transcript:
            with st.expander("View Raw Transcript (Step 1 Output - Combined from Chunks)"): # Clarified label
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
        if st.session_state.refined_transcript:
             with st.expander("View Refined Transcript (Step 2 Output)"):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_to_use)
        st.write("") # Spacer
        dl_cols = st.columns(3)
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"
        if st.session_state.suggested_filename: fname_base = st.session_state.suggested_filename
        else: fname_base = default_fname
        with dl_cols[0]: st.download_button(label="⬇️ Notes (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]: st.download_button(label="⬇️ Notes (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.refined_transcript:
                refined_fname_base = fname_base.replace("_notes", "_refined_transcript") if "_notes" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the speaker-diarized and corrected transcript from Step 2.")
            else: st.button("Refined Tx N/A", disabled=True, use_container_width=True, help="Refined transcript is only available after successful audio processing.")
    elif not st.session_state.processing:
        st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# History Section
# (No changes needed)
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                st.code(entry['notes'][:300] + ("..." if len(entry['notes']) > 300 else ""), language=None)
                st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                if i < len(st.session_state.history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    # Reset state before starting
    st.session_state.processing = True
    st.session_state.generating_filename = False
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.suggested_filename = None
    st.session_state.raw_transcript = None
    st.session_state.refined_transcript = None
    st.session_state.processed_audio_chunk_references = [] # <-- RESET CHUNK LIST
    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    # --- MOVED: Cleanup list initialization ---
    processed_audio_chunk_references = [] # Local list for cleanup within this run

    # --- Use st.status for progress updates ---
    with st.status("🚀 Initializing process...", expanded=True) as status:
        try: # Outer try-finally for overall process and cleanup
            # State & Input Retrieval
            status.update(label="⚙️ Reading inputs and settings...")
            meeting_type = st.session_state.selected_meeting_type
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            user_prompt_text = st.session_state.current_prompt_text
            general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
            earnings_call_topics_text = st.session_state.earnings_call_topics.strip() if meeting_type == "Earnings Call" else ""
            actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

            # Initialize Models
            status.update(label="🧠 Initializing AI models...")
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            # Initialize Transcript Variable
            final_transcript_for_notes = transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            # Validation
            status.update(label="✔️ Validating inputs...")
            if actual_input_type == "Paste Text" and not final_transcript_for_notes: raise ValueError("Text input is empty.")
            elif actual_input_type == "Upload PDF" and not final_transcript_for_notes: raise ValueError("PDF processing failed or returned empty text.")
            elif actual_input_type == "Upload Audio" and not audio_file_obj: raise ValueError("No audio file uploaded.")
            if meeting_type == "Custom" and not user_prompt_text.strip(): raise ValueError("Custom Prompt is required but empty.")

            # ==============================================
            # --- STEP 1 & 2: Audio Processing Pipeline ---
            # ==============================================
            if actual_input_type == "Upload Audio":
                st.session_state.uploaded_audio_info = audio_file_obj # Store original info if needed

                # --- Load Audio with pydub ---
                status.update(label=f"🔊 Loading audio file '{audio_file_obj.name}'...")
                audio_bytes = audio_file_obj.getvalue()
                audio_format = os.path.splitext(audio_file_obj.name)[1].lower().replace('.', '')
                # Common pydub format mappings (add more if needed)
                if audio_format == 'm4a': audio_format = 'mp4'
                elif audio_format == 'ogg': audio_format = 'ogg' # Or 'opus' if specifically opus encoded
                elif audio_format == 'aac': audio_format = 'aac'

                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                except Exception as audio_load_err:
                     # Give a more helpful error if ffmpeg is likely missing
                    if "ffmpeg" in str(audio_load_err).lower() or "Couldn't find ffmpeg or avconv" in str(audio_load_err):
                        raise ValueError(f"❌ **Error:** Could not load audio. `ffmpeg` or `libav` might be missing from your system or PATH. Please install it. (Original error: {audio_load_err})")
                    else:
                        raise ValueError(f"❌ Could not load audio file using pydub (format: {audio_format}). Error: {audio_load_err}")

                # --- Chunking ---
                # Aim for ~35 minutes per chunk (adjust as needed)
                chunk_length_ms = 35 * 60 * 1000
                chunks = make_chunks(audio, chunk_length_ms)
                num_chunks = len(chunks)
                status.update(label=f"🔪 Splitting audio into {num_chunks} chunk(s)...")

                all_transcripts = []
                # --- Process Each Chunk ---
                for i, chunk in enumerate(chunks):
                    chunk_num = i + 1
                    status.update(label=f"🔄 Processing Chunk {chunk_num}/{num_chunks}...")

                    # Export chunk to a temporary file (use WAV for compatibility, FLAC preferred if supported well)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                        chunk.export(temp_chunk_file.name, format="wav")
                        temp_chunk_path = temp_chunk_file.name

                    chunk_file_ref = None # Reset for each chunk
                    try:
                        # Upload Chunk
                        status.update(label=f"☁️ Uploading Chunk {chunk_num}/{num_chunks}...")
                        chunk_display_name = f"chunk_{chunk_num}_of_{num_chunks}_{int(time.time())}_{audio_file_obj.name}"
                        chunk_file_ref = genai.upload_file(path=temp_chunk_path, display_name=chunk_display_name)
                        processed_audio_chunk_references.append(chunk_file_ref) # Add to list for potential cleanup later

                        # Poll for Chunk Readiness
                        status.update(label=f"⏳ Waiting for Chunk {chunk_num}/{num_chunks} to process...")
                        polling_start = time.time()
                        while chunk_file_ref.state.name == "PROCESSING":
                            if time.time() - polling_start > 600: # 10 min timeout per chunk
                                raise TimeoutError(f"Audio processing timed out for chunk {chunk_num}.")
                            time.sleep(5) # Poll more frequently for chunks
                            chunk_file_ref = genai.get_file(chunk_file_ref.name)
                        if chunk_file_ref.state.name != "ACTIVE":
                            raise Exception(f"Audio chunk {chunk_num} processing failed. Final state: {chunk_file_ref.state.name}")

                        # Transcribe Chunk
                        status.update(label=f"✍️ Step 1: Transcribing Chunk {chunk_num}/{num_chunks} using {st.session_state.selected_transcription_model_display_name}...")
                        t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                        t_response = transcription_model.generate_content(
                            [t_prompt, chunk_file_ref],
                            generation_config=transcription_gen_config,
                        )

                        if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                            all_transcripts.append(t_response.text.strip())
                            status.update(label=f"✍️ Transcription complete for Chunk {chunk_num}/{num_chunks}!")
                        elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                            raise Exception(f"Transcription blocked for chunk {chunk_num}: {t_response.prompt_feedback.block_reason}")
                        else:
                            # Don't raise an error for empty transcript, just add empty string or warning
                            st.warning(f"⚠️ Transcription for chunk {chunk_num} returned empty response. Skipping.", icon="🤔")
                            all_transcripts.append("") # Append empty string to maintain order if needed

                    except Exception as chunk_err:
                        # Clean up the *current* chunk's temp file if it exists
                        if os.path.exists(temp_chunk_path): os.remove(temp_chunk_path)
                        # Re-raise the error to stop processing
                        raise Exception(f"❌ Error processing chunk {chunk_num}: {chunk_err}") from chunk_err
                    finally:
                        # --- IMPORTANT: Clean up the temp chunk file ---
                        if os.path.exists(temp_chunk_path): os.remove(temp_chunk_path)
                        # --- IMPORTANT: Clean up the *uploaded* chunk file from cloud ---
                        # We do this after *each* chunk is processed successfully or fails,
                        # using the main finally block's cleanup is also an option, but immediate cleanup is safer.
                        if chunk_file_ref:
                            try:
                                status.update(label=f"🗑️ Cleaning up cloud file for chunk {chunk_num}...")
                                genai.delete_file(chunk_file_ref.name)
                                # Remove from the list *if deleted successfully* here, otherwise outer finally handles it
                                if chunk_file_ref in processed_audio_chunk_references:
                                    processed_audio_chunk_references.remove(chunk_file_ref)
                            except Exception as del_err:
                                st.warning(f"⚠️ Failed to delete cloud file for chunk {chunk_num} ({chunk_file_ref.name}): {del_err}. Will retry cleanup later.", icon="🗑️")


                # --- Combine Transcripts ---
                status.update(label="🧩 Combining chunk transcripts...")
                st.session_state.raw_transcript = "\n\n".join(all_transcripts) # Join with double newline
                final_transcript_for_notes = st.session_state.raw_transcript
                status.update(label="✅ Step 1: Full Transcription Complete!")

                # --- Step 2: Refinement (Operates on the *combined* transcript) ---
                if final_transcript_for_notes: # Only refine if we got some transcript back
                    try:
                        status.update(label=f"🧹 Step 2: Refining combined transcript using {st.session_state.selected_refinement_model_display_name}...")
                        refinement_prompt = f"""Please refine the following raw audio transcript: ... [Refinement Prompt Content - same as before] ... **Refined Transcript:** """ # Keep prompt content
                        refinement_prompt = f"""Please refine the following raw audio transcript:

                        **Raw Transcript:**
                        ```
                        {st.session_state.raw_transcript}
                        ```

                        **Instructions:**
                        1.  **Identify Speakers:** Assign consistent labels (e.g., Speaker 1, Speaker 2). Place the label on a new line before the speaker's turn.
                        2.  **Translate to English:** Convert any non-English speech found within the transcript to English, ensuring it fits naturally within the conversation.
                        3.  **Correct Errors:** Fix spelling mistakes and grammatical errors. Use the overall conversation context to correct potentially misheard words or phrases.
                        4.  **Format:** Ensure clear separation between speaker turns using the speaker labels. Maintain the original conversational flow and content.
                        5.  **Output:** Provide *only* the refined, speaker-diarized, translated, and corrected transcript text. Do not add any introduction, summary, or commentary before or after the transcript.

                        **Additional Context (Optional - use for understanding terms, names, etc.):**
                        {general_context if general_context else "None provided."}

                        **Refined Transcript:**
                        """
                        r_response = refinement_model.generate_content(
                            refinement_prompt,
                            generation_config=refinement_gen_config,
                        )
                        if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                            st.session_state.refined_transcript = r_response.text.strip()
                            final_transcript_for_notes = st.session_state.refined_transcript # Use refined version for notes
                            status.update(label="🧹 Step 2: Refinement complete!")
                        elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                            st.warning(f"⚠️ Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Refinement blocked. Proceeding with raw transcript.")
                        else:
                            st.warning("🤔 Refinement step returned empty response. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Refinement failed. Proceeding with raw transcript.")
                    except Exception as refine_err:
                         st.warning(f"❌ Error during Step 2 (Refinement): {refine_err}. Using raw transcript for notes.", icon="⚠️")
                         status.update(label="⚠️ Refinement error. Proceeding with raw transcript.")
                else:
                    status.update(label="⚠️ Skipping Refinement (Step 2) as raw transcript is empty.")


            # =============================
            # --- STEP 3: Generate Notes ---
            # =============================
            if not final_transcript_for_notes:
                 raise ValueError("No transcript available (from text, PDF, or audio processing) to generate notes.")

            # --- Determine Final Prompt Text ---
            status.update(label="📝 Preparing final prompt for note generation...")
            final_prompt_for_api = None
            api_payload_parts = []
            # (Prompt creation logic remains the same)
            if meeting_type == "Custom":
                final_prompt_for_api = user_prompt_text
                api_payload_parts = [final_prompt_for_api, f"\n\n--- TRANSCRIPT START ---\n{final_transcript_for_notes}\n--- TRANSCRIPT END ---"]
            elif meeting_type == "Expert Meeting":
                final_prompt_for_api = create_expert_meeting_prompt(final_transcript_for_notes, general_context)
                api_payload_parts = [final_prompt_for_api]
            elif meeting_type == "Earnings Call":
                 final_prompt_for_api = create_earnings_call_prompt(final_transcript_for_notes, earnings_call_topics_text, general_context)
                 api_payload_parts = [final_prompt_for_api]
            else: raise ValueError(f"Invalid meeting type '{meeting_type}' for prompt generation.")
            if not final_prompt_for_api: raise ValueError("Failed to determine the final prompt for note generation.")

            # --- Generate Notes API Call ---
            try:
                status.update(label=f"✨ Step 3: Generating notes using {st.session_state.selected_notes_model_display_name}...")
                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=main_gen_config,
                )
                # (Response handling remains the same)
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    st.session_state.generated_notes = response.text.strip()
                    st.session_state.edited_notes_text = st.session_state.generated_notes
                    add_to_history(st.session_state.generated_notes)
                    st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                    status.update(label="✅ Notes generated successfully!", state="complete")
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    st.session_state.error_message = f"⚠️ Note generation blocked: {response.prompt_feedback.block_reason}."
                    status.update(label=f"❌ Blocked: {response.prompt_feedback.block_reason}", state="error")
                elif response:
                    st.session_state.error_message = "🤔 AI returned empty response during note generation."
                    status.update(label="❌ Error: AI returned empty response.", state="error")
                else:
                    st.session_state.error_message = "😥 Note generation failed (No response from API)."
                    status.update(label="❌ Error: Note generation failed (No response).", state="error")

            except Exception as api_call_err:
                st.session_state.error_message = f"❌ Error during Step 3 (API Call for Notes): {api_call_err}"
                status.update(label=f"❌ Error: {api_call_err}", state="error")

        except Exception as e:
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        # --- Outer FINALLY block: Handles final cleanup of any *remaining* chunk files ---
        finally:
            st.session_state.processing = False # Mark processing as finished
            # --- Cloud Audio Chunk Cleanup (Catch-all) ---
            # Use the local list captured at the start of the try block
            if processed_audio_chunk_references:
                st.toast(f"☁️ Performing final cleanup of {len(processed_audio_chunk_references)} remaining cloud chunk file(s)...", icon="🗑️")
                for file_ref in processed_audio_chunk_references:
                    try:
                        genai.delete_file(file_ref.name)
                    except Exception as final_cleanup_error:
                        # Log warning but continue trying to delete others
                        st.warning(f"Final cloud audio chunk cleanup failed for {file_ref.name}: {final_cleanup_error}", icon="⚠️")
                st.session_state.processed_audio_chunk_references = [] # Clear the state list too

            st.rerun() # Rerun to display final results or errors

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

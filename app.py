# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
import docx
import re
from pydub import AudioSegment
from pydub.utils import make_chunks
import copy

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Keep CSS concise for example

# --- Define Available Models & Meeting Types ---
AVAILABLE_MODELS = { # ... Models ...
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
DEFAULT_NOTES_MODEL_NAME = next((k for k, v in AVAILABLE_MODELS.items() if "2.5" in k),
                                next((k for k, v in AVAILABLE_MODELS.items() if "1.5 Pro" in k), list(AVAILABLE_MODELS.keys())[0]))
DEFAULT_TRANSCRIPTION_MODEL_NAME = next((k for k, v in AVAILABLE_MODELS.items() if "Flash" in k), list(AVAILABLE_MODELS.keys())[0])
DEFAULT_REFINEMENT_MODEL_NAME = DEFAULT_NOTES_MODEL_NAME

MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]
DEFAULT_EARNINGS_CALL_MODE = EARNINGS_CALL_MODES[0]
SECTOR_OPTIONS = ["Other / Manual Topics", "IT Services", "QSR"]
DEFAULT_SECTOR = SECTOR_OPTIONS[0]
SECTOR_TOPICS = { # ... Topics ...
    "IT Services": """...""", "QSR": """..."""
}

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try: # ... genai configuration ...
    genai.configure(api_key=API_KEY)
    # ... Generation Configs ...
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Prompts Definitions ---
PROMPTS = { # ... Prompts ...
    "Expert Meeting": { "...": "..." }, "Earnings Call": { "...": "..." }, "Custom": "..."
}
EXPERT_MEETING_OPTIONS = [ # ... Options ...
    "Option 1...", "Option 2...", "Option 3..."
]
DEFAULT_EXPERT_MEETING_OPTION = EXPERT_MEETING_OPTIONS[1]
EXPERT_MEETING_SUMMARY_PROMPT_KEY = "Summary Prompt (for Option 3)"

# --- Initialize Session State ---
default_state = { # ... State ...
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'add_context_enabled': False,
    'context_input': '',
     # Add other necessary state variables...
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value # Initialize state

# --- Helper Functions ---
# ... (All helper functions definitions remain the same) ...
def extract_text_from_pdf(pdf_file_stream): pass
def update_topic_template(): pass
def format_prompt_safe(prompt_template, **kwargs): pass
def create_docx(text): pass
def get_current_input_data(): pass
def validate_inputs(): pass
def handle_edit_toggle(): pass
def get_prompt_display_text(for_display_only=False): pass
def clear_all_state(): pass
def generate_suggested_filename(notes_content, meeting_type): pass
def add_to_history(notes): pass
def restore_note_from_history(index): pass


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Container 1: Meeting Setup ---
with st.container(border=True): # Line ~198
    # --- FIX: Ensure subsequent code is indented ---
    st.subheader("1. Meeting Setup")
    col1, col2 = st.columns(2)
    with col1:
        # Meeting Type
        st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                 index=MEETING_TYPES.index(st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)),
                 on_change=lambda: st.session_state.update(current_prompt_text=""))

        # Conditional Options
        meeting_type = st.session_state.get('selected_meeting_type')
        if meeting_type == "Expert Meeting":
            st.radio(
                "Expert Meeting Note Style:",
                options=EXPERT_MEETING_OPTIONS, key="expert_meeting_prompt_option",
                index=EXPERT_MEETING_OPTIONS.index(st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)),
                help="Choose output: Strict Q&A, Natural Q&A, or Natural Q&A + Summary.",
                on_change=lambda: st.session_state.update(current_prompt_text="")
            )
        elif meeting_type == "Earnings Call":
            st.radio(
                "Mode:", options=EARNINGS_CALL_MODES, key="earnings_call_mode", horizontal=True,
                index=EARNINGS_CALL_MODES.index(st.session_state.get('earnings_call_mode', DEFAULT_EARNINGS_CALL_MODE)),
                help="Generate notes from scratch or enrich existing ones."
            )

    with col2:
        st.selectbox("Notes/Enrichment Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model for final output.")
        st.selectbox("Transcription Model (Audio):", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model for audio transcription.")
        st.selectbox("Refinement Model (Audio):", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model for audio refinement.")
    # --- End of Indented Block for Container 1 ---

st.divider() # This is now correctly outside Container 1

# --- Container 2: Input ---
with st.container(border=True):
    # --- FIX: Ensure subsequent code is indented ---
    show_enrich_input = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                         st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")

    if show_enrich_input:
        st.subheader("2a. Existing Notes Input (for Enrichment)")
        st.text_area("Paste your existing notes here:", height=150, key="existing_notes_input",
                     placeholder="Paste the notes you want to enrich...")
        st.markdown("---")
        st.subheader("2b. Source Transcript Input")
    else:
        st.subheader("2. Source Input")

    # Common Source Input Widgets
    st.radio(label="Source input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.get('input_method_radio', 'Paste Text')
    if input_type_ui == "Paste Text": st.text_area("Paste source transcript:", height=150, key="text_input", placeholder="Paste transcript source...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")
    # --- End of Indented Block for Container 2 ---


# --- Container 3: Configuration ---
with st.container(border=True):
    # --- FIX: Ensure subsequent code is indented ---
    st.subheader("3. Configuration")
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1: # Topics (Earnings Call Only)
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            st.markdown("**Earnings Call Topics**")
            current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
            # Selectbox for loading template
            new_sector = st.selectbox("Load Template (Optional):", options=SECTOR_OPTIONS, key="selected_sector", index=SECTOR_OPTIONS.index(current_sector), label_visibility="collapsed")
            if new_sector != current_sector:
                update_topic_template()
                st.rerun()
            # Text area to display/edit topics
            st.text_area("Topics (Edit below):",
                         value=st.session_state.get("earnings_call_topics", ""),
                         key="earnings_call_topics", # Keep key simple if no complex callbacks needed
                         height=120,
                         placeholder="Enter topics manually or load a template...",
                         help="Guides structure for new notes or focuses enrichment.")
        else:
            st.caption("Topic selection only available for Earnings Calls.")

    with col_cfg2: # Context & Prompt Edit Toggle
        st.markdown("**Additional Context & Prompt**")
        add_context = st.checkbox("Add General Context", key="add_context_enabled")
        st.text_area(
            "Context Details:",
            value=st.session_state.get("context_input", ""),
            key="context_input",
            height=60,
            placeholder="Enable checkbox to add context...",
            disabled=not add_context
        )

        st.write("") # Spacer

        # View/Edit Prompt Checkbox
        meeting_type = st.session_state.get('selected_meeting_type')
        is_enrich_mode = (meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
        if meeting_type != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled",
                         disabled=is_enrich_mode,
                         on_change=handle_edit_toggle,
                         help="View/edit the default prompt. Disabled in Enrichment mode.")
             if is_enrich_mode:
                 st.caption("Prompt editing disabled in Enrichment mode.")
    # --- End of Indented Block for Container 3 ---


# --- Container 4: Prompt Editor (Conditional) ---
# (Conditional logic and content remain the same)
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and st.session_state.get('selected_meeting_type') != "Custom" and not \
                    (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes"))

if show_prompt_area:
    with st.container(border=True):
        # --- FIX: Ensure subsequent code is indented ---
        prompt_title = "Final Prompt Editor" if st.session_state.get('selected_meeting_type') != "Custom" else "Custom Final Prompt (Required)"
        st.subheader(prompt_title)
        default_prompt_for_display = get_prompt_display_text(for_display_only=True)
        current_value = st.session_state.current_prompt_text
        prompt_to_display = current_value if current_value else default_prompt_for_display
        st.text_area(
            label="Prompt Text:", value=prompt_to_display, key="current_prompt_text", height=300,
            label_visibility="collapsed",
            help="Edit the prompt..." if st.session_state.get('selected_meeting_type') != "Custom" else "Enter your custom prompt...",
            disabled=False
        )
        if st.session_state.get('selected_meeting_type') != "Custom":
             st.caption("Editing enabled. Placeholders `{transcript}` and `{context_section}` will be filled.")
        else:
             st.caption("Placeholders `{transcript}` and `{context_section}` will be automatically filled.")
        # --- End of Indented Block for Container 4 ---

# --- Generate Button ---
# (Logic remains the same)
st.divider()
is_valid, error_msg = validate_inputs()
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes based on current inputs and settings."
generate_button = st.button("🚀 Generate / Enrich Notes", type="primary", use_container_width=True,
                            disabled=st.session_state.processing or st.session_state.generating_filename or not is_valid,
                            help=generate_tooltip)


# --- Output Section ---
# (Output section logic remains the same)
output_container = st.container(border=True)
with output_container:
    # --- FIX: Ensure subsequent code is indented ---
    if st.session_state.get('generating_filename'): st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.get('error_message'): st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.get('generated_notes'):
        # ... Output display logic ...
        is_enrich_output = (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
        # ... (Rest of Output Display) ...
    elif not st.session_state.get('processing'):
        st.markdown("<p class='initial-prompt'>Configure inputs above and click 'Generate / Enrich Notes'.</p>", unsafe_allow_html=True)
    # --- End of Indented Block for Output Container ---


# --- History Section ---
# (History section logic remains the same)
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    # --- FIX: Ensure subsequent code is indented ---
    if not st.session_state.get('history'): st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
             with st.container():
                 # ... History entry display ...
                 st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                 # ... (rest of history item display) ...
                 st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                 if i < len(st.session_state.history) - 1: st.divider()
    # --- End of Indented Block for History Expander ---


# --- Processing Logic ---
# (Processing logic remains the same as the previous corrected version)
if generate_button:
    # ... Start processing logic ...
    is_valid_on_click, error_msg_on_click = validate_inputs()
    # ... (rest of button click logic) ...


if st.session_state.get('processing') and not st.session_state.get('generating_filename'):
    # ... (Main processing logic) ...
    processed_audio_chunk_references = []
    operation_desc = "Generating Notes"
    # ... (rest of processing) ...

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

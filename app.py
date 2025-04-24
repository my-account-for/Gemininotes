# --- Required Imports ---
# ... (Imports remain the same) ...
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
# ... (Page config remains the same) ...
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
# ... (CSS remains the same) ...
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

# --- Define Available Models & Meeting Types ---
# ... (Definitions remain the same) ...
AVAILABLE_MODELS = {
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
SECTOR_TOPICS = {
    "IT Services": """...""",
    "QSR": """..."""
}

# --- Load API Key and Configure Gemini Client ---
# ... (API key loading remains the same) ...
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    # ... (genai configuration) ...
    genai.configure(api_key=API_KEY)
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    summary_gen_config = {"temperature": 0.6, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    enrichment_gen_config = {"temperature": 0.4, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    refinement_gen_config = {"temperature": 0.3, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()


# --- Prompts Definitions ---
# ... (Prompts remain the same) ...
PROMPTS = { "Expert Meeting": { "...": "..." }, "Earnings Call": { "...": "..." }, "Custom": "..." }
EXPERT_MEETING_OPTIONS = [ "Option 1...", "Option 2...", "Option 3..." ]
DEFAULT_EXPERT_MEETING_OPTION = EXPERT_MEETING_OPTIONS[1]
EXPERT_MEETING_SUMMARY_PROMPT_KEY = "Summary Prompt (for Option 3)"

# --- Initialize Session State ---
# ... (Initialization remains the same) ...
default_state = {
    # ... other keys ...
    'add_context_enabled': False,
    'context_input': '',
    # ... other keys ...
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value


# --- Helper Functions ---
# ... (All helper functions remain the same) ...
def extract_text_from_pdf(pdf_file_stream): # ...
    pass
def update_topic_template(): # ...
    pass
def format_prompt_safe(prompt_template, **kwargs): # ...
    pass
def create_docx(text): # ...
    pass
def get_current_input_data(): # ...
    pass
def validate_inputs(): # ...
    pass
def handle_edit_toggle(): # ...
    pass
def get_prompt_display_text(for_display_only=False): # ...
    pass
def clear_all_state(): # ...
    pass
def generate_suggested_filename(notes_content, meeting_type): # ...
    pass
def add_to_history(notes): # ...
    pass
def restore_note_from_history(index): # ...
    pass


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Container 1: Meeting Setup ---
# ... (Meeting Setup container remains the same) ...
with st.container(border=True):
    st.subheader("1. Meeting Setup")
    # ... Meeting Type, Mode/Style, Models ...

st.divider()

# --- Container 2: Input ---
# ... (Input container remains the same, including conditional Existing Notes) ...
with st.container(border=True):
    show_enrich_input = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                         st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
    if show_enrich_input:
        st.subheader("2a. Existing Notes Input (for Enrichment)")
        # ... Existing Notes Text Area ...
        st.markdown("---")
        st.subheader("2b. Source Transcript Input")
    else:
        st.subheader("2. Source Input")
    # ... Source Input Widgets (Radio, Text, Uploaders) ...

# --- Container 3: Configuration ---
with st.container(border=True):
    st.subheader("3. Configuration")
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1: # Topics (Earnings Call Only)
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            st.markdown("**Earnings Call Topics**")
            # ... Sector Selectbox and Topic Text Area ...
            current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
            new_sector = st.selectbox("Load Template (Optional):", options=SECTOR_OPTIONS, key="selected_sector", index=SECTOR_OPTIONS.index(current_sector), label_visibility="collapsed")
            if new_sector != current_sector:
                update_topic_template()
                st.rerun()
            st.text_area("Topics (Edit below):",
                         value=st.session_state.get("earnings_call_topics", ""),
                         key="earnings_call_topics", # Use direct state key if no specific on_change needed
                         height=120,
                         placeholder="Enter topics manually or load a template...",
                         help="Guides structure for new notes or focuses enrichment.")
        else:
            st.caption("Topic selection only available for Earnings Calls.")

    with col_cfg2: # Context & Prompt Edit Toggle
        st.markdown("**Additional Context & Prompt**")
        # --- FIX: Make Checkbox Control Disabled State of Text Area ---
        add_context = st.checkbox("Add General Context", key="add_context_enabled")
        st.text_area(
            "Context Details:",
            value=st.session_state.get("context_input", ""), # Read from state
            key="context_input", # Write to state
            height=60, # Keep height consistent
            placeholder="Enable checkbox to add context...",
            disabled=not add_context # Use the return value of checkbox directly
        )
        # -------------------------------------------------------------

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


# --- Container 4: Prompt Editor (Conditional) ---
# ... (Prompt Editor container remains the same) ...
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and st.session_state.get('selected_meeting_type') != "Custom" and not \
                    (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes"))
if show_prompt_area:
    with st.container(border=True):
        # ... Prompt Editor Logic ...

# --- Generate Button ---
# ... (Generate Button remains the same) ...
st.divider()
is_valid, error_msg = validate_inputs()
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes based on current inputs and settings."
generate_button = st.button("🚀 Generate / Enrich Notes", type="primary", use_container_width=True,
                            disabled=st.session_state.processing or st.session_state.generating_filename or not is_valid,
                            help=generate_tooltip)


# --- Output Section ---
# ... (Output section remains the same) ...
output_container = st.container(border=True)
with output_container:
    # ... Output display logic ...

# --- History Section ---
# ... (History section remains the same) ...
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    # ... History display logic ...

# --- Processing Logic ---
# ... (Processing logic remains the same) ...
if generate_button:
    # ... Start processing logic ...
    is_valid_on_click, error_msg_on_click = validate_inputs()
    # ... (rest of button click logic) ...


if st.session_state.get('processing') and not st.session_state.get('generating_filename'):
    # ... (Main processing logic, audio handling, prompt formatting, API call, etc.) ...
    pass # Keep existing logic here

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

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
from pydub import AudioSegment
from pydub.utils import make_chunks
import copy

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
# (CSS remains the same)
st.markdown("""
<style>
    /* ... CSS styles ... */
</style>
""", unsafe_allow_html=True)

# --- Define Available Models & Meeting Types ---
AVAILABLE_MODELS = {
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
DEFAULT_NOTES_MODEL_NAME = next((k for k, v in AVAILABLE_MODELS.items() if "2.5" in k),
                                next((k for k, v in AVAILABLE_MODELS.items() if "1.5 Pro" in k), list(AVAILABLE_MODELS.keys())[0]))
DEFAULT_TRANSCRIPTION_MODEL_NAME = next((k for k, v in AVAILABLE_MODELS.items() if "Flash" in k), list(AVAILABLE_MODELS.keys())[0])
DEFAULT_REFINEMENT_MODEL_NAME = DEFAULT_NOTES_MODEL_NAME # Often same as notes model

MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]
DEFAULT_EARNINGS_CALL_MODE = EARNINGS_CALL_MODES[0]

# --- Sector-Specific Topics ---
SECTOR_OPTIONS = ["Other / Manual Topics", "IT Services", "QSR"]
DEFAULT_SECTOR = SECTOR_OPTIONS[0]
SECTOR_TOPICS = {
    "IT Services": """Future investments related comments (Including GenAI, AI, Data, Cloud, etc):
Capital allocation:
Talent supply chain related comments:
Org structure change:
Other comments:
Short-term comments:
- Guidance:
- Order booking:
- Impact of macro slowdown:
- Vertical wise comments:""",
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
    # --- Generation Configs ---
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    summary_gen_config = {"temperature": 0.6, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    enrichment_gen_config = {"temperature": 0.4, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    refinement_gen_config = {"temperature": 0.3, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Prompts Definitions ---
# (Prompts remain the same as the previous version)
PROMPTS = {
    "Expert Meeting": {
         "Option 1: Existing (Detailed & Strict)": """...""", # Keep prompts concise in example
        "Option 2: Less Verbose (Default)": """...""",
        "Summary Prompt (for Option 3)": """..."""
    },
    "Earnings Call": {
        "Generate New Notes": """...""",
        "Enrich Existing Notes": """..."""
    },
    "Custom": "{user_custom_prompt}\n\n--- TRANSCRIPT START ---\n{transcript}\n--- TRANSCRIPT END ---\n{context_section}"
}

EXPERT_MEETING_OPTIONS = [
    "Option 1: Existing (Detailed & Strict)",
    "Option 2: Less Verbose (Default)",
    "Option 3: Option 2 + Executive Summary"
]
DEFAULT_EXPERT_MEETING_OPTION = EXPERT_MEETING_OPTIONS[1]
EXPERT_MEETING_SUMMARY_PROMPT_KEY = "Summary Prompt (for Option 3)"

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None,
    'add_context_enabled': False,
    'selected_notes_model_display_name': DEFAULT_NOTES_MODEL_NAME,
    'selected_transcription_model_display_name': DEFAULT_TRANSCRIPTION_MODEL_NAME,
    'selected_refinement_model_display_name': DEFAULT_REFINEMENT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'expert_meeting_prompt_option': DEFAULT_EXPERT_MEETING_OPTION,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'selected_sector': DEFAULT_SECTOR,
    'earnings_call_topics': SECTOR_TOPICS.get(DEFAULT_SECTOR, ""), # Initialize topics
    'earnings_call_mode': DEFAULT_EARNINGS_CALL_MODE,
    'existing_notes_input': "",
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': []
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
# (All helper functions remain largely the same - extract_pdf, update_topic, format_prompt, create_docx, get_input, validate, etc.)
def extract_text_from_pdf(pdf_file_stream):
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def update_topic_template():
    # Now called when sector changes OR at start of processing
    selected_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if selected_sector in SECTOR_TOPICS:
        # Only update if the template is different from current text or empty
        if st.session_state.get('earnings_call_topics',"") != SECTOR_TOPICS[selected_sector]:
            st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
    # Don't clear manual input if "Other" is selected

def format_prompt_safe(prompt_template, **kwargs):
    formatted_prompt = copy.deepcopy(prompt_template)
    try:
        placeholders = re.findall(r"\{([^}]+)\}", formatted_prompt)
        for key in placeholders:
            value = kwargs.get(key, f"[MISSING: {key}]")
            str_value = str(value) if value is not None else ""
            formatted_prompt = formatted_prompt.replace("{" + key + "}", str_value)
        return formatted_prompt
    except Exception as e:
        st.error(f"Prompt formatting error: {e}")
        return f"# Error formatting prompt: {e}"

def create_docx(text):
    document = docx.Document(); [document.add_paragraph(line) for line in text.split('\n')]; buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()

def get_current_input_data():
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

def validate_inputs():
    # Use .get() for safer access during validation
    input_method = st.session_state.get('input_method_radio', 'Paste Text')
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    custom_prompt = st.session_state.get('current_prompt_text', '')

    if input_method == "Paste Text" and not st.session_state.get('text_input','').strip():
        return False, "Please paste the source transcript text."
    if input_method == "Upload PDF" and st.session_state.get('pdf_uploader') is None:
        return False, "Please upload a source PDF file."
    if input_method == "Upload Audio" and st.session_state.get('audio_uploader') is None:
        return False, "Please upload a source audio file."

    if meeting_type == "Custom" and not custom_prompt.strip():
         return False, "Custom prompt cannot be empty for 'Custom' meeting type."
    if meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
        if not st.session_state.get('existing_notes_input',"").strip():
            return False, "Please provide your existing notes for enrichment."

    return True, ""

def handle_edit_toggle():
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = "" # Clear edits if toggled off

def get_prompt_display_text(for_display_only=False):
    # (Same as previous version)
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)

    if not for_display_only and st.session_state.get('view_edit_prompt_enabled') and meeting_type != "Custom" and st.session_state.get('current_prompt_text'):
        return st.session_state.current_prompt_text

    display_text = ""
    temp_context = st.session_state.get('context_input','').strip() if st.session_state.get('add_context_enabled') else None
    input_type = st.session_state.get('input_method_radio', 'Paste Text')
    placeholder = "[TRANSCRIPT WILL APPEAR HERE]"
    format_kwargs = {
        'transcript': placeholder,
        'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{temp_context}\n---" if temp_context else ""
    }
    prompt_template_to_display = None

    try:
        if meeting_type == "Expert Meeting":
            expert_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
            if expert_option == "Option 1: Existing (Detailed & Strict)":
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
            else:
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

            if prompt_template_to_display:
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                 if expert_option == "Option 3: Option 2 + Executive Summary":
                     summary_prompt_preview = PROMPTS["Expert Meeting"][EXPERT_MEETING_SUMMARY_PROMPT_KEY].split("---")[0]
                     display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step generated *after* these notes, using a separate prompt starting like:\n'''\n{summary_prompt_preview.strip()}\n'''"
            else:
                display_text = "# Error: Could not find prompt template for display."

        elif meeting_type == "Earnings Call":
             prompt_template_to_display = PROMPTS["Earnings Call"]["Generate New Notes"]
             user_topics_text_for_display = st.session_state.get('earnings_call_topics', "")
             topic_instructions_preview = ""
             if user_topics_text_for_display and user_topics_text_for_display.strip():
                 topic_instructions_preview = f"Structure notes under user-specified headings (e.g., {user_topics_text_for_display.splitlines()[0]}...)\n- **Other Key Points**"
             else:
                 topic_instructions_preview = "Identify logical main themes...\n- **Other Key Points**"
             format_kwargs['topic_instructions'] = topic_instructions_preview
             display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)

        elif meeting_type == "Custom":
             audio_note = ("\n# NOTE FOR AUDIO: ...") # Keep concise
             default_custom = "# Enter your custom prompt...\n# Use {transcript} and {context_section} placeholders."
             current_or_default = st.session_state.current_prompt_text if not for_display_only and st.session_state.current_prompt_text else default_custom
             display_text = current_or_default + (audio_note if st.session_state.get('input_method_radio') == 'Upload Audio' else "")
             return display_text

        else:
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
             return "Error generating prompt preview."

        if input_type == "Upload Audio" and meeting_type != "Custom":
             audio_note = ("# NOTE FOR AUDIO: ...") # Keep concise
             display_text = audio_note + display_text

    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview."

    return display_text

def clear_all_state():
    # (Same as before)
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
    st.session_state.expert_meeting_prompt_option = DEFAULT_EXPERT_MEETING_OPTION
    st.session_state.input_method_radio = 'Paste Text'
    st.session_state.text_input = ""
    st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None
    st.session_state.context_input = ""
    st.session_state.add_context_enabled = False
    st.session_state.selected_sector = DEFAULT_SECTOR
    st.session_state.earnings_call_topics = SECTOR_TOPICS.get(DEFAULT_SECTOR, "") # Reset topics
    st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_enabled = False
    st.session_state.earnings_call_mode = DEFAULT_EARNINGS_CALL_MODE
    st.session_state.existing_notes_input = ""
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
    st.session_state.processed_audio_chunk_references = []
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type):
    # (Same as before)
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        # ... (rest of filename generation logic) ...
    except Exception as e:
        st.warning(f"Filename gen error: {e}", icon="⚠️")
        return None
    finally:
        st.session_state.generating_filename = False # Ensure this is always set back

def add_to_history(notes):
    # (Same as before)
    if not notes: return
    try:
        # ... (history logic) ...
         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
         new_entry = {"timestamp": timestamp, "notes": notes}
         current_history = st.session_state.get('history', [])
         if not isinstance(current_history, list):
             st.warning("History state was not a list, resetting.", icon="⚠️"); current_history = []
         current_history.insert(0, new_entry)
         st.session_state.history = current_history[:3] # Keep only last 3
    except Exception as e: st.warning(f"⚠️ Error updating note history: {e}", icon="❗")


def restore_note_from_history(index):
    # (Same as before)
    if 0 <= index < len(st.session_state.history):
        # ... (restore logic) ...
         entry = st.session_state.history[index]
         st.session_state.generated_notes = entry["notes"]
         st.session_state.edited_notes_text = entry["notes"]
         st.session_state.edit_notes_enabled = False
         st.session_state.suggested_filename = None
         st.session_state.error_message = None
         st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Container 1: Meeting Setup ---
with st.container(border=True):
    st.subheader("1. Meeting Setup")
    col1, col2 = st.columns(2)
    with col1:
        # Meeting Type
        st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                 index=MEETING_TYPES.index(st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)),
                 on_change=lambda: st.session_state.update(current_prompt_text="")) # Clear edits

        # Conditional Options
        meeting_type = st.session_state.get('selected_meeting_type')
        if meeting_type == "Expert Meeting":
            st.radio(
                "Expert Meeting Note Style:",
                options=EXPERT_MEETING_OPTIONS, key="expert_meeting_prompt_option",
                index=EXPERT_MEETING_OPTIONS.index(st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)),
                help="Choose output: Strict Q&A, Natural Q&A, or Natural Q&A + Summary.",
                on_change=lambda: st.session_state.update(current_prompt_text="") # Clear edits
            )
        elif meeting_type == "Earnings Call":
            st.radio(
                "Mode:", options=EARNINGS_CALL_MODES, key="earnings_call_mode", horizontal=True,
                index=EARNINGS_CALL_MODES.index(st.session_state.get('earnings_call_mode', DEFAULT_EARNINGS_CALL_MODE)),
                help="Generate notes from scratch or enrich existing ones."
                # No need for callback here, the UI below will react
            )

    with col2:
        st.selectbox("Notes/Enrichment Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model for final output.")
        st.selectbox("Transcription Model (Audio):", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model for audio transcription.")
        st.selectbox("Refinement Model (Audio):", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model for audio refinement.")

# --- Container 2: Input ---
with st.container(border=True):
    # --- Conditional Input Area for Existing Notes ---
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

    # --- Common Source Input Widgets ---
    st.radio(label="Source input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.get('input_method_radio', 'Paste Text')
    if input_type_ui == "Paste Text": st.text_area("Paste source transcript:", height=150, key="text_input", placeholder="Paste transcript source...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

# --- Container 3: Configuration ---
with st.container(border=True):
    st.subheader("3. Configuration")
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1: # Topics (Earnings Call Only) & Context
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            st.markdown("**Earnings Call Topics**")
            current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
            new_sector = st.selectbox("Load Template (Optional):", options=SECTOR_OPTIONS, key="selected_sector", index=SECTOR_OPTIONS.index(current_sector), label_visibility="collapsed")
            if new_sector != current_sector:
                update_topic_template()
                # No rerun needed here, text_area below reads directly from state
            st.text_area("Topics (Edit below):",
                         value=st.session_state.get("earnings_call_topics", ""), # Use .get for safety
                         key="earnings_call_topics", # Update state directly on edit
                         height=120,
                         placeholder="Enter topics manually or load a template...",
                         help="Guides structure for new notes or focuses enrichment.")
        else:
            st.caption("Topic selection only available for Earnings Calls.")

    with col_cfg2: # Context & Prompt Edit Toggle
        st.markdown("**Additional Context & Prompt**")
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.get('add_context_enabled'):
            st.text_area("Context Details:", height=60, key="context_input", placeholder="Company Name, Ticker...")

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
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and st.session_state.get('selected_meeting_type') != "Custom" and not \
                    (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes"))

if show_prompt_area:
    with st.container(border=True):
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

# --- Generate Button ---
st.divider()
is_valid, error_msg = validate_inputs()
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes based on current inputs and settings."
generate_button = st.button("🚀 Generate / Enrich Notes",
                            type="primary",
                            use_container_width=True,
                            disabled=st.session_state.processing or st.session_state.generating_filename or not is_valid,
                            help=generate_tooltip)

# --- Output Section ---
# (Output section logic remains the same)
output_container = st.container(border=True)
with output_container:
    if st.session_state.get('generating_filename'): st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.get('error_message'): st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.get('generated_notes'):
        is_enrich_output = (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
        output_title = "✅ Enriched Notes" if is_enrich_output else "✅ Generated Notes"
        st.subheader(output_title)

        if st.session_state.get('raw_transcript'):
            with st.expander("View Raw Source Transcript (Step 1)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
        if st.session_state.get('refined_transcript'):
             with st.expander("View Refined Source Transcript (Step 2)", expanded=bool(st.session_state.refined_transcript)):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)

        st.checkbox("Edit Output", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.get('edit_notes_enabled') else st.session_state.generated_notes

        is_expert_meeting_summary = (st.session_state.get('selected_meeting_type') == "Expert Meeting" and
                                     st.session_state.get('expert_meeting_prompt_option') == "Option 3: Option 2 + Executive Summary" and
                                     "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n" in notes_content_to_use)

        if st.session_state.get('edit_notes_enabled'):
            st.text_area("Editable Output:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            if is_expert_meeting_summary:
                 # ... (display split summary) ...
                  try:
                     notes_part, summary_part = notes_content_to_use.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n", 1)
                     st.markdown("### Detailed Notes (Q&A Format)")
                     st.markdown(notes_part)
                     st.markdown("---")
                     st.markdown("### Executive Summary")
                     st.markdown(summary_part)
                  except ValueError:
                     st.markdown(notes_content_to_use) # Fallback
            else:
                 st.markdown(notes_content_to_use)

        # Download Buttons
        dl_cols = st.columns([1,1,1.5]) # Adjust column widths slightly
        output_type_label = "enriched_notes" if is_enrich_output else "notes"
        default_fname = f"{st.session_state.get('selected_meeting_type', 'meeting').lower().replace(' ', '_')}_{output_type_label}"
        fname_base = st.session_state.get('suggested_filename', default_fname)
        with dl_cols[0]: st.download_button(label=f"⬇️ Output (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]: st.download_button(label=f"⬇️ Output (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.get('refined_transcript'):
                refined_fname_base = fname_base.replace(f"_{output_type_label}", "_refined_transcript") if f"_{output_type_label}" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Source Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the refined source transcript.")
            else: st.button("Refined Tx N/A", disabled=True, use_container_width=True)

    elif not st.session_state.get('processing'):
        st.markdown("<p class='initial-prompt'>Configure inputs above and click 'Generate / Enrich Notes'.</p>", unsafe_allow_html=True)


# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    # (Logic remains the same)
    if not st.session_state.get('history'): st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
             with st.container():
                # ... (history display) ...
                 st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                 display_note = entry['notes']
                 summary_separator = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
                 preview_text = ""
                 if summary_separator in display_note:
                      notes_part, summary_part = display_note.split(summary_separator, 1)
                      preview_text = "\n".join(notes_part.strip().splitlines()[:3]) + "\n... (+ Summary)"
                 else:
                     preview_text = "\n".join(display_note.strip().splitlines()[:3]) + "..."

                 st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
                 st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                 if i < len(st.session_state.history) - 1: st.divider()

# --- Processing Logic ---
# (Logic remains the same as the previous corrected version)
if generate_button:
    is_valid_on_click, error_msg_on_click = validate_inputs()
    if not is_valid_on_click:
        st.session_state.error_message = f"Validation Error: {error_msg_on_click}"
        st.rerun()
    else:
        st.session_state.processing = True
        st.session_state.generating_filename = False
        st.session_state.generated_notes = None
        st.session_state.edited_notes_text = ""
        st.session_state.edit_notes_enabled = False
        st.session_state.error_message = None
        st.session_state.suggested_filename = None
        st.session_state.raw_transcript = None
        st.session_state.refined_transcript = None
        st.session_state.processed_audio_chunk_references = []
        st.rerun()

if st.session_state.get('processing') and not st.session_state.get('generating_filename'):
    processed_audio_chunk_references = []

    operation_desc = "Generating Notes"
    if st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
        operation_desc = "Enriching Notes"

    with st.status(f"🚀 Initializing {operation_desc} process...", expanded=True) as status:
        try:
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process:
                raise ValueError(f"Input validation failed: {error_msg_process}")

            status.update(label="⚙️ Reading inputs and settings...")
            # --- Read state using .get() for safety ---
            meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
            expert_meeting_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
            notes_model_id = AVAILABLE_MODELS[st.session_state.get('selected_notes_model_display_name', DEFAULT_NOTES_MODEL_NAME)]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.get('selected_transcription_model_display_name', DEFAULT_TRANSCRIPTION_MODEL_NAME)]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.get('selected_refinement_model_display_name', DEFAULT_REFINEMENT_MODEL_NAME)]
            user_edited_or_custom_prompt = st.session_state.get('current_prompt_text', '').strip()
            general_context = st.session_state.get('context_input', '').strip() if st.session_state.get('add_context_enabled') else None
            earnings_mode = st.session_state.get('earnings_call_mode', DEFAULT_EARNINGS_CALL_MODE)
            user_existing_notes = st.session_state.get('existing_notes_input', '').strip() if earnings_mode == "Enrich Existing Notes" else None
            actual_input_type, source_transcript_data, source_audio_file_obj = get_current_input_data() # Reads state internally

            # --- Update and read topics AFTER reading meeting type ---
            if meeting_type == "Earnings Call":
                # Call update logic based on selected_sector state
                update_topic_template()
                earnings_call_topics_text = st.session_state.get("earnings_call_topics", "").strip()
            else:
                 earnings_call_topics_text = ""

            status.update(label="🧠 Initializing AI models...")
            # ... (model initialization) ...
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            final_source_transcript = source_transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            # --- Audio Processing (Step 1 & 2) ---
            if actual_input_type == "Upload Audio":
                # ... [Existing Audio Processing Logic For Source - ensure final_source_transcript is updated] ...
                 status.update(label=f"🔊 Loading source audio file '{source_audio_file_obj.name}'...")
                 audio_bytes = source_audio_file_obj.getvalue()
                 # ... rest of audio processing ...
                 # ... update final_source_transcript based on success/failure of refinement ...


            # --- Notes Generation / Enrichment (Step 3) ---
            if not final_source_transcript:
                 raise ValueError("No source transcript available to generate or enrich notes.")

            status.update(label=f"📝 Preparing final prompt for {operation_desc}...")
            # ... [Existing prompt selection/formatting logic - uses earnings_call_topics_text safely] ...
            # ... [API Call] ...
            # ... [Handle Response & Summary Step] ...
            # ... [Post-generation steps] ...


        except Exception as e:
            st.session_state.error_message = f"❌ Processing Error: {e}"
            if status: status.update(label=f"❌ Error: {e}", state="error") # Update status only if it exists
            # Don't rerun here, let it finish to display error

        finally:
            st.session_state.processing = False
            # --- Audio Chunk Cleanup ---
            if processed_audio_chunk_references:
                 # ... [Cleanup Logic] ...
                 st.toast(f"☁️ Performing final cleanup...", icon="🗑️")
                 refs_to_delete = list(processed_audio_chunk_references)
                 for file_ref in refs_to_delete:
                    try:
                        genai.delete_file(file_ref.name)
                        processed_audio_chunk_references.remove(file_ref)
                    except Exception as final_cleanup_error:
                        st.warning(f"Final cloud file cleanup failed for {file_ref.name}: {final_cleanup_error}", icon="⚠️")
                 st.session_state.processed_audio_chunk_references = []

            # Rerun only if NO error occurred during processing
            if not st.session_state.get('error_message'):
                st.rerun()
            # If an error occurred, the script ends here, and the error message will be displayed
            # because error_message was set in the except block.

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

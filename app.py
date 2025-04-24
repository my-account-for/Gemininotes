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

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
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
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash", "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume)": "gemini-1.5-flash-8b", "Gemini 2.0 Flash (Next Gen Speed)": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite (Low Latency)": "gemini-2.0-flash-lite", "Gemini 2.5 Flash Preview (Adaptive)": "gemini-2.5-flash-preview-04-17",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
DEFAULT_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_KEY)
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,
    'selected_model_display_name': DEFAULT_MODEL_NAME, 'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '', 'earnings_call_topics': '', 'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    # 'processing_step': None, # Removed
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

# --- Restored ORIGINAL Expert Meeting Prompt ---
def create_expert_meeting_prompt(transcript, context=None):
    """Creates the prompt for 'Expert Meeting' type (Q&A format)."""
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

# --- MODIFIED Earnings Call Prompt ---
def create_earnings_call_prompt(transcript, user_topics=None, context=None):
    """Creates the prompt for 'Earnings Call' type, tailored for investment firms, allowing user topics."""
    topic_instructions = ""
    if user_topics:
        topic_list_str = "\n".join(f"- **{topic.strip()}**" for topic in user_topics)
        topic_instructions = ( f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n"
                               f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered above)\n\n"
                               f"Place details under the most appropriate heading. If a user topic isn't discussed, state 'Not discussed'.")
    else: # No topics provided by user
        topic_instructions = ( f"Since no specific topics were provided, first identify the logical main themes discussed (e.g., Financials, Strategy, Outlook, Q&A). Use these themes as **bold headings**.\n"
                               f"Include a final mandatory section:\n- **Other Key Points** (for important info not covered in main themes)\n\n"
                               f"Place details under the most appropriate heading.")
    prompt_parts = [ "You are an expert AI assistant creating DETAILED notes from an earnings call transcript for an investment firm.",
                     "Output MUST be comprehensive, factual notes, capturing all critical financial and strategic information.",
                     "**Formatting Requirements (Mandatory):**\n- US$ for dollars (US$2.5M), % for percentages.\n- State comparison periods (+5% YoY, -2% QoQ).\n- Represent fiscal periods accurately (Q3 FY25).\n- Use common abbreviations (CEO, KPI).\n- Use bullet points under headings.\n- Each bullet = complete sentence with distinct info.\n- Capture ALL numbers, names, data accurately.\n- Use quotes \"\" for significant statements.\n- DO NOT summarize or interpret unless part of the structure.",
                     "\n**Note Structure:**", "- **Call Participants:** (List names/titles or 'Not specified')", topic_instructions,
                     "\n**CRITICAL:** Ensure accuracy and adhere strictly to structure and formatting.", "\n---",
                     (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""), ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL NOTES:**\n"); return "\n".join(filter(None, prompt_parts))

def create_docx(text):
    document = docx.Document(); [document.add_paragraph(line) for line in text.split('\n')]; buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()

def get_current_input_data():
    """Helper to get transcript/audio file based on input method and state."""
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

# --- Function to update prompt display text (Removed on_change, called only when rendering) ---
def get_prompt_display_text():
    """Generates the appropriate prompt text for display/editing, does NOT modify state."""
    meeting_type = st.session_state.selected_meeting_type
    display_text = "" # Default to empty

    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        user_topics = None
        if meeting_type == "Earnings Call":
             topics_str = st.session_state.earnings_call_topics.strip()
             if topics_str: user_topics = [topic.strip() for topic in topics_str.split(',')]

        prompt_func = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else lambda t, c: create_earnings_call_prompt(t, user_topics, c)
        placeholder = "[TRANSCRIPT WILL APPEAR HERE]" if input_type != "Upload Audio" else None

        if input_type == "Upload Audio":
             base_prompt = prompt_func(transcript=None, context=temp_context)
             display_text = ("# NOTE FOR AUDIO: Default uses 2 steps (Transcribe then Notes).\n"
                             "# If edited, ensure your prompt handles the audio file.\n"
                             "####################################\n\n" + base_prompt)
        else:
            display_text = prompt_func(transcript=placeholder, context=temp_context)

    elif meeting_type == "Custom":
         # For custom, show the current state or a placeholder if empty
         display_text = st.session_state.current_prompt_text or "# Enter your custom prompt here...\n# For audio, include transcription instructions."

    return display_text

def clear_all_state():
    # Reset selections and inputs
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_model_display_name = DEFAULT_MODEL_NAME
    st.session_state.input_method_radio = 'Paste Text'
    st.session_state.text_input = ""
    st.session_state.pdf_uploader = None # Clear state ref
    st.session_state.audio_uploader = None # Clear state ref
    st.session_state.context_input = ""
    st.session_state.add_context_enabled = False
    st.session_state.earnings_call_topics = "" # Clear specific topics
    st.session_state.current_prompt_text = "" # Clear prompt text
    st.session_state.view_edit_prompt_enabled = False
    # Reset outputs
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.processing = False
    st.session_state.suggested_filename = None
    st.session_state.uploaded_audio_info = None # Clear cloud ref
    st.session_state.history = [] # Clear history
    st.toast("Inputs and outputs cleared!", icon="🧹")
    # Note: We don't call update_prompt_display_text here anymore. The UI will naturally clear.

def generate_suggested_filename(notes_content, meeting_type):
    # (Keep function as is)
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True; st.toast("💡 Generating filename...", icon="⏳")
        filename_model = genai.GenerativeModel("gemini-1.5-flash")
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "")
        filename_prompt = (f"Suggest filename: YYYYMMDD_ClientOrTopic_MeetingType. Date={today_date}. Type='{mt_cleaned}'. Max 3 words topic. Output ONLY filename.\nNOTES:{notes_content[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if re.match(r"\d{8}_[\w\-\.]+_\w+", s_name): st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename sugg '{s_name}' bad format.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    # (Keep function as is)
    if not notes: return; timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); new_entry = {"timestamp": timestamp, "notes": notes};
    current_history = st.session_state.get('history', []); current_history.insert(0, new_entry); st.session_state.history = current_history[:3]

def restore_note_from_history(index):
    # (Keep function as is)
    if 0 <= index < len(st.session_state.history): entry = st.session_state.history[index]; st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]; \
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None; st.session_state.error_message = None; st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a: st.subheader("Meeting Details"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True) # Removed on_change
        with col1b: st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary")
    st.divider(); st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed") # Removed on_change
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")
    st.divider(); col3a, col3b = st.columns(2); selected_mt = st.session_state.selected_meeting_type
    with col3a: # Topics (Earnings Call only) / Context (All)
        if selected_mt == "Earnings Call": st.text_area("Earnings Call Topics (Optional, comma-sep):", key="earnings_call_topics", height=75, placeholder="E.g., Guidance, Segment Perf...")
        st.checkbox("Add General Context", key="add_context_enabled") # Removed on_change
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=75, key="context_input", placeholder="Company Name, Ticker...")
    with col3b: # View/Edit Prompt Checkbox
        if selected_mt != "Custom": st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled") # Removed on_change

# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and selected_mt != "Custom") or (selected_mt == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if selected_mt != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_title); caption = "Edit prompt..." if selected_mt != "Custom" else "Enter prompt..."
        # Generate default prompt text for display *here* if needed and state is empty
        default_display_prompt = ""
        if not st.session_state.current_prompt_text and selected_mt != "Custom":
             default_display_prompt = get_prompt_display_text() # Call helper to generate default
        elif selected_mt == "Custom" and not st.session_state.current_prompt_text:
             default_display_prompt = "# Enter your custom prompt here...\n# For audio, include transcription instructions."

        st.caption(caption); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text or default_display_prompt, key="current_prompt_text", height=350, label_visibility="collapsed")

# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# Output Section
output_container = st.container(border=True)
with output_container:
    # (Output display logic remains largely the same)
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: st.info("⏳ Processing request...", icon="🧠")
    elif st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes"); st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_to_use)
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"; fname_base = st.session_state.suggested_filename or default_fname
        st.write(""); col_btn_dl1, col_btn_dl2 = st.columns(2)
        with col_btn_dl1: st.download_button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with col_btn_dl2: st.download_button(label="⬇️ Markdown", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history:
        st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                # Use st.code for better preview formatting and scrollbars if needed
                st.code(entry['notes'][:300] + ("..." if len(entry['notes']) > 300 else ""), language=None)
                st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}",
                          on_click=restore_note_from_history, args=(i,))
                # --- Correct indentation for the divider ---
                if i < len(st.session_state.history) - 1:
                    st.divider()
                # --- End correction ---
# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generating_filename = False; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False; st.session_state.error_message = None
    st.session_state.suggested_filename = None; st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    try: # Outer try
        # State & Input Retrieval
        meeting_type = st.session_state.selected_meeting_type; selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled = st.session_state.view_edit_prompt_enabled; user_prompt_text = st.session_state.current_prompt_text
        general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        user_topics_str = st.session_state.earnings_call_topics.strip() if meeting_type == "Earnings Call" else ""; user_topic_list = [t.strip() for t in user_topics_str.split(',')] if user_topics_str else None
        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # Validation
        if actual_input_type == "Paste Text" and not transcript_data: raise ValueError("Text area empty.")
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: raise ValueError("PDF error.")
        elif actual_input_type == "Upload Audio" and not audio_file_obj: raise ValueError("No audio uploaded.")
        if meeting_type == "Custom" and not user_prompt_text.strip(): raise ValueError("Custom Prompt empty.")

        # Initialize variables
        final_prompt_for_api = None; processed_audio_file_ref = None; api_payload_parts = []
        prompt_was_edited_or_custom = (meeting_type == "Custom" or view_edit_enabled); notes_model_id = selected_model_id
        transcribed_text_for_notes = transcript_data

        # Inner Try for pre-processing, prompt-building, audio handling
        try:
            # --- Handle Audio Upload & Transcription (if applicable) ---
            if actual_input_type == "Upload Audio":
                if not audio_file_obj: raise ValueError("Audio missing.")
                st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️"); audio_bytes = audio_file_obj.getvalue(); temp_file_path = None
                try: # Tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj.name)[1]) as tf: tf.write(audio_bytes); temp_file_path = tf.name
                    if temp_file_path: processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}")
                    else: raise Exception("Temp file failed.")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path) # Disk cleanup
                st.session_state.uploaded_audio_info = processed_audio_file_ref # Store cloud ref
                st.toast(f"🎧 Processing audio...", icon="⏳"); polling_start = time.time() # Polling
                while processed_audio_file_ref.state.name == "PROCESSING":
                    if time.time() - polling_start > 300: raise TimeoutError("Audio timeout.")
                    time.sleep(5); processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio state: {processed_audio_file_ref.state.name}")
                st.toast(f"🎧 Audio ready!", icon="✅")

                # --- TWO-STEP LOGIC ---
                if not prompt_was_edited_or_custom:
                    st.toast(f"✍️ Transcribing audio...", icon="⏳")
                    t_prompt = "Transcribe accurately. Output only raw text."; t_model = genai.GenerativeModel(selected_model_id)
                    t_response = t_model.generate_content([t_prompt, processed_audio_file_ref], generation_config=transcription_gen_config, safety_settings=safety_settings)
                    if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                        transcribed_text_for_notes = t_response.text.strip(); st.toast("✍️ Transcription complete!", icon="✅"); actual_input_type = "Generated Transcript"
                    else: raise Exception("Audio transcription failed.")
            # --- End Audio Pre-processing ---

            # --- Determine Final Prompt Text ---
            if not st.session_state.error_message: # Check error before proceeding
                if prompt_was_edited_or_custom: final_prompt_for_api = user_prompt_text.split("#######")[-1].strip() # Clean note
                else: # Generate default prompt
                    if meeting_type == "Expert Meeting": prompt_function = create_expert_meeting_prompt
                    elif meeting_type == "Earnings Call": prompt_function = lambda t, c: create_earnings_call_prompt(t, user_topic_list, c)
                    else: raise ValueError(f"Invalid meeting type '{meeting_type}' for default prompt.")
                    if transcribed_text_for_notes: final_prompt_for_api = prompt_function(transcribed_text_for_notes, general_context)
                    else: st.session_state.error_message = "Error: Source text unavailable."

            if not final_prompt_for_api and not st.session_state.error_message: st.session_state.error_message = "Error: Failed to determine prompt."

            # --- Prepare API Payload ---
            if not st.session_state.error_message and final_prompt_for_api:
                api_payload_parts.append(final_prompt_for_api)
                if actual_input_type == "Upload Audio" and prompt_was_edited_or_custom:
                    if processed_audio_file_ref: api_payload_parts.append(processed_audio_file_ref)
                    else: st.session_state.error_message = "Error: Audio ref missing."
            elif not st.session_state.error_message: st.session_state.error_message = "Error: Prompt not ready."

        except Exception as pre_process_err:
            st.session_state.error_message = f"❌ Pre-processing Error: {pre_process_err}"
            if st.session_state.uploaded_audio_info: try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None; except Exception: pass


        # --- Generate Notes API Call ---
        if not st.session_state.error_message and api_payload_parts:
            try: # Inner try for API call
                st.toast(f"📝 Generating notes...", icon="✨")
                model = genai.GenerativeModel(model_name=notes_model_id, safety_settings=safety_settings, generation_config=main_gen_config)
                response = model.generate_content(api_payload_parts)

                # Handle Response
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    st.session_state.generated_notes = response.text.strip(); st.session_state.edited_notes_text = st.session_state.generated_notes
                    add_to_history(st.session_state.generated_notes); st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                    st.toast("🎉 Notes generated!", icon="✅")
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.session_state.error_message = f"⚠️ Response blocked: {response.prompt_feedback.block_reason}."
                elif response: st.session_state.error_message = "🤔 AI returned empty response."
                else: st.session_state.error_message = "😥 Generation failed (No response)."

            except Exception as api_call_err: st.session_state.error_message = f"❌ API Call Error: {api_call_err}"

        # --- Cloud Audio Cleanup (always attempt if ref exists) ---
        if st.session_state.uploaded_audio_info:
            try: st.toast("☁️ Cleaning cloud audio...", icon="🗑️"); genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None
            except Exception as e: st.warning(f"Cloud cleanup failed: {e}", icon="⚠️")
        # End of inner try...except and cleanup

    # --- Outer FINALLY block: Always runs ---
    finally:
        st.session_state.processing = False
        # --- Attempt Cloud Audio Cleanup if reference exists ---
        # Correct indentation for try/except block
        if st.session_state.uploaded_audio_info:
            try:
                # This check might be redundant if error handling inside the main try already cleared it,
                # but it's safe to attempt deletion again here if the ref still exists.
                st.toast("☁️ Final check: Cleaning cloud audio...", icon="🗑️",) # Optional feedback
                genai.delete_file(st.session_state.uploaded_audio_info.name)
                st.session_state.uploaded_audio_info = None # Clear ref after successful deletion
            except Exception as final_cleanup_error:
                # Log or display a warning if final cleanup fails, but don't stop the rerun
                st.warning(f"Final cloud audio cleanup failed: {final_cleanup_error}", icon="⚠️")
        # --- End correction ---
        st.rerun() # Rerun to display final results or errors

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

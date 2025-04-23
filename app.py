# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
from dotenv import load_dotenv
import PyPDF2
import docx # Still needed for DOCX fallback
# Import for copy button might still fail if installation didn't work, keeping it commented for now
# from streamlit_copy_to_clipboard import st_copy_to_clipboard

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS Injection ---
# Ensures a polished UI matching previous iterations
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
    .stButton>button.secondary-button { background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB;
        width: auto; padding: 0.5rem 1rem; margin-right: 0.5rem; font-weight: 500; }
    .stButton>button.secondary-button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
     /* Download Buttons */
    .stDownloadButton > button { border-radius: 0.5rem; padding: 0.6rem 1.2rem; font-weight: 500; background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB; transition: background-color 0.2s ease-in-out; width: auto; margin-top: 0; margin-right: 0.5rem;} /* Add margin-right */
    .stDownloadButton > button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
    /* Output Area Styling */
    .output-container { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1.5rem; margin-top: 1.5rem; min-height: 150px; }
    .output-container .stMarkdown { background-color: transparent; border: none; padding: 0; color: #374151; font-size: 1rem; line-height: 1.6; }
    .output-container .stMarkdown h3, .output-container .stMarkdown h4, .output-container .stMarkdown strong { color: #111827; font-weight: 600; }
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem; }
    .output-container .initial-prompt { color: #6B7280; font-style: italic; text-align: center; padding-top: 2rem; }
    /* Prompt Edit Area */
    #prompt-edit-area textarea { font-family: monospace; font-size: 0.9rem; line-height: 1.4; background-color: #FDFDFD; }
    /* Footer */
    footer { text-align: center; color: #9CA3AF; font-size: 0.8rem; padding-top: 2rem; padding-bottom: 1rem; }
    footer a { color: #6B7280; text-decoration: none; }
    footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# --- Define Available Models ---
AVAILABLE_MODELS = {
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume)": "gemini-1.5-flash-8b",
    "Gemini 2.0 Flash (Next Gen Speed)": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite (Low Latency)": "gemini-2.0-flash-lite",
    "Gemini 2.5 Flash Preview (Adaptive)": "gemini-2.5-flash-preview-04-17",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
DEFAULT_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
     DEFAULT_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
     if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]


# --- Define Meeting Types ---
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]


# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_KEY)
    generation_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()


# --- Initialize Session State ---
default_state = {
    'processing': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,
    'selected_model_display_name': DEFAULT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text',
    'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'edit_notes_enabled': False, 'edited_notes_text': "",
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from PDF, updates session state on error."""
    try:
        pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts); return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e: st.session_state.error_message = f"📄 PDF Read Error: {e}"; return None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def create_expert_meeting_prompt(transcript, context=None):
    """Creates the prompt for 'Expert Meeting'."""
    prompt_parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:",
        "\n**Structure:**",
        "- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.",
        "- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.",
        "  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).",
        "  - **Answers:** Use bullet points directly below the question. Each bullet MUST be a complete sentence with one distinct fact. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or action items.",
        "\n**Additional Instructions:**",
        "- Accuracy is paramount. Capture ALL facts precisely.",
        "- Be clear and concise.",
        "- Include ONLY information present in the transcript.",
        "- If a section (like Opening Overview) isn't present, OMIT it.",
        "\n---",
        (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n")
    return "\n".join(filter(None, prompt_parts))

def create_earnings_call_prompt(transcript, context=None):
    """Creates the prompt for 'Earnings Call' with STRONGER structure enforcement."""
    prompt_parts = [
        "You are a financial analyst tasked with summarizing an earnings call transcript. Your output MUST be structured notes.",
        "Analyze the entire transcript and extract key information, numerical data, guidance, strategic comments, and management sentiment.",
        "Present the information using the EXACT headings and subheadings provided below. You MUST categorize all relevant comments under the correct heading.",
        "\n**Mandatory Structure:**",
        "- **Call Participants:** (List names and titles mentioned. If none mentioned, state 'Not specified')",
        "- **Opening Remarks/CEO Statement:** (Summarize key themes, vision, achievements/challenges mentioned.)",
        "- **Financial Highlights:** (List specific Revenue, Profitability, EPS, Margins, etc. Include numbers and comparisons (YoY/QoQ) EXACTLY as stated.)",
        "- **Segment Performance:** (If discussed, detail performance by business unit, geography, or product line.)",
        "- **Key Business Updates/Strategy:** (Summarize new initiatives, partnerships, market position, M&A activity discussed.)",
        "\n**Industry-Specific Categorization (Apply ONLY ONE section based on company type identified from the transcript):**",
        "\n  **>>> If IT Services Topics Discussed <<<**",
        "    *(Scan the transcript for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Future Investments / Capital Allocation:** (List all mentions of R&D, technology spend, acquisitions, buybacks, dividends.)",
        "    - **Talent Supply Chain:** (List all comments on hiring, attrition, utilization, training, location strategy.)",
        "    - **Org Structure Changes:** (List any mentions of leadership changes, reorganizations.)",
        "    - **Short-term Outlook & Demand:**",
        "      - **Guidance:** (List specific quarterly/annual targets for revenue, margin, EPS, etc.)",
        "      - **Order Booking / Pipeline:** (List comments on deal wins, TCV, book-to-bill, pipeline health.)",
        "      - **Macro Impact:** (Summarize comments on economic slowdown effects, client spending changes.)",
        "    - **Other Key IT Comments:** (List comments on Cloud, AI, digital transformation, major client verticals, etc.)",
        "\n  **>>> If QSR (Quick Service Restaurant) Topics Discussed <<<**",
         "    *(Scan the transcript for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Customer Proposition / Menu Strategy:** (List comments on new products, value offers, marketing, loyalty programs.)",
        "    - **Business Update (Operations):** (List SSSG/Comps, Traffic, Average Check/Ticket, Price increases mentioned.)",
        "    - **Unit Economics / Store Performance:** (List comments on restaurant margins, cost pressures like food/labor.)",
        "    - **Store Network:** (List comments on store openings, closures, remodels, domestic/international strategy.)",
        "    - **Other Key QSR Comments:** (List comments on digital sales, delivery, technology, drive-thru.)",
        "  *(If neither IT nor QSR specific topics are dominant, OMIT this entire Industry-Specific section)*",
        "\n- **Q&A Session Summary:**",
        "  - Summarize key analyst questions and management's core responses.",
        "  - Use this format STRICTLY: Q: [Concise Analyst Question Topic] / A: [Bulleted list of key points from management response]",
        "  - Focus on new information or clarifications.",
        "- **Guidance Summary (Reiterate/Confirm):** (Provide a final consolidated view of all forward-looking guidance mentioned.)",
        "- **Closing Remarks:** (Summarize final key message, if any.)",
        "\n**CRITICAL Instructions:**",
        "- Adhere STRICTLY to the headings and subheadings defined above.",
        "- Categorize every relevant point from the transcript under the appropriate heading.",
        "- Extract direct quotes for impactful statements using quotation marks.",
        "- Be factual and objective. DO NOT interpret or add external info.",
        "- If a standard section (like Segment Performance) was not discussed, state 'Not discussed'.",
        "- If neither IT nor QSR specific sections apply, OMIT that entire block.",
        "- Ensure all numerical data is captured accurately.",
        "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY:**\n")
    return "\n".join(filter(None, prompt_parts))

def create_docx(text):
    """Creates a Word document (.docx) in memory from text."""
    document = docx.Document()
    for line in text.split('\n'): document.add_paragraph(line)
    buffer = io.BytesIO(); document.save(buffer); buffer.seek(0)
    return buffer.getvalue()

def get_current_input_data():
    """Helper to get transcript/audio file based on input method and state."""
    input_type = st.session_state.input_method_radio
    transcript = None; audio_file = None
    if input_type == "Paste Text": transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file is not None: transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
    elif input_type == "Upload Audio": audio_file = st.session_state.audio_uploader
    return input_type, transcript, audio_file

# --- Function to generate and update the prompt text area ---
def update_prompt_display_text():
    """Generates the appropriate prompt based on selections and updates session state."""
    meeting_type = st.session_state.selected_meeting_type
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        prompt_func = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
        placeholder = "[TRANSCRIPT WILL BE INSERTED HERE]" if input_type != "Upload Audio" else None
        if input_type == "Upload Audio":
             base_prompt = prompt_func(transcript=None, context=temp_context)
             st.session_state.current_prompt_text = ("# NOTE FOR AUDIO INPUT: The '1. Transcribe first...' wrapper will be added *unless* you edit this prompt.\n"
                                                    "# If edited, ensure your prompt handles the audio file.\n"
                                                    "####################################\n\n" + base_prompt)
        else: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state.current_prompt_text: st.session_state.current_prompt_text = "# Enter your custom prompt here...\n# For audio, include transcription instructions."
    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom": st.session_state.current_prompt_text = ""

# --- Function to clear state ---
def clear_all_state():
    """Resets session state for inputs and outputs."""
    st.session_state.text_input = ""; st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None; st.session_state.context_input = ""
    st.session_state.add_context_enabled = False; st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_enabled = False; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.processing = False
    update_prompt_display_text(); st.toast("Inputs and outputs cleared!", icon="🧹")


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Input Section ---
with st.container(border=True):
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            st.radio(label="Select Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                     help="Choose meeting type. 'Custom' requires your own prompt.", on_change=update_prompt_display_text)
        with col1b:
            st.subheader("AI Model")
            st.selectbox(label="Choose model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name",
                         help="Select Gemini model. Preview/Experimental models may vary.")
    with col_main_2:
        st.subheader("") # Align button
        st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, help="Reset all inputs and outputs.", type="secondary") # Use secondary style?

    st.divider()
    st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio",
             horizontal=True, label_visibility="collapsed", on_change=update_prompt_display_text)
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...\nExample:\nSpeaker A: ...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider()
    col3a, col3b = st.columns(2)
    with col3a: # Context
        st.checkbox("Add Context (Optional)", key="add_context_enabled", help="Provide background info.", on_change=update_prompt_display_text)
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=100, key="context_input", on_change=update_prompt_display_text, placeholder="E.g., Attendees:...")
    with col3b: # View/Edit Prompt Checkbox
        if st.session_state.selected_meeting_type != "Custom":
            st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled", help="View/modify the AI prompt.", on_change=update_prompt_display_text)

# --- Prompt Display/Input Area (Conditional) ---
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
                   (st.session_state.selected_meeting_type == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_area_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_area_title)
        caption_text = "Edit prompt... For audio, 'Transcribe first...' added unless edited." if st.session_state.selected_meeting_type != "Custom" else "Enter prompt... For audio, include transcription instructions."
        st.caption(caption_text)
        st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# --- Generate Button ---
st.write("")
generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: st.info("⏳ Processing...", icon="🧠")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        # Edit Notes Checkbox
        st.checkbox("Edit Notes", key="edit_notes_enabled", help="Enable editing the generated notes before download.")
        if st.session_state.edit_notes_enabled:
             st.text_area("Editable Notes:", value=st.session_state.edited_notes_text, key="edited_notes_text", height=400, label_visibility="collapsed")
             notes_content_to_use = st.session_state.edited_notes_text
        else:
            st.markdown(st.session_state.generated_notes)
            notes_content_to_use = st.session_state.generated_notes

        # Action Buttons (Removed Copy, kept TXT/MD)
        st.write("") # Spacer
        col_btn_dl1, col_btn_dl2 = st.columns(2)
        with col_btn_dl1:
            st.download_button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes.txt", mime="text/plain", key='download-txt', help="Download as Plain Text", use_container_width=True)
        with col_btn_dl2:
            st.download_button(label="⬇️ Markdown", data=notes_content_to_use, file_name=f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes.md", mime="text/markdown", key='download-md', help="Download as Markdown File", use_container_width=True)

    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.rerun()

if st.session_state.processing:
    # --- Use try...finally to ensure processing state reset ---
    try:
        # Retrieve state
        meeting_type = st.session_state.selected_meeting_type
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled = st.session_state.view_edit_prompt_enabled
        user_prompt_text = st.session_state.current_prompt_text # Get latest from state (potentially edited)
        final_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None

        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # --- Validation ---
        if actual_input_type == "Paste Text" and not transcript_data: st.session_state.error_message = "⚠️ Text area is empty."
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_message = "⚠️ No PDF uploaded or failed to extract text."
        elif actual_input_type == "Upload Audio" and not audio_file_obj: st.session_state.error_message = "⚠️ No audio file uploaded."
        if meeting_type == "Custom" and not user_prompt_text.strip(): st.session_state.error_message = "⚠️ Custom Prompt cannot be empty."

        # --- Determine Final Prompt ---
        final_prompt_for_api = None
        prompt_was_edited_or_custom = (meeting_type == "Custom" or view_edit_enabled)

        if not st.session_state.error_message:
            if prompt_was_edited_or_custom:
                final_prompt_for_api = user_prompt_text
                final_prompt_for_api = final_prompt_for_api.split("####################################\n\n")[-1] # Clean note
            else: # Generate default prompt
                prompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
                if actual_input_type != "Upload Audio":
                     if transcript_data: final_prompt_for_api = prompt_function(transcript_data, final_context)
                     else: st.session_state.error_message = "Error: Transcript data missing." # Should be caught by validation
                else: # Audio: Add wrapper automatically ONLY if not custom/edited
                    base_prompt = prompt_function(transcript=None, context=final_context)
                    final_prompt_for_api = ("1. First, accurately transcribe the provided audio file.\n"
                                            "2. Then, using the transcription, create notes based on:\n---\n"
                                            f"{base_prompt}")

        # --- API Call Section ---
        if not st.session_state.error_message and final_prompt_for_api and (transcript_data or audio_file_obj):
            # --- Inner try...except for specific API/processing errors ---
            try:
                st.toast(f"🧠 Generating with {st.session_state.selected_model_display_name}...", icon="✨")
                model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=generation_config)
                response = None; api_payload = None; processed_audio_file_ref = None

                if actual_input_type == "Upload Audio":
                    if not audio_file_obj: raise ValueError("Audio file object missing.")
                    st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️")
                    audio_bytes = audio_file_obj.getvalue()
                    # --- Use file=audio_bytes --- Corrected Call ---
                    processed_audio_file_ref = genai.upload_file(
                        file=audio_bytes, # Pass bytes directly to 'file'
                        display_name=f"audio_{int(time.time())}_{audio_file_obj.name}",
                        mime_type=audio_file_obj.type
                    )
                    # --- End Correction ---
                    st.session_state.uploaded_audio_info = processed_audio_file_ref # Store ref
                    # Polling...
                    polling_start_time = time.time()
                    while processed_audio_file_ref.state.name == "PROCESSING":
                        if time.time() - polling_start_time > 300: raise TimeoutError("Audio processing timeout.")
                        st.toast(f"🎧 Processing '{audio_file_obj.name}'...", icon="⏳"); time.sleep(5)
                        processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                    if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio processing failed/unexpected state: {processed_audio_file_ref.state.name}")
                    st.toast(f"🎧 Audio ready!", icon="✅")
                    api_payload = [final_prompt_for_api, processed_audio_file_ref] # List payload
                else: # Text or PDF
                    if not transcript_data: raise ValueError("Transcript data missing.")
                    api_payload = final_prompt_for_api # String payload

                # Generate Content
                if api_payload: response = model.generate_content(api_payload)
                else: raise ValueError("API Payload construction failed.")

                # Handle Response
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    st.session_state.generated_notes = response.text.strip()
                    st.session_state.edited_notes_text = st.session_state.generated_notes # Init editor
                    st.toast("🎉 Notes generated!", icon="✅")
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     st.session_state.error_message = f"⚠️ AI response blocked. Reason: {response.prompt_feedback.block_reason}."
                elif response: st.session_state.error_message = "🤔 AI returned empty response."
                else: st.session_state.error_message = "😥 AI generation failed (No response)."

                # Cleanup Audio only AFTER successful API call
                if actual_input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None; st.toast("☁️ Temp audio cleaned up.", icon="🗑️")
                    except Exception as delete_err: st.warning(f"Could not delete temp audio: {delete_err}", icon="⚠️")

            except Exception as e: # Catch API/processing errors
                st.session_state.error_message = f"❌ Processing Error: {e}"
                if st.session_state.uploaded_audio_info: # Attempt cleanup on inner error
                    try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None
                    except Exception: pass
        # End of inner try...except

    # --- FINALLY block: Always runs after the outer try ---
    finally:
        st.session_state.processing = False
        # Rerun ONLY if error needs displaying OR if notes were generated successfully
        # This prevents rerunning just for validation errors before processing attempt
        if st.session_state.error_message or st.session_state.generated_notes:
             st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by [Google Gemini](https://deepmind.google.technologies/gemini/) | App by SynthNotes AI")

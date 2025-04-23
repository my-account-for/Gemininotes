# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELSimport io
import time
import tempfile
from datetime import datetime # For history timestamp & default date
from dotenv: DEFAULT_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
MEETING import load_dotenv
import PyPDF2
import docx # Still needed for DOCX fallback
import re #_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_ For cleaning filename suggestions

# --- Page Configuration ---
st.set_page_config(
    page_titleTYPES[0]

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY =="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed" os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
    /* Overall App Background🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure( */
    .stApp { background: linear-gradient(to bottom right, #F0F2F6api_key=API_KEY)
    filename_gen_config = {"temperature": 0.2,, #FFFFFF); }
    /* Main content area */
    .main .block-container { padding:  "max_output_tokens": 50, "response_mime_type": "text/plain"}
2rem; max-width: 1000px; margin: auto; }
    /* General Container    main_gen_config = {"temperature": 0.7, "top_p": 1.0 Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock, "top_k": 32, "max_output_tokens": 8192, ""][style*="border"] {
         background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 0.75rem;
         padding: 1.5response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORYrem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; }
    /* Headers */
_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as    h1 { color: #111827; font-weight: 700; text-align: center; margin-bottom: 0.5rem; }
    h2, h3 { color: #1F2937; font-weight: 600; border-bottom:  e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False,1px solid #E5E7EB; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    /* App Subtitle - Adjust selector index if layout changes */
    .main . 'processing_step': None, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,block-container > div:nth-child(3) > div > div > div > p { text-align
    'selected_model_display_name': DEFAULT_MODEL_NAME, 'selected_meeting_type':: center; color: #4B5563; font-size: 1.1rem; margin-bottom: 2rem; }
    /* Input Widgets */
    .stTextInput textarea, .stFile DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'Uploader div[data-testid="stFileUploaderDropzone"], .stTextArea textarea {
        border-radius: pdf_uploader': None, 'audio_uploader': None,
    'context_input': '', 'edit_notes_enabled0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB': False, 'edited_notes_text': "", 'suggested_filename': None, 'history': [],
}
for key, value in default_state.items():
    if key not in st.session_state;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05); transition: all 0.2s ease; }
    .stTextInput textarea:focus, .: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        pdf_file_stream.seek(0stFileUploader div[data-testid="stFileUploaderDropzone"]:focus-within, .stTextArea textarea:focus {
        border-color: #007AFF; box-shadow: inset 0 ); pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        1px 2px rgba(0, 0, 0, 0.05), 0 text = "\n".join(text_parts); return text.strip() if text else None
    except Py0 0 3px rgba(0, 122, 255, 0.2PDF2.errors.PdfReadError as e: st.session_state.error_message = f"📄);
        background-color: #FFFFFF; }
    .stFileUploader p { font-size: 0 PDF Read Error: {e}"; return None
    except Exception as e: st.session_state.error_.95rem; color: #4B5563; }
    /* Radio Buttons */
    div[role="radiogroup"] > label { background-color: #FFFFFF; border: 1px solidmessage = f"⚙️ PDF Extraction Error: {e}"; return None

def create_expert_meeting_prompt #D1D5DB; border-radius: 0.5rem;
        padding: 0.(transcript, context=None):
    prompt_parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:", "\n**Structure:**",
        "- **Opening overview or Expert6rem 1rem; margin-right: 0.5rem; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.03); background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture
        display: inline-block; margin-bottom: 0.5rem; }
    div[role="radiogroup"] label:hover { border-color: #9CA3AF; }
    div[role="radi ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.",
        "- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.",
        "  - **Questions:** Extractogroup"] input[type="radio"]:checked + div { background-color: #EFF6FF; border-color: #007AFF; color: #005ECB;
        font-weight:  clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold500; box-shadow: 0 1px 3px rgba(0, 122).",
        "  - **Answers:** Use bullet points directly below the question. Each bullet MUST be a complete sentence with one distinct fact. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub, 255, 0.1); }
    /* Checkbox styling */
    .stCheckbox-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or action items.",
        "\n**Additional { margin-top: 1rem; padding: 0.5rem; background-color: #F9FAFB Instructions:**",
        "- Accuracy is paramount. Capture ALL facts precisely.", "- Be clear and concise.",
        "-; border-radius: 0.5rem; }
    .stCheckbox label span { font-weight: Include ONLY information present in the transcript.", "- If a section (like Opening Overview) isn't present, OMIT it.", 500; color: #374151; }
    /* Selectbox Styling */

        "\n---", (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript    .stSelectbox > div { border-radius: 0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB; }
    .stSelectbox > div else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**:focus-within { border-color: #007AFF; box-shadow: 0 0 \n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n"); return "\n".join(filter(None, prompt_parts))

def create_earnings_call_0 3px rgba(0, 122, 255, 0.2); }prompt(transcript, context=None):
    prompt_parts = [
        "You are a financial analyst tasked
    /* Button Styling */
    .stButton > button { border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: 600; transition with summarizing an earnings call transcript. Your output MUST be structured notes.",
        "Analyze the entire transcript and extract key: all 0.2s ease-in-out; border: none; width: 100%; information, numerical data, guidance, strategic comments, and management sentiment.",
        "Present the information using the EXACT headings and subheadings }
    .stButton > button[kind="primary"] { background-color: #007AFF; color provided below. You MUST categorize all relevant comments under the correct heading.",
        "\n**Mandatory Structure:**",
        "- **Call Participants:** (List names and titles mentioned. If none mentioned, state 'Not specified')",
: white; box-shadow: 0 4px 6px rgba(0, 122, 255, 0.1), 0 1px 3px rgba(0, 0        "- **Opening Remarks/CEO Statement:** (Summarize key themes, vision, achievements/challenges mentioned.)",
, 0, 0.08); }
    .stButton > button[kind="primary"]:hover        "- **Financial Highlights:** (List specific Revenue, Profitability, EPS, Margins, etc. Include numbers and { background-color: #005ECB; box-shadow: 0 7px 14 comparisons (YoY/QoQ) EXACTLY as stated.)",
        "- **Segment Performance:** (If discussed, detail performance by business unit, geography, or product line.)",
        "- **Key Business Updates/Strategypx rgba(0, 122, 255, 0.1), 0 3:** (Summarize new initiatives, partnerships, market position, M&A activity discussed.)",
        "\n**Industrypx 6px rgba(0, 0, 0, 0.08); transform: translateY(--Specific Categorization (Apply ONLY ONE section based on company type identified from the transcript):**",
        "\n1px); }
    .stButton > button[kind="primary"]:focus { box-shadow: 0 0   **>>> If IT Services Topics Discussed <<<**",
        "    *(Scan the transcript for these specific topics0 3px rgba(0, 122, 255, 0.4); outline and categorize comments STRICTLY under these subheadings)*",
        "    - **Future Investments / Capital Allocation:** (: none; }
    .stButton > button:disabled, .stButton > button[kind="primary"]:disabled { background-color: #D1D5DB; color: #6B7280; boxList all mentions of R&D, technology spend, acquisitions, buybacks, dividends.)",
        "    --shadow: none; transform: none; cursor: not-allowed; }
     /* Secondary Button styling for Clear */ **Talent Supply Chain:** (List all comments on hiring, attrition, utilization, training, location strategy.)",
        "    - **Org Structure Changes:** (List any mentions of leadership changes, reorganizations.)",
        "
    .stButton>button[type="secondary"], .stButton>button.secondary-button { background-    - **Short-term Outlook & Demand:**",
        "      - **Guidance:** (List specific quarterlycolor: #F3F4F6; color: #1F2937; border: 1/annual targets for revenue, margin, EPS, etc.)",
        "      - **Order Booking / Pipeline:**px solid #D1D5DB;
        width: auto; padding: 0.5rem 1rem (List comments on deal wins, TCV, book-to-bill, pipeline health.)",
        "      ; margin-right: 0.5rem; font-weight: 500; }
    .- **Macro Impact:** (Summarize comments on economic slowdown effects, client spending changes.)",
        "    - **Other Key IT Comments:** (List comments on Cloud, AI, digital transformation, major client verticals, etc.)",
        "\n  stButton>button[type="secondary"]:hover, .stButton>button.secondary-button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
     /***>>> If QSR (Quick Service Restaurant) Topics Discussed <<<**",
         "    *(Scan the transcript Download Buttons */
    .stDownloadButton > button { border-radius: 0.5rem; padding: for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Customer Proposition / 0.6rem 1.2rem; font-weight: 500; background-color: Menu Strategy:** (List comments on new products, value offers, marketing, loyalty programs.)",
        "    - #F3F4F6; color: #1F2937; border: 1px solid **Business Update (Operations):** (List SSSG/Comps, Traffic, Average Check/Ticket, Price #D1D5DB; transition: background-color 0.2s ease-in-out; width: auto increases mentioned.)",
        "    - **Unit Economics / Store Performance:** (List comments on restaurant margins, cost; margin-top: 0; margin-right: 0.5rem;} /* Add margin-right */ pressures like food/labor.)",
        "    - **Store Network:** (List comments on store openings, closures
    .stDownloadButton > button:hover { background-color: #E5E7EB; border-, remodels, domestic/international strategy.)",
        "    - **Other Key QSR Comments:** (Listcolor: #9CA3AF; }
    /* Output Area Styling */
    .output-container { background comments on digital sales, delivery, technology, drive-thru.)",
        "  *(If neither IT nor Q-color: #F9FAFB; border: 1px solid #E5E7EB; border-SR specific topics are dominant, OMIT this entire Industry-Specific section)*",
        "\n- **Q&A Session Summary:**",
        "  - Summarize key analyst questions and management's core responses.",
        radius: 0.75rem; padding: 1.5rem; margin-top: 1.5rem;"  - Use this format STRICTLY: Q: [Concise Analyst Question Topic] / A: [Bulleted min-height: 150px; }
    .output-container .stMarkdown { background-color: transparent; border: none; padding: 0; color: #374151; font-size: list of key points from management response]",
        "  - Focus on new information or clarifications.",
        "- ** 1rem; line-height: 1.6; }
    .output-container .stMarkdown h3, .output-container .stMarkdown h4, .output-container .stMarkdown strong { color: #Guidance Summary (Reiterate/Confirm):** (Provide a final consolidated view of all forward-looking guidance mentioned.)",
        "- **Closing Remarks:** (Summarize final key message, if any.)",
        "\n**CRITICAL Instructions:**",
        "- Adhere STRICTLY to the headings and subheadings defined above.",
        "- Categorize every111827; font-weight: 600; }
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem; }
    .output-container .initial-prompt { color: #6B7280; font-style: italic relevant point from the transcript under the appropriate heading.",
        "- Extract direct quotes for impactful statements using quotation marks.",
        "- Be factual and objective. DO NOT interpret or add external info.",
        "- If a standard section (like Segment Performance) was not discussed, state 'Not discussed'.",
        "- If neither IT nor QSR specific sections; text-align: center; padding-top: 2rem; }
    /* Prompt Edit Area */
    #prompt-edit-area textarea { font-family: monospace; font-size: 0.9rem; line- apply, OMIT that entire block.",
        "- Ensure all numerical data is captured accurately.",
        "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" ifheight: 1.4; background-color: #FDFDFD; }
    /* History Styling */ transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT
    .history-entry { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
    .history-entry:last-child { border-:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGbottom: none; }
    .history-entry pre { background-color: #f0f2f6; paddingS CALL SUMMARY:**\n"); return "\n".join(filter(None, prompt_parts))

def create_docx(: 0.5rem; border-radius: 0.25rem; max-height: 150px; overflow-y: auto; }
    /* Footer */
    footer { text-align:text):
    document = docx.Document(); [document.add_paragraph(line) for line in text center; color: #9CA3AF; font-size: 0.8rem; padding: 2.split('\n')]
    buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()

# --- Corrected get_current_input_data function ---
def getrem 0 1rem 0; }
    footer a { color: #6B7280_current_input_data():
    """Helper to get transcript/audio file based on input method and state."""
    input; text-decoration: none; }
    footer a:hover { color: #007AFF; text_type = st.session_state.input_method_radio
    transcript = None; audio_file =-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Define Available None
    if input_type == "Paste Text":
        transcript = st.session_state.text_ Models & Meeting Types (Keep as is) ---
AVAILABLE_MODELS = {
    "Gemini 1input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session.5 Flash (Fast & Versatile)": "gemini-1.5-flash", "Gemini _state.pdf_uploader
        if pdf_file is not None:
            try:
                # Ind1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Geminientation fixed here
                transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue())) 1.5 Flash-8B (High Volume)": "gemini-1.5-flash-8
            except Exception as e:
                st.session_state.error_message = f"Error processing PDFb", "Gemini 2.0 Flash (Next Gen Speed)": "gemini-2.0- upload: {e}"
                transcript = None # Ensure transcript is None on error
    elif input_type == "flash",
    "Gemini 2.0 Flash-Lite (Low Latency)": "gemini-2.0-flash-lite", "Gemini 2.5 Flash Preview (Adaptive)": "geminiUpload Audio":
        audio_file = st.session_state.audio_uploader
    return input_type, transcript-2.5-flash-preview-04-17",
    "Gemini 2.5, audio_file
# --- End of corrected function ---

def update_prompt_display_text():
    meeting_type Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-0 = st.session_state.selected_meeting_type
    if st.session_state.view_edit3-25",
}
DEFAULT_MODEL_NAME = "Gemini 2.5 Pro Exp._prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state. Preview (Enhanced Reasoning)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_MODEL_NAMEcontext_input.strip() if st.session_state.add_context_enabled else None
        input_type = st = "Gemini 1.5 Flash (Fast & Versatile)"
MEETING_TYPES = ["Expert Meeting.session_state.input_method_radio
        prompt_func = create_expert_meeting_prompt if", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]

# meeting_type == "Expert Meeting" else create_earnings_call_prompt
        placeholder = "[TRANSCRIPT ...]" if --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI input_type != "Upload Audio" else None
        if input_type == "Upload Audio":
             base_prompt =_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_ prompt_func(transcript=None, context=temp_context)
             st.session_state.current_KEY)
    filename_gen_config = {"temperature": 0.2, "max_output_tokensprompt_text = ("# NOTE FOR AUDIO...\n#######\n\n" + base_prompt)
        ": 50, "response_mime_type": "text/plain"}
    main_gen_configelse: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state. = {"temperature": 0.7, "top_p": 1.0, "top_k":current_prompt_text: st.session_state.current_prompt_text = "# Enter custom prompt..."
 32, "max_output_tokens": 8192, "response_mime_type":    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"} # Simpler config for transcription
    safety_settings = [{"category st.session_state.current_prompt_text = ""

def clear_all_state():
    st": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_model_display_name = DEFAULT_MODEL_NAME
    st.session_state.input__HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXmethod_radio = 'Paste Text'
    st.session_state.text_input = ""; st.sessionUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨");_state.pdf_uploader = None
    st.session_state.audio_uploader = None; st. st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'processing_step': None, 'generating_filename': False, 'generated_notes': None, 'error_session_state.context_input = ""
    st.session_state.add_context_enabled = False; st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_enabled = False; st.session_state.generated_notes = None
    st.sessionmessage': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.processing = False
    st.session
    'selected_model_display_name': DEFAULT_MODEL_NAME, 'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_state.suggested_filename = None; st.session_state.uploaded_audio_info = None
    st.session_state.history = []; st.session_state.processing_step = None
    update_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '', 'edit_notes_enabled_prompt_display_text(); st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested': False, 'edited_notes_text': "", 'suggested_filename': None,
    'history': [],
}
for key, value in default_state.items():
    if key not in st.session_filename(notes_content, meeting_type):
    if not notes_content: return None
    try_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text:
        st.session_state.generating_filename = True; st.toast("💡 Generating filename...", icon="⏳")
        filename_model = genai.GenerativeModel("gemini-1.5-flash_from_pdf(pdf_file_stream):
    try:
        pdf_file_stream.seek")
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text__type.replace(" ", "")
        filename_prompt = (f"Analyze notes. Suggest filename: YYYYparts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts); return text.strip() if text else None
    MMDD_ClientOrTopic_MeetingType. Use {today_date}. "
                           f"Extract main clientexcept PyPDF2.errors.PdfReadError as e: st.session_state.error_message = f/topic. Use CamelCase/underscores. Type: '{mt_cleaned}'. Max 3 words topic. "
                           f"Examples: {today_date}_AcmeCorp_{mt_cleaned}. Output ONLY filename.\n"📄 PDF Read Error: {e}"; return None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def create_expert_meeting\nNOTES:\n{notes_content[:1500]}")
        response = filename_model.generate_content_prompt(transcript, context=None):
    prompt_parts = [
        "You are an expert meeting(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if note-taker analyzing an expert consultation or similar focused meeting.",
        # ... (rest of prompt as before) ...
         response and hasattr(response, 'text') and response.text:
            s_name = re.sub((f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    r'[^\w\-.]', '_', response.text.strip())[:100]
            if re]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context.match(r"\d{8}_[\w\-\.]+_\w+", s_name): st.toast("💡, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n"); return "\ Filename suggestion ready!", icon="✅"); return s_name
            else: st.warning(f"Filename suggestion '{s_name}' bad format.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedbackn".join(filter(None, prompt_parts))

def create_earnings_call_prompt(transcript, context=None):
    prompt_parts = [
        "You are a financial analyst tasked with summarizing an earnings') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response. call transcript. Your output MUST be structured notes.",
        # ... (rest of prompt as before) ...
        prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not(f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""), gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY = False

def add_to_history(notes):
    if not notes: return
    timestamp = datetime.now().:**\n"); return "\n".join(filter(None, prompt_parts))

def create_docx(text):
strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"timestamp": timestamp    document = docx.Document(); [document.add_paragraph(line) for line in text.split('\, "notes": notes}
    current_history = st.session_state.get('history', [])
    currentn')]
    buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return_history.insert(0, new_entry); st.session_state.history = current_history[:3 buffer.getvalue()

def get_current_input_data():
    """Helper to get transcript/audio file based]

def restore_note_from_history(index):
    if 0 <= index < len(st on input method and state."""
    input_type = st.session_state.input_method_radio
    transcript = None; audio_file = None
    if input_type == "Paste Text":
        transcript.session_state.history):
        entry = st.session_state.history[index]
        st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":["notes"]
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None
        st.session_state.error_message = None; st.toast
        pdf_file = st.session_state.pdf_uploader
        # Correct logic: Check if pdf_file exists before calling getvalue()
        if pdf_file is not None:
            try:
                # --- CORRECT(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI --- INDENTATION ---
                transcript = extract_text_from_pdf(io.BytesIO(pdf_file
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.getvalue()))
                # --- END CORRECTION ---
            except Exception as e: # Catch potential errors during get.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1avalue/BytesIO too
                st.session_state.error_message = f"Error processing PDF upload: {e, col1b = st.columns(2)
        with col1a: st.subheader("Meeting Details}"
                transcript = None # Ensure transcript is None on error
    elif input_type == "Upload Audio":
        audio_file = st.session_state.audio_uploader
    return input_type, transcript,"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True, on_change=update_prompt_display_text)
        with col1b: audio_file


def update_prompt_display_text():
    # (Keep function as is)
     st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.meeting_type = st.session_state.selected_meeting_type
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_statecontext_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        prompt_func = create_expert_meeting_prompt if, use_container_width=True, type="secondary")
    st.divider(); st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key meeting_type == "Expert Meeting" else create_earnings_call_prompt
        placeholder = "[TRANSCRIPT ...]" if input="input_method_radio", horizontal=True, label_visibility="collapsed", on_change=update_prompt_type != "Upload Audio" else None
        if input_type == "Upload Audio":
             base_prompt = prompt_func(transcript=None, context=temp_context)
             st.session_state.current_display_text)
    input_type_ui = st.session_state.input_method_radio_prompt_text = ("# NOTE FOR AUDIO...\n#######\n\n" + base_prompt)

    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=        else: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state.current_    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")
    st.divider(); col3a, col3bprompt_text: st.session_state.current_prompt_text = "# Enter custom prompt..."
    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom": st.session_state.current_prompt_text = ""

def clear_all_state():
    # (Keep = st.columns(2) # Optional Elements
    with col3a: # Context
        st.checkbox function as is)
    st.session_state.text_input = ""; st.session_state.pdf("Add Context", key="add_context_enabled", on_change=update_prompt_display_text)_uploader = None
    st.session_state.audio_uploader = None; st.session_state.
        # --- CORRECTED INDENTATION ---
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=100, key="context_input", on_context_input = ""
    st.session_state.add_context_enabled = False; st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_change=update_prompt_display_text, placeholder="Attendees...")
        # --- END CORRECTION ---
    enabled = False; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.sessionwith col3b: # View/Edit Prompt Checkbox
        # --- CORRECTED INDENTATION ---
        if_state.error_message = None; st.session_state.processing = False
    st.session_state.suggest st.session_state.selected_meeting_type != "Custom":
            st.checkbox("View/Edit Prompt",ed_filename = None; st.session_state.uploaded_audio_info = None
    st.session key="view_edit_prompt_enabled", on_change=update_prompt_display_text)
        # --- END_state.history = []; st.session_state.processing_step = None
    update_prompt_display CORRECTION ---

# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
_text(); st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type):
    # (Keep function as is)
    if not notes_content                   (st.session_state.selected_meeting_type == "Custom")
if show_prompt_area: return None
    try:
        st.session_state.generating_filename = True; st.toast:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
("💡 Generating filename...", icon="⏳")
        filename_model = genai.GenerativeModel("gemini-1        st.subheader(prompt_title); caption = "Edit prompt..." if st.session_state.selected_meeting_type.5-flash")
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "")
        filename_prompt = (f"Analyze notes. Suggest != "Custom" else "Enter prompt..."
        st.caption(caption); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height filename: YYYYMMDD_ClientOrTopic_MeetingType. Use {today_date}. "
                           =350, label_visibility="collapsed")

# Generate Button
st.write(""); generate_button =f"Extract main client/topic. Use CamelCase/underscores. Type: '{mt_cleaned}'. Max 3 words topic. "
                           f"Examples: {today_date}_AcmeCorp_{mt_cleaned}. st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# --- Output Section ---
output Output ONLY filename.\n\nNOTES:\n{notes_content[:1500]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_container = st.container(border=True)
with output_container:
    st.markdown('<div_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text: class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: status_message = st.session_state.processing_step or "⏳ Processing..."; st.info
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip(status_message, icon="🧠")
    elif st.session_state.generating_filename: st.())[:100]
            if re.match(r"\d{8}_[\w\-\.]+_\w+", s_name): st.toast("💡 Filename suggestion ready!", icon="✅"); return s_info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: stname
            else: st.warning(f"Filename suggestion '{s_name}' bad format.", icon="⚠️"); return.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); Notes")
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except_to_use = st.session_state.edited_notes_text if st.session_state.edit Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use,    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    # key="edited_notes_text", height=400, label_visibility="collapsed")
        else: (Keep function as is)
    if not notes: return
    timestamp = datetime.now().strftime("%Y-%m st.markdown(notes_content_to_use)
        default_fname = f"{st.session_-%d %H:%M:%S")
    new_entry = {"timestamp": timestamp, "notes": notesstate.selected_meeting_type.lower().replace(' ', '_')}_notes"; fname_base = st.session_state}
    current_history = st.session_state.get('history', [])
    current_history.insert(.suggested_filename or default_fname
        st.write(""); col_btn_dl1, col_0, new_entry); st.session_state.history = current_history[:3]

def restore_note_frombtn_dl2 = st.columns(2)
        with col_btn_dl1: st.download_history(index):
    # (Keep function as is)
    if 0 <= index < len(st._button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_session_state.history):
        entry = st.session_state.history[index]
        st.width=True)
        with col_btn_dl2: st.download_button(label="⬇️session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False; st.session_ Markdown", data=notes_content_to_use, file_name=f"{fname_base}.md",state.suggested_filename = None
        st.session_state.error_message = None; st. mime="text/markdown", key='download-md', use_container_width=True)
    else:toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3.")
with st.container(border=True): # Input Section
    col_main_1, col_)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a: st.subheader("Meeting Detailswith st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**"); st.markdown(f"```\n{entry['notes'][:200]}...\n```")
"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True, on_change=update_prompt_display_text)
        with col1b:                st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_click=restore_note_from_history, args=(i,))
                if i < len(st.session_state.history) - 1: st.divider()


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generating_filename = False; st_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary")
    st.divider(); st.subheader("Source Input")
.session_state.processing_step = None
    st.session_state.generated_notes = None; st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed", on_change=update_prompt = False
    st.session_state.error_message = None; st.session_state.suggested_display_text)
    input_type_ui = st.session_state.input_method_radio_filename = None
    st.rerun()

if st.session_state.processing and not st.
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=session_state.generating_filename:
    try: # Outer try
        # Retrieve state & inputs
        meeting_150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
type = st.session_state.selected_meeting_type
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled =    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg st.session_state.view_edit_prompt_enabled
        user_prompt_text = st.session','flac','aac'], key="audio_uploader")
    st.divider(); col3a, col3_state.current_prompt_text
        final_context = st.session_state.context_input.b = st.columns(2) # Optional Elements
    with col3a: # Context
        st.checkbox("Add Context", key="add_context_enabled", on_change=update_prompt_display_textstrip() if st.session_state.add_context_enabled else None
        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # Validation
        )
        # --- CORRECTED INDENTATION ---
        if st.session_state.add_context_enabled:
if actual_input_type == "Paste Text" and not transcript_data: st.session_state.error            st.text_area("Context Details:", height=100, key="context_input", on_change=update_message = "⚠️ Text area empty."
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_message_prompt_display_text, placeholder="Attendees...")
        # --- END CORRECTION ---
    with col3b = "⚠️ PDF error."
        elif actual_input_type == "Upload Audio" and not audio_file: # View/Edit Prompt Checkbox
        # --- CORRECTED INDENTATION ---
        if st.session__obj: st.session_state.error_message = "⚠️ No audio uploaded."
        if meeting_state.selected_meeting_type != "Custom":
            st.checkbox("View/Edit Prompt", key="view_type == "Custom" and not user_prompt_text.strip(): st.session_state.error_message = "⚠️ Custom Prompt empty."

        # Initialize variables for API call
        final_prompt_for_api = Noneedit_prompt_enabled", on_change=update_prompt_display_text)
        # --- END CORRECTION ---


        api_payload_parts = [] # Use a list to build the payload
        processed_audio_file_ref# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
                   (st. = None # Cloud reference
        transcribed_text_for_notes = transcript_data # Use original transcript unless overwrittensession_state.selected_meeting_type == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if st.session by audio step

        if not st.session_state.error_message:
            # --- Inner Try for pre_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_title); caption = "Edit prompt..." if st.session_state.selected_meeting_type != "Custom"-processing, prompt-building, audio handling ---
            try:
                # Handle Audio Upload & Potential Transcription FIRST
                if actual_input_type == "Upload Audio":
                    if not audio_file_obj: raise else "Enter prompt..."
        st.caption(caption); st.text_area(label="Prompt Text:", value=st ValueError("Audio file missing.")
                    st.session_state.processing_step = f"☁️ Uploading '{audio_file_obj.name}'..."
                    st.rerun() # Show status

                    audio_.session_state.current_prompt_text, key="current_prompt_text", height=350bytes = audio_file_obj.getvalue(); temp_file_path = None
                    try: # Tempfile handling
, label_visibility="collapsed")

# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_fileprocessing or st.session_state.generating_filename)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-_obj.name)[1]) as temp_file:
                            temp_file.write(audio_bytes); temp_file_path = temp_file.name
                        if temp_file_path: processed_audio_container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: statusfile_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}")
                        else: raise Exception_message = st.session_state.processing_step or "⏳ Processing..."; st.info(status_message, icon="🧠")
    elif st.session_state.generating_filename: st.info("⏳ Generating("Temp file creation failed.")
                    finally:
                        if temp_file_path and os.path.exists( filename...", icon="💡")
    elif st.session_state.error_message: st.error(sttemp_file_path): os.remove(temp_file_path) # Disk cleanup

                    st.session.session_state.error_message, icon="🚨"); st.session_state.error_message = None_state.uploaded_audio_info = processed_audio_file_ref # Store cloud ref for potential later cleanup
                    # Polling...
                    st.session_state.processing_step = f"🎧 Processing audio..."
                    st.
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = strerun() # Show status
                    polling_start = time.time()
                    while processed_audio_file_ref.state.name == "PROCESSING":
                        # --- CORRECTED INDENTATION ---
                        if time.time().session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st - polling_start > 300:
                            raise TimeoutError("Audio processing timed out after 5 minutes.")
                        .text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_# --- END CORRECTION ---
                        time.sleep(5); processed_audio_file_ref = genai.get_to_use)
        default_fname = f"{st.session_state.selected_meeting_type.file(processed_audio_file_ref.name) # Refresh state

                    if processed_audio_file_lower().replace(' ', '_')}_notes"; fname_base = st.session_state.suggested_filename or default_ref.state.name != "ACTIVE": raise Exception(f"Audio processing failed or has unexpected state: {processed_audiofname
        st.write(""); col_btn_dl1, col_btn_dl2 = st.columns_file_ref.state.name}")
                    st.toast(f"🎧 Audio ready!", icon="✅")

                    (2)
        with col_btn_dl1: st.download_button(label="⬇️ TX# --- TWO-STEP LOGIC Check: Default Prompt for Audio? ---
                    prompt_was_edited_or_customT", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with col = (meeting_type == "Custom" or view_edit_enabled)
                    if not prompt_was__btn_dl2: st.download_button(label="⬇️ Markdown", data=notes_content_edited_or_custom:
                        # Default Audio -> Two Step: Transcribe First
                        st.session_state.to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='processing_step = f"✍️ Transcribing audio..."
                        st.rerun() # Show status
                        download-md', use_container_width=True)
    else: st.markdown("<p class='initialtranscription_prompt = "Transcribe the provided audio file accurately. Output only the raw transcribed text."
                        transcription_model = genai.GenerativeModel(model_name=selected_model_id, safety_settings-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if=safety_settings, generation_config=transcription_gen_config)
                        transcript_response = transcription_model.generate not st.session_state.history: st.caption("No history yet.")
    else:
        for_content([transcription_prompt, processed_audio_file_ref])

                        if transcript_response and hasattr i, entry in enumerate(st.session_state.history):
            with st.container():
                st(transcript_response, 'text') and transcript_response.text.strip():
                            transcribed_text_.markdown(f"**#{i+1} - {entry['timestamp']}**"); st.markdown(f"```\n{entry['notes'][:200]}...\n```")
                st.button(f"Viewfor_notes = transcript_response.text.strip() # OVERWRITE with transcript
                            st.toast("✍/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_️ Transcription complete!", icon="✅")
                            actual_input_type = "Generated Transcript" # Mark as textfrom_history, args=(i,))
                if i < len(st.session_state.history) --based now
                            # Cloud file object might be deleted now if only needed for transcription, or keep for later cleanup 1: st.divider()

# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generating_filename = False; st.session_state
                            # Let's keep st.session_state.uploaded_audio_info for now, cleanup happens later
.processing_step = None
    st.session_state.generated_notes = None; st.session_                        else:
                             st.session_state.error_message = "Error: Failed to transcribe audio."
                             state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.suggested_filename = None# Need to stop processing if transcription fails
                             raise Exception("Audio transcription failed.")
                    # If custom/edited prompt,
    st.rerun()

if st.session_state.processing and not st.session_state. we proceed with the single call using the audio ref later
                # --- End Audio Pre-processing ---

                # --- Determinegenerating_filename:
    try: # Outer try
        # --- State & Input Retrieval ---
        meeting_type = Final Prompt Text ---
                prompt_was_edited_or_custom = (meeting_type == "Custom" or view_ st.session_state.selected_meeting_type
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled = st.edit_enabled) # Re-check needed? No harm.
                if not st.session_state.error_messagesession_state.view_edit_prompt_enabled
        user_prompt_text = st.session_state:
                     if prompt_was_edited_or_custom:
                         final_prompt_for_api =.current_prompt_text
        final_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        actual_input_type, transcript_ user_prompt_text.split("####################################\n\n")[-1] # Clean potential note
                         data, audio_file_obj = get_current_input_data()

        # --- Validation ---
        # If it was audio + custom prompt, we still need the audio file ref later
                     else: # Default prompt logicif actual_input_type == "Paste Text" and not transcript_data: st.session_state.error
                         prompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings__message = "⚠️ Text area empty."
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_messagecall_prompt
                         # Use transcribed_text_for_notes (which is original transcript for Text/PDF, or generated transcript for default Audio)
                         if transcribed_text_for_notes:
                             final_prompt_for_api = "⚠️ PDF error."
        elif actual_input_type == "Upload Audio" and not audio_file = prompt_function(transcribed_text_for_notes, final_context)
                         else:
                             _obj: st.session_state.error_message = "⚠️ No audio uploaded."
        if meeting_type == "Custom" and not user_prompt_text.strip(): st.session_state.error_message# This case implies Text/PDF input failed earlier, or audio transcription failed
                              st.session_state.error_ = "⚠️ Custom Prompt empty."

        # --- Initialize variables ---
        final_prompt_for_api = None
        processed_audio_file_ref = None # Stores the cloud reference for audio
        temp_file_path =message = "Error: Source text unavailable for note generation."

                # --- Prepare API Payload ---
                if not st. None # Stores local temp file path for audio
        api_payload_parts = [] # Use a list to buildsession_state.error_message and final_prompt_for_api:
                    api_payload_parts.append(final_prompt_for_api) # Add the prompt text first
                    # Add audio object ONLY if it the payload
        prompt_was_edited_or_custom = (meeting_type == "Custom" or view's audio input AND it's a custom/edited prompt
                    if actual_input_type == "Upload_edit_enabled)
        notes_model_id = selected_model_id # Model for the main notes generation

        if not st.session_state.error_message:
            # --- Inner Try for pre-processing, prompt Audio" and prompt_was_edited_or_custom:
                        if processed_audio_file_ref:
                            api_payload_parts.append(processed_audio_file_ref)
                        else:
                            st.session-building, audio handling ---
            try:
                st.session_state.processing_step = "🧠 Initializing..."_state.error_message = "Error: Audio reference missing for custom audio prompt."
                    elif actual_input_type not
                st.rerun() # Show initial status quickly

                # --- Handle Audio Upload & Transcription (if applicable) ---
 in ["Upload Audio", "Generated Transcript", "Paste Text", "Upload PDF"]:
                        # Safety check for unexpected                if actual_input_type == "Upload Audio":
                    if not audio_file_obj: raise ValueError input type state
                        st.session_state.error_message = f"Error: Unexpected input type '{actual_("Audio missing.")
                    st.session_state.processing_step = f"☁️ Uploading '{audio_file_obj.name}'..."
                    st.rerun()

                    audio_bytes = audio_fileinput_type}' for final payload."
                elif not st.session_state.error_message: # If_obj.getvalue(); temp_file_path = None
                    try: # Tempfile handling
                        with temp prompt is None/empty but no error yet
                     st.session_state.error_message = "Error: Failed to determinefile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj. the prompt for the AI."


            except Exception as pre_process_err:
                st.session_statename)[1]) as temp_file:
                            temp_file.write(audio_bytes); temp_file_path = temp_file.name
                        if temp_file_path: processed_audio_file_ref.error_message = f"❌ Pre-processing Error: {pre_process_err}"
                # Attempt cleanup = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}")
                        else: raise Exception("Temp file creation failed.") if audio ref exists
                if st.session_state.uploaded_audio_info:
                     try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path): os..uploaded_audio_info = None
                     except Exception: pass


            # --- Generate Notes API Call (only if noremove(temp_file_path) # Disk cleanup

                    st.session_state.uploaded_audio_info errors and payload is ready) ---
            if not st.session_state.error_message and api_payload_ = processed_audio_file_ref # Store cloud ref for potential later cleanup
                    # Polling...
                    parts:
                try:
                    st.session_state.processing_step = f"📝 Generating notes..."
                    stst.session_state.processing_step = f"🎧 Processing audio..."
                    st.rerun() # Show status
.rerun() # Show status
                    model = genai.GenerativeModel(model_name=selected_                    polling_start = time.time()
                    # --- CORRECTED INDENTATION IN POLLING LOOP ---
                    while processed_audio_file_ref.state.name == "PROCESSING":
                        if time.time()model_id, safety_settings=safety_settings, generation_config=main_gen_config)
                     - polling_start > 300:
                            raise TimeoutError("Audio processing timeout after 5 minutes.")
                        timeresponse = model.generate_content(api_payload_parts) # Send the constructed payload list

                    # Handle Response
                    if response and hasattr(response, 'text') and response.text and response.text.strip():.sleep(5) # Check status every 5 seconds
                        processed_audio_file_ref = genai.get
                        st.session_state.generated_notes = response.text.strip()
                        st.session__file(processed_audio_file_ref.name) # Refresh state
                    # --- END CORRECTION ---
                    state.edited_notes_text = st.session_state.generated_notes
                        add_to_history(st.session_state.generated_notes)
                        st.session_state.suggested_filename =if processed_audio_file_ref.state.name != "ACTIVE":
                        raise Exception(f"Audio processing failed or ended in unexpected state: {processed_audio_file_ref.state.name}")
                    st.toast generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                        st.toast("(f"🎧 Audio ready!", icon="✅")

                    # --- TWO-STEP LOGIC Check: Default Prompt🎉 Notes generated!", icon="✅")
                    elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         st.session_state.error_message = f"⚠️ Response blocked: for Audio? ---
                    if not prompt_was_edited_or_custom:
                        st.session_state {response.prompt_feedback.block_reason}."
                    elif response: st.session_state.error.processing_step = f"✍️ Transcribing audio..."
                        st.rerun() # Show status
                        transcription_prompt = "Transcribe the provided audio file accurately. Output only the raw transcribed text."
                        # Use selected_message = "🤔 AI returned empty response."
                    else: st.session_state.error_message = model for transcription too, or maybe force Flash? Let's use selected for now.
                        transcription_model = genai "😥 Generation failed (No response)."

                except Exception as api_call_err: # Catch API call errors
                    .GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generationst.session_state.error_message = f"❌ API Call Error: {api_call_err}"

            #_config=transcription_gen_config)
                        transcript_response = transcription_model.generate_content([ --- Cloud Audio File Cleanup (if reference still exists) ---
            # This runs *after* the main API call attempttranscription_prompt, processed_audio_file_ref])

                        if transcript_response and hasattr(transcript_response,, regardless of inner success/failure
            if st.session_state.uploaded_audio_info:
                try 'text') and transcript_response.text.strip():
                            transcript_data = transcript_response.text.strip() #:
                    st.toast("☁️ Cleaning up cloud audio...", icon="🗑️")
                    genai.delete OVERWRITE with transcript
                            st.toast("✍️ Transcription complete!", icon="✅")
                            actual_input_type_file(st.session_state.uploaded_audio_info.name)
                    st.session_state. = "Generated Transcript" # Mark as text-based now
                            # Cloud audio object no longer needed for the *notes* calluploaded_audio_info = None
                except Exception as e: st.warning(f"Cloud cleanup failed: {e}", icon="⚠️")


        # --- End of inner try...except ---
    # --- Outer FIN in this path
                        else:
                            raise Exception("Failed to transcribe audio.")
                    # --- End Two-Step Logic ---ALLY block: Always runs ---
    finally:
        st.session_state.processing = False
        st.session_
                # --- End Audio Pre-processing ---

                # --- Determine Final Prompt Text (if not already custom) ---
                if not prompt_was_edited_or_custom:
                    prompt_function = create_expertstate.processing_step = None # Clear step
        # Rerun to display final results or errors
        #_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
                    if This ensures the UI updates after processing is flagged as false
        st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

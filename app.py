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
from streamlit_extras.copy_button import copy_button # CORRECT: Import for the copy button

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
    /* ... CSS styles ... */
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
    "Gemini 2.0 Flash (Fast & Versatile)": "gemini-2.0-flash-lite",
    "Gemini 2.5 Flash (Fast & Versatile)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (paid)": "gemini-2.5-pro",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "gemini-2.5-pro",
}
# Ensure default models exist in the available list
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro (paid)"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"

DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 2.5 Flash (Fast & Versatile)"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]

DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 2.5 Flash (Fast & Versatile)"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]

MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]
DEFAULT_EARNINGS_CALL_MODE = EARNINGS_CALL_MODES[1]

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
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("### 🔑 API Key Not Found!", icon="🚨")
    st.stop()
try:
    genai.configure(api_key=API_KEY)
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    summary_gen_config = {"temperature": 0.6, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    enrichment_gen_config = {"temperature": 0.4, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    refinement_gen_config = {"temperature": 0.3, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e:
    st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨")
    st.stop()


# --- Prompts Definitions (with Notion Formatting) ---
PROMPTS = {
    "Expert Meeting": {
        "Option 1: Existing (Detailed & Strict)": """You are an expert meeting note-taker analyzing a transcript for an investment firm. Your final output must be formatted for Notion.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If present, include this FIRST as standard bullet points.
- **Q&A format (Notion Toggle List):** Structure the main body as a Notion-style toggle list.
  - **Question (Toggle Header):** Format each question as a toggle header. Example: `> **Q: What is the market outlook?**`
  - **Answer (Inside Toggle):** The answer must be indented under the question toggle. Use standard bullet points (`- `) for the answer. Each bullet MUST be a complete sentence representing one single, distinct factual point. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets.

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- Adhere strictly to one fact per bullet point.
- Include ONLY information present in the transcript. DO NOT add interpretations, summaries, or conclusions not explicitly stated.
- If a section (like Opening Overview) isn't present, OMIT it.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Notion Toggle Format - Strict):**
""",
        "Option 2: Less Verbose (Default)": """You are an expert meeting note-taker analyzing a transcript for an investment firm. Your final output must be formatted for Notion.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If present, include this FIRST as standard bullet points.
- **Q&A format (Notion Toggle List):** Structure the main body as a Notion-style toggle list.
  - **Question (Toggle Header):** Format each question as a toggle header. Example: `> **Q: What is the market outlook?**`
  - **Answer (Inside Toggle):** The answer must be indented under the question toggle. Use standard bullet points (`- `) for the answer. Each bullet should convey specific factual information using clear, complete sentences. While focusing on distinct facts, you can combine closely related or consecutive points into a single sentence IF it enhances readability AND does not reduce factual detail.

**Additional Instructions:**
- Capture ALL specifics (data, names, examples, $, %, etc.).
- Do not add interpretations, summaries, or action items not explicitly stated.
- Include ONLY information present in the transcript.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Notion Toggle Format - Concise):**
""",
        "Summary Prompt (for Option 3)": """IMPORTANT: Enclose the entire final executive summary in a single Notion-style 'light bulb' callout block (using the `> 💡` syntax at the start of the block).

Based ONLY on the detailed 'GENERATED NOTES' provided below, create a concise executive summary highlighting the most significant insights, findings, or critical points discussed.

**Format (Inside the Callout Block):**
1.  Identify the main themes or key topics discussed in the notes (e.g., **GenAI Impact**, **Vendor Landscape**, **Genpact Specifics**). Create a clear, concise heading for each theme using bold text.
2.  Under each heading, use primary bullet points (`- `) to list the most significant insights, findings, or critical points related to that theme.
3.  Each bullet point should represent a single, distinct key takeaway. DO NOT use indented sub-bullets.

**Instructions:**
- Aim for a total summary length of approximately 500-1000 words.
- Maintain an objective and professional tone.
- Do not introduce any information or conclusions not explicitly supported by the GENERATED NOTES provided below.

---
**GENERATED NOTES (Input for Summary):**
{generated_notes}
---

**EXECUTIVE SUMMARY (Notion Callout Format):**
"""
    },
    "Earnings Call": { # Kept as standard markdown as toggles might be less readable here
        "Generate New Notes": """You are an expert AI assistant creating DETAILED yet CONCISE notes from an earnings call transcript for an investment firm.
Output MUST be comprehensive, factual notes, capturing all critical financial and strategic information with **crisp language**.

**Formatting Requirements (Mandatory):**
- US$ for dollars (US$2.5M), % for percentages.
- State comparison periods (+5% YoY, -2% QoQ).
- Represent fiscal periods accurately (Q3 FY25).
- Use common abbreviations (CEO, KPI).
- Use bullet points under headings.
- Each bullet = complete sentence with distinct info. **Be concise.**
- Capture ALL numbers, names, data accurately.
- Use quotes "" for significant statements.
- **AVOID redundant phrasing and unnecessary elaboration.** Focus on the core facts.
- **DO NOT summarize or interpret unless part of the structure or explicitly stated in the call.**
- **DO NOT add information not mentioned in the transcript.**

**Note Structure:**
- **Call Participants:** (List names/titles or 'Not specified')
{topic_instructions}

**CRITICAL:** Ensure accuracy and adhere strictly to structure and formatting. Prioritize factual content delivered concisely.
---
**EARNINGS CALL TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED EARNINGS CALL NOTES:**
""",
        "Enrich Existing Notes": """You are an expert AI assistant tasked with enriching existing earnings call notes using a provided source transcript.
Your goal is to identify significant financial, strategic, or forward-looking details mentioned in the **Source Transcript** that are MISSING from the **User's Existing Notes** and relevant to the specified **Topic Structure**. Integrate these missing details **accurately and concisely** into the existing notes, maintaining the overall structure and tone.

**Inputs:**
1.  **User's Existing Notes:** The notes provided by the user.
2.  **Source Transcript:** The full earnings call transcript.
3.  **Topic Structure:** Headings provided by the user (or logically derived if none provided) to guide the enrichment focus.
4.  **Additional Context:** Optional background information.

**Process:**
1.  Thoroughly read the **Source Transcript**.
2.  Carefully review the **User's Existing Notes** against the **Topic Structure**.
3.  Identify KEY information (specific financial figures, guidance updates, strategic initiatives, significant quotes, competitive remarks, Q&A points) present in the **Source Transcript** but ABSENT or INSUFFICIENTLY DETAILED in the **User's Existing Notes** under the relevant topics. **Focus on significant details.**
4.  Integrate these identified missing details into the appropriate sections of the **User's Existing Notes** using **crisp and concise language**.
    - Add new bullet points where necessary.
    - Augment existing bullet points ONLY if the addition is directly related, factual, and adds significant value (e.g., adding a specific percentage change). Avoid making existing points overly verbose.
    - Maintain the formatting requirements (US$, %, YoY/QoQ, FY periods).
    - Use quotes "" for direct significant statements added from the transcript.
    - Ensure added points are factual and directly from the transcript.
5.  If a topic in the **Topic Structure** is completely missing from the **User's Existing Notes** but discussed in the transcript, add the heading and relevant bullet points from the transcript **concisely**.
6.  Output the **Complete Enriched Notes**, incorporating the additions. DO NOT output commentary about the changes made.

**Formatting Requirements for Added Information:**
- US$ for dollars (US$2.5M), % for percentages.
- State comparison periods (+5% YoY, -2% QoQ).
- Represent fiscal periods accurately (Q3 FY25).
- Use common abbreviations (CEO, KPI).
- New points should be complete sentences, **written concisely.**
- **AVOID redundant phrasing or unnecessary elaboration when adding information.**
- **DO NOT add interpretation or summarization beyond what was in the original notes or clearly stated in the transcript.**
- **DO NOT add information not found in the Source Transcript.**

---
**TOPIC STRUCTURE (Focus enrichment on these areas):**
{topic_instructions}
---
**USER'S EXISTING NOTES (Input):**
{existing_notes}
---
**SOURCE TRANSCRIPT (Input):**
{transcript}
---
{context_section}
---
**COMPLETE ENRICHED NOTES (Output):**
"""
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
    'previous_selected_sector': DEFAULT_SECTOR,
    'earnings_call_topics': '',
    'earnings_call_mode': DEFAULT_EARNINGS_CALL_MODE,
    'existing_notes_input': "",
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': [],
    'earnings_call_topics_initialized': False,
    'speaker_1_name': "", # New state for speaker names
    'speaker_2_name': "", # New state for speaker names
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        pdf_file_stream.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])
        return text.strip() if text else None
    except Exception as e:
        st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"
        return None

def update_topic_template():
    selected_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if selected_sector in SECTOR_TOPICS and selected_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
        st.toast(f"Loaded topic template for {selected_sector}", icon="📝")
    st.session_state.current_prompt_text = ""

if not st.session_state.earnings_call_topics_initialized:
    initial_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if initial_sector in SECTOR_TOPICS and initial_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[initial_sector]
    else:
        st.session_state.earnings_call_topics = ""
    st.session_state.earnings_call_topics_initialized = True

current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
prev_sector = st.session_state.get('previous_selected_sector', DEFAULT_SECTOR)
if current_sector != prev_sector:
    update_topic_template()
    st.session_state.previous_selected_sector = current_sector

def format_prompt_safe(prompt_template, **kwargs):
    formatted_prompt = copy.deepcopy(prompt_template)
    try:
        placeholders = re.findall(r"\{([^}]+)\}", formatted_prompt)
        for key in placeholders:
            value = kwargs.get(key, f"[DEBUG: MISSING_PLACEHOLDER_{key}]")
            str_value = str(value) if value is not None else ""
            formatted_prompt = formatted_prompt.replace("{" + key + "}", str_value)
        return formatted_prompt
    except Exception as e:
        st.error(f"Prompt formatting error: {e}")
        return f"# Error formatting prompt template: {e}\nOriginal Template:\n{prompt_template}"

def create_docx(text):
    document = docx.Document()
    for line in text.split('\n'):
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def get_current_input_data():
    input_type = st.session_state.input_method_radio
    transcript = None
    audio_file = None
    if input_type == "Paste Text":
        transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file is not None:
            try:
                transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
            except Exception as e:
                st.session_state.error_message = f"Error processing PDF: {e}"
                transcript = None
    elif input_type == "Upload Audio":
        audio_file = st.session_state.audio_uploader
    return input_type, transcript, audio_file

def validate_inputs():
    input_method = st.session_state.get('input_method_radio', 'Paste Text')
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    custom_prompt = st.session_state.get('current_prompt_text', "").strip()
    view_edit_enabled = st.session_state.get('view_edit_prompt_enabled', False)

    if input_method == "Paste Text" and not st.session_state.get('text_input', "").strip():
        return False, "Please paste the source transcript text."
    if input_method == "Upload PDF" and st.session_state.get('pdf_uploader') is None:
        return False, "Please upload a source PDF file."
    if input_method == "Upload Audio" and st.session_state.get('audio_uploader') is None:
        return False, "Please upload a source audio file."

    if meeting_type == "Custom":
         if not custom_prompt:
             return False, "Custom prompt cannot be empty for 'Custom' meeting type."
    elif meeting_type == "Earnings Call":
        if st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
            if not st.session_state.get('existing_notes_input',"").strip():
                return False, "Please provide your existing notes for enrichment."
        if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                 return False, "Edited prompt is missing the required {transcript} placeholder."
            if "{topic_instructions}" not in custom_prompt and meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Generate New Notes":
                 return False, "Edited Earnings Call prompt is missing {topic_instructions}."
    elif meeting_type == "Expert Meeting":
         if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                return False, "Edited Expert Meeting prompt is missing {transcript}."
    return True, ""

def handle_edit_toggle():
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = ""

def get_prompt_display_text(for_display_only=False):
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    if not for_display_only and st.session_state.get('view_edit_prompt_enabled', False) and meeting_type != "Custom" and st.session_state.get('current_prompt_text', "").strip():
        return st.session_state.current_prompt_text
    display_text, temp_context = "", st.session_state.get('context_input',"").strip() if st.session_state.get('add_context_enabled') else None
    input_type, transcript_placeholder = st.session_state.get('input_method_radio', 'Paste Text'), "{transcript}"
    context_placeholder_section = f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{temp_context}\n---" if temp_context else ""
    format_kwargs, prompt_template_to_display = {'transcript': transcript_placeholder, 'context_section': context_placeholder_section}, None
    try:
        if meeting_type == "Expert Meeting":
            expert_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
            prompt_key = "Option 1: Existing (Detailed & Strict)" if expert_option == "Option 1: Existing (Detailed & Strict)" else "Option 2: Less Verbose (Default)"
            prompt_template_to_display = PROMPTS["Expert Meeting"][prompt_key]
            if prompt_template_to_display:
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                 if expert_option == "Option 3: Option 2 + Executive Summary":
                     summary_prompt_preview = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY, "Summary prompt not found.").split("---")[0]
                     display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step using a separate prompt:\n'''\n{summary_prompt_preview.strip()}\n'''"
            else: display_text = "# Error: Could not find prompt template for Expert Meeting display."
        elif meeting_type == "Earnings Call":
             prompt_template_to_display = PROMPTS["Earnings Call"]["Generate New Notes"]
             earnings_call_topics_text = st.session_state.get('earnings_call_topics', "")
             topic_instructions = ""
             if earnings_call_topics_text and earnings_call_topics_text.strip():
                 formatted_topics = [f"- **{line.strip().strip(':')}**" if line.strip() and not line.strip().startswith(('-', '*', '#')) else line.strip() for line in earnings_call_topics_text.split('\n')]
                 topic_instructions = f"Structure notes under:\n" + "\n".join(formatted_topics) + "\n\n- **Other Key Points** (MANDATORY)"
             else: topic_instructions = "Identify logical main themes (e.g., Financials, Outlook) and use them as bold headings. Include a final mandatory section: - **Other Key Points**."
             format_kwargs["topic_instructions"] = topic_instructions
             display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
        elif meeting_type == "Custom":
             audio_note = "\n# NOTE: For audio, your custom prompt will receive a *refined transcript*."
             default_custom = "# Enter your custom prompt... Use {transcript} and {context_section}."
             display_text = st.session_state.get('current_prompt_text', default_custom) + (audio_note if input_type == 'Upload Audio' else "")
             return display_text
        else:
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
             return "Error generating prompt preview."
        if input_type == "Upload Audio" and meeting_type != "Custom":
             audio_note = "# NOTE FOR AUDIO: This prompt is used in Step 3 with the *refined* transcript.\n\n"
             display_text = audio_note + display_text
    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview: {e}"
    return display_text

def clear_all_state():
    """Resets most session state variables to their defaults."""
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
    st.session_state.expert_meeting_prompt_option = DEFAULT_EXPERT_MEETING_OPTION
    st.session_state.input_method_radio = 'Paste Text'
    st.session_state.text_input, st.session_state.pdf_uploader, st.session_state.audio_uploader = "", None, None
    st.session_state.context_input, st.session_state.add_context_enabled = "", False
    st.session_state.selected_sector, st.session_state.previous_selected_sector = DEFAULT_SECTOR, DEFAULT_SECTOR
    st.session_state.earnings_call_topics = SECTOR_TOPICS.get(DEFAULT_SECTOR, "") if DEFAULT_SECTOR != "Other / Manual Topics" else ""
    st.session_state.earnings_call_topics_initialized = True
    st.session_state.current_prompt_text, st.session_state.view_edit_prompt_enabled = "", False
    st.session_state.earnings_call_mode, st.session_state.existing_notes_input = DEFAULT_EARNINGS_CALL_MODE, ""
    st.session_state.generated_notes, st.session_state.edited_notes_text, st.session_state.edit_notes_enabled = None, "", False
    st.session_state.error_message, st.session_state.processing, st.session_state.generating_filename = None, False, False
    st.session_state.suggested_filename, st.session_state.uploaded_audio_info, st.session_state.history = None, None, []
    st.session_state.raw_transcript, st.session_state.refined_transcript, st.session_state.processed_audio_chunk_references = None, None, []
    st.session_state.speaker_1_name, st.session_state.speaker_2_name = "", "" # Reset speaker names
    st.toast("Inputs and outputs cleared!", icon="🧹")
    st.rerun()

def generate_suggested_filename(notes_content, meeting_type):
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        today_date = datetime.now().strftime("%Y%m%d")
        mt_cleaned = meeting_type.replace(" ", "_").lower()
        notes_preview = notes_content.split("\n\n---\n\n> 💡", 1)[0] if "> 💡" in notes_content else notes_content
        filename_prompt = (f"Suggest a concise filename (max 5 words, use underscores_not_spaces). Start with {today_date}_{mt_cleaned}. Base on key topics/names from these notes. Output ONLY the filename string (e.g., {today_date}_{mt_cleaned}_topic.txt). NOTES:\n{notes_preview[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())
            s_name = re.sub(r'_+', '_', s_name).strip('_')[:100]
            if not s_name.startswith(today_date): s_name = f"{today_date}_{s_name}"
            if s_name:
                st.toast("💡 Filename suggested!", icon="✅")
                return s_name
    except Exception as e:
        st.warning(f"Filename generation error: {e}", icon="⚠️")
    finally:
        st.session_state.generating_filename = False
    return None

def add_to_history(notes):
    if not notes: return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"timestamp": timestamp, "notes": notes}
        current_history = st.session_state.get('history', [])
        if not isinstance(current_history, list): current_history = []
        current_history.insert(0, new_entry)
        st.session_state.history = current_history[:3]
    except Exception as e:
        st.warning(f"⚠️ Error updating note history: {e}", icon="❗")

def restore_note_from_history(index):
    if 0 <= index < len(st.session_state.history):
        entry = st.session_state.history[index]
        st.session_state.generated_notes = entry["notes"]
        st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False
        st.session_state.suggested_filename, st.session_state.error_message = None, None
        st.session_state.raw_transcript, st.session_state.refined_transcript = None, None
        st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")
        st.rerun()

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, Notion-ready notes.")

# --- Settings Container ---
with st.container(border=True):
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        st.subheader("Meeting & Model Settings")
        col1a, col1b = st.columns(2)
        with col1a:
            # Meeting Type
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                     on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))

            # NEW: Speaker Name Inputs
            st.text_input("Speaker 1 Name (Optional):", key="speaker_1_name", placeholder="e.g., John Doe - Expert")
            st.text_input("Speaker 2 Name (Optional):", key="speaker_2_name", placeholder="e.g., Jane Smith - Analyst")

        with col1b:
            # Model Selection
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_transcription_model_display_name', DEFAULT_TRANSCRIPTION_MODEL_NAME)), help="Model for audio-to-text (Step 1).")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_refinement_model_display_name', DEFAULT_REFINEMENT_MODEL_NAME)), help="Model for cleaning transcript & adding speakers (Step 2).")
            st.selectbox("Notes Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_notes_model_display_name', DEFAULT_NOTES_MODEL_NAME)), help="Model for generating final notes (Step 3).")

    with col_main_2:
        st.subheader("") # Spacer
        st.button("🧹 Clear All Inputs & Outputs", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

    # Conditional options based on meeting type
    st.markdown("---")
    if st.session_state.get('selected_meeting_type') == "Expert Meeting":
         st.radio(
            "Expert Meeting Note Style:", options=EXPERT_MEETING_OPTIONS, key="expert_meeting_prompt_option",
            help="Choose output format. All options generate Notion-friendly toggle lists.", horizontal=True,
            on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False)
        )
    elif st.session_state.get('selected_meeting_type') == "Earnings Call":
         st.radio(
            "Mode:", options=EARNINGS_CALL_MODES, key="earnings_call_mode", horizontal=True,
            help="Generate notes from scratch or enrich existing ones.",
            on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False)
        )

st.divider()

# --- Input Sections ---
with st.container(border=True):
    is_enrich_mode = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                      st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")

    if is_enrich_mode:
        st.subheader("Existing Notes & Source Transcript")
        st.text_area("1. Paste your existing notes here:", height=200, key="existing_notes_input", placeholder="Paste the notes you want to enrich...")
        st.markdown("---")
        st.markdown("**2. Provide the Source Transcript Input (Text, PDF, or Audio)**")
    else:
        st.subheader("Source Input (Transcript or Audio)")

    st.radio("Source type:", ("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.get('input_method_radio', 'Paste Text')
    if input_type_ui == "Paste Text":
        st.text_area("Paste source transcript:", height=150, key="text_input", placeholder="Paste transcript source...")
    elif input_type_ui == "Upload PDF":
        st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else:
        st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.markdown("---")
    st.subheader("Topics & Context")
    col3a, col3b = st.columns(2)
    with col3a:
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            st.selectbox("Select Sector (for Topic Template):", options=SECTOR_OPTIONS, key="selected_sector")
            st.text_area("Earnings Call Topics (Edit below):", key="earnings_call_topics", height=150, placeholder="Enter topics manually or select a sector to load a template...")
        else:
             st.caption("Topic selection is available for Earnings Calls.")
    with col3b:
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.get('add_context_enabled'):
            st.text_area("Context Details:", height=75, key="context_input", placeholder="e.g., Company Name, Date, Key Competitors...")
        st.write("")
        if st.session_state.get('selected_meeting_type') != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled",
                         disabled=is_enrich_mode, on_change=handle_edit_toggle,
                         help="Advanced: View/edit the base prompt. Disabled in Enrichment mode.")

# --- Prompt Area (Conditional Display) ---
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and not is_enrich_mode)

if show_prompt_area:
    with st.container(border=True):
        st.subheader("Final Prompt Editor")
        base_template_text = get_prompt_display_text(for_display_only=True)
        if st.session_state.view_edit_prompt_enabled and not st.session_state.current_prompt_text.strip():
             st.session_state.current_prompt_text = base_template_text
        st.text_area("Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# --- Generate Button ---
st.write("")
is_valid, error_msg = validate_inputs()
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes."
generate_button_label = "🚀 Enrich Notes" if is_enrich_mode else "🚀 Generate Notes"
generate_button = st.button(generate_button_label, type="primary", use_container_width=True,
                            disabled=st.session_state.processing or st.session_state.generating_filename or not is_valid,
                            help=generate_tooltip)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    if st.session_state.get('processing'):
        op_desc = "Enriching notes" if is_enrich_mode else "Generating notes"
        st.info(f"⏳ Processing... Currently {op_desc}. Please wait.", icon="⏳")
    elif st.session_state.get('generating_filename'):
        st.info("⏳ Generating suggested filename...", icon="💡")
    elif st.session_state.get('error_message'):
        st.error(st.session_state.error_message, icon="🚨")
    elif st.session_state.get('generated_notes'):
        output_title = "✅ Enriched Notes" if is_enrich_mode else "✅ Generated Notes"
        st.subheader(output_title)

        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        
        # --- CORRECTED: Copy Button and Edit Toggle ---
        col_out_1, col_out_2 = st.columns([1, 4])
        with col_out_1:
             copy_button(notes_content_to_use, "📋 Copy Notes")
        with col_out_2:
             st.checkbox("Edit Output", key="edit_notes_enabled", help="Toggle to manually edit the generated notes below.")

        if st.session_state.get('edit_notes_enabled'):
            st.text_area("Editable Output:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            # The markdown renderer in Streamlit won't show toggles/callouts perfectly, but the raw text will be correct for Notion
            st.markdown(notes_content_to_use)

        st.markdown("---")
        with st.expander("View Source Transcripts & Download Options"):
            if st.session_state.get('raw_transcript'):
                st.text_area("Raw Source Transcript (Step 1 Output)", st.session_state.raw_transcript, height=200, disabled=True)
            if st.session_state.get('refined_transcript'):
                st.text_area("Refined Source Transcript (Step 2 Output)", st.session_state.refined_transcript, height=300, disabled=True)
            
            st.write("")
            dl_cols = st.columns(3)
            fname_base = st.session_state.get('suggested_filename', "synthnotes_output")
            with dl_cols[0]:
                st.download_button("⬇️ Output (.txt)", notes_content_to_use, f"{fname_base}.txt", "text/plain", use_container_width=True)
            with dl_cols[1]:
                st.download_button("⬇️ Output (.md)", notes_content_to_use, f"{fname_base}.md", "text/markdown", use_container_width=True)
            with dl_cols[2]:
                if st.session_state.get('refined_transcript'):
                    st.download_button("⬇️ Refined Tx (.txt)", st.session_state.refined_transcript, f"{fname_base}_refined_transcript.txt", "text/plain", use_container_width=True)
                else:
                    st.button("Refined Tx N/A", disabled=True, use_container_width=True)
                
    elif not st.session_state.get('processing'):
        st.markdown("<p class='initial-prompt'>Configure inputs and click 'Generate' to start.</p>", unsafe_allow_html=True)


# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    history = st.session_state.get('history', [])
    if not history:
        st.caption("No generated notes in history for this session.")
    else:
        for i, entry in enumerate(history):
             st.markdown(f"**#{i+1} - {entry.get('timestamp', 'N/A')}**")
             preview_text = "\n".join(entry.get('notes', '').strip().splitlines()[:5]) + "..."
             st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
             st.button(f"Restore Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
             if i < len(history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    is_valid_on_click, error_msg_on_click = validate_inputs()
    if not is_valid_on_click:
        st.session_state.error_message = f"Validation Error: {error_msg_on_click}"
        st.session_state.processing = False
        st.rerun()
    else:
        st.session_state.processing = True
        st.session_state.generating_filename, st.session_state.generated_notes = False, None
        st.session_state.edited_notes_text, st.session_state.edit_notes_enabled = "", False
        st.session_state.error_message, st.session_state.suggested_filename = None, None
        st.session_state.raw_transcript, st.session_state.refined_transcript = None, None
        st.session_state.processed_audio_chunk_references = []
        st.rerun()

if st.session_state.get('processing') and not st.session_state.get('generating_filename') and not st.session_state.get('error_message'):
    processed_audio_chunk_references = []
    is_enrich_mode = st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes"
    operation_desc = "Enriching Notes" if is_enrich_mode else "Generating Notes"

    with st.status(f"🚀 Initializing {operation_desc} process...", expanded=True) as status:
        try:
            status.update(label="⚙️ Reading inputs and settings...")
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process:
                raise ValueError(f"Input validation failed: {error_msg_process}")

            # Get settings from session state
            meeting_type = st.session_state.selected_meeting_type
            expert_meeting_option = st.session_state.expert_meeting_prompt_option
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            general_context = st.session_state.get('context_input', "").strip() if st.session_state.add_context_enabled else None
            actual_input_type, source_transcript_data, source_audio_file_obj = get_current_input_data()
            
            # --- Get Speaker Names from UI ---
            speaker_1_name = st.session_state.get('speaker_1_name', '').strip()
            speaker_2_name = st.session_state.get('speaker_2_name', '').strip()

            status.update(label="🧠 Initializing AI models...")
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            status.update(label="✍️ Acquiring source transcript...")
            final_source_transcript = source_transcript_data
            st.session_state.raw_transcript = source_transcript_data
            st.session_state.refined_transcript = None

            if actual_input_type == "Upload Audio":
                # Full audio processing loop
                if source_audio_file_obj is None:
                     raise ValueError("Audio file selected but no file object found.")
                st.session_state.uploaded_audio_info = source_audio_file_obj
                status.update(label=f"🔊 Loading source audio '{source_audio_file_obj.name}'...")
                audio_bytes, file_extension = source_audio_file_obj.getvalue(), os.path.splitext(source_audio_file_obj.name)[1].lower().replace('.', '')
                audio_format = 'mp4' if file_extension == 'm4a' else file_extension
                
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                except Exception as audio_load_err:
                     if "ffmpeg" in str(audio_load_err).lower():
                         raise ValueError(f"❌ **Error:** Could not load audio. `ffmpeg` might be missing on the server. Ensure `ffmpeg` is in your packages.txt file. (Original error: {audio_load_err})")
                     else:
                         raise ValueError(f"❌ Could not load audio file. Ensure it is valid. Error: {audio_load_err}")

                chunks = make_chunks(audio, 50 * 60 * 1000)
                status.update(label=f"🔪 Splitting audio into {len(chunks)} chunk(s)...")

                transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
                transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
                all_transcripts = []
                for i, chunk in enumerate(chunks):
                    chunk_num = i + 1
                    status.update(label=f"🔄 Processing Source Chunk {chunk_num}/{len(chunks)}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                        chunk.export(temp_chunk_file.name, format="wav")
                        chunk_file_ref = genai.upload_file(path=temp_chunk_file.name)
                        processed_audio_chunk_references.append(chunk_file_ref)
                        while chunk_file_ref.state.name == "PROCESSING":
                            time.sleep(10)
                            chunk_file_ref = genai.get_file(chunk_file_ref.name)
                        if chunk_file_ref.state.name != "ACTIVE":
                            raise Exception(f"Audio chunk {chunk_num} failed. State: {chunk_file_ref.state.name}")
                        t_response = transcription_model.generate_content(["Transcribe accurately.", chunk_file_ref], generation_config=transcription_gen_config)
                        all_transcripts.append(t_response.text.strip() if t_response and hasattr(t_response, 'text') else "")
                
                st.session_state.raw_transcript = "\n\n".join(all_transcripts).strip()
                final_source_transcript = st.session_state.raw_transcript
                status.update(label="✅ Step 1: Full Source Transcription Complete!")

            # --- Step 2: Transcript Refinement (with Speaker Name Logic) ---
            should_refine = (actual_input_type == "Upload Audio") or \
                            (meeting_type == "Expert Meeting" and actual_input_type in ["Paste Text", "Upload PDF"])

            if final_source_transcript and should_refine:
                status.update(label=f"🧹 Step 2: Refining transcript...")
                
                speaker_instructions = "Assign consistent generic labels (e.g., Speaker 1, Speaker 2, Interviewer)."
                if speaker_1_name and speaker_2_name:
                    speaker_instructions = f"The speakers are '{speaker_1_name}' and '{speaker_2_name}'. Please use these names as labels."
                elif speaker_1_name or speaker_2_name:
                    single_name = speaker_1_name or speaker_2_name
                    speaker_instructions = f"One of the primary speakers is '{single_name}'. Please use this name as a label where appropriate."
                
                refinement_prompt = f"""Please refine the following raw transcript.

                **Raw Transcript:**
                ```
                {str(st.session_state.raw_transcript or "")}
                ```

                **Instructions:**
                1.  **Identify Speakers:** {speaker_instructions} Place the label on a new line before their speech.
                2.  **Correct & Clarify:** Fix obvious transcription errors and improve readability. Do not change the core meaning.
                3.  **Format:** Ensure clean separation between speaker turns. Output only the refined transcript text.

                **Additional Context (Optional):**
                {general_context if general_context else "None provided."}

                **Refined Transcript:**
                """
                r_response = refinement_model.generate_content(refinement_prompt, generation_config=refinement_gen_config, safety_settings=safety_settings)

                if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                    st.session_state.refined_transcript = r_response.text.strip()
                    final_source_transcript = st.session_state.refined_transcript
                    status.update(label="🧹 Step 2: Refinement complete!")
                else:
                    st.warning("⚠️ Refinement failed. Using raw transcript.", icon="⚠️")
                    status.update(label="⚠️ Refinement failed. Proceeding with raw transcript.")
            
            if not final_source_transcript:
                 raise ValueError("No source transcript available to generate notes.")
                 
            # --- Step 3: Prepare and Execute Final Prompt ---
            status.update(label=f"📝 Preparing final prompt for {operation_desc}...")
            # This unified logic reuses the display function to build the correct prompt
            final_api_prompt = get_prompt_display_text(for_display_only=True)
            final_api_prompt = format_prompt_safe(final_api_prompt, transcript=final_source_transcript, context_section=f"\n**ADDITIONAL CONTEXT:**\n{general_context}\n---" if general_context else "")
            
            status.update(label=f"✨ Step 3: {operation_desc}...")
            response = notes_model.generate_content(final_api_prompt, generation_config=main_gen_config, safety_settings=safety_settings)

            if not (response and hasattr(response, 'text') and response.text.strip()):
                raise Exception(f"{operation_desc} failed or returned empty. Block Reason: {getattr(response.prompt_feedback, 'block_reason', 'N/A')}")
            
            generated_content = response.text.strip()
            
            is_expert_summary_step = (meeting_type == "Expert Meeting" and expert_meeting_option == "Option 3: Option 2 + Executive Summary")
            if is_expert_summary_step:
                status.update(label="✨ Step 3b: Generating Executive Summary...")
                summary_prompt = format_prompt_safe(PROMPTS["Expert Meeting"][EXPERT_MEETING_SUMMARY_PROMPT_KEY], generated_notes=generated_content)
                summary_response = notes_model.generate_content(summary_prompt, generation_config=summary_gen_config, safety_settings=safety_settings)
                if summary_response and hasattr(summary_response, 'text') and summary_response.text.strip():
                    # The prompt asks for Notion format, so we just append it
                    st.session_state.generated_notes = f"{generated_content}\n\n---\n\n{summary_response.text.strip()}"
                else:
                    st.warning("⚠️ Summary generation failed. Providing detailed notes only.", icon="⚠️")
                    st.session_state.generated_notes = generated_content
            else:
                st.session_state.generated_notes = generated_content

            status.update(label=f"✅ {operation_desc} completed successfully!", state="complete")
            st.session_state.edited_notes_text = st.session_state.generated_notes
            add_to_history(st.session_state.generated_notes)

            status.update(label="💡 Suggesting filename...")
            suggested_fname = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
            st.session_state.suggested_filename = suggested_fname

        except Exception as e:
             st.session_state.error_message = f"❌ Processing Error: {e}"
             status.update(label=f"❌ Error: {e}", state="error")
        finally:
            st.session_state.processing = False
            if processed_audio_chunk_references:
                 st.toast(f"☁️ Cleaning up {len(processed_audio_chunk_references)} cloud audio chunk(s)...", icon="🗑️")
                 for file_ref in processed_audio_chunk_references:
                    try: genai.delete_file(file_ref.name)
                    except Exception as cleanup_error: st.warning(f"Cleanup failed for {file_ref.name}: {cleanup_error}")
            st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

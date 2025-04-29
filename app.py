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
# (CSS remains the same as in your original code)
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
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro", "Gemini 2.0 Flash (Fast & Versatile)": "gemini-2.0-flash-lite",
    "Gemini 2.5 Pro (paid)": "gemini-2.5-pro-preview-03-25", # Placeholder, might need actual model ID
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25", # Keep using this if available
}
# Ensure default models exist in the available list
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"

DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 2.0 Flash (Fast & Versatile)"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

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


# --- Prompts Definitions (with Earnings Call conciseness update) ---
PROMPTS = {
    "Expert Meeting": {
         "Option 1: Existing (Detailed & Strict)": """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.
- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.
  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).
  - **Answers:** Use bullet points directly below the question. **Each bullet MUST be a complete sentence representing one single, distinct factual point.** Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. **DO NOT add interpretations, summaries, conclusions, or action items not explicitly stated in the transcript.**

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- Be clear and concise, adhering strictly to one fact per bullet point.
- Include ONLY information present in the transcript. DO NOT add external information.
- If a section (like Opening Overview) isn't present, OMIT it.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Strict):**
""",
        "Option 2: Less Verbose (Default)": """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple, direct language. DO NOT summarize.
- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.
	- **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).
	- **Answers:** Use bullet points directly below the question.
		- Each bullet point should convey specific factual information using clear, complete sentences.
		- **Strive for natural sentence flow. While focusing on distinct facts, combine closely related details or sequential points into a single sentence where it enhances readability and avoids excessive choppiness, without adding interpretation or summarization.**
		- Capture ALL specifics (data, names, examples, $, %, etc.).
		- DO NOT use sub-bullets or section headers within answers.
		- **DO NOT add interpretations, summaries, conclusions, or action items not explicitly stated in the transcript.**

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- **Write clearly and concisely, avoiding unnecessary words. Favor informative sentences over overly simplistic ones.**
- Include ONLY information present in the transcript. DO NOT add external information.
- If a section (like Opening Overview) isn't present, OMIT it.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Concise):**
""",
        "Summary Prompt (for Option 3)": """Based ONLY on the detailed 'GENERATED NOTES (Q&A Format - Concise)' provided below, create a concise executive summary highlighting the most significant insights, findings, or critical points discussed.

**Format:**
1.  Identify the main themes or key topics discussed in the notes (e.g., **GenAI Impact**, **Vendor Landscape**, **Genpact Specifics**). Create a clear, concise heading for each theme using bold text.
2.  Under each heading, use primary bullet points (`- `) to list the most significant insights, findings, or critical points related to that theme.
3.  **Crucially: Each bullet point should represent a single, distinct key takeaway or significant piece of information.** DO NOT use indented sub-bullets or nested lists. If a point has multiple important facets, break them down into separate primary bullet points under the same theme heading.
4.  Focus on synthesizing the key takeaways from the detailed Q&A points. These bullets should represent crucial insights. **DO NOT list minor details or repeat verbatim points from the Q&A.**

**Instructions:**
- Aim for a total summary length of approximately 500-1000 words.
- Maintain an objective and professional tone, reflecting the expert's views accurately.
- Ensure the summary accurately reflects the content and emphasis of the detailed notes it is based on.
- **Do not introduce any information, conclusions, or opinions not explicitly supported by the GENERATED NOTES provided below.**
- **DO NOT hallucinate or invent details.**

---
**GENERATED NOTES (Input for Summary):**
{generated_notes}
---

**EXECUTIVE SUMMARY:**
"""
    },
    "Earnings Call": {
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
    'previous_selected_sector': DEFAULT_SECTOR, # Tracks previous sector
    'earnings_call_topics': '', # Unified key for topics
    'earnings_call_mode': DEFAULT_EARNINGS_CALL_MODE,
    'existing_notes_input': "",
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': [],
    'earnings_call_topics_initialized': False # Initialize flag
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
    """
    Updates the earnings_call_topics state based on the selected sector.
    Only updates if the sector has changed and is a template sector (not 'Other').
    """
    selected_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if selected_sector in SECTOR_TOPICS and selected_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
        st.toast(f"Loaded topic template for {selected_sector}", icon="📝")
    # Clear the edited prompt text when loading a new template automatically
    st.session_state.current_prompt_text = ""


# --- Initialize topics on first run ---
if not st.session_state.earnings_call_topics_initialized:
    initial_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if initial_sector in SECTOR_TOPICS and initial_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[initial_sector]
    else:
        st.session_state.earnings_call_topics = ""
    st.session_state.earnings_call_topics_initialized = True

# --- Detect Sector Change and Update Topics ---
current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
prev_sector = st.session_state.get('previous_selected_sector', DEFAULT_SECTOR)

if current_sector != prev_sector:
    update_topic_template()
    st.session_state.previous_selected_sector = current_sector
    # No rerun needed here, changes reflected in UI rendering

def format_prompt_safe(prompt_template, **kwargs):
    """Safely formats a prompt string, replacing placeholders."""
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
    """Creates a DOCX file in memory from text."""
    document = docx.Document()
    for line in text.split('\n'):
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def get_current_input_data():
    """Gets transcript data based on selected input method."""
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
    """Validates required inputs based on selected options."""
    input_method = st.session_state.get('input_method_radio', 'Paste Text')
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    custom_prompt = st.session_state.get('current_prompt_text', "").strip()
    view_edit_enabled = st.session_state.get('view_edit_prompt_enabled', False)

    # Check source input
    if input_method == "Paste Text" and not st.session_state.get('text_input', "").strip():
        return False, "Please paste the source transcript text."
    if input_method == "Upload PDF" and st.session_state.get('pdf_uploader') is None:
        return False, "Please upload a source PDF file."
    if input_method == "Upload Audio" and st.session_state.get('audio_uploader') is None:
        return False, "Please upload a source audio file."

    # Check meeting type specific requirements
    if meeting_type == "Custom":
         if not custom_prompt:
             return False, "Custom prompt cannot be empty for 'Custom' meeting type."
    elif meeting_type == "Earnings Call":
        if st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
            if not st.session_state.get('existing_notes_input',"").strip():
                return False, "Please provide your existing notes for enrichment (in the dedicated input box)."
        # Check edited prompt for required placeholders if edit is enabled
        if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                 return False, "Edited prompt is missing the required {transcript} placeholder."
            if "{topic_instructions}" not in custom_prompt and meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Generate New Notes":
                 # topic_instructions is specific to Generate New Notes mode
                 return False, "Edited Earnings Call prompt is missing the required {topic_instructions} placeholder."
    elif meeting_type == "Expert Meeting":
         # Check edited prompt for required placeholders if edit is enabled
         if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                return False, "Edited Expert Meeting prompt is missing the required {transcript} placeholder."


    return True, ""

def handle_edit_toggle():
    """Clears custom prompt edits if 'View/Edit Prompt' is toggled off for non-Custom types."""
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = ""

def get_prompt_display_text(for_display_only=False):
    """
    Generates the prompt text for display/editing, ensuring it uses the
    correct structure and placeholders identical to the processing logic.
    """
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)

    # If editing is enabled AND there's text in the editor state, show that text.
    # Exception: If 'for_display_only' is True, force regeneration from template.
    if not for_display_only \
       and st.session_state.get('view_edit_prompt_enabled', False) \
       and meeting_type != "Custom" \
       and st.session_state.get('current_prompt_text', "").strip():
        return st.session_state.current_prompt_text

    # Generate the default prompt view from the base template
    display_text = ""
    temp_context = st.session_state.get('context_input',"").strip() if st.session_state.get('add_context_enabled') else None
    input_type = st.session_state.get('input_method_radio', 'Paste Text')
    transcript_placeholder = "{transcript}"
    context_placeholder_section = f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{temp_context}\n---" if temp_context else ""

    format_kwargs = {
        'transcript': transcript_placeholder,
        'context_section': context_placeholder_section
    }
    prompt_template_to_display = None

    try:
        if meeting_type == "Expert Meeting":
            expert_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
            if expert_option == "Option 1: Existing (Detailed & Strict)":
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
            else: # Option 2 and 3 use Option 2 template
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

            if prompt_template_to_display:
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                 if expert_option == "Option 3: Option 2 + Executive Summary":
                     summary_prompt_preview = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY, "Summary prompt not found.").split("---")[0]
                     display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step generated *after* these notes, using a separate prompt starting like:\n'''\n{summary_prompt_preview.strip()}\n'''"
            else:
                display_text = "# Error: Could not find prompt template for Expert Meeting display."

        elif meeting_type == "Earnings Call":
             prompt_template_to_display = PROMPTS["Earnings Call"]["Generate New Notes"]
             earnings_call_topics_text = st.session_state.get('earnings_call_topics', "")

             # Replicate the topic formatting logic from the main processing block
             topic_instructions = ""
             if earnings_call_topics_text and earnings_call_topics_text.strip():
                 formatted_topics = []
                 for line in earnings_call_topics_text.split('\n'):
                     trimmed_line = line.strip()
                     if trimmed_line and not trimmed_line.startswith(('-', '*', '#')):
                         formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                     elif trimmed_line:
                         formatted_topics.append(trimmed_line)
                 topic_list_str = "\n".join(formatted_topics)
                 topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                       f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                       f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
             else: # No topics provided
                 topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                       f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                       f"Place all relevant details under the most appropriate heading.")
             format_kwargs["topic_instructions"] = topic_instructions
             display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)

        elif meeting_type == "Custom":
             audio_note = ("\n# NOTE FOR AUDIO: If using audio, the system will first chunk and transcribe,\n"
                           "# then *refine* the full transcript (speaker ID, translation, corrections).\n"
                           "# Your custom prompt below will receive this *refined transcript* as the primary text input.\n"
                           "# Design your prompt accordingly.\n")
             default_custom = "# Enter your custom prompt here...\n# Use {transcript} and {context_section} placeholders.\n# Example: Summarize this meeting:\n# {transcript}\n# {context_section}"
             # For Custom mode, the text area's value IS the prompt, so we just display it or the default
             display_text = st.session_state.get('current_prompt_text', default_custom) + (audio_note if input_type == 'Upload Audio' else "")
             return display_text # Return directly

        else:
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' encountered for prompt preview.")
             return "Error generating prompt preview."

        # Add audio processing note if applicable (but not for Custom)
        if input_type == "Upload Audio" and meeting_type != "Custom":
             audio_note = ("# NOTE FOR AUDIO: 3-step process (Chunked Transcribe -> Refine -> Notes).\n"
                           "# This prompt (or your edited version) is used in Step 3 with the *refined* transcript.\n"
                           "####################################\n\n")
             display_text = audio_note + display_text

    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview: Review inputs/prompt structure.\nDetails: {e}"

    return display_text

def clear_all_state():
    """Resets most session state variables to their defaults."""
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
    st.session_state.previous_selected_sector = DEFAULT_SECTOR # Reset tracker
    # Reset earnings call topics based on the default sector
    st.session_state.earnings_call_topics = SECTOR_TOPICS.get(DEFAULT_SECTOR, "") if DEFAULT_SECTOR != "Other / Manual Topics" else ""
    st.session_state.earnings_call_topics_initialized = True # Mark as initialized here too
    st.session_state.current_prompt_text = "" # Clear edited prompt
    st.session_state.view_edit_prompt_enabled = False
    st.session_state.earnings_call_mode = DEFAULT_EARNINGS_CALL_MODE
    st.session_state.existing_notes_input = ""
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.processing = False
    st.session_state.generating_filename = False
    st.session_state.suggested_filename = None
    st.session_state.uploaded_audio_info = None
    st.session_state.history = []
    st.session_state.raw_transcript = None
    st.session_state.refined_transcript = None
    st.session_state.processed_audio_chunk_references = []
    st.toast("Inputs and outputs cleared!", icon="🧹")
    st.rerun()

def generate_suggested_filename(notes_content, meeting_type):
    """Suggests a filename using a fast LLM based on notes content."""
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True # Indicate processing start
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        today_date = datetime.now().strftime("%Y%m%d")
        mt_cleaned = meeting_type.replace(" ", "_").lower()

        notes_preview = notes_content
        summary_marker = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
        if summary_marker in notes_content:
            notes_preview = notes_content.split(summary_marker)[0]

        filename_prompt = (f"Suggest a concise filename (max 5 words including type, use underscores_not_spaces). Start with Date={today_date}, Type='{mt_cleaned}'. Base suggestion on key topics/company names from the start of these notes. Output ONLY the filename string (e.g., {today_date}_{mt_cleaned}_topic_name.txt). NOTES_PREVIEW:\n{notes_preview[:1000]}")

        response = filename_model.generate_content(
            filename_prompt,
            generation_config=filename_gen_config,
            safety_settings=safety_settings
        )

        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())
            s_name = re.sub(r'_+', '_', s_name)
            s_name = s_name.strip('_')
            s_name = s_name[:100]

            if not s_name.startswith(today_date):
                s_name = f"{today_date}_{s_name}"

            if s_name:
                st.toast("💡 Filename suggested!", icon="✅")
                return s_name
            else:
                st.warning(f"Filename suggestion was empty or invalid after cleaning: '{response.text}'", icon="⚠️")
                return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            st.warning(f"Filename suggestion blocked: {response.prompt_feedback.block_reason}", icon="⚠️")
            return None
        else:
            st.warning("Could not generate filename suggestion (empty response).", icon="⚠️")
            return None
    except Exception as e:
        st.warning(f"Filename generation error: {e}", icon="⚠️")
        return None
    finally:
        st.session_state.generating_filename = False

def add_to_history(notes):
    """Adds the generated notes to the history in session state."""
    if not notes: return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"timestamp": timestamp, "notes": notes}
        current_history = st.session_state.get('history', [])
        if not isinstance(current_history, list):
            st.warning("History state was not a list, resetting.", icon="⚠️")
            current_history = []
        current_history.insert(0, new_entry)
        st.session_state.history = current_history[:3]
    except Exception as e:
        st.warning(f"⚠️ Error updating note history: {e}", icon="❗")

def restore_note_from_history(index):
    """Restores selected notes from history to the main output area."""
    if 0 <= index < len(st.session_state.history):
        entry = st.session_state.history[index]
        st.session_state.generated_notes = entry["notes"]
        st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False
        st.session_state.suggested_filename = None
        st.session_state.error_message = None
        st.session_state.raw_transcript = None
        st.session_state.refined_transcript = None
        st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")
        st.rerun()

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Settings Container ---
with st.container(border=True):
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            mt_options = list(MEETING_TYPES)
            mt_default = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
            mt_index = mt_options.index(mt_default) if mt_default in mt_options else 0
            # Clear edits/toggle when meeting type changes
            st.radio("Meeting Type:", options=mt_options, key="selected_meeting_type", horizontal=True,
                     index=mt_index,
                     on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))

            # Conditional options
            if st.session_state.get('selected_meeting_type') == "Expert Meeting":
                 em_options = list(EXPERT_MEETING_OPTIONS)
                 em_default = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
                 em_index = em_options.index(em_default) if em_default in em_options else 0
                 st.radio(
                    "Expert Meeting Note Style:",
                    options=em_options, key="expert_meeting_prompt_option", index=em_index,
                    help="Choose output format: Strict Q&A, Natural Q&A, or Natural Q&A + Summary.",
                    on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False) # Clear edits/toggle
                )
            elif st.session_state.get('selected_meeting_type') == "Earnings Call":
                 ec_options = list(EARNINGS_CALL_MODES)
                 ec_default = st.session_state.get('earnings_call_mode', DEFAULT_EARNINGS_CALL_MODE)
                 ec_index = ec_options.index(ec_default) if ec_default in ec_options else 0
                 # Clear edits/toggle when mode changes
                 st.radio(
                    "Mode:", options=ec_options, key="earnings_call_mode", horizontal=True, index=ec_index,
                    help="Generate notes from scratch or enrich existing ones.",
                    on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False)
                )

        with col1b:
            st.subheader("AI Model Selection")
            trans_options = list(AVAILABLE_MODELS.keys())
            trans_default = st.session_state.get('selected_transcription_model_display_name', DEFAULT_TRANSCRIPTION_MODEL_NAME)
            trans_index = trans_options.index(trans_default) if trans_default in trans_options else 0
            st.selectbox("Transcription Model:", options=trans_options, key="selected_transcription_model_display_name", index=trans_index, help="Model for audio transcription (Step 1). Used only if audio is uploaded.")

            refine_options = list(AVAILABLE_MODELS.keys())
            refine_default = st.session_state.get('selected_refinement_model_display_name', DEFAULT_REFINEMENT_MODEL_NAME)
            refine_index = refine_options.index(refine_default) if refine_default in refine_options else 0
            st.selectbox("Refinement Model:", options=refine_options, key="selected_refinement_model_display_name", index=refine_index, help="Model for audio refinement (Step 2). Used only if audio is uploaded.")

            notes_options = list(AVAILABLE_MODELS.keys())
            notes_default = st.session_state.get('selected_notes_model_display_name', DEFAULT_NOTES_MODEL_NAME)
            notes_index = notes_options.index(notes_default) if notes_default in notes_options else 0
            st.selectbox("Notes/Enrichment Model:", options=notes_options, key="selected_notes_model_display_name", index=notes_index, help="Model for final output generation (Step 3).")

    with col_main_2:
        st.subheader("") # Spacer
        st.button("🧹 Clear All Inputs & Outputs", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

st.divider()

# --- Input Sections ---
with st.container(border=True):
    is_enrich_mode = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                      st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")

    if is_enrich_mode:
        st.subheader("Existing Notes Input (Required for Enrichment)")
        st.text_area("Paste your existing notes here:", height=200, key="existing_notes_input",
                     placeholder="Paste the notes you want to enrich...",
                     help="These notes will be used as the base for enrichment.",
                     value=st.session_state.get("existing_notes_input", ""))
        st.markdown("---")
        st.subheader("Source Transcript Input (Text, PDF, or Audio)")
    else:
        st.subheader("Source Input (Transcript or Audio)")

    # Source Input Widgets
    input_options = ("Paste Text", "Upload PDF", "Upload Audio")
    input_default = st.session_state.get('input_method_radio', 'Paste Text')
    input_index = input_options.index(input_default) if input_default in input_options else 0
    st.radio(label="Source input type:", options=input_options, key="input_method_radio", horizontal=True, label_visibility="collapsed", index=input_index)
    input_type_ui = st.session_state.get('input_method_radio', 'Paste Text')
    if input_type_ui == "Paste Text":
        st.text_area("Paste source transcript:", height=150, key="text_input",
                     placeholder="Paste transcript source...", value=st.session_state.get("text_input", ""))
    elif input_type_ui == "Upload PDF":
        st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else: # Upload Audio
        st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.markdown("---")

    # Topics & Context Section
    st.subheader("Topics & Context")
    col3a, col3b = st.columns(2)

    with col3a: # Topics
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            sector_options = list(SECTOR_OPTIONS)
            sector_default = st.session_state.get('selected_sector', DEFAULT_SECTOR)
            sector_index = sector_options.index(sector_default) if sector_default in sector_options else 0
            st.selectbox("Select Sector (for Topic Template):",
                         options=sector_options, key="selected_sector", index=sector_index,
                         help="Loads a topic template. Select 'Other' to keep/edit manually.")

            st.text_area("Earnings Call Topics (Edit below):",
                         value=st.session_state.get("earnings_call_topics", ""), key="earnings_call_topics",
                         height=150, placeholder="Enter topics manually or select a sector to load a template...",
                         help="Guides structure for new notes or focuses enrichment. Edit freely.")
        else:
             st.caption("Topic selection/editing is available for Earnings Calls.")

    with col3b: # Context & Prompt Edit Toggle
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.get('add_context_enabled'):
            st.text_area("Context Details:", height=75, key="context_input",
                         placeholder="E.g., Company Name, Ticker, Date, Key Competitors...",
                         value=st.session_state.get("context_input", ""))

        st.write("")
        selected_mt = st.session_state.get('selected_meeting_type')
        disable_edit_checkbox = is_enrich_mode or (selected_mt == "Custom")

        if selected_mt != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled",
                         disabled=disable_edit_checkbox,
                         on_change=handle_edit_toggle, # Clears edits if toggled OFF
                         help="View/edit the base prompt used for generation. Disabled in Enrichment mode & Custom type.")
             if is_enrich_mode:
                 st.caption("Prompt editing is disabled in Enrichment mode.")

# --- Prompt Area (Conditional Display) ---
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and
                    st.session_state.get('selected_meeting_type') != "Custom" and not is_enrich_mode)

if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Final Prompt Editor" if st.session_state.get('selected_meeting_type') != "Custom" else "Custom Final Prompt (Required)"
        st.subheader(prompt_title)

        # Get the base template text (correctly formatted)
        base_template_text = get_prompt_display_text(for_display_only=True)
        # If editing is enabled and the current edit state is empty, populate it with the base template
        if st.session_state.view_edit_prompt_enabled and not st.session_state.current_prompt_text.strip():
             st.session_state.current_prompt_text = base_template_text

        # Display the content of current_prompt_text (either user edits or the populated base)
        st.text_area(
            label="Prompt Text:",
            value=st.session_state.current_prompt_text,
            key="current_prompt_text", # Let user edits update the state directly
            height=350,
            label_visibility="collapsed",
            help="Edit the prompt used for note generation. Ensure required placeholders like {transcript} are present." if st.session_state.get('selected_meeting_type') != "Custom" else "Enter your full custom prompt. Use {transcript} and {context_section} placeholders.",
            disabled=False
        )

        if st.session_state.get('selected_meeting_type') != "Custom":
             st.caption("Editing enabled. Ensure required placeholders like `{transcript}` (and `{topic_instructions}` for Earnings Calls) are present. Placeholders like `{context_section}` will be filled if context is added.")
        else:
             st.caption("Placeholders `{transcript}` and `{context_section}` will be automatically filled during processing.")


# --- Generate Button ---
st.write("")
is_valid, error_msg = validate_inputs() # Validation now checks placeholders in edited prompts
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes based on current inputs and settings."
generate_button_label = "🚀 Enrich Notes" if is_enrich_mode else "🚀 Generate Notes"

generate_button = st.button(generate_button_label,
                            type="primary",
                            use_container_width=True,
                            disabled=st.session_state.get('processing') or st.session_state.get('generating_filename') or not is_valid,
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
        # Clear error after displaying if desired:
        # st.session_state.error_message = None
    elif st.session_state.get('generated_notes'):
        output_title = "✅ Enriched Notes" if is_enrich_mode else "✅ Generated Notes"
        st.subheader(output_title)

        # Transcript previews
        if st.session_state.get('raw_transcript'):
            with st.expander("View Raw Source Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True, key="raw_transcript_view")
        if st.session_state.get('refined_transcript'):
             with st.expander("View Refined Source Transcript (Step 2 Output)", expanded=True):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True, key="refined_transcript_view")

        # Edit and display notes
        st.checkbox("Edit Output", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        is_expert_meeting_summary = (st.session_state.get('selected_meeting_type') == "Expert Meeting" and
                                     st.session_state.get('expert_meeting_prompt_option') == "Option 3: Option 2 + Executive Summary" and
                                     "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n" in notes_content_to_use)
        if st.session_state.get('edit_notes_enabled'):
            st.text_area("Editable Output:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            if is_expert_meeting_summary:
                 try:
                     notes_part, summary_part = notes_content_to_use.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n", 1)
                     st.markdown("### Detailed Notes (Q&A Format)")
                     st.markdown(notes_part)
                     st.markdown("---")
                     st.markdown("### Executive Summary")
                     st.markdown(summary_part)
                 except ValueError:
                     st.markdown(notes_content_to_use)
            else:
                 st.markdown(notes_content_to_use)

        # Download buttons
        st.write("")
        dl_cols = st.columns(3)
        output_type_label = "enriched_notes" if is_enrich_mode else "notes"
        default_fname_base = f"{st.session_state.get('selected_meeting_type', 'meeting').lower().replace(' ', '_')}_{output_type_label}"
        fname_base = st.session_state.get('suggested_filename', default_fname_base)
        with dl_cols[0]:
            st.download_button(label=f"⬇️ Output (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]:
            st.download_button(label=f"⬇️ Output (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.get('refined_transcript'):
                refined_fname_base = fname_base.replace(f"_{output_type_label}", "_refined_transcript") if f"_{output_type_label}" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the speaker-diarized and corrected source transcript from Step 2 (if audio was processed).")
            else:
                st.button("Refined Tx N/A", disabled=True, use_container_width=True, help="Refined transcript is only available after successful audio processing.")

    elif not st.session_state.get('processing'):
        st.markdown("<p class='initial-prompt'>Configure inputs above and click 'Generate / Enrich Notes' to start.</p>", unsafe_allow_html=True)


# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    history = st.session_state.get('history', [])
    if not history:
        st.caption("No generated notes in history for this session.")
    else:
        for i, entry in enumerate(history):
             with st.container():
                st.markdown(f"**#{i+1} - {entry.get('timestamp', 'N/A')}**")
                display_note = entry.get('notes', '')
                summary_separator = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
                preview_text = ""
                if summary_separator in display_note:
                     notes_part = display_note.split(summary_separator, 1)[0]
                     preview_text = "\n".join(notes_part.strip().splitlines()[:3]) + "\n... (+ Executive Summary)"
                else:
                    preview_text = "\n".join(display_note.strip().splitlines()[:4]) + "..."
                st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
                st.button(f"Restore Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                if i < len(history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    # Re-validate just before starting processing
    is_valid_on_click, error_msg_on_click = validate_inputs()
    if not is_valid_on_click:
        st.session_state.error_message = f"Validation Error: {error_msg_on_click}"
        st.session_state.processing = False # Ensure processing flag is off
        st.rerun() # Rerun to show validation error immediately
    else:
        # Set flags and clear previous results
        st.session_state.processing = True
        st.session_state.generating_filename = False
        st.session_state.generated_notes = None
        st.session_state.edited_notes_text = ""
        st.session_state.edit_notes_enabled = False
        st.session_state.error_message = None # Clear previous errors
        st.session_state.suggested_filename = None
        st.session_state.raw_transcript = None
        st.session_state.refined_transcript = None
        st.session_state.processed_audio_chunk_references = []
        st.rerun() # Rerun to show the status indicator and hide old results

# --- Processing Block ---
if st.session_state.get('processing') and not st.session_state.get('generating_filename') and not st.session_state.get('error_message'):

    processed_audio_chunk_references = []
    operation_desc = "Generating Notes"
    if st.session_state.get('selected_meeting_type') == "Earnings Call" and \
       st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
        operation_desc = "Enriching Notes"

    with st.status(f"🚀 Initializing {operation_desc} process...", expanded=True) as status:
        try:
            # --- 1. Read Inputs and Settings ---
            status.update(label="⚙️ Reading inputs and settings...")
            # Re-validate inside status (belt-and-suspenders)
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process:
                raise ValueError(f"Input validation failed just before processing: {error_msg_process}")

            # Get settings from session state
            meeting_type = st.session_state.selected_meeting_type
            expert_meeting_option = st.session_state.expert_meeting_prompt_option
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            view_edit_enabled = st.session_state.get('view_edit_prompt_enabled', False)
            edited_prompt_text = st.session_state.current_prompt_text.strip()

            # Determine if an edited prompt should be used
            use_edited_prompt = view_edit_enabled and edited_prompt_text and meeting_type != "Custom" and not is_enrich_mode

            # Get custom prompt if applicable
            custom_prompt_text = edited_prompt_text if meeting_type == "Custom" else None
            if meeting_type == "Custom" and not custom_prompt_text:
                 raise ValueError("Custom prompt is required for 'Custom' meeting type but is empty.")

            general_context = st.session_state.get('context_input', "").strip() if st.session_state.get('add_context_enabled') else None
            earnings_mode = st.session_state.get('earnings_call_mode')
            user_existing_notes = st.session_state.get('existing_notes_input', "").strip() if is_enrich_mode else None
            actual_input_type, source_transcript_data, source_audio_file_obj = get_current_input_data()

            if meeting_type == "Earnings Call":
                earnings_call_topics_text = st.session_state.get("earnings_call_topics", "").strip()
            else:
                 earnings_call_topics_text = ""

            # --- 2. Initialize AI Models ---
            status.update(label="🧠 Initializing AI models...")
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            # --- 3. Process Input Source (Audio handling) ---
            final_source_transcript = source_transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            if actual_input_type == "Upload Audio":
                if source_audio_file_obj is None:
                     raise ValueError("Audio file selected but no file object found.")
                st.session_state.uploaded_audio_info = source_audio_file_obj
                status.update(label=f"🔊 Loading source audio file '{source_audio_file_obj.name}'...")
                # (Audio loading and chunking code...)
                audio_bytes = source_audio_file_obj.getvalue()
                file_extension = os.path.splitext(source_audio_file_obj.name)[1].lower().replace('.', '')
                audio_format = file_extension
                if audio_format == 'm4a': audio_format = 'mp4'
                elif audio_format in ['oga']: audio_format = 'ogg'
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                except Exception as audio_load_err:
                     if "ffmpeg" in str(audio_load_err).lower() or "Couldn't find ffmpeg or avconv" in str(audio_load_err):
                         raise ValueError(f"❌ **Error:** Could not load audio. `ffmpeg` or `libav` might be missing. Install it (check https://ffmpeg.org/download.html). (Original error: {audio_load_err})")
                     else:
                         raise ValueError(f"❌ Could not load audio file using pydub (format: '{audio_format}'). Ensure file is valid. Error: {audio_load_err}")

                chunk_length_ms = 50 * 60 * 1000
                chunks = make_chunks(audio, chunk_length_ms)
                num_chunks = len(chunks)
                status.update(label=f"🔪 Splitting source audio into {num_chunks} chunk(s)...")

                # --- Step 1: Transcription per chunk ---
                all_transcripts = []
                processed_audio_chunk_references = []
                for i, chunk in enumerate(chunks):
                    # (Chunk processing, upload, transcription loop...)
                    chunk_num = i + 1
                    status.update(label=f"🔄 Processing Source Chunk {chunk_num}/{num_chunks}...")
                    temp_chunk_path = None
                    chunk_file_ref = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                            chunk.export(temp_chunk_file.name, format="wav")
                            temp_chunk_path = temp_chunk_file.name

                        status.update(label=f"☁️ Uploading Source Chunk {chunk_num}/{num_chunks}...")
                        chunk_display_name = f"chunk_{chunk_num}_of_{num_chunks}_{int(time.time())}_{source_audio_file_obj.name}"
                        chunk_file_ref = genai.upload_file(path=temp_chunk_path, display_name=chunk_display_name)
                        processed_audio_chunk_references.append(chunk_file_ref)

                        status.update(label=f"⏳ Waiting for Source Chunk {chunk_num}/{num_chunks} API processing...")
                        polling_start_time = time.time()
                        timeout_seconds = 600
                        while chunk_file_ref.state.name == "PROCESSING":
                            if time.time() - polling_start_time > timeout_seconds:
                                raise TimeoutError(f"Audio processing timed out for chunk {chunk_num} after {timeout_seconds}s.")
                            time.sleep(10)
                            chunk_file_ref = genai.get_file(chunk_file_ref.name)

                        if chunk_file_ref.state.name != "ACTIVE":
                            raise Exception(f"Audio chunk {chunk_num} processing failed. State: {chunk_file_ref.state.name}")

                        status.update(label=f"✍️ Step 1: Transcribing Source Chunk {chunk_num}/{num_chunks}...")
                        t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                        t_response = transcription_model.generate_content(
                            [t_prompt, chunk_file_ref],
                            generation_config=transcription_gen_config, safety_settings=safety_settings
                        )

                        if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                            all_transcripts.append(t_response.text.strip())
                        elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                            raise Exception(f"Transcription blocked for chunk {chunk_num}: {t_response.prompt_feedback.block_reason}")
                        else:
                            st.warning(f"⚠️ Source Transcription for chunk {chunk_num} returned empty. Skipping.", icon="🤔")
                            all_transcripts.append("")

                    except Exception as chunk_err:
                        raise Exception(f"❌ Error processing source audio chunk {chunk_num}: {chunk_err}") from chunk_err
                    finally:
                        if temp_chunk_path and os.path.exists(temp_chunk_path):
                            try: os.remove(temp_chunk_path)
                            except OSError as remove_err: st.warning(f"Could not remove temp chunk file {temp_chunk_path}: {remove_err}")

                # Combine transcripts
                status.update(label="🧩 Combining source chunk transcripts...")
                st.session_state.raw_transcript = "\n\n".join(all_transcripts).strip()

                # --- DEBUG PRINTS After Step 1 ---
                print("-" * 20 + " DEBUG: After Step 1 " + "-" * 20)
                print(f"Type of st.session_state.raw_transcript: {type(st.session_state.raw_transcript)}")
                raw_transcript_content = st.session_state.raw_transcript
                if raw_transcript_content:
                    print(f"Raw transcript length: {len(raw_transcript_content)}")
                    print(f"Raw transcript snippet (first 300 chars): {raw_transcript_content[:300]}")
                    print(f"Raw transcript is considered True in Python: {bool(raw_transcript_content)}")
                else:
                     print("Raw transcript is EMPTY or None")
                print("-" * 55)
                # --- END DEBUG PRINTS ---

                final_source_transcript = st.session_state.raw_transcript
                status.update(label="✅ Step 1: Full Source Transcription Complete!")

                # --- Step 2: Refinement (with Debug prints) ---
                print(f"\n>>> DEBUG: Checking condition 'if final_source_transcript:'")
                print(f"    Value of final_source_transcript is None: {final_source_transcript is None}")
                if isinstance(final_source_transcript, str):
                    print(f"    Value of final_source_transcript (str) length: {len(final_source_transcript)}")
                    print(f"    Condition evaluates to: {bool(final_source_transcript)}")
                else:
                    print(f"    Value of final_source_transcript type: {type(final_source_transcript)}")
                    print(f"    Condition evaluates to: {bool(final_source_transcript)}")

                if final_source_transcript: # Check if transcription yielded something non-empty
                    print(">>> DEBUG: Condition TRUE. Entering Step 2 Refinement block.")
                    try:
                        status.update(label=f"🧹 Step 2: Refining source transcript using {st.session_state.selected_refinement_model_display_name}...")
                        if not isinstance(st.session_state.raw_transcript, str):
                             print(f"!!! WARNING: st.session_state.raw_transcript is NOT a string before creating refinement prompt. Type: {type(st.session_state.raw_transcript)}")
                             current_raw_transcript_for_prompt = str(st.session_state.raw_transcript)
                        else:
                             current_raw_transcript_for_prompt = st.session_state.raw_transcript

                        refinement_prompt = f"""Please refine the following raw audio transcript:

                        **Raw Transcript:**
                        ```
                        {current_raw_transcript_for_prompt}
                        ```

                        **Instructions:**
                        1.  **Identify Speakers:** Assign consistent labels (e.g., Speaker 1, Speaker 2, Interviewer, Expert Name if identifiable). Place the label on a new line before the speaker's turn (e.g., `Speaker 1:`). If you cannot distinguish speakers reliably, use a single label like 'SPEAKER'.
                        2.  **Translate to English:** Convert any substantial non-English speech found within the transcript to English, ensuring it fits naturally within the conversation. Preserve original names or technical terms if translation is uncertain.
                        3.  **Correct Errors & Improve Readability:** Fix obvious spelling mistakes and grammatical errors. Use the overall conversation context to correct potentially misheard words or phrases where confident. Preserve technical terms or names if unsure. Remove excessive filler words (like 'um', 'uh', 'like') only if they severely hinder readability, but retain the natural conversational flow. Do not paraphrase or summarize.
                        4.  **Format:** Ensure clear separation between speaker turns using the speaker labels followed by a colon and a newline. Use standard paragraph breaks for longer turns if appropriate.
                        5.  **Output:** Provide *only* the refined, speaker-diarized, translated (where applicable), and corrected transcript text. Do not add any introduction, summary, or commentary before or after the transcript itself.

                        **Additional Context (Optional - use for understanding terms, names, etc.):**
                        {general_context if general_context else "None provided."}

                        **Refined Transcript:**
                        """
                        print(">>> DEBUG: Refinement prompt created. Calling generate_content...")
                        r_response = refinement_model.generate_content(
                            refinement_prompt,
                            generation_config=refinement_gen_config, safety_settings=safety_settings
                        )
                        print(">>> DEBUG: Refinement generate_content call returned.")

                        if r_response and hasattr(r_response, 'text') and r_response.text and r_response.text.strip():
                            print(">>> DEBUG: Refinement response received and has text content.")
                            st.session_state.refined_transcript = r_response.text.strip()
                            final_source_transcript = st.session_state.refined_transcript # Update for Step 3
                            status.update(label="🧹 Step 2: Source Refinement complete!")
                            print(f">>> DEBUG: Refined transcript length: {len(st.session_state.refined_transcript)}")
                        elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                            print(f">>> DEBUG: Refinement BLOCKED. Reason: {r_response.prompt_feedback.block_reason}")
                            st.warning(f"⚠️ Source Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Source Refinement blocked. Proceeding with raw transcript.")
                        else:
                            print(">>> DEBUG: Refinement response was empty or had no text.")
                            st.warning("🤔 Source Refinement step returned empty response. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Source Refinement failed. Proceeding with raw transcript.")
                    except Exception as refine_err:
                         print(f">>> DEBUG: EXCEPTION during Step 2 Refinement: {refine_err}")
                         st.warning(f"❌ Error during Step 2 (Source Refinement): {refine_err}. Using raw transcript for notes.", icon="⚠️")
                         status.update(label="⚠️ Source Refinement error. Proceeding with raw transcript.")
                else:
                    print(">>> DEBUG: Condition FALSE. SKIPPING Step 2 Refinement block.")
                    status.update(label="⚠️ Skipping Source Refinement (Step 2) as raw transcript is empty or invalid.")
            # End of Audio Processing Block

            # --- 4. Prepare Final Prompt for Notes/Enrichment ---
            if not final_source_transcript:
                 raise ValueError("No source transcript available (from text, PDF, or audio processing) to generate or enrich notes.")

            status.update(label=f"📝 Preparing final prompt for {operation_desc}...")
            final_api_prompt = None
            api_payload_parts = []
            prompt_template_base_text = None # Store the base template text if needed
            gen_config_to_use = main_gen_config

            format_kwargs = {
                'transcript': final_source_transcript,
                'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{general_context}\n---" if general_context else ""
            }

            # Determine which prompt to use based on priority
            if meeting_type == "Custom":
                prompt_template_base_text = custom_prompt_text
                final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                api_payload_parts = [final_api_prompt]
                status.update(label=f"📝 Using user's Custom prompt...")

            elif is_enrich_mode: # Enrichment mode
                prompt_template_base_text = PROMPTS["Earnings Call"]["Enrich Existing Notes"]
                gen_config_to_use = enrichment_gen_config
                # Format topic instructions for enrichment focus
                topic_instructions = ""
                if earnings_call_topics_text:
                    formatted_topics = [f"- {line.strip()}" for line in earnings_call_topics_text.split('\n') if line.strip()]
                    topic_list_str = "\n".join(formatted_topics)
                    topic_instructions = (f"Focus enrichment primarily on details related to the following user-specified topic structure:\n{topic_list_str}\n\n"
                                        f"Also incorporate any other highly significant financial or strategic points found in the transcript, potentially under an 'Other Key Points' section if they don't fit the provided structure.")
                else: # No specific topics provided
                    topic_instructions = (f"Since no specific topics were provided, identify the logical main themes in the transcript (e.g., Financials, Strategy, Outlook, Q&A) and enrich the user's existing notes based on significant information related to those themes.\n"
                                          f"Include any other highly significant points under an 'Other Key Points' section if relevant.")
                format_kwargs["topic_instructions"] = topic_instructions
                if user_existing_notes is None: raise ValueError("Existing notes required for Enrichment mode but not found.")
                format_kwargs["existing_notes"] = user_existing_notes
                final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                api_payload_parts = [final_api_prompt]
                status.update(label=f"📝 Using standard Enrich prompt...")

            elif use_edited_prompt: # Edit enabled and text area has content
                 prompt_template_base_text = edited_prompt_text # Use the edited text as the template
                 # Re-generate and add topic instructions if it's Generate New Notes Earnings Call
                 if meeting_type == "Earnings Call" and earnings_mode == "Generate New Notes":
                     topic_instructions = "" # Calculate as before
                     if earnings_call_topics_text:
                         formatted_topics = []
                         for line in earnings_call_topics_text.split('\n'):
                             trimmed_line = line.strip()
                             if trimmed_line and not trimmed_line.startswith(('-', '*', '#')): formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                             elif trimmed_line: formatted_topics.append(trimmed_line)
                         topic_list_str = "\n".join(formatted_topics)
                         topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                               f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                               f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
                     else: # No topics provided
                         topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                               f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                               f"Place all relevant details under the most appropriate heading.")
                     format_kwargs["topic_instructions"] = topic_instructions
                 # Format the *edited* prompt text
                 final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                 api_payload_parts = [final_api_prompt]
                 status.update(label=f"📝 Using edited prompt for {operation_desc}...")

            else: # Standard path (Not Custom, Not Enrich, Not Edited)
                 if meeting_type == "Expert Meeting":
                    if expert_meeting_option == "Option 1: Existing (Detailed & Strict)":
                        prompt_template_base_text = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
                    else: # Option 2 and 3 use Option 2 template
                        prompt_template_base_text = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]
                 elif meeting_type == "Earnings Call": # Must be Generate New Notes mode here
                     prompt_template_base_text = PROMPTS["Earnings Call"]["Generate New Notes"]
                     gen_config_to_use = main_gen_config
                     # Format topic instructions
                     topic_instructions = "" # Calculate as before
                     if earnings_call_topics_text:
                         formatted_topics = []
                         for line in earnings_call_topics_text.split('\n'):
                             trimmed_line = line.strip()
                             if trimmed_line and not trimmed_line.startswith(('-', '*', '#')): formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                             elif trimmed_line: formatted_topics.append(trimmed_line)
                         topic_list_str = "\n".join(formatted_topics)
                         topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                               f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                               f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
                     else: # No topics provided
                         topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                               f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                               f"Place all relevant details under the most appropriate heading.")
                     format_kwargs["topic_instructions"] = topic_instructions
                 else:
                     raise ValueError(f"Unhandled meeting type '{meeting_type}' in standard prompt selection logic.")

                 if not prompt_template_base_text:
                     raise ValueError(f"Could not find standard prompt template (Meeting: {meeting_type}, Option/Mode: {expert_meeting_option if meeting_type=='Expert Meeting' else earnings_mode}).")

                 final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                 api_payload_parts = [final_api_prompt]
                 status.update(label=f"📝 Using standard prompt for {operation_desc}...")


            if not final_api_prompt or not api_payload_parts:
                raise ValueError("Failed to prepare the final prompt for the API call.")

            # --- 5. Execute API Call (Notes/Enrichment) ---
            try:
                status.update(label=f"✨ Step 3: {operation_desc} using {st.session_state.selected_notes_model_display_name}...")
                # Optional Debug: Print final prompt before sending
                # print("-" * 20 + " FINAL PROMPT TO API " + "-" * 20)
                # print(final_api_prompt)
                # print("-" * 60)

                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=gen_config_to_use,
                    safety_settings=safety_settings
                )

                # --- 6. Process Response & Handle Summary Step ---
                generated_content = None
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    generated_content = response.text.strip()
                    status.update(label=f"✅ Initial {operation_desc} successful!")

                    # Check if Expert Meeting Option 3 (Summary step) is needed
                    is_expert_summary_step = (meeting_type == "Expert Meeting" and
                                              expert_meeting_option == "Option 3: Option 2 + Executive Summary" and
                                              not use_edited_prompt) # Don't summarize if prompt was edited

                    if is_expert_summary_step:
                        status.update(label=f"✨ Step 3b: Generating Executive Summary...")
                        summary_prompt_template = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY)
                        if not summary_prompt_template:
                             st.warning("⚠️ Could not find summary prompt template. Skipping summary step.", icon="❗")
                             st.session_state.generated_notes = generated_content
                             status.update(label="⚠️ Summary Prompt Missing. Only detailed notes generated.", state="warning")
                        else:
                            summary_kwargs = {'generated_notes': generated_content}
                            summary_prompt = format_prompt_safe(summary_prompt_template, **summary_kwargs)
                            try:
                                summary_response = notes_model.generate_content(
                                    summary_prompt,
                                    generation_config=summary_gen_config,
                                    safety_settings=safety_settings
                                )
                                if summary_response and hasattr(summary_response, 'text') and summary_response.text.strip():
                                    summary_text = summary_response.text.strip()
                                    st.session_state.generated_notes = f"{generated_content}\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{summary_text}"
                                    status.update(label="✅ Notes and Summary generated successfully!", state="complete")
                                else:
                                    reason = "Blocked" if hasattr(summary_response, 'prompt_feedback') else "Empty Response"
                                    st.warning(f"⚠️ Summary generation failed or was blocked ({reason}). Only detailed notes provided.", icon="⚠️")
                                    st.session_state.generated_notes = generated_content
                                    status.update(label="⚠️ Summary Failed/Blocked. Only detailed notes generated.", state="warning")
                            except Exception as summary_err:
                                 st.warning(f"❌ Error during summary generation step: {summary_err}. Only detailed notes provided.", icon="⚠️")
                                 st.session_state.generated_notes = generated_content
                                 status.update(label="⚠️ Summary Error. Only detailed notes generated.", state="warning")
                    else:
                        st.session_state.generated_notes = generated_content
                        status.update(label=f"✅ {operation_desc} completed successfully!", state="complete")

                    # --- 7. Post-generation Steps ---
                    if st.session_state.generated_notes:
                        st.session_state.edited_notes_text = st.session_state.generated_notes # Populate editor
                        add_to_history(st.session_state.generated_notes) # Add to history

                        status.update(label="💡 Suggesting filename...")
                        fname_label = meeting_type.replace(" ","_")
                        if is_enrich_mode: fname_label = "Enriched_Earnings_Call"
                        suggested_fname = generate_suggested_filename(st.session_state.generated_notes, fname_label)
                        st.session_state.suggested_filename = suggested_fname

                # Handle API errors/blocks for the main notes/enrichment call
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    st.session_state.error_message = f"⚠️ {operation_desc} was blocked by the API. Reason: {response.prompt_feedback.block_reason}. Please modify inputs or prompt if applicable."
                    status.update(label=f"❌ Blocked: {response.prompt_feedback.block_reason}", state="error")
                elif response is not None:
                    st.session_state.error_message = f"🤔 The AI returned an empty response during {operation_desc}. Try adjusting the input or model."
                    status.update(label="❌ Error: AI returned empty response.", state="error")
                else: # No response object at all
                    st.session_state.error_message = f"😥 The {operation_desc} API call failed (No response received). Check connection or API status."
                    status.update(label="❌ Error: API call failed (No response).", state="error")

            except Exception as api_call_err:
                 st.session_state.error_message = f"❌ Error during Step 3 ({operation_desc} API Call): {api_call_err}"
                 status.update(label=f"❌ Error during API call: {api_call_err}", state="error")

        except Exception as e:
             st.session_state.error_message = f"❌ Processing Error: {e}"
             status.update(label=f"❌ Error: {e}", state="error")

        finally:
            # --- 8. Cleanup ---
            st.session_state.processing = False # Mark processing as finished regardless of outcome
            if processed_audio_chunk_references:
                 st.toast(f"☁️ Performing final cleanup of {len(processed_audio_chunk_references)} cloud audio chunk(s)...", icon="🗑️")
                 refs_to_delete = list(processed_audio_chunk_references)
                 for file_ref in refs_to_delete:
                    try:
                        if file_ref and hasattr(file_ref, 'name'):
                           genai.delete_file(file_ref.name)
                           processed_audio_chunk_references.remove(file_ref)
                        else:
                            st.warning(f"Skipping cleanup for invalid file reference: {file_ref}", icon="⚠️")
                    except Exception as final_cleanup_error:
                        st.warning(f"Final cloud audio chunk cleanup failed for {getattr(file_ref, 'name', 'Unknown File')}: {final_cleanup_error}", icon="⚠️")
                 st.session_state.processed_audio_chunk_references = []
            st.rerun() # Rerun one last time to update UI


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

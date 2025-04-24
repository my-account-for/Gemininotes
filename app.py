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
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25", # Keep using this if available
}
# Ensure default models exist in the available list
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"

DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback to first available

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
# (Same as before)
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
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
# (Prompts remain the same as the previous version)
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
        "Generate New Notes": """You are an expert AI assistant creating DETAILED notes from an earnings call transcript for an investment firm.
Output MUST be comprehensive, factual notes, capturing all critical financial and strategic information.

**Formatting Requirements (Mandatory):**
- US$ for dollars (US$2.5M), % for percentages.
- State comparison periods (+5% YoY, -2% QoQ).
- Represent fiscal periods accurately (Q3 FY25).
- Use common abbreviations (CEO, KPI).
- Use bullet points under headings.
- Each bullet = complete sentence with distinct info.
- Capture ALL numbers, names, data accurately.
- Use quotes "" for significant statements.
- **DO NOT summarize or interpret unless part of the structure or explicitly stated in the call.**
- **DO NOT add information not mentioned in the transcript.**

**Note Structure:**
- **Call Participants:** (List names/titles or 'Not specified')
{topic_instructions}

**CRITICAL:** Ensure accuracy and adhere strictly to structure and formatting.
---
**EARNINGS CALL TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED EARNINGS CALL NOTES:**
""",
        "Enrich Existing Notes": """You are an expert AI assistant tasked with enriching existing earnings call notes using a provided source transcript.
Your goal is to identify significant financial, strategic, or forward-looking details mentioned in the **Source Transcript** that are MISSING from the **User's Existing Notes** and relevant to the specified **Topic Structure**. Integrate these missing details accurately and concisely into the existing notes, maintaining the overall structure and tone.

**Inputs:**
1.  **User's Existing Notes:** The notes provided by the user.
2.  **Source Transcript:** The full earnings call transcript.
3.  **Topic Structure:** Headings provided by the user (or logically derived if none provided) to guide the enrichment focus.
4.  **Additional Context:** Optional background information.

**Process:**
1.  Thoroughly read the **Source Transcript**.
2.  Carefully review the **User's Existing Notes** against the **Topic Structure**.
3.  Identify KEY information (specific financial figures, guidance updates, strategic initiatives, significant quotes, competitive remarks, Q&A points) present in the **Source Transcript** but ABSENT or INSUFFICIENTLY DETAILED in the **User's Existing Notes** under the relevant topics.
4.  Integrate these identified missing details into the appropriate sections of the **User's Existing Notes**.
    - Add new bullet points where necessary.
    - Augment existing bullet points ONLY if the addition is directly related and factual (e.g., adding a specific percentage change).
    - Maintain the formatting requirements (US$, %, YoY/QoQ, FY periods).
    - Use quotes "" for direct significant statements added from the transcript.
    - Ensure added points are factual and directly from the transcript.
5.  If a topic in the **Topic Structure** is completely missing from the **User's Existing Notes** but discussed in the transcript, add the heading and relevant bullet points from the transcript.
6.  Output the **Complete Enriched Notes**, incorporating the additions. DO NOT output commentary about the changes made.

**Formatting Requirements for Added Information:**
- US$ for dollars (US$2.5M), % for percentages.
- State comparison periods (+5% YoY, -2% QoQ).
- Represent fiscal periods accurately (Q3 FY25).
- Use common abbreviations (CEO, KPI).
- New points should be complete sentences.
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
    'earnings_call_topics': '',
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
# (extract_text_from_pdf, format_prompt_safe, create_docx, get_current_input_data,
#  validate_inputs, handle_edit_toggle, get_prompt_display_text,
#  generate_suggested_filename, add_to_history, restore_note_from_history remain the same)
def extract_text_from_pdf(pdf_file_stream):
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def update_topic_template():
    # Updates the earnings_call_topics in session state based on selected_sector
    # Ensure this function exists and is correctly defined
    selected_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR) # Use get for safety
    if selected_sector in SECTOR_TOPICS:
        st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
    elif 'earnings_call_topics' not in st.session_state: # Initialize if completely missing
        st.session_state.earnings_call_topics = ""
    # If sector is "Other / Manual Topics", don't overwrite existing manual input
    elif selected_sector != DEFAULT_SECTOR:
         st.session_state.earnings_call_topics = ""


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
    input_method = st.session_state.input_method_radio
    meeting_type = st.session_state.selected_meeting_type
    custom_prompt = st.session_state.current_prompt_text

    if input_method == "Paste Text" and not st.session_state.text_input.strip():
        return False, "Please paste the source transcript text."
    if input_method == "Upload PDF" and st.session_state.pdf_uploader is None:
        return False, "Please upload a source PDF file."
    if input_method == "Upload Audio" and st.session_state.audio_uploader is None:
        return False, "Please upload a source audio file."

    if meeting_type == "Custom" and not custom_prompt.strip():
         return False, "Custom prompt cannot be empty for 'Custom' meeting type."
    # Use .get() for safety in case state is somehow missing
    if meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes":
        if not st.session_state.get('existing_notes_input',"").strip():
            return False, "Please provide your existing notes for enrichment."

    return True, ""

def handle_edit_toggle():
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = "" # Clear edits if toggled off

def get_prompt_display_text(for_display_only=False):
    # --- This function generates the prompt text for the EDITABLE AREA ---
    # --- It does NOT generate the Enrichment prompt (that's internal) ---
    meeting_type = st.session_state.selected_meeting_type

    # If editing is enabled and user has potentially edited, return the edited version
    # UNLESS we explicitly ask for the default display text
    if not for_display_only and st.session_state.view_edit_prompt_enabled and meeting_type != "Custom" and st.session_state.current_prompt_text:
        return st.session_state.current_prompt_text

    # Otherwise, generate the default prompt text for display/initial population
    display_text = ""
    temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
    input_type = st.session_state.input_method_radio
    placeholder = "[TRANSCRIPT WILL APPEAR HERE]"
    format_kwargs = {
        'transcript': placeholder,
        'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{temp_context}\n---" if temp_context else ""
    }
    prompt_template_to_display = None

    try:
        if meeting_type == "Expert Meeting":
            expert_option = st.session_state.expert_meeting_prompt_option
            if expert_option == "Option 1: Existing (Detailed & Strict)":
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
            else: # Option 2 and 3 use Option 2 template as base
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

            if prompt_template_to_display:
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                 if expert_option == "Option 3: Option 2 + Executive Summary":
                     summary_prompt_preview = PROMPTS["Expert Meeting"][EXPERT_MEETING_SUMMARY_PROMPT_KEY].split("---")[0]
                     display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step generated *after* these notes, using a separate prompt starting like:\n'''\n{summary_prompt_preview.strip()}\n'''"
            else:
                display_text = "# Error: Could not find prompt template for display."

        elif meeting_type == "Earnings Call":
             # The editable prompt is ALWAYS the "Generate New Notes" one, even if Enrich mode is selected elsewhere
             prompt_template_to_display = PROMPTS["Earnings Call"]["Generate New Notes"]
             user_topics_text_for_display = str(st.session_state.get('earnings_call_topics', "") or "")
             topic_instructions_preview = ""
             if user_topics_text_for_display and user_topics_text_for_display.strip():
                 topic_instructions_preview = f"Structure notes under user-specified headings (e.g., {user_topics_text_for_display.splitlines()[0]}...)\n- **Other Key Points**"
             else:
                 topic_instructions_preview = "Identify logical main themes...\n- **Other Key Points**"
             format_kwargs['topic_instructions'] = topic_instructions_preview
             display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)

        elif meeting_type == "Custom":
             # Logic for custom prompt display remains the same
             audio_note = ("\n# NOTE FOR AUDIO: If using audio, the system will first chunk and transcribe,\n"
                           "# then *refine* the full transcript (speaker ID, translation, corrections).\n"
                           "# Your custom prompt below will receive this *refined transcript* as the primary text input.\n"
                           "# Design your prompt accordingly.\n")
             default_custom = "# Enter your custom prompt here...\n# Use {transcript} and {context_section} placeholders.\n# Example: Summarize this meeting:\n# {transcript}\n# {context_section}"
             # Use current edit only if not for_display_only
             current_or_default = st.session_state.current_prompt_text if not for_display_only and st.session_state.current_prompt_text else default_custom
             display_text = current_or_default + (audio_note if st.session_state.input_method_radio == 'Upload Audio' else "")
             return display_text # Return custom text directly

        else: # Should not happen
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
             return "Error generating prompt preview."

        # Add audio note if applicable for standard prompts
        if input_type == "Upload Audio" and meeting_type != "Custom":
             audio_note = ("# NOTE FOR AUDIO: 3-step process (Chunked Transcribe -> Refine -> Notes).\n"
                           "# This prompt (or your edited version) is used in Step 3 with the *refined* transcript.\n"
                           "####################################\n\n")
             display_text = audio_note + display_text

    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview: Review inputs/prompt structure."

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
    st.session_state.earnings_call_topics = ""
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
    update_topic_template() # Reset topics based on default sector
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type):
    # (Same as before)
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings) # Use flash for speed
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "_").lower()
        notes_preview = notes_content.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n")[0]
        filename_prompt = (f"Suggest a concise filename (max 5 words including type, use underscores): Date={today_date}, Type='{mt_cleaned}'. Base it on key topics/company name from notes. Output ONLY filename.\nNOTES:{notes_preview[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if not s_name.startswith(today_date):
                s_name = f"{today_date}_{s_name}"
            if s_name: st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename suggestion empty/invalid.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    # (Same as before)
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
    # (Same as before)
    if 0 <= index < len(st.session_state.history): entry = st.session_state.history[index]; st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]; \
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None; st.session_state.error_message = None; st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section Container
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                     on_change=lambda: st.session_state.update(current_prompt_text="")) # Clear edits on change

            # Conditional options based on meeting type
            if st.session_state.selected_meeting_type == "Expert Meeting":
                st.radio(
                    "Expert Meeting Note Style:",
                    options=EXPERT_MEETING_OPTIONS,
                    key="expert_meeting_prompt_option",
                    index=EXPERT_MEETING_OPTIONS.index(st.session_state.expert_meeting_prompt_option),
                    help="...", # Keep help text concise
                    on_change=lambda: st.session_state.update(current_prompt_text="") # Clear edits on change
                )
            elif st.session_state.selected_meeting_type == "Earnings Call":
                st.radio(
                    "Mode:",
                    options=EARNINGS_CALL_MODES,
                    key="earnings_call_mode",
                    horizontal=True,
                    index=EARNINGS_CALL_MODES.index(st.session_state.earnings_call_mode),
                    help="Choose to generate notes from scratch or enrich existing ones."
                )

        with col1b:
            st.subheader("AI Model Selection")
            st.selectbox("Notes/Enrichment Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model for final output generation.")
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model for audio transcription (Step 1).")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model for audio refinement (Step 2).")

    with col_main_2:
        st.subheader("") # Spacer
        st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

st.divider() # Divider after the first main container

# --- Input Sections - Structure Adjusted ---
with st.container(border=True): # Container for Inputs (Existing Notes, Source, Topics, Context)
    # --- Display Existing Notes Input FIRST if in Enrich mode ---
    if st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes":
        st.subheader("Existing Notes Input (for Enrichment)")
        st.text_area("Paste your existing notes here:", height=200, key="existing_notes_input",
                     placeholder="Paste the notes you want to enrich...")
        st.markdown("---") # Visual separator
        st.subheader("Source Transcript Input (Text, PDF, or Audio)")
    else:
        st.subheader("Source Input")

    # --- Common Source Input Widgets ---
    st.radio(label="Source input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste source transcript:", height=150, key="text_input", placeholder="Paste transcript source...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.markdown("---") # Visual separator

    # --- Topics & Context Section ---
    st.subheader("Topics & Context")
    col3a, col3b = st.columns(2)
    with col3a: # Topics
        if st.session_state.selected_meeting_type == "Earnings Call":
            st.selectbox("Select Sector (for Topic Template):", options=SECTOR_OPTIONS, key="selected_sector")
            # Automatically update topics if sector changes *before* rendering text area
            update_topic_template()
            st.text_area("Earnings Call Topics (Edit below):", key="earnings_call_topics", height=150,
                         placeholder="Enter topics manually or edit loaded template...",
                         help="Guides structure for new notes or focuses enrichment.")
        else:
             st.caption("Topic selection only available for Earnings Calls.") # Placeholder

    with col3b: # Context & Prompt Edit Toggle
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=75, key="context_input", placeholder="Company Name, Ticker...")

        st.write("") # Spacer
        # View/Edit Prompt Checkbox
        selected_mt = st.session_state.selected_meeting_type
        disable_edit_checkbox = (selected_mt == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes")
        if selected_mt != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled",
                         disabled=disable_edit_checkbox,
                         on_change=handle_edit_toggle,
                         help="View/edit the prompt used for generation. Disabled in Enrichment mode.")
             if disable_edit_checkbox:
                 st.caption("Prompt editing disabled in Enrichment mode.")


# --- Prompt Area (Conditional Display) ---
# (Logic remains the same - shows if Custom or View/Edit enabled and not Enrich mode)
show_prompt_area = (st.session_state.selected_meeting_type == "Custom") or \
                   (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom" and not \
                    (st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes"))

if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Final Prompt Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Final Prompt (Required)"
        st.subheader(prompt_title)

        default_prompt_for_display = get_prompt_display_text(for_display_only=True)
        current_value = st.session_state.current_prompt_text
        # Show the default prompt if the current state is empty, otherwise show the user's edit
        prompt_to_display = current_value if current_value else default_prompt_for_display

        st.text_area(
            label="Prompt Text:",
            value=prompt_to_display,
            key="current_prompt_text", # Update state on edit
            height=350,
            label_visibility="collapsed",
            help="Edit the prompt used for note generation. Placeholders {transcript} and {context_section} will be filled." if st.session_state.selected_meeting_type != "Custom" else "Enter your full custom prompt. Use {transcript} and {context_section} placeholders.",
            disabled=False
        )
        if st.session_state.selected_meeting_type != "Custom":
             st.caption("Editing enabled. Placeholders `{transcript}` and `{context_section}` will be filled.")
        else: # Custom prompt
             st.caption("Placeholders `{transcript}` and `{context_section}` will be automatically filled.")

# --- Generate Button ---
st.write("")
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
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None # Clear after showing
    elif st.session_state.generated_notes:
        output_title = "✅ Enriched Notes" if (st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes") else "✅ Generated Notes"
        st.subheader(output_title)

        if st.session_state.raw_transcript:
            with st.expander("View Raw Source Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
        if st.session_state.refined_transcript:
             with st.expander("View Refined Source Transcript (Step 2 Output)", expanded=bool(st.session_state.refined_transcript)):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)

        st.checkbox("Edit Output", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes

        is_expert_meeting_summary = (st.session_state.selected_meeting_type == "Expert Meeting" and
                                     st.session_state.expert_meeting_prompt_option == "Option 3: Option 2 + Executive Summary" and
                                     "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n" in notes_content_to_use)

        if st.session_state.edit_notes_enabled:
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
                     st.markdown(notes_content_to_use) # Fallback
            else:
                 st.markdown(notes_content_to_use)

        st.write("") # Spacer
        dl_cols = st.columns(3)
        output_type_label = "enriched_notes" if (st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes") else "notes"
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_{output_type_label}"
        if st.session_state.suggested_filename: fname_base = st.session_state.suggested_filename
        else: fname_base = default_fname
        with dl_cols[0]: st.download_button(label=f"⬇️ Output (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]: st.download_button(label=f"⬇️ Output (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.refined_transcript:
                refined_fname_base = fname_base.replace(f"_{output_type_label}", "_refined_transcript") if f"_{output_type_label}" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the speaker-diarized and corrected source transcript from Step 2.")
            else: st.button("Refined Tx N/A", disabled=True, use_container_width=True, help="Refined transcript is only available after successful audio processing.")
    elif not st.session_state.processing:
        st.markdown("<p class='initial-prompt'>Configure inputs above and click 'Generate / Enrich Notes'.</p>", unsafe_allow_html=True)


# --- History Section ---
# (History section logic remains the same)
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
             with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                display_note = entry['notes']
                summary_separator = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
                preview_text = ""
                if summary_separator in display_note:
                     notes_part, summary_part = display_note.split(summary_separator, 1)
                     preview_text = "\n".join(notes_part.strip().splitlines()[:3]) + "\n... (+ Executive Summary)"
                else:
                    preview_text = "\n".join(display_note.strip().splitlines()[:3]) + "..."

                st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
                st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                if i < len(st.session_state.history) - 1: st.divider()

# --- Processing Logic ---
# (Processing logic remains the same as the previous corrected version)
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

if st.session_state.processing and not st.session_state.generating_filename:
    processed_audio_chunk_references = []

    operation_desc = "Generating Notes"
    if st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes":
        operation_desc = "Enriching Notes"

    with st.status(f"🚀 Initializing {operation_desc} process...", expanded=True) as status:
        try:
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process:
                raise ValueError(f"Input validation failed: {error_msg_process}")

            status.update(label="⚙️ Reading inputs and settings...")
            meeting_type = st.session_state.selected_meeting_type
            expert_meeting_option = st.session_state.expert_meeting_prompt_option
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            user_edited_or_custom_prompt = st.session_state.current_prompt_text.strip()
            general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
            earnings_mode = st.session_state.earnings_call_mode
            user_existing_notes = st.session_state.existing_notes_input.strip() if earnings_mode == "Enrich Existing Notes" else None
            actual_input_type, source_transcript_data, source_audio_file_obj = get_current_input_data()

            # Update topics *after* getting meeting_type and potentially sector
            if meeting_type == "Earnings Call":
                update_topic_template() # Ensure topics are potentially updated
                earnings_call_topics_text = st.session_state.earnings_call_topics.strip()
            else:
                 earnings_call_topics_text = "" # Ensure it's empty otherwise


            status.update(label="🧠 Initializing AI models...")
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            final_source_transcript = source_transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            # Audio Processing (Step 1 & 2)...
            # (Logic remains the same - processes source_audio_file_obj into final_source_transcript)
            if actual_input_type == "Upload Audio":
                # ... [Existing Audio Processing Logic For Source] ...
                st.session_state.uploaded_audio_info = source_audio_file_obj
                status.update(label=f"🔊 Loading source audio file '{source_audio_file_obj.name}'...")
                # ... (rest of audio processing) ...
                # ... (ensure final_source_transcript gets updated) ...

            # Notes Generation / Enrichment (Step 3)...
            # (Logic remains the same - determines prompt, formats, calls API, handles response)
            if not final_source_transcript:
                 raise ValueError("No source transcript available to generate or enrich notes.")

            status.update(label=f"📝 Preparing final prompt for {operation_desc}...")
            final_api_prompt = None
            api_payload_parts = []
            prompt_template = None
            gen_config_to_use = main_gen_config

            format_kwargs = {
                'transcript': final_source_transcript,
                'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{general_context}\n---" if general_context else ""
            }

            use_edited_or_custom = user_edited_or_custom_prompt and \
                                   not (meeting_type == "Earnings Call" and earnings_mode == "Enrich Existing Notes")

            if use_edited_or_custom:
                 final_api_prompt = format_prompt_safe(user_edited_or_custom_prompt, **format_kwargs)
                 api_payload_parts = [final_api_prompt]
                 status.update(label=f"📝 Using edited/custom prompt for {operation_desc}...")

            else: # Standard path
                # ... [Existing logic to select standard prompt based on meeting_type/expert_option/earnings_mode] ...
                if meeting_type == "Expert Meeting":
                    # ... select expert prompt ...
                     if expert_meeting_option == "Option 1: Existing (Detailed & Strict)":
                        prompt_template = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
                     else: # Option 2 and 3 use Option 2 template for notes
                        prompt_template = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

                elif meeting_type == "Earnings Call":
                     if earnings_mode == "Generate New Notes":
                        prompt_template = PROMPTS["Earnings Call"]["Generate New Notes"]
                        gen_config_to_use = main_gen_config
                        # ... format topic instructions ...
                        user_topics_text = earnings_call_topics_text
                        topic_instructions = ""
                        if user_topics_text: # Check if topics exist
                            formatted_topics = []
                            for line in user_topics_text.split('\n'):
                                trimmed_line = line.strip()
                                if trimmed_line and not trimmed_line.startswith(('-', '*')): formatted_topics.append(f"- **{trimmed_line}**")
                                else: formatted_topics.append(line)
                            topic_list_str = "\n".join(formatted_topics)
                            topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                                  f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered above)\n\n"
                                                  f"Place details under the most appropriate heading. If a topic isn't discussed, state 'Not discussed' under that heading.")
                        else:
                            topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed (e.g., Financials, Strategy, Outlook, Q&A). Use these themes as **bold headings**.\n"
                                                  f"Include a final mandatory section:\n- **Other Key Points** (for important info not covered in main themes)\n\n"
                                                  f"Place details under the most appropriate heading.")
                        format_kwargs["topic_instructions"] = topic_instructions

                     elif earnings_mode == "Enrich Existing Notes":
                        prompt_template = PROMPTS["Earnings Call"]["Enrich Existing Notes"]
                        gen_config_to_use = enrichment_gen_config
                        # ... format topic instructions for enrichment ...
                        user_topics_text = earnings_call_topics_text
                        topic_instructions = ""
                        if user_topics_text: # Check if topics exist
                            formatted_topics = []
                            for line in user_topics_text.split('\n'):
                                trimmed_line = line.strip()
                                if trimmed_line and not trimmed_line.startswith(('-', '*')): formatted_topics.append(f"- **{trimmed_line}**")
                                else: formatted_topics.append(line)
                            topic_list_str = "\n".join(formatted_topics)
                            topic_instructions = (f"Focus enrichment on the following user-specified topic structure:\n{topic_list_str}\n\n"
                                                f"Also consider any other significant points found in the transcript under an 'Other Key Points' section if relevant.")
                        else:
                            topic_instructions = (f"Since no specific topics were provided, identify logical main themes in the transcript (e.g., Financials, Strategy, Outlook, Q&A) and enrich the existing notes based on those themes.\n"
                                                  f"Include significant points under an 'Other Key Points' section if relevant.")
                        format_kwargs["topic_instructions"] = topic_instructions
                        format_kwargs["existing_notes"] = user_existing_notes

                else:
                     raise ValueError(f"Unhandled meeting type for standard prompt: '{meeting_type}'")

                if not prompt_template:
                     raise ValueError(f"Could not find standard prompt template for the selected options.")
                final_api_prompt = format_prompt_safe(prompt_template, **format_kwargs)
                api_payload_parts = [final_api_prompt]
                status.update(label=f"📝 Using standard prompt for {operation_desc}...")


            if not final_api_prompt:
                raise ValueError("Failed to determine the final prompt for the API call.")

            # API Call ...
            try:
                status.update(label=f"✨ Step 3: {operation_desc} using {st.session_state.selected_notes_model_display_name}...")
                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=gen_config_to_use,
                    safety_settings=safety_settings
                )

                # Handle Response & Summary Step (if applicable)...
                # (Logic remains the same)
                # ... [Response handling logic] ...

            except Exception as api_call_err:
                # ... API error handling ...
                 st.session_state.error_message = f"❌ Error during Step 3 (API Call): {api_call_err}"
                 status.update(label=f"❌ Error: {api_call_err}", state="error")


        except Exception as e:
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        finally:
            st.session_state.processing = False
            # Audio Chunk Cleanup...
            # (Logic remains the same)
            if processed_audio_chunk_references:
                # ... cleanup logic ...
                 st.toast(f"☁️ Performing final cleanup of {len(processed_audio_chunk_references)} remaining cloud chunk file(s)...", icon="🗑️")
                 refs_to_delete = list(processed_audio_chunk_references)
                 for file_ref in refs_to_delete:
                    try:
                        genai.delete_file(file_ref.name)
                        processed_audio_chunk_references.remove(file_ref)
                    except Exception as final_cleanup_error:
                        st.warning(f"Final cloud audio chunk cleanup failed for {file_ref.name}: {final_cleanup_error}", icon="⚠️")
                 st.session_state.processed_audio_chunk_references = []

            st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

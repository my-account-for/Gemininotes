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

# --- A New "Base" Prompt for Expert Meeting Instructions ---
# This is the new, single source of truth for the core note-taking rules.
# When a user edits the prompt for an "Expert Meeting", they will be editing this content.
EXPERT_MEETING_CHUNK_BASE = """### **NOTES STRUCTURE**

(1.) Opening overview or Expert background (Optional):

If the transcript begins with an overview, agenda, or expert introduction, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple, direct language. DO NOT summarize.

Please omit any introduction around Janchor Partners and focus only on the expert’s background or the overview.

(2.) Q&A Format: Structure the main body STRICTLY in Question/Answer format.

(2.A) Questions:

Extract clear questions from the transcript.

Rephrase slightly ONLY for clarity when needed.

Format each question clearly in bold.

Combine follow-up questions with their main question so they appear together as a single grouped question.

When possible, bring forward related future questions on the same topic so the Q&A is easy to follow and minimizes back-and-forth repetition.

(2.B) Answers:

Place answers directly below the corresponding question in bullet points.

Each bullet should convey a specific factual detail in a complete, natural-sounding sentence.

Combine closely related or sequential details into a single sentence where it improves flow, but never omit any detail.

Capture all specifics: data, names, examples, monetary values, percentages, timelines, etc.

DO NOT use sub-bullets or section headers within answers.

DO NOT add summaries, interpretations, conclusions, or action items that are not explicitly stated.

Maintain clarity while ensuring completeness and precision of the information.

Additional Instructions:

Accuracy is paramount. Capture every factual detail exactly as stated.

Completeness over brevity: Always include all details, even if minor. Err on the side of too much detail rather than too little.

Use clear, concise, and natural language, avoiding unnecessary filler.

Include ONLY what is present in the transcript. DO NOT add external context.

If a section (like Opening Overview) isn’t present, simply OMIT it."""


# --- Prompts for Long Transcript Chunking (Now as Wrappers) ---
PROMPT_INITIAL = """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.
Generate detailed, factual notes from the provided meeting transcript. You will process the transcript sequentially. For every Question/Answer pair you identify, you must generate notes following the structure below.

---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting. continuing a note-taking task. Your goal is to process the new transcript chunk provided below, using the context from the previous chunk to ensure perfect continuity.

### **CONTEXT FROM PREVIOUS CHUNK**
{context_package}

---
### **PRIMARY INSTRUCTIONS**

First, review the context. Because of the overlap in the transcript, you will see text that was already processed. **Locate the "Last Question Processed" from the context and begin your work from the *first NEW question and answer* that follows it.**

From that point forward, you must adhere to the following procedure for every Q&A pair in the remainder of the chunk.

---
{base_instructions}
---
**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""


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
    "Gemini 2.0 Flash": "gemini-2.0-flash-lite",
    "Gemini 2.5 Flash": "gemini-2.5-flash", "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro"
DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 2.5 Flash"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]
DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 2.5 Flash Lite"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]
DEFAULT_EARNINGS_CALL_MODE = EARNINGS_CALL_MODES[1]
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
    main_gen_config = {"temperature": 0.5, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    summary_gen_config = {"temperature": 0.6, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    refinement_gen_config = {"temperature": 0.2, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    safety_settings_relaxed = [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

except Exception as e:
    st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨")
    st.stop()


# --- Prompts Definitions (now with unified base instructions) ---
PROMPTS = {
    "Expert Meeting": {
        "Option 1: Existing (Detailed & Strict)": """You are an expert meeting note-taker. Your primary goal is COMPLETE and ACCURATE information capture. Do not summarize or omit details.
Generate factual notes from the provided transcript.

{base_instructions}
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Strict & Detailed):**
""",
        "Option 2: Less Verbose (Default)": """You are an expert meeting note-taker. Your primary goal is COMPLETE and ACCURATE information capture. Do not summarize or omit details.
Generate factual notes from the provided transcript, striving for natural sentence flow but **never sacrificing factual detail for brevity.**

{base_instructions}
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Complete & Detailed):**
""",
        "Summary Prompt (for Option 3)": """Based ONLY on the detailed 'GENERATED NOTES' provided below, create a concise executive summary highlighting the MOST significant insights, findings, or critical points.

**Format:**
1.  Identify the main themes discussed. Create a clear, bold heading for each theme.
2.  Under each heading, use bullet points to list the most significant insights.
3.  Each bullet point should represent a single, distinct key takeaway. Do not use sub-bullets.

**Instructions:**
- Focus on synthesizing key takeaways from the detailed points. Do not list minor details.
- Maintain an objective, professional tone.
- Do not introduce any information or conclusions not explicitly supported by the notes provided below.
---
**GENERATED NOTES (Input for Summary):**
{generated_notes}
---
**EXECUTIVE SUMMARY:**
"""
    },
    "Earnings Call": {
        "Generate New Notes": """You are an expert AI assistant creating DETAILED and COMPREHENSIVE notes from an earnings call transcript for an investment firm. Your primary goal is completeness.

**Formatting Requirements (Mandatory):**
- Use US$ for dollars (US$2.5M), % for percentages, and state comparison periods (+5% YoY, -2% QoQ).
- Represent fiscal periods accurately (Q3 FY25).
- Use bullet points under headings. Each bullet must be a complete sentence.
- Use crisp, professional language, but **ensure every financial figure, strategic detail, and forward-looking statement is captured. Brevity must not come at the cost of completeness.**

**CRITICAL INSTRUCTIONS:**
- **Capture ALL Details:** Your primary task is to extract all factual information. Do not summarize, generalize, or omit any data points, names, or significant qualitative statements.
- **Mandatory:** Capture ALL quantitative data (e.g., revenue figures, growth percentages, guidance numbers) and qualitative statements (e.g., strategic priorities, competitive landscape comments, significant quotes) with precision. Do not generalize financial results.
- **No Interpretation:** Your role is to extract and structure information, not to interpret it.
- **Fact-Based Only:** Do not add information not mentioned in the transcript.

**Note Structure:**
- **Call Participants:** (List names/titles)
{topic_instructions}
---
**EARNINGS CALL TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED EARNINGS CALL NOTES (Detailed):**
""",
        "Enrich Existing Notes": """You are an expert AI assistant tasked with enriching existing earnings call notes using a provided source transcript.
Your goal is to identify significant financial, strategic, or forward-looking details mentioned in the **Source Transcript** that are MISSING from the **User's Existing Notes** and relevant to the specified **Topic Structure**. Integrate these missing details **accurately and concisely** into the existing notes, maintaining the overall structure and tone.

**Process:**
1.  Thoroughly read the **Source Transcript**.
2.  Carefully review the **User's Existing Notes** against the **Topic Structure**.
3.  Identify KEY information (specific financial figures, guidance updates, strategic initiatives, significant quotes, competitive remarks, Q&A points) present in the **Source Transcript** but ABSENT or INSUFFICIENTLY DETAILED in the **User's Existing Notes** under the relevant topics.
4.  Integrate these identified missing details into the appropriate sections of the **User's Existing Notes** using **crisp and concise language**.
5.  Output the **Complete Enriched Notes**, incorporating the additions. DO NOT output commentary about the changes made.
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
REFINE_ONLY_OPTION = "Option 4: Refine Transcript Only"
EXPERT_MEETING_OPTIONS = [ "Option 1: Existing (Detailed & Strict)", "Option 2: Less Verbose (Default)", "Option 3: Option 2 + Executive Summary", REFINE_ONLY_OPTION ]
DEFAULT_EXPERT_MEETING_OPTION = EXPERT_MEETING_OPTIONS[1]
EXPERT_MEETING_SUMMARY_PROMPT_KEY = "Summary Prompt (for Option 3)"


# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None, 'uploaded_audio_info': None, 'add_context_enabled': False,
    'selected_notes_model_display_name': DEFAULT_NOTES_MODEL_NAME, 'selected_transcription_model_display_name': DEFAULT_TRANSCRIPTION_MODEL_NAME,
    'selected_refinement_model_display_name': DEFAULT_REFINEMENT_MODEL_NAME, 'selected_meeting_type': DEFAULT_MEETING_TYPE, 'expert_meeting_prompt_option': DEFAULT_EXPERT_MEETING_OPTION,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "", 'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '', 'selected_sector': DEFAULT_SECTOR, 'previous_selected_sector': DEFAULT_SECTOR, 'earnings_call_topics': '', 'earnings_call_mode': DEFAULT_EARNINGS_CALL_MODE,
    'existing_notes_input': "", 'edit_notes_enabled': False, 'edited_notes_text': "", 'suggested_filename': None, 'history': [], 'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': [], 'earnings_call_topics_initialized': False, 'speaker_1_name': "", 'speaker_2_name': "",
    'enable_refinement_step': False,
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def chunk_text_by_words(text, chunk_size=4000, overlap=200):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def _create_context_from_notes(notes_text):
    if not notes_text or not notes_text.strip():
        return ""
    questions = re.findall(r"(\*\*.*?\*\*)", notes_text)
    if not questions:
        return "" 
    last_question = questions[-1]
    answer_match = re.search(re.escape(last_question) + r"(.*?)(?=\*\*|$)", notes_text, re.DOTALL)
    last_answer = ""
    if answer_match:
        last_answer = answer_match.group(1).strip()
    context_package = (
        f"-   **Last Question Processed:** {last_question}\n"
        f"-   **Last Answer Provided:**\n{last_answer}"
    )
    return context_package.strip()

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
    transcript, audio_file = None, None
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
    # No validation needed for expert meeting edits as we now construct the full prompt in the backend
    return True, ""

def handle_edit_toggle():
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = ""

def get_prompt_display_text(for_display_only=False):
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    
    if meeting_type == "Expert Meeting" and st.session_state.get('expert_meeting_prompt_option') == REFINE_ONLY_OPTION:
        return "# Refine Transcript Only mode is active.\nThis mode does not use a final note-generation prompt. It will only perform transcription and refinement."

    # For expert meetings, if the user is editing, we show them ONLY the base instructions to edit.
    if meeting_type == "Expert Meeting" and st.session_state.get('view_edit_prompt_enabled', False):
        return EXPERT_MEETING_CHUNK_BASE

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
            prompt_template_wrapper = PROMPTS["Expert Meeting"][prompt_key]
            
            # For display, we inject the default base instructions into the wrapper
            full_prompt_template = prompt_template_wrapper.format(base_instructions=EXPERT_MEETING_CHUNK_BASE)
            
            if full_prompt_template:
                 display_text = format_prompt_safe(full_prompt_template, **format_kwargs)
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
             audio_note = "\n# NOTE: For audio, your custom prompt will receive a *refined transcript*." if st.session_state.get('enable_refinement_step') else ""
             default_custom = "# Enter your custom prompt... Use {transcript} and {context_section}."
             display_text = st.session_state.get('current_prompt_text', default_custom) + audio_note
             return display_text
        else:
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
             return "Error generating prompt preview."
        
        if st.session_state.get('enable_refinement_step'):
             refinement_note = "# NOTE: This prompt will be used with the *refined* transcript from Step 2.\n\n"
             display_text = refinement_note + display_text

    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview: {e}"
    return display_text

def clear_all_state():
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
    st.session_state.speaker_1_name, st.session_state.speaker_2_name = "", ""
    st.session_state.enable_refinement_step = False
    st.toast("Inputs and outputs cleared!", icon="🧹")
    st.rerun()

def generate_suggested_filename(notes_content, meeting_type, is_refine_only=False):
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        today_date = datetime.now().strftime("%Y%m%d")
        
        if is_refine_only:
             mt_cleaned = "refined_transcript"
        else:
             mt_cleaned = meeting_type.replace(" ", "_").lower()

        summary_marker = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
        if summary_marker in notes_content:
            notes_preview = notes_content.split(summary_marker)[0]
        else:
            notes_preview = notes_content

        filename_prompt = (f"Suggest a concise filename (max 5 words, use underscores_not_spaces). Start with {today_date}_{mt_cleaned}. Base on key topics/names from this text. Output ONLY the filename string (e.g., {today_date}_{mt_cleaned}_topic.txt). TEXT:\n{notes_preview[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config={"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"})
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
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

is_refine_only_mode = (
    st.session_state.get('selected_meeting_type') == "Expert Meeting" and
    st.session_state.get('expert_meeting_prompt_option') == REFINE_ONLY_OPTION
)

with st.container(border=True):
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        st.subheader("Meeting & Model Settings")
        col1a, col1b = st.columns(2)
        with col1a:
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True,
                     on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))
            st.text_input("Speaker 1 Name (Optional):", key="speaker_1_name", placeholder="e.g., John Doe - Expert")
            st.text_input("Speaker 2 Name (Optional):", key="speaker_2_name", placeholder="e.g., Jane Smith - Analyst")
        with col1b:
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_transcription_model_display_name', DEFAULT_TRANSCRIPTION_MODEL_NAME)), help="Model for audio-to-text (Step 1).")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_refinement_model_display_name', DEFAULT_REFINEMENT_MODEL_NAME)), help="Model for cleaning transcript & adding speakers (Step 2).")
            st.selectbox("Notes Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", index=list(AVAILABLE_MODELS.keys()).index(st.session_state.get('selected_notes_model_display_name', DEFAULT_NOTES_MODEL_NAME)), help="Model for generating final notes (Step 3).", disabled=is_refine_only_mode)
    
    with col_main_2:
        st.subheader("")
        st.button("🧹 Clear All Inputs & Outputs", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")
        
        refinement_value = True if is_refine_only_mode else st.session_state.get('enable_refinement_step', (st.session_state.input_method_radio == 'Upload Audio'))
        st.checkbox("Enable Transcript Refinement", key="enable_refinement_step",
                    help="Recommended for audio. Cleans transcript and adds speaker labels. This step is mandatory for 'Refine Transcript Only' mode.",
                    value=refinement_value,
                    disabled=is_refine_only_mode)

    st.markdown("---")
    if st.session_state.get('selected_meeting_type') == "Expert Meeting":
         st.radio("Expert Meeting Note Style:", options=EXPERT_MEETING_OPTIONS, key="expert_meeting_prompt_option", horizontal=True, on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))
    elif st.session_state.get('selected_meeting_type') == "Earnings Call":
         st.radio("Mode:", options=EARNINGS_CALL_MODES, key="earnings_call_mode", horizontal=True, on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))

st.divider()

with st.container(border=True):
    is_enrich_mode = (st.session_state.get('selected_meeting_type') == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
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
        st.checkbox("Add General Context", key="add_context_enabled", disabled=is_refine_only_mode)
        if st.session_state.get('add_context_enabled'):
            st.text_area("Context Details:", height=75, key="context_input", placeholder="e.g., Company Name, Date, Key Competitors...")
        st.write("")
        if st.session_state.get('selected_meeting_type') != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled", disabled=(is_enrich_mode or is_refine_only_mode), on_change=handle_edit_toggle)

show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or (st.session_state.get('view_edit_prompt_enabled') and not is_enrich_mode and not is_refine_only_mode)
if show_prompt_area:
    with st.container(border=True):
        st.subheader("Final Prompt Editor")
        base_template_text = get_prompt_display_text(for_display_only=True)
        if st.session_state.view_edit_prompt_enabled and not st.session_state.current_prompt_text.strip():
             st.session_state.current_prompt_text = base_template_text
        st.text_area("Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

st.write("")
is_valid, error_msg = validate_inputs()
generate_tooltip = error_msg if not is_valid else "Refine or generate notes."

if is_refine_only_mode:
    generate_button_label = "🚀 Refine Transcript"
elif is_enrich_mode:
    generate_button_label = "🚀 Enrich Notes"
else:
    generate_button_label = "🚀 Generate Notes"

generate_button = st.button(generate_button_label, type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename or not is_valid, help=generate_tooltip)

output_container = st.container(border=True)
with output_container:
    if st.session_state.get('processing'):
        st.info(f"⏳ Processing... Please wait.", icon="⏳")
    elif st.session_state.get('error_message'):
        st.error(st.session_state.error_message, icon="🚨")
    elif st.session_state.get('generated_notes'):
        output_header = "✅ Refined Transcript" if is_refine_only_mode else "✅ Generated Notes"
        st.subheader(output_header)
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        st.checkbox("Edit Output", key="edit_notes_enabled")
        if st.session_state.get('edit_notes_enabled'):
            st.text_area("Editable Output:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            st.markdown(f"```\n{notes_content_to_use}\n```" if is_refine_only_mode else notes_content_to_use)
        st.markdown("---")
        with st.expander("View Source Transcripts & Download Options"):
            if st.session_state.get('raw_transcript'):
                st.text_area("Raw Source (Step 1 Output)", st.session_state.raw_transcript, height=200, disabled=True)
            if st.session_state.get('refined_transcript') and not is_refine_only_mode:
                st.text_area("Refined Transcript (Step 2 Output)", st.session_state.refined_transcript, height=300, disabled=True)
            st.write("")
            dl_cols = st.columns(3)
            fname_base = st.session_state.get('suggested_filename', "synthnotes_output")
            dl_button_label = "⬇️ Refined Tx (.txt)" if is_refine_only_mode else "⬇️ Output (.txt)"
            with dl_cols[0]:
                st.download_button(dl_button_label, notes_content_to_use, f"{fname_base}.txt", "text/plain", use_container_width=True)
            with dl_cols[1]:
                 st.download_button("⬇️ Output (.md)", notes_content_to_use, f"{fname_base}.md", "text/markdown", use_container_width=True, disabled=is_refine_only_mode)
            with dl_cols[2]:
                if st.session_state.get('refined_transcript') and not is_refine_only_mode:
                    st.download_button("⬇️ Refined Tx (.txt)", st.session_state.refined_transcript, f"{fname_base}_refined_transcript.txt", "text/plain", use_container_width=True)
                else:
                    st.button("Refined Tx N/A", disabled=True, use_container_width=True)
    else:
        st.markdown("<p class='initial-prompt'>Configure inputs and click 'Generate' to start.</p>", unsafe_allow_html=True)

with st.expander("📜 Recent History (Last 3)", expanded=False):
    history = st.session_state.get('history', [])
    if not history:
        st.caption("No generated items in history for this session.")
    else:
        for i, entry in enumerate(history):
             st.markdown(f"**#{i+1} - {entry.get('timestamp', 'N/A')}**")
             preview_text = "\n".join(entry.get('notes', '').strip().splitlines()[:5]) + "..."
             st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
             st.button(f"Restore Item #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
             if i < len(history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True
    st.session_state.error_message = None
    st.session_state.generated_notes = None
    st.rerun()

if st.session_state.get('processing'):
    processed_audio_chunk_references = []
    
    is_refine_only_flow = (st.session_state.selected_meeting_type == "Expert Meeting" and st.session_state.expert_meeting_prompt_option == REFINE_ONLY_OPTION)
    is_enrich_flow = st.session_state.selected_meeting_type == "Earnings Call" and st.session_state.earnings_call_mode == "Enrich Existing Notes"
    
    if is_refine_only_flow:
        operation_desc = "Refining Transcript"
    elif is_enrich_flow:
        operation_desc = "Enriching Notes"
    else:
        operation_desc = "Generating Notes"

    with st.status(f"🚀 {operation_desc} in progress...", expanded=True) as status:
        try:
            status.update(label="⚙️ Validating inputs...")
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process: raise ValueError(f"Input validation failed: {error_msg_process}")
            
            meeting_type = st.session_state.selected_meeting_type
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            actual_input_type, source_transcript_data, source_audio_file_obj = get_current_input_data()
            speaker_1_name = st.session_state.get('speaker_1_name', '').strip()
            speaker_2_name = st.session_state.get('speaker_2_name', '').strip()

            status.update(label="🧠 Initializing AI models...")
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings_relaxed)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings_relaxed)

            transcript_to_process = None
            
            # Step 1: Get Text from Source
            if actual_input_type == "Upload Audio":
                status.update(label="🎤 Step 1: Transcribing Audio...")
                if source_audio_file_obj is None: raise ValueError("Audio file not found.")
                audio_bytes = source_audio_file_obj.getvalue()
                audio_format = os.path.splitext(source_audio_file_obj.name)[1].lower().replace('.', '')
                if audio_format == 'm4a': audio_format = 'mp4'
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                chunks = make_chunks(audio, 5 * 60 * 1000)
                all_transcripts = []
                for i, chunk in enumerate(chunks):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                        chunk.export(temp_chunk_file.name, format="wav")
                        chunk_file_ref = genai.upload_file(path=temp_chunk_file.name)
                        processed_audio_chunk_references.append(chunk_file_ref)
                        while chunk_file_ref.state.name == "PROCESSING": time.sleep(5)
                        chunk_file_ref = genai.get_file(chunk_file_ref.name)
                        if chunk_file_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk processing failed.")
                        t_response = transcription_model.generate_content(["Transcribe this audio.", chunk_file_ref], generation_config=transcription_gen_config)
                        all_transcripts.append(t_response.text.strip() if t_response and hasattr(t_response, 'text') else "")
                st.session_state.raw_transcript = "\n\n".join(all_transcripts).strip()
                if not st.session_state.raw_transcript:
                    raise ValueError("Audio transcription failed or produced no text.")
                transcript_to_process = st.session_state.raw_transcript
                status.update(label="✅ Step 1: Transcription Complete!")
            else:
                status.update(label="📄 Step 1: Loading Text...")
                st.session_state.raw_transcript = source_transcript_data
                transcript_to_process = source_transcript_data
                status.update(label="✅ Step 1: Text Loaded!")

            if not transcript_to_process: raise ValueError("No source transcript available for processing.")
            
            # --- PATH A: REFINE ONLY ---
            if is_refine_only_flow:
                status.update(label=f"🧹 Step 2: Refining Transcript...")
                speaker_instructions = "Assign consistent generic labels (e.g., Speaker 1, Speaker 2)."
                if speaker_1_name and speaker_2_name:
                    speaker_instructions = f"The speakers are '{speaker_1_name}' and '{speaker_2_name}'. Use these names as labels."
                elif speaker_1_name or speaker_2_name:
                    speaker_instructions = f"One speaker is '{speaker_1_name or speaker_2_name}'. Use this name as a label."
                
                refinement_prompt = f"""You are an AI assistant that cleans up and formats transcripts. Refine the following source transcript.

                **Instructions:**
                1.  **Identify and Label Speakers:** {speaker_instructions} Ensure each speaker's turn starts on a new line with their label (e.g., `Speaker 1:`).
                2.  **Correct & Clarify:** Fix obvious spelling mistakes, grammatical errors, or transcription artifacts (if any).
                3.  **Improve Readability:** Ensure clean separation between speaker turns and use standard paragraph formatting. Do not summarize or change the meaning.
                4.  **Output ONLY the refined transcript text.**

                **Source Transcript:**
                ```
                {transcript_to_process}
                ```
                **Refined Transcript:**
                """
                r_response = refinement_model.generate_content(refinement_prompt, generation_config=refinement_gen_config)
                if not (r_response and hasattr(r_response, 'text') and r_response.text.strip()):
                    raise ValueError(f"Refinement process failed or produced no text. Model response: {r_response.text if r_response else 'No response'}")
                
                st.session_state.refined_transcript = r_response.text.strip()
                st.session_state.generated_notes = st.session_state.refined_transcript
                st.session_state.edited_notes_text = st.session_state.generated_notes
                add_to_history(st.session_state.generated_notes)
                st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type, is_refine_only=True)
                status.update(label="✅ Refinement Complete!", state="complete")

            # --- PATH B: FULL NOTE GENERATION ---
            else:
                final_source_transcript = transcript_to_process
                st.session_state.refined_transcript = None

                if st.session_state.get('enable_refinement_step'):
                    status.update(label=f"🧹 Step 2: Refining Transcript...")
                    speaker_instructions = "Assign consistent generic labels (e.g., Speaker 1, Speaker 2)."
                    if speaker_1_name and speaker_2_name:
                        speaker_instructions = f"The speakers are '{speaker_1_name}' and '{speaker_2_name}'. Use these names as labels."
                    elif speaker_1_name or speaker_2_name:
                        speaker_instructions = f"One speaker is '{speaker_1_name or speaker_2_name}'. Use this name as a label."

                    refinement_prompt = f"""You are an AI assistant that cleans up and formats transcripts. Refine the following source transcript.

                    **Instructions:**
                    1.  **Identify and Label Speakers:** {speaker_instructions} Ensure each speaker's turn starts on a new line with their label (e.g., `Speaker 1:`).
                    2.  **Correct & Clarify:** Fix obvious spelling mistakes, grammatical errors, or transcription artifacts (if any).
                    3.  **Improve Readability:** Ensure clean separation between speaker turns and use standard paragraph formatting. Do not summarize or change the meaning.
                    4.  **Output ONLY the refined transcript text.**

                    **Source Transcript:**
                    ```
                    {transcript_to_process}
                    ```
                    **Refined Transcript:**
                    """
                    r_response = refinement_model.generate_content(refinement_prompt, generation_config=refinement_gen_config)
                    if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                        st.session_state.refined_transcript = r_response.text.strip()
                        final_source_transcript = st.session_state.refined_transcript
                        status.update(label="✅ Step 2: Refinement Complete!")
                    else:
                        status.update(label="⚠️ Step 2: Refinement failed. Proceeding with original transcript.")
                else:
                    status.update(label="⏭️ Step 2: Refinement skipped by user.")

                status.update(label=f"📝 Step 3: Generating Notes...")
                generated_content = ""
                
                # --- UNIFIED PROMPT LOGIC ---
                if meeting_type == "Expert Meeting":
                    # 1. Determine which base instructions to use: user's edit or the default.
                    if st.session_state.get('view_edit_prompt_enabled', False) and st.session_state.get('current_prompt_text', "").strip():
                        base_instructions = st.session_state.current_prompt_text
                    else:
                        base_instructions = EXPERT_MEETING_CHUNK_BASE

                    word_count = len(final_source_transcript.split())
                    CHUNK_THRESHOLD = 3800
                    use_chunking = (word_count > CHUNK_THRESHOLD)

                    if use_chunking:
                        status.update(label=f"📝 Long transcript detected ({word_count} words). Activating chunking.")
                        chunks = chunk_text_by_words(final_source_transcript, chunk_size=4000, overlap=200)
                        all_notes, context_package = [], ""
                        for i, chunk in enumerate(chunks):
                            status.update(label=f"🧠 Processing Chunk {i+1}/{len(chunks)}...")
                            prompt = PROMPT_INITIAL.format(base_instructions=base_instructions, chunk_text=chunk) if i == 0 else PROMPT_CONTINUATION.format(base_instructions=base_instructions, context_package=context_package, chunk_text=chunk)
                            chunk_response = notes_model.generate_content(prompt, generation_config=main_gen_config)
                            notes_for_chunk = chunk_response.text.strip() if chunk_response and hasattr(chunk_response, 'text') else ""
                            if not notes_for_chunk:
                                st.warning(f"⚠️ Chunk {i+1} returned empty. Skipping.")
                                continue
                            all_notes.append(notes_for_chunk)
                            context_package = _create_context_from_notes(notes_for_chunk)
                        generated_content = "\n\n".join(all_notes).strip()
                    else: # Single pass for Expert Meeting
                        expert_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
                        prompt_key = "Option 1: Existing (Detailed & Strict)" if expert_option == "Option 1: Existing (Detailed & Strict)" else "Option 2: Less Verbose (Default)"
                        single_pass_template_wrapper = PROMPTS["Expert Meeting"][prompt_key]
                        
                        prompt_template = single_pass_template_wrapper.format(base_instructions=base_instructions)
                        
                        context = f"**CONTEXT:**\n{st.session_state.get('context_input', '')}" if st.session_state.get('add_context_enabled') and st.session_state.get('context_input') else ""
                        final_prompt = format_prompt_safe(prompt_template, transcript=final_source_transcript, context_section=context)
                        response = notes_model.generate_content(final_prompt, generation_config=main_gen_config)
                        if not (response and hasattr(response, 'text') and response.text.strip()):
                            raise Exception(f"Note generation failed or returned empty. Model response: {response.text if response else 'No response'}")
                        generated_content = response.text.strip()
                
                else: # Logic for Earnings Call, Custom
                    # This logic remains as it was, since chunking/editing problem was specific to Expert Meetings
                    prompt_template = get_prompt_display_text(for_display_only=False)
                    topic_instructions = ""
                    if meeting_type == "Earnings Call":
                        topics = st.session_state.get('earnings_call_topics', '')
                        if topics:
                            formatted_topics = [f"- **{line.strip().strip(':')}**" if line.strip() and not line.strip().startswith(('-', '*', '#')) else line.strip() for line in topics.split('\n')]
                            topic_instructions = f"Structure notes under:\n" + "\n".join(formatted_topics) + "\n\n- **Other Key Points** (MANDATORY)"
                    context = f"**CONTEXT:**\n{st.session_state.get('context_input', '')}" if st.session_state.get('add_context_enabled') and st.session_state.get('context_input') else ""
                    final_prompt = format_prompt_safe(prompt_template, transcript=final_source_transcript, topic_instructions=topic_instructions, existing_notes=st.session_state.get('existing_notes_input', ''), context_section=context, user_custom_prompt=st.session_state.get('current_prompt_text', ''))
                    response = notes_model.generate_content(final_prompt, generation_config=main_gen_config)
                    if not (response and hasattr(response, 'text') and response.text.strip()):
                        raise Exception(f"Note generation failed or returned empty. Model response: {response.text if response else 'No response'}")
                    generated_content = response.text.strip()
                
                # --- Handle Executive Summary ---
                if meeting_type == "Expert Meeting" and st.session_state.expert_meeting_prompt_option == "Option 3: Option 2 + Executive Summary":
                    status.update(label="📄 Generating Executive Summary...")
                    summary_prompt = format_prompt_safe(PROMPTS["Expert Meeting"][EXPERT_MEETING_SUMMARY_PROMPT_KEY], generated_notes=generated_content)
                    summary_response = notes_model.generate_content(summary_prompt, generation_config=summary_gen_config)
                    if summary_response and hasattr(summary_response, 'text') and summary_response.text.strip():
                        st.session_state.generated_notes = f"{generated_content}\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{summary_response.text.strip()}"
                    else:
                        st.session_state.generated_notes = generated_content
                else:
                    st.session_state.generated_notes = generated_content
                
                st.session_state.edited_notes_text = st.session_state.generated_notes
                add_to_history(st.session_state.generated_notes)
                st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                status.update(label="✅ Success!", state="complete")

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

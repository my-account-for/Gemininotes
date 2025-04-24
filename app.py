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
from pydub import AudioSegment # <-- ADDED for audio chunking
from pydub.utils import make_chunks # <-- ADDED for audio chunking
import copy # To avoid modifying original prompts

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
    # Generation configs will be used in generate_content calls
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    summary_gen_config = {"temperature": 0.6, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"} # Slightly different for summary
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    refinement_gen_config = {"temperature": 0.3, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()


# --- Prompts Definitions ---

# Structure to hold prompts
PROMPTS = {
    "Expert Meeting": {
        # OPTION 1: The original, very strict format
        "Option 1: Existing (Detailed & Strict)": """You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.
Generate detailed, factual notes from the provided meeting transcript.
Follow this specific structure EXACTLY:

**Structure:**
- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.
- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.
  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).
  - **Answers:** Use bullet points directly below the question. **Each bullet MUST be a complete sentence representing one single, distinct factual point.** Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or action items.

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- Be clear and concise, adhering strictly to one fact per bullet point.
- Include ONLY information present in the transcript.
- If a section (like Opening Overview) isn't present, OMIT it.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Strict):**
""",

        # OPTION 2: Less verbose, allows combining closely related points per bullet
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
		- DO NOT add interpretations, summaries, conclusions, or action items.

**Additional Instructions:**
- Accuracy is paramount. Capture ALL facts precisely.
- **Write clearly and concisely, avoiding unnecessary words. Favor informative sentences over overly simplistic ones.**
- Include ONLY information present in the transcript.
- If a section (like Opening Overview) isn't present, OMIT it.
---
**MEETING TRANSCRIPT:**
{transcript}
---
{context_section}
---
**GENERATED NOTES (Q&A Format - Concise):**
""",

        # OPTION 3: Uses Option 2 for notes, THEN adds a summary. This prompt is for the *summary step*.
        # ***** UPDATED SUMMARY PROMPT BELOW *****
        "Summary Prompt (for Option 3)": """Based ONLY on the detailed 'GENERATED NOTES (Q&A Format - Concise)' provided below, create a concise executive summary highlighting the most significant insights, findings, or critical points discussed.

**Format:**
1.  Identify the main themes or key topics discussed in the notes (e.g., **GenAI Impact**, **Vendor Landscape**, **Genpact Specifics**). Create a clear, concise heading for each theme using bold text.
2.  Under each heading, use primary bullet points (`- `) to list the most significant insights, findings, or critical points related to that theme.
3.  **Crucially: Each bullet point should represent a single, distinct key takeaway or significant piece of information.** DO NOT use indented sub-bullets or nested lists. If a point has multiple important facets, break them down into separate primary bullet points under the same theme heading.
4.  Focus on synthesizing the key takeaways from the detailed Q&A points. These bullets should represent crucial insights, not just a repetition of individual facts. Aim for clear, impactful statements.

**Instructions:**
- Aim for a total summary length of approximately 500-1000 words.
- Maintain an objective and professional tone, reflecting the expert's views accurately.
- Ensure the summary accurately reflects the content and emphasis of the detailed notes it is based on.
- Do not introduce any information, conclusions, or opinions not explicitly supported by the GENERATED NOTES provided below.

---
**GENERATED NOTES (Input for Summary):**
{generated_notes}
---

**EXECUTIVE SUMMARY:**
"""
        # ***** END OF UPDATED SUMMARY PROMPT *****
    },
    "Earnings Call": """You are an expert AI assistant creating DETAILED notes from an earnings call transcript for an investment firm.
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
- DO NOT summarize or interpret unless part of the structure.

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
    "Custom": "{user_custom_prompt}\n\n--- TRANSCRIPT START ---\n{transcript}\n--- TRANSCRIPT END ---\n{context_section}" # Simplified placeholder usage
}

# Define the user-facing names for the Expert Meeting options
EXPERT_MEETING_OPTIONS = [
    "Option 1: Existing (Detailed & Strict)",
    "Option 2: Less Verbose (Default)",
    "Option 3: Option 2 + Executive Summary" # This represents the combined process
]
DEFAULT_EXPERT_MEETING_OPTION = EXPERT_MEETING_OPTIONS[1] # Option 2 is default

# Internal key for the summary prompt used by Option 3 logic
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
    'expert_meeting_prompt_option': DEFAULT_EXPERT_MEETING_OPTION, # <-- Use defined default
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'selected_sector': DEFAULT_SECTOR,
    'earnings_call_topics': '',
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None,
    'processed_audio_chunk_references': []
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def update_topic_template():
    selected_sector = st.session_state.selected_sector
    if selected_sector in SECTOR_TOPICS: st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
    else: st.session_state.earnings_call_topics = ""

def format_prompt_safe(prompt_template, **kwargs):
    """Safely formats a string template, replacing missing keys with placeholders."""
    # Use a copy to avoid modifying the original template
    formatted_prompt = copy.deepcopy(prompt_template)
    try:
        # Find all placeholders like {key}
        placeholders = re.findall(r"\{([^}]+)\}", formatted_prompt)
        for key in placeholders:
            # Use the provided value if it exists, otherwise use a placeholder string
            value = kwargs.get(key, f"[MISSING: {key}]")
            # Ensure value is a string before replacing
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

def get_prompt_display_text():
    """Generates the appropriate prompt text for display/editing."""
    meeting_type = st.session_state.selected_meeting_type
    display_text = ""

    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
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
                # Determine which prompt template to display based on selection
                if expert_option == "Option 1: Existing (Detailed & Strict)":
                    prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
                else: # For Option 2 and Option 3, display the Option 2 template (as it's the base for notes)
                    prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

                if prompt_template_to_display:
                     display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                     if expert_option == "Option 3: Option 2 + Executive Summary":
                         # Append note about the separate summary step prompt (which isn't shown here)
                         summary_prompt_preview = PROMPTS["Expert Meeting"][EXPERT_MEETING_SUMMARY_PROMPT_KEY].split("---")[0] # Show first part
                         display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step generated *after* these notes, using a separate prompt starting like:\n'''\n{summary_prompt_preview.strip()}\n'''"
                else:
                    display_text = "# Error: Could not find prompt template for display."

            elif meeting_type == "Earnings Call":
                 prompt_template_to_display = PROMPTS["Earnings Call"]
                 user_topics_text_for_display = str(st.session_state.get('earnings_call_topics', "") or "")
                 topic_instructions_preview = "" # Generate preview version of topics
                 if user_topics_text_for_display and user_topics_text_for_display.strip():
                     # Simplified topic formatting for preview
                     topic_instructions_preview = f"Structure notes under user-specified headings (e.g., {user_topics_text_for_display.splitlines()[0]}...)\n- **Other Key Points**"
                 else:
                     topic_instructions_preview = "Identify logical main themes...\n- **Other Key Points**"
                 format_kwargs['topic_instructions'] = topic_instructions_preview
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)

            else: # Should not happen if meeting_type != "Custom"
                 st.error(f"Internal Error: Invalid meeting type '{meeting_type}' for prompt preview.")
                 return "Error generating prompt preview."

            if input_type == "Upload Audio":
                 audio_note = ("# NOTE FOR AUDIO: 3-step process (Chunked Transcribe -> Refine -> Notes).\n"
                               "# This prompt is used in Step 3 with the *refined* transcript.\n"
                               "####################################\n\n")
                 display_text = audio_note + display_text

        except Exception as e:
             st.error(f"Error generating prompt preview: {e}")
             display_text = f"# Error generating preview: Review inputs/prompt structure. Details logged if possible."

    elif meeting_type == "Custom":
         audio_note = ("\n# NOTE FOR AUDIO: If using audio, the system will first chunk and transcribe,\n"
                       "# then *refine* the full transcript (speaker ID, translation, corrections).\n"
                       "# Your custom prompt below will receive this *refined transcript* as the primary text input.\n"
                       "# Design your prompt accordingly.\n")
         default_custom = "# Enter your custom prompt here..."
         current_or_default = st.session_state.current_prompt_text or default_custom
         display_text = current_or_default + (audio_note if st.session_state.input_method_radio == 'Upload Audio' else "")

    return display_text


def clear_all_state():
    # Reset selections and inputs
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
    st.session_state.expert_meeting_prompt_option = DEFAULT_EXPERT_MEETING_OPTION # <-- RESET NEW STATE
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
    # Reset outputs
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
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings) # Use flash for speed
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "")
        # Extract initial part if summary exists
        notes_preview = notes_content.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n")[0]
        filename_prompt = (f"Suggest filename: YYYYMMDD_ClientOrTopic_MeetingType. Date={today_date}. Type='{mt_cleaned}'. Max 3 words topic. Output ONLY filename.\nNOTES:{notes_preview[:1000]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if s_name: st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename suggestion empty/invalid.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
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
    if 0 <= index < len(st.session_state.history): entry = st.session_state.history[index]; st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]; \
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None; st.session_state.error_message = None; st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True)
            # --- Conditional Expert Meeting Option ---
            if st.session_state.selected_meeting_type == "Expert Meeting":
                st.radio(
                    "Expert Meeting Note Style:",
                    options=EXPERT_MEETING_OPTIONS, # Use the defined list for user choices
                    key="expert_meeting_prompt_option",
                    index=EXPERT_MEETING_OPTIONS.index(DEFAULT_EXPERT_MEETING_OPTION), # Set default index
                    help="Choose the desired output format.\n"
                         "- Option 1: Very strict, one fact per bullet.\n"
                         "- Option 2: More natural flow, combines related points.\n"
                         "- Option 3: Option 2 notes + Executive Summary (using updated prompt)." # Updated help text slightly
                )
            # --- End Conditional ---
        with col1b:
            st.subheader("AI Model Selection")
            st.selectbox("Notes Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model used for final note generation (and summary if applicable).")
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model used for Audio Transcription (Step 1 - per chunk). Faster models recommended.")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model used for Audio Refinement (Step 2 - full transcript). More capable models recommended.")

    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

    st.divider(); st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider(); col3a, col3b = st.columns(2); selected_mt = st.session_state.selected_meeting_type

    with col3a: # Topics / Context
        if selected_mt == "Earnings Call":
            st.selectbox("Select Sector (for Topic Template):", options=SECTOR_OPTIONS, key="selected_sector", on_change=update_topic_template)
            st.text_area("Earnings Call Topics (Edit below):", key="earnings_call_topics", height=150, placeholder="Enter topics manually or edit loaded template...")
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=75, key="context_input", placeholder="Company Name, Ticker...")
    with col3b: # View/Edit Prompt Checkbox
        if selected_mt != "Custom": st.checkbox("View/Edit Final Notes Prompt", key="view_edit_prompt_enabled")

# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and selected_mt != "Custom") or (selected_mt == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Final Notes Prompt Preview/Editor" if selected_mt != "Custom" else "Custom Final Notes Prompt (Required)"
        st.subheader(prompt_title)
        display_prompt_text = get_prompt_display_text()
        # If Custom, allow editing the text area directly, otherwise just display
        if selected_mt == "Custom":
            prompt_value = st.session_state.current_prompt_text if st.session_state.current_prompt_text else display_prompt_text
            st.text_area(label="Prompt Text:", value=prompt_value, key="current_prompt_text", height=350, label_visibility="collapsed", help="This prompt is used for the *final* step of generating notes.")
        else: # For non-custom, just display the generated prompt preview
             st.text_area(label="Prompt Text:", value=display_prompt_text, key="current_prompt_text_display", height=350, label_visibility="collapsed", help="This prompt is used for the *final* step of generating notes. Edit via options above.", disabled=True)
             st.session_state.current_prompt_text = display_prompt_text # Store the previewed prompt for potential use, though direct editing is disabled


# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# Output Section
output_container = st.container(border=True)
with output_container:
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None # Clear after showing
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        if st.session_state.raw_transcript:
            with st.expander("View Raw Transcript (Step 1 Output - Combined from Chunks)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
        if st.session_state.refined_transcript:
             with st.expander("View Refined Transcript (Step 2 Output)"):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)

        # Display Notes / Summary
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes

        # Check if summary is expected (Expert Meeting Option 3)
        is_option3_summary = (st.session_state.selected_meeting_type == "Expert Meeting" and
                              st.session_state.expert_meeting_prompt_option == "Option 3: Option 2 + Executive Summary" and
                              "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n" in notes_content_to_use)

        if st.session_state.edit_notes_enabled:
            st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            # If Option 3, potentially split display
            if is_option3_summary:
                 try:
                     notes_part, summary_part = notes_content_to_use.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n", 1)
                     st.markdown("### Detailed Notes (Q&A Format)")
                     st.markdown(notes_part)
                     st.markdown("---")
                     st.markdown("### Executive Summary")
                     st.markdown(summary_part)
                 except ValueError: # Should not happen if separator is present
                     st.markdown(notes_content_to_use) # Fallback to single block if split fails
            else:
                 st.markdown(notes_content_to_use) # Display as single block


        st.write("") # Spacer
        dl_cols = st.columns(3)
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"
        if st.session_state.suggested_filename: fname_base = st.session_state.suggested_filename
        else: fname_base = default_fname
        with dl_cols[0]: st.download_button(label="⬇️ Notes (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]: st.download_button(label="⬇️ Notes (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.refined_transcript:
                refined_fname_base = fname_base.replace("_notes", "_refined_transcript") if "_notes" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the speaker-diarized and corrected transcript from Step 2.")
            else: st.button("Refined Tx N/A", disabled=True, use_container_width=True, help="Refined transcript is only available after successful audio processing.")
    elif not st.session_state.processing:
        st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# History Section
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**")
                # Display logic might need adjustment if summaries are included
                display_note = entry['notes']
                summary_separator = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
                if summary_separator in display_note:
                     display_note = display_note.split(summary_separator, 1)[0] + "\n... (Summary omitted in preview)"

                st.code(display_note[:300] + ("..." if len(display_note) > 300 else ""), language=None)
                st.button(f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                if i < len(st.session_state.history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    # Reset state before starting
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
    processed_audio_chunk_references = [] # Local list for cleanup within this run

    with st.status("🚀 Initializing process...", expanded=True) as status:
        try: # Outer try-finally for overall process and cleanup
            # State & Input Retrieval
            status.update(label="⚙️ Reading inputs and settings...")
            meeting_type = st.session_state.selected_meeting_type
            expert_meeting_option = st.session_state.expert_meeting_prompt_option # Get selected option
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            user_custom_prompt_text = st.session_state.current_prompt_text # Used only for Custom type
            general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
            earnings_call_topics_text = st.session_state.earnings_call_topics.strip() if meeting_type == "Earnings Call" else ""
            actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

            # Initialize Models
            status.update(label="🧠 Initializing AI models...")
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            # Initialize Transcript Variable
            final_transcript_for_notes = transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            # Validation
            status.update(label="✔️ Validating inputs...")
            if actual_input_type == "Paste Text" and not final_transcript_for_notes: raise ValueError("Text input is empty.")
            elif actual_input_type == "Upload PDF" and not final_transcript_for_notes: raise ValueError("PDF processing failed or returned empty text.")
            elif actual_input_type == "Upload Audio" and not audio_file_obj: raise ValueError("No audio file uploaded.")
            if meeting_type == "Custom" and not user_custom_prompt_text.strip(): raise ValueError("Custom Prompt is required but empty.")
            if meeting_type == "Expert Meeting" and not expert_meeting_option: raise ValueError("Expert Meeting option not selected.")

            # ==============================================
            # --- STEP 1 & 2: Audio Processing Pipeline ---
            # ==============================================
            if actual_input_type == "Upload Audio":
                st.session_state.uploaded_audio_info = audio_file_obj

                # --- Load Audio with pydub ---
                status.update(label=f"🔊 Loading audio file '{audio_file_obj.name}'...")
                audio_bytes = audio_file_obj.getvalue()
                audio_format = os.path.splitext(audio_file_obj.name)[1].lower().replace('.', '')
                if audio_format == 'm4a': audio_format = 'mp4'
                elif audio_format == 'ogg': audio_format = 'ogg'
                elif audio_format == 'aac': audio_format = 'aac'

                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                except Exception as audio_load_err:
                    if "ffmpeg" in str(audio_load_err).lower() or "Couldn't find ffmpeg or avconv" in str(audio_load_err):
                        raise ValueError(f"❌ **Error:** Could not load audio. `ffmpeg` or `libav` might be missing from your system or PATH. Please install it. (Original error: {audio_load_err})")
                    else:
                        raise ValueError(f"❌ Could not load audio file using pydub (format: {audio_format}). Error: {audio_load_err}")

                # --- Chunking ---
                chunk_length_ms = 35 * 60 * 1000 # ~35 minutes
                chunks = make_chunks(audio, chunk_length_ms)
                num_chunks = len(chunks)
                status.update(label=f"🔪 Splitting audio into {num_chunks} chunk(s)...")

                all_transcripts = []
                # --- Process Each Chunk ---
                for i, chunk in enumerate(chunks):
                    chunk_num = i + 1
                    status.update(label=f"🔄 Processing Chunk {chunk_num}/{num_chunks}...")
                    temp_chunk_path = None # Define here for finally block access
                    chunk_file_ref = None # Define here for finally block access
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                            chunk.export(temp_chunk_file.name, format="wav")
                            temp_chunk_path = temp_chunk_file.name

                        # Upload Chunk
                        status.update(label=f"☁️ Uploading Chunk {chunk_num}/{num_chunks}...")
                        chunk_display_name = f"chunk_{chunk_num}_of_{num_chunks}_{int(time.time())}_{audio_file_obj.name}"
                        chunk_file_ref = genai.upload_file(path=temp_chunk_path, display_name=chunk_display_name)
                        # DO NOT ADD TO SESSION STATE HERE. Use local var `processed_audio_chunk_references`
                        processed_audio_chunk_references.append(chunk_file_ref)

                        # Poll for Chunk Readiness
                        status.update(label=f"⏳ Waiting for Chunk {chunk_num}/{num_chunks} to process...")
                        polling_start = time.time()
                        while chunk_file_ref.state.name == "PROCESSING":
                            if time.time() - polling_start > 600: raise TimeoutError(f"Audio processing timed out for chunk {chunk_num}.")
                            time.sleep(5)
                            chunk_file_ref = genai.get_file(chunk_file_ref.name)
                        if chunk_file_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {chunk_num} processing failed. Final state: {chunk_file_ref.state.name}")

                        # Transcribe Chunk
                        status.update(label=f"✍️ Step 1: Transcribing Chunk {chunk_num}/{num_chunks} using {st.session_state.selected_transcription_model_display_name}...")
                        t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                        t_response = transcription_model.generate_content(
                            [t_prompt, chunk_file_ref],
                            generation_config=transcription_gen_config, safety_settings=safety_settings
                        )

                        if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                            all_transcripts.append(t_response.text.strip())
                            status.update(label=f"✍️ Transcription complete for Chunk {chunk_num}/{num_chunks}!")
                        elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                            raise Exception(f"Transcription blocked for chunk {chunk_num}: {t_response.prompt_feedback.block_reason}")
                        else:
                            st.warning(f"⚠️ Transcription for chunk {chunk_num} returned empty response. Skipping.", icon="🤔")
                            all_transcripts.append("")

                    except Exception as chunk_err:
                        raise Exception(f"❌ Error processing chunk {chunk_num}: {chunk_err}") from chunk_err
                    finally:
                        if temp_chunk_path and os.path.exists(temp_chunk_path): os.remove(temp_chunk_path)
                        if chunk_file_ref:
                            try:
                                status.update(label=f"🗑️ Cleaning up cloud file for chunk {chunk_num}...")
                                genai.delete_file(chunk_file_ref.name)
                                if chunk_file_ref in processed_audio_chunk_references:
                                    processed_audio_chunk_references.remove(chunk_file_ref)
                            except Exception as del_err:
                                st.warning(f"⚠️ Failed to delete cloud file for chunk {chunk_num} ({chunk_file_ref.name}): {del_err}. Will retry cleanup later.", icon="🗑️")


                # --- Combine Transcripts ---
                status.update(label="🧩 Combining chunk transcripts...")
                st.session_state.raw_transcript = "\n\n".join(all_transcripts)
                final_transcript_for_notes = st.session_state.raw_transcript
                status.update(label="✅ Step 1: Full Transcription Complete!")

                # --- Step 2: Refinement ---
                if final_transcript_for_notes:
                    try:
                        status.update(label=f"🧹 Step 2: Refining combined transcript using {st.session_state.selected_refinement_model_display_name}...")
                        refinement_prompt = f"""Please refine the following raw audio transcript:

                        **Raw Transcript:**
                        ```
                        {st.session_state.raw_transcript}
                        ```

                        **Instructions:**
                        1.  **Identify Speakers:** Assign consistent labels (e.g., Speaker 1, Speaker 2). Place the label on a new line before the speaker's turn.
                        2.  **Translate to English:** Convert any non-English speech found within the transcript to English, ensuring it fits naturally within the conversation.
                        3.  **Correct Errors:** Fix spelling mistakes and grammatical errors. Use the overall conversation context to correct potentially misheard words or phrases.
                        4.  **Format:** Ensure clear separation between speaker turns using the speaker labels. Maintain the original conversational flow and content.
                        5.  **Output:** Provide *only* the refined, speaker-diarized, translated, and corrected transcript text. Do not add any introduction, summary, or commentary before or after the transcript.

                        **Additional Context (Optional - use for understanding terms, names, etc.):**
                        {general_context if general_context else "None provided."}

                        **Refined Transcript:**
                        """
                        r_response = refinement_model.generate_content(
                            refinement_prompt,
                            generation_config=refinement_gen_config, safety_settings=safety_settings
                        )
                        if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                            st.session_state.refined_transcript = r_response.text.strip()
                            final_transcript_for_notes = st.session_state.refined_transcript
                            status.update(label="🧹 Step 2: Refinement complete!")
                        elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                            st.warning(f"⚠️ Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Refinement blocked. Proceeding with raw transcript.")
                        else:
                            st.warning("🤔 Refinement step returned empty response. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Refinement failed. Proceeding with raw transcript.")
                    except Exception as refine_err:
                         st.warning(f"❌ Error during Step 2 (Refinement): {refine_err}. Using raw transcript for notes.", icon="⚠️")
                         status.update(label="⚠️ Refinement error. Proceeding with raw transcript.")
                else:
                    status.update(label="⚠️ Skipping Refinement (Step 2) as raw transcript is empty.")


            # =============================
            # --- STEP 3: Generate Notes ---
            # =============================
            if not final_transcript_for_notes:
                 raise ValueError("No transcript available (from text, PDF, or audio processing) to generate notes.")

            status.update(label="📝 Preparing final prompt for note generation...")
            final_prompt_for_notes_api = None
            api_payload_parts = [] # For sending content to API
            notes_prompt_key_to_use = None # Determine which key for the first call

            # --- Format Prompt Based on Meeting Type ---
            format_kwargs = {
                'transcript': final_transcript_for_notes,
                'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{general_context}\n---" if general_context else ""
            }

            if meeting_type == "Expert Meeting":
                # Determine the prompt key for the *initial* note generation
                if expert_meeting_option == "Option 1: Existing (Detailed & Strict)":
                    notes_prompt_key_to_use = "Option 1: Existing (Detailed & Strict)"
                elif expert_meeting_option == "Option 2: Less Verbose (Default)":
                    notes_prompt_key_to_use = "Option 2: Less Verbose (Default)"
                elif expert_meeting_option == "Option 3: Option 2 + Executive Summary":
                    # Option 3 uses Option 2's prompt for the notes part
                    notes_prompt_key_to_use = "Option 2: Less Verbose (Default)"
                else:
                    raise ValueError(f"Invalid Expert Meeting option selected: {expert_meeting_option}")

                prompt_template = PROMPTS["Expert Meeting"].get(notes_prompt_key_to_use)
                if not prompt_template:
                     raise ValueError(f"Could not find prompt template for key: {notes_prompt_key_to_use}")
                final_prompt_for_notes_api = format_prompt_safe(prompt_template, **format_kwargs)
                api_payload_parts = [final_prompt_for_notes_api]

            elif meeting_type == "Earnings Call":
                 prompt_template = PROMPTS["Earnings Call"]
                 user_topics_text = earnings_call_topics_text
                 topic_instructions = ""
                 if user_topics_text and user_topics_text.strip():
                     formatted_topics = []
                     for line in user_topics_text.strip().split('\n'):
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
                 final_prompt_for_notes_api = format_prompt_safe(prompt_template, **format_kwargs)
                 api_payload_parts = [final_prompt_for_notes_api]

            elif meeting_type == "Custom":
                 prompt_template = PROMPTS["Custom"]
                 format_kwargs["user_custom_prompt"] = user_custom_prompt_text
                 # Note: context_section is already part of the Custom template
                 final_prompt_for_notes_api = format_prompt_safe(prompt_template, **format_kwargs)
                 api_payload_parts = [final_prompt_for_notes_api] # Send the formatted prompt directly
            else:
                raise ValueError(f"Invalid meeting type '{meeting_type}' for prompt generation.")

            if not final_prompt_for_notes_api:
                raise ValueError("Failed to determine the final prompt for note generation.")

            # --- Generate Notes API Call (Step 3 / First Call for Option 3) ---
            try:
                status.update(label=f"✨ Step 3: Generating notes using {st.session_state.selected_notes_model_display_name}...")
                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=main_gen_config,
                    safety_settings=safety_settings
                )

                # --- Handle Response (and Option 3 Summary) ---
                generated_notes_content = None
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    generated_notes_content = response.text.strip()
                    status.update(label="✅ Initial notes generated!")

                    # --- Option 3: Generate Summary ---
                    if meeting_type == "Expert Meeting" and expert_meeting_option == "Option 3: Option 2 + Executive Summary":
                        status.update(label=f"✨ Step 3b: Generating Executive Summary using {st.session_state.selected_notes_model_display_name}...")
                        # Use the dedicated summary prompt key (which has been updated)
                        summary_prompt_template = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY)
                        if not summary_prompt_template:
                             raise ValueError(f"Could not find summary prompt template with key: {EXPERT_MEETING_SUMMARY_PROMPT_KEY}")

                        # Format the summary prompt with the notes generated above
                        summary_kwargs = {'generated_notes': generated_notes_content}
                        summary_prompt = format_prompt_safe(summary_prompt_template, **summary_kwargs)

                        try:
                            summary_response = notes_model.generate_content(
                                summary_prompt,
                                generation_config=summary_gen_config, # Use summary config
                                safety_settings=safety_settings
                            )

                            if summary_response and hasattr(summary_response, 'text') and summary_response.text.strip():
                                summary_text = summary_response.text.strip()
                                # Combine notes and summary
                                st.session_state.generated_notes = f"{generated_notes_content}\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{summary_text}"
                                status.update(label="✅ Notes and Summary generated successfully!", state="complete")
                            elif hasattr(summary_response, 'prompt_feedback') and summary_response.prompt_feedback.block_reason:
                                st.warning(f"⚠️ Summary generation blocked: {summary_response.prompt_feedback.block_reason}. Only detailed notes provided.", icon="⚠️")
                                if 'generated_notes' not in st.session_state or not st.session_state.generated_notes: # Check if notes already set
                                     st.session_state.generated_notes = generated_notes_content # Fallback to just notes
                                status.update(label="⚠️ Summary Blocked. Only detailed notes generated.", state="warning")
                            else:
                                st.warning("🤔 AI returned empty response during summary generation. Only detailed notes provided.", icon="⚠️")
                                if 'generated_notes' not in st.session_state or not st.session_state.generated_notes:
                                     st.session_state.generated_notes = generated_notes_content # Fallback to just notes
                                status.update(label="⚠️ Summary Failed. Only detailed notes generated.", state="warning")

                        except Exception as summary_err:
                            st.warning(f"❌ Error during summary generation: {summary_err}. Only detailed notes provided.", icon="⚠️")
                            if 'generated_notes' not in st.session_state or not st.session_state.generated_notes:
                                 st.session_state.generated_notes = generated_notes_content # Fallback
                            status.update(label="⚠️ Summary Error. Only detailed notes generated.", state="warning")
                    else:
                        # Not Option 3, or summary failed - just use the generated notes
                        if 'generated_notes' not in st.session_state or not st.session_state.generated_notes: # Avoid overwriting if already set by failed summary
                             st.session_state.generated_notes = generated_notes_content
                             status.update(label="✅ Notes generated successfully!", state="complete")

                    # --- Post-generation steps (common) ---
                    if st.session_state.generated_notes: # Check if we have notes (either solo or combined)
                        st.session_state.edited_notes_text = st.session_state.generated_notes
                        add_to_history(st.session_state.generated_notes)
                        st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)

                # Handle errors from the *initial* notes generation call
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    st.session_state.error_message = f"⚠️ Note generation blocked: {response.prompt_feedback.block_reason}."
                    status.update(label=f"❌ Blocked: {response.prompt_feedback.block_reason}", state="error")
                elif response:
                    st.session_state.error_message = "🤔 AI returned empty response during note generation."
                    status.update(label="❌ Error: AI returned empty response.", state="error")
                else:
                    st.session_state.error_message = "😥 Note generation failed (No response from API)."
                    status.update(label="❌ Error: Note generation failed (No response).", state="error")

            except Exception as api_call_err:
                st.session_state.error_message = f"❌ Error during Step 3 (API Call for Notes): {api_call_err}"
                status.update(label=f"❌ Error: {api_call_err}", state="error")

        except Exception as e:
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        finally:
            st.session_state.processing = False # Mark processing as finished
            # --- Cloud Audio Chunk Cleanup (Catch-all) ---
            if processed_audio_chunk_references:
                st.toast(f"☁️ Performing final cleanup of {len(processed_audio_chunk_references)} remaining cloud chunk file(s)...", icon="🗑️")
                for file_ref in processed_audio_chunk_references:
                    try: genai.delete_file(file_ref.name)
                    except Exception as final_cleanup_error: st.warning(f"Final cloud audio chunk cleanup failed for {file_ref.name}: {final_cleanup_error}", icon="⚠️")
                # Clear the session state list as well, even if deletion failed for some
                st.session_state.processed_audio_chunk_references = []

            st.rerun() # Rerun to display final results or errors

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

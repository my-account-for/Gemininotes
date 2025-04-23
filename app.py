# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
from dotenv import load_dotenv
import PyPDF2
# import docx # <-- REMOVED import for DOCX generation

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS Injection ---
# (Keep CSS as is)
st.markdown("""
<style>
    /* ... Keep all previous CSS ... */
    /* Footer */
    footer { text-align: center; color: #9CA3AF; font-size: 0.8rem; padding-top: 2rem; padding-bottom: 1rem; }
    footer a { color: #6B7280; text-decoration: none; }
    footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# --- Define Available Models ---
AVAILABLE_MODELS = {
    # Stable Models
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume)": "gemini-1.5-flash-8b",
    # Newer/Preview Models
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
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
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
    'context_input': ''
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
    # (Prompt definition remains the same)
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
    # (Prompt definition remains the same)
    prompt_parts = [
        "You are a financial analyst tasked with summarizing an earnings call transcript. Your output MUST be structured notes.",
        "Analyze the entire transcript and extract key information, numerical data, guidance, strategic comments, and management sentiment.",
        "Present the information using the EXACT headings and subheadings provided below. You MUST categorize all relevant comments under the correct heading.",
        "\n**Mandatory Structure:**",
        "- **Call Participants:** ...",
        "- **Opening Remarks/CEO Statement:** ...",
        "- **Financial Highlights:** ...",
        "- **Segment Performance:** ...",
        "- **Key Business Updates/Strategy:** ...",
        "\n**Industry-Specific Categorization (Apply ONLY ONE section based on company type identified from the transcript):**",
        "\n  **>>> If IT Services Topics Discussed <<<**",
        "    *(Scan the transcript for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Future Investments / Capital Allocation:** ...",
        "    - **Talent Supply Chain:** ...",
        "    - **Org Structure Changes:** ...",
        "    - **Short-term Outlook & Demand:**",
        "      - **Guidance:** ...",
        "      - **Order Booking / Pipeline:** ...",
        "      - **Macro Impact:** ...",
        "    - **Other Key IT Comments:** ...",
        "\n  **>>> If QSR (Quick Service Restaurant) Topics Discussed <<<**",
         "    *(Scan the transcript for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Customer Proposition / Menu Strategy:** ...",
        "    - **Business Update (Operations):** ...",
        "    - **Unit Economics / Store Performance:** ...",
        "    - **Store Network:** ...",
        "    - **Other Key QSR Comments:** ...",
        "  *(If neither IT nor QSR specific topics are dominant, OMIT this entire Industry-Specific section)*",
        "\n- **Q&A Session Summary:** ...",
        "- **Guidance Summary (Reiterate/Confirm):** ...",
        "- **Closing Remarks:** ...",
        "\n**CRITICAL Instructions:** ...",
        "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY:**\n")
    return "\n".join(filter(None, prompt_parts))

# --- REMOVED create_docx function ---

def get_current_input_data():
    """Helper to get transcript/audio file based on input method and state."""
    input_type = st.session_state.input_method_radio
    transcript = None; audio_file = None
    if input_type == "Paste Text": transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file: transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
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
             st.session_state.current_prompt_text = (
                 "# NOTE FOR AUDIO INPUT: The '1. Transcribe first...' wrapper will be added *unless* you edit this prompt.\n"
                 "# If edited, ensure your prompt handles the audio file.\n"
                 "####################################\n\n" + base_prompt
             )
        else: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state.current_prompt_text:
              st.session_state.current_prompt_text = "# Enter your custom prompt here...\n# For audio, include transcription instructions."
    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        st.session_state.current_prompt_text = ""


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Input Section ---
with st.container(border=True):
    # (UI structure remains the same)
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Meeting Details")
        st.radio(label="Select Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type",
                 index=MEETING_TYPES.index(st.session_state.selected_meeting_type), horizontal=True,
                 help="Choose meeting type. 'Custom' requires your own prompt.", on_change=update_prompt_display_text)
    with col1b:
        st.subheader("AI Model")
        st.selectbox(label="Choose model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name",
                     index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_display_name),
                     help="Select Gemini model. Preview/Experimental models may vary.")

    st.divider()
    st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio",
             horizontal=True, label_visibility="collapsed", on_change=update_prompt_display_text)
    input_type = st.session_state.input_method_radio
    if input_type == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input")
    elif input_type == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider()
    col3a, col3b = st.columns(2)
    with col3a: # Context
        st.checkbox("Add Context (Optional)", key="add_context_enabled", help="Provide background info.", on_change=update_prompt_display_text)
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=100, key="context_input", on_change=update_prompt_display_text)
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
        caption_text = "Edit the prompt below. For audio, 'Transcribe first...' is added unless edited." if st.session_state.selected_meeting_type != "Custom" else "Enter the full prompt. For audio, include transcription instructions."
        st.caption(caption_text)
        st.text_area("Prompt Text:", key="current_prompt_text", height=350, label_visibility="collapsed")

# --- Generate Button ---
st.write("")
generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True) # Marker for CSS
    if st.session_state.processing: st.info("⏳ Processing...", icon="🧠")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        # --- Display the notes first ---
        st.markdown(st.session_state.generated_notes)
        # --- Reverted to TXT Download ---
        st.download_button(
             label="⬇️ Download Notes (.txt)",
             data=st.session_state.generated_notes, # Download the raw text
             file_name=f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes.txt", # .txt extension
             mime="text/plain", # Plain text MIME type
             key='download-txt' # Key for the button
         )
        # --- End of TXT download change ---
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True
    st.session_state.generated_notes = None
    st.session_state.error_message = None
    st.rerun()

if st.session_state.processing:
    # Retrieve state
    meeting_type = st.session_state.selected_meeting_type
    selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
    view_edit_enabled = st.session_state.view_edit_prompt_enabled
    user_prompt_text = st.session_state.current_prompt_text
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
            # Clean the note added for audio display if present
            final_prompt_for_api = final_prompt_for_api.split("####################################\n\n")[-1]
        else: # Generate default prompt
            prompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
            if actual_input_type != "Upload Audio":
                 if transcript_data: final_prompt_for_api = prompt_function(transcript_data, final_context)
                 else: st.session_state.error_message = "Error: Transcript data missing for non-audio input."
            else: # Audio: Add wrapper automatically ONLY if not custom/edited
                base_prompt = prompt_function(transcript=None, context=final_context)
                final_prompt_for_api = ("1. First, accurately transcribe the provided audio file.\n"
                                        "2. Then, using the transcription, create notes based on:\n---\n"
                                        f"{base_prompt}")

    # --- API Call ---
    if not st.session_state.error_message and final_prompt_for_api and (transcript_data or audio_file_obj):
        try:
            st.toast(f"🧠 Generating with {st.session_state.selected_model_display_name}...", icon="✨")
            model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=generation_config)
            response = None; api_payload = None; processed_audio_file_ref = None

            if actual_input_type == "Upload Audio":
                 if not audio_file_obj: raise ValueError("Audio file object missing.")
                 st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️")
                 audio_bytes = audio_file_obj.getvalue()
                 processed_audio_file_ref = genai.upload_file(content=audio_bytes, display_name=f"audio_{int(time.time())}", mime_type=audio_file_obj.type)
                 st.session_state.uploaded_audio_info = processed_audio_file_ref
                 # Polling...
                 polling_start_time = time.time()
                 while processed_audio_file_ref.state.name == "PROCESSING":
                     if time.time() - polling_start_time > 300: raise TimeoutError("Audio processing timeout.")
                     st.toast(f"🎧 Processing '{audio_file_obj.name}'...", icon="⏳"); time.sleep(5)
                     processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                 if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio processing failed/unexpected state: {processed_audio_file_ref.state.name}")
                 st.toast(f"🎧 Audio ready!", icon="✅")
                 api_payload = [final_prompt_for_api, processed_audio_file_ref]
            else: # Text or PDF
                if not transcript_data: raise ValueError("Transcript data missing.")
                api_payload = final_prompt_for_api

            # Generate Content
            if api_payload: response = model.generate_content(api_payload)
            else: raise ValueError("API Payload construction failed.")

            # Handle Response (Added check for empty text explicitly)
            if response and hasattr(response, 'text') and response.text and response.text.strip():
                 st.session_state.generated_notes = response.text.strip() # Store stripped text
                 st.toast("🎉 Notes generated!", icon="✅")
            # Check for candidates and parts if text is missing (more robust error checking)
            elif response and not hasattr(response, 'text') and hasattr(response, 'candidates') and response.candidates:
                 st.session_state.error_message = f"🤔 AI response received, but no usable text found. Finish reason: {response.candidates[0].finish_reason}. Safety: {response.candidates[0].safety_ratings}"
            elif response: st.session_state.error_message = "🤔 AI returned empty response." # Fallback empty check
            else: st.session_state.error_message = "😥 AI generation failed (No response)."

            # Cleanup Audio only AFTER successful API call
            if actual_input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None; st.toast("☁️ Temp audio cleaned up.", icon="🗑️")
                except Exception as delete_err: st.warning(f"Could not delete temp audio: {delete_err}", icon="⚠️")

        except Exception as e:
            st.session_state.error_message = f"❌ API/Processing Error: {e}"
            # Attempt audio cleanup on error
            if st.session_state.uploaded_audio_info:
                try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None
                except Exception: pass

    # --- Finish processing ---
    st.session_state.processing = False
    # Rerun only if error needs displaying, otherwise state update handles UI
    if st.session_state.error_message or not st.session_state.generated_notes:
        st.rerun()


# --- Footer ---
st.divider()
st.caption("Powered by [Google Gemini](https://deepmind.google/technologies/gemini/) | App by SynthNotes AI")

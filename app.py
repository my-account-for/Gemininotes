# --- Required Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
import tempfile
from datetime import datetime # For default date in filename
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
    /* ... Keep previous CSS ... */
    footer { text-align: center; color: #9CA3AF; font-size: 0.8rem; padding: 2rem 0 1rem 0; }
    footer a { color: #6B7280; text-decoration: none; }
    footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Define Available Models & Meeting Types (Keep as is) ---
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
    # Separate config for potential filename generation (simpler task)
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, # Add flag for filename step
    'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,
    'selected_model_display_name': DEFAULT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text',
    'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'edit_notes_enabled': False, 'edited_notes_text': "",
    'suggested_filename': None, # State for filename
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    # (Keep as is)
    try:
        pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts); return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e: st.session_state.error_message = f"📄 PDF Read Error: {e}"; return None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def create_expert_meeting_prompt(transcript, context=None):
    # (Keep prompt function as is)
    prompt_parts = [ "You are an expert meeting note-taker...", (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""), ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n"); return "\n".join(filter(None, prompt_parts))

def create_earnings_call_prompt(transcript, context=None):
    # (Keep prompt function as is)
    prompt_parts = [ "You are a financial analyst...", (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""), ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY:**\n"); return "\n".join(filter(None, prompt_parts))

def create_docx(text):
    # (Keep as is)
    document = docx.Document(); [document.add_paragraph(line) for line in text.split('\n')]
    buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()

def get_current_input_data():
    # (Keep as is)
    input_type = st.session_state.input_method_radio
    transcript = None; audio_file = None
    if input_type == "Paste Text": transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file is not None: transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
    elif input_type == "Upload Audio": audio_file = st.session_state.audio_uploader
    return input_type, transcript, audio_file

def update_prompt_display_text():
    # (Keep as is)
    meeting_type = st.session_state.selected_meeting_type
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        prompt_func = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
        placeholder = "[TRANSCRIPT ...]" if input_type != "Upload Audio" else None
        if input_type == "Upload Audio":
             base_prompt = prompt_func(transcript=None, context=temp_context)
             st.session_state.current_prompt_text = ("# NOTE FOR AUDIO...\n#######\n\n" + base_prompt)
        else: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state.current_prompt_text: st.session_state.current_prompt_text = "# Enter custom prompt..."
    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom": st.session_state.current_prompt_text = ""

def clear_all_state():
    # (Keep as is) - Need to reset suggested_filename too
    st.session_state.text_input = ""; st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None; st.session_state.context_input = ""
    st.session_state.add_context_enabled = False; st.session_state.current_prompt_text = ""
    st.session_state.view_edit_prompt_enabled = False; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.processing = False
    st.session_state.suggested_filename = None # Reset filename
    update_prompt_display_text(); st.toast("Inputs and outputs cleared!", icon="🧹")

# --- Function to Generate Filename ---
def generate_suggested_filename(notes_content, meeting_type):
    """Uses Gemini to suggest a filename based on notes content."""
    if not notes_content:
        return None
    try:
        st.session_state.generating_filename = True # Set flag
        st.toast("💡 Generating filename suggestion...", icon="⏳")
        # Use a faster/cheaper model if available and suitable
        filename_model_id = "gemini-1.5-flash" # Or keep selected model if preferred
        filename_model = genai.GenerativeModel(filename_model_id)

        # Construct prompt for filename generation
        today_date = datetime.now().strftime("%Y%m%d")
        mt_cleaned = meeting_type.replace(" ", "") # e.g., ExpertMeeting
        filename_prompt = (
            f"Analyze the following meeting notes. Suggest a concise filename in the format YYYYMMDD_ClientOrTopic_MeetingType. "
            f"Use today's date ({today_date}) if no specific date is clearly mentioned in the notes. "
            f"Extract the main client name, company name, or primary topic discussed. Use CamelCase or underscores for the topic part. Use '{mt_cleaned}' for the MeetingType part. "
            f"Examples: {today_date}_AcmeCorp_{mt_cleaned}, {today_date}_PricingStrategy_{mt_cleaned}, {today_date}_CompetitorXYZ_{mt_cleaned}. "
            f"Output ONLY the filename string, nothing else.\n\n"
            f"NOTES:\n{notes_content[:1500]}" # Limit notes length sent for filename generation
        )

        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config)

        if response and hasattr(response, 'text') and response.text:
            suggested_name = response.text.strip()
            # Basic validation/cleaning
            suggested_name = re.sub(r'[^\w\-.]', '_', suggested_name) # Replace invalid chars with underscore
            if len(suggested_name) > 100: # Limit length
                suggested_name = suggested_name[:100]
            # Check if it somewhat matches the expected pattern (optional)
            if re.match(r"\d{8}_\w+_\w+", suggested_name):
                st.toast(f"💡 Filename suggestion: {suggested_name}.txt/.md", icon="✅")
                return suggested_name # Return only the base name
            else:
                st.warning("AI filename suggestion wasn't in the expected format, using default.", icon="⚠️")
                return None # Fallback if format is wrong
        else:
             st.warning("Could not generate filename suggestion from AI.", icon="⚠️")
             return None
    except Exception as e:
        st.warning(f"Error generating filename: {e}", icon="⚠️")
        return None
    finally:
         st.session_state.generating_filename = False # Reset flag


# --- Streamlit App UI (Keep as is) ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1: col1a, col1b = st.columns(2) # Meeting Type & Model
    with col1a: st.subheader("Meeting Details"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True, on_change=update_prompt_display_text)
    with col1b: st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary")
    st.divider()
    st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed", on_change=update_prompt_display_text)
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")
    st.divider()
    col3a, col3b = st.columns(2) # Optional Elements
   # Row 3: Optional Elements
    col3a, col3b = st.columns(2)
    with col3a: # Context
        st.checkbox("Add Context (Optional)", key="add_context_enabled", help="Provide background info.", on_change=update_prompt_display_text)
        # Correct indentation for the st.text_area within the if block
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=100, key="context_input", on_change=update_prompt_display_text, placeholder="E.g., Attendees: Alice, Bob\nProject: Q3 Launch")
    with col3b: # View/Edit Prompt Checkbox
        # Correct indentation for the st.checkbox within the if block
        if st.session_state.selected_meeting_type != "Custom":
            st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled", help="View/modify the AI prompt.", on_change=update_prompt_display_text)
# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
                   (st.session_state.selected_meeting_type == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_area_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_area_title); caption_text = "Edit prompt..." if st.session_state.selected_meeting_type != "Custom" else "Enter prompt..."
        st.caption(caption_text); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# Generate Button
st.write("")
generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: st.info("⏳ Generating notes...", icon="🧠")
    elif st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡") # Show separate status
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        if st.session_state.edit_notes_enabled:
             st.text_area("Editable Notes:", value=st.session_state.edited_notes_text, key="edited_notes_text", height=400, label_visibility="collapsed")
             notes_content_to_use = st.session_state.edited_notes_text
        else: st.markdown(st.session_state.generated_notes); notes_content_to_use = st.session_state.generated_notes

        # Determine filename
        default_filename_base = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"
        filename_base = st.session_state.suggested_filename or default_filename_base

        # Action Buttons
        st.write("")
        col_btn_dl1, col_btn_dl2 = st.columns(2)
        with col_btn_dl1: st.download_button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{filename_base}.txt", mime="text/plain", key='download-txt', help="Download as Plain Text", use_container_width=True)
        with col_btn_dl2: st.download_button(label="⬇️ Markdown", data=notes_content_to_use, file_name=f"{filename_base}.md", mime="text/markdown", key='download-md', help="Download as Markdown File", use_container_width=True)
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    # Reset relevant outputs
    st.session_state.processing = True; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.suggested_filename = None # Reset filename
    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename: # Ensure filename gen isn't running
    # --- Outer try...finally for overall processing state ---
    try:
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
                final_prompt_for_api = final_prompt_for_api.split("####################################\n\n")[-1]
            else: # Generate default prompt
                prompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
                if actual_input_type != "Upload Audio":
                     if transcript_data: final_prompt_for_api = prompt_function(transcript_data, final_context)
                     else: st.session_state.error_message = "Error: Transcript data missing."
                else: # Audio: Add wrapper automatically ONLY if not custom/edited
                    base_prompt = prompt_function(transcript=None, context=final_context)
                    # --- Refined Audio Prompt Wrapper ---
                    final_prompt_for_api = (
                        f"You are provided with an audio file and accompanying context/instructions below.\n"
                        f"1. First, accurately transcribe the *entire* audio file.\n"
                        f"2. After transcription, use the *full transcript text* you generated to create detailed notes based *only* on that transcript, following the specific structure and instructions provided below.\n"
                        f"Ensure the final output only contains the structured notes based on the transcript content.\n"
                        f"---\n"
                        f"[BEGIN SPECIFIC STRUCTURE/INSTRUCTIONS]\n"
                        f"{base_prompt}\n" # This includes the ending like "**GENERATED NOTES:**"
                        f"[END SPECIFIC STRUCTURE/INSTRUCTIONS]\n"
                        f"---"
                    )


        # --- API Call Section ---
        if not st.session_state.error_message and final_prompt_for_api and (transcript_data or audio_file_obj):
            # --- Inner try...except for specific API/processing errors ---
            try:
                st.toast(f"🧠 Generating notes with {st.session_state.selected_model_display_name}...", icon="✨")
                model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=main_gen_config) # Use main config
                response = None; api_payload = None; processed_audio_file_ref = None; temp_file_path = None

                if actual_input_type == "Upload Audio":
                    if not audio_file_obj: raise ValueError("Audio file object missing.")
                    st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️")
                    audio_bytes = audio_file_obj.getvalue()
                    # Use tempfile
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj.name)[1]) as temp_file:
                            temp_file.write(audio_bytes); temp_file_path = temp_file.name
                        if temp_file_path:
                            processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}", mime_type=audio_file_obj.type)
                        else: raise Exception("Failed to create temporary file.")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path) # Clean up disk temp file

                    st.session_state.uploaded_audio_info = processed_audio_file_ref # Store cloud ref
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
                    st.session_state.edited_notes_text = st.session_state.generated_notes
                    st.toast("🎉 Notes generated!", icon="✅")
                    # --- Trigger filename generation ---
                    st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                    # --- End trigger ---
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     st.session_state.error_message = f"⚠️ AI response blocked. Reason: {response.prompt_feedback.block_reason}."
                elif response: st.session_state.error_message = "🤔 AI returned empty response."
                else: st.session_state.error_message = "😥 AI generation failed (No response)."

                # Cleanup Cloud Audio File Object only AFTER successful call
                if actual_input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None; st.toast("☁️ Temp cloud audio cleaned up.", icon="🗑️")
                    except Exception as delete_err: st.warning(f"Could not delete temp cloud audio: {delete_err}", icon="⚠️")

            except Exception as e: # Catch API/processing errors
                st.session_state.error_message = f"❌ Processing Error: {e}"
                if st.session_state.uploaded_audio_info: # Attempt cleanup on inner error
                    try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None
                    except Exception: pass
        # End of inner try...except

    # --- FINALLY block: Always runs ---
    finally:
        st.session_state.processing = False
        # Rerun ONLY if error needs displaying OR if notes were generated successfully
        # The UI will update based on state changes after the rerun
        if st.session_state.error_message or st.session_state.generated_notes:
             st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by [Google Gemini](https://deepmind.google.technologies/gemini/) | App by SynthNotes AI")

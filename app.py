import streamlit as st
import google.generativeai as genai
import os
import io
import time
from dotenv import load_dotenv
import PyPDF2

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS Injection (Keep as is) ---
st.markdown("""
<style>
    /* ... Keep all previous CSS ... */
    /* Add styling for prompt text area if needed */
    #prompt-edit-area textarea {
        font-family: monospace; /* Use monospace for prompts */
        font-size: 0.9rem;
        line-height: 1.4;
        background-color: #FDFDFD; /* Slightly different bg */
    }
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
     if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
        DEFAULT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]


# --- Define Meeting Types ---
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"] # Added Custom
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]


# --- Load API Key and Configure Gemini Client ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("### 🔑 API Key Not Found!", icon="🚨")
    st.markdown("Please ensure `GEMINI_API_KEY` is set in your environment or `.env` file.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    generation_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e:
     st.error(f"### 💥 Error Configuring Google AI Client", icon="🚨"); st.error(f"Details: {e}"); st.stop()


# --- Initialize Session State ---
default_state = {
    'processing': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_context_enabled': False,
    'selected_model_display_name': DEFAULT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, # State for view/edit checkbox
    'current_prompt_text': "",       # State for the prompt text area
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from PDF, updates session state on error."""
    try:
        pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts)
        return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e: st.session_state.error_message = f"📄 PDF Read Error: {e}"; return None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

def create_expert_meeting_prompt(transcript, context=None):
    """Creates the prompt for 'Expert Meeting'."""
    # (Prompt definition remains the same as before)
    prompt_parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:",
        "\n**Structure:**",
        "- **Opening overview or Expert background (Optional):** ...", # Truncated for brevity
        "- **Q&A format:** ...",
        "\n**Additional Instructions:** ...",
        "\n---",
        (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n")
    return "\n".join(filter(None, prompt_parts))

def create_earnings_call_prompt(transcript, context=None):
    """Creates the prompt for 'Earnings Call'."""
    # (Prompt definition remains the same as before)
    prompt_parts = [
        "You are a financial analyst summarizing an earnings call transcript.",
        "Generate detailed, factual notes, extracting key information and statements.",
        "Structure the notes logically...",
        "\n**General Structure to Follow:** ...",
        "\n**Industry-Specific Sections:** ...",
        "\n- **Q&A Session Summary:** ...",
        "\n**Additional Instructions:** ...",
        "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY:**\n")
    return "\n".join(filter(None, prompt_parts))

def get_current_input_data():
    """Helper to get transcript/audio file based on input method and state."""
    input_type = st.session_state.get("input_method_radio", "Paste Text")
    transcript = None
    audio_file = None

    if input_type == "Paste Text":
        transcript = st.session_state.get("text_input", "").strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.get("pdf_uploader")
        if pdf_file:
            pdf_stream = io.BytesIO(pdf_file.getvalue())
            # Use existing error handling within the function
            transcript = extract_text_from_pdf(pdf_stream)
            # If extraction failed, error is already in session_state
    elif input_type == "Upload Audio":
        audio_file = st.session_state.get("audio_uploader") # Return the file object

    return input_type, transcript, audio_file

# --- Function to generate and update the prompt text area ---
def update_prompt_display_text():
    """Generates the appropriate prompt based on selections and updates session state."""
    meeting_type = st.session_state.selected_meeting_type
    # Only generate if view/edit is enabled and meeting type is NOT custom
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        # Get potential context (even if context area not shown yet, use state)
        temp_context = None
        if st.session_state.add_context_enabled:
             temp_context = st.session_state.get("context_input", "").strip()

        # Get current transcript/audio selection (don't need the actual data here, just type)
        input_type, _, _ = get_current_input_data()

        if meeting_type == "Expert Meeting":
            # Pass placeholder transcript for prompt structure view
            prompt_func = create_expert_meeting_prompt
            placeholder = "[MEETING TRANSCRIPT WILL BE INSERTED HERE]" if input_type != "Upload Audio" else None
        elif meeting_type == "Earnings Call":
            prompt_func = create_earnings_call_prompt
            placeholder = "[EARNINGS CALL TRANSCRIPT WILL BE INSERTED HERE]" if input_type != "Upload Audio" else None
        else: # Should not happen if called correctly
            st.session_state.current_prompt_text = ""
            return

        # For audio, generate the base prompt without the wrapper for viewing
        if input_type == "Upload Audio":
             base_prompt = prompt_func(transcript=None, context=temp_context)
             # Add a note for the user about audio
             st.session_state.current_prompt_text = (
                 "# NOTE FOR AUDIO INPUT: The '1. Transcribe first...' wrapper will be added *unless* you edit this prompt.\n"
                 "# If you edit, ensure your prompt instructs the model how to handle the audio file.\n"
                 "####################################\n\n" + base_prompt
             )
        else:
            st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)

    elif meeting_type == "Custom":
         # For custom, the text area IS the source, don't overwrite user input unless empty
         if not st.session_state.current_prompt_text: # Set placeholder only if empty
              st.session_state.current_prompt_text = "# Enter your custom prompt here...\n# Remember to include instructions for transcription if using audio input."
    else:
        # If view/edit is disabled or type changed from custom, clear the prompt text
        st.session_state.current_prompt_text = ""


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Input Section ---
with st.container(border=True):
    # Row 1: Meeting Type and Model Selection
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Meeting Details")
        st.radio( # Directly update session state on change
            "Select Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type",
            index=MEETING_TYPES.index(st.session_state.selected_meeting_type), horizontal=True,
            help="Choose meeting type for tailored note structure. 'Custom' requires your own prompt.",
            on_change=update_prompt_display_text # Regenerate prompt text if type changes
        )
    with col1b:
        st.subheader("AI Model")
        st.selectbox( # Directly update session state on change
            "Choose model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name",
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_display_name),
            help="Select Gemini model. Preview/Experimental models may have different quotas/stability."
        )

    st.divider()

    # Row 2: Source Input
    st.subheader("Source Input")
    st.radio( # Update session state directly
        "Input type:", ("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio",
        horizontal=True, label_visibility="collapsed",
        on_change=update_prompt_display_text # Regenerate prompt text if input type changes (for audio note)
    )
    # Define widgets (values accessed via keys later)
    input_type = st.session_state.input_method_radio # Get current selection
    if input_type == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input")
    elif input_type == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider()

    # Row 3: Optional Elements (Context and Prompt View/Edit)
    col3a, col3b = st.columns(2) # Layout optional elements side-by-side
    with col3a:
        # Context Checkbox & Area
        st.checkbox( # Updates session state directly
             "Add Context (Optional)", key="add_context_enabled",
             help="Provide background info like attendees, goals, etc.",
             on_change=update_prompt_display_text # Regenerate prompt text if context added/removed
        )
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=100, key="context_input",
                         on_change=update_prompt_display_text) # Regenerate prompt text if context changes

    with col3b:
        # View/Edit Prompt Checkbox (only if NOT Custom meeting type)
        if st.session_state.selected_meeting_type != "Custom":
            st.checkbox( # Updates session state directly
                "View/Edit Prompt", key="view_edit_prompt_enabled",
                help="View the prompt that will be sent to the AI and modify it if needed.",
                on_change=update_prompt_display_text # Trigger prompt generation/clearing
            )

# --- Prompt Display/Input Area (Conditional) ---
# Show if EITHER (View/Edit is checked for non-custom types) OR (meeting type IS custom)
if (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
   (st.session_state.selected_meeting_type == "Custom"):
    with st.container(border=True):
        prompt_area_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_area_title)
        if st.session_state.selected_meeting_type != "Custom":
            st.caption("The prompt below will be used. You can edit it directly. If using audio, the 'Transcribe first...' wrapper will be added automatically *unless* you edit this text.")
        else:
             st.caption("Enter the full prompt the AI should follow. If using audio input, you *must* include instructions for transcription within your prompt.")

        # Text area bound to session state
        st.text_area(
            "Prompt Text:", key="current_prompt_text", # Value comes from/goes to session state
            height=350, label_visibility="collapsed",
            element_id="prompt-edit-area" # For CSS targeting
        )


# --- Generate Button ---
st.write("")
generate_button = st.button(
    "🚀 Generate Notes", type="primary", use_container_width=True,
    disabled=st.session_state.processing
)

# --- Output Section (Keep as is) ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True) # Marker for CSS
    if st.session_state.processing: st.info("⏳ Processing...", icon="🧠")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.markdown(st.session_state.generated_notes)
        st.download_button(label="⬇️ Download Notes (.txt)", data=st.session_state.generated_notes,
                           file_name=f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes.txt",
                           mime="text/plain", key='download-txt')
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True
    st.session_state.generated_notes = None
    st.session_state.error_message = None
    st.rerun()

if st.session_state.processing:
    # Retrieve selections and inputs from state
    meeting_type = st.session_state.selected_meeting_type
    input_type = st.session_state.input_method_radio
    selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
    view_edit_enabled = st.session_state.view_edit_prompt_enabled
    user_prompt_text = st.session_state.current_prompt_text # Get edited/custom prompt

    # Get input data (transcript or audio file object)
    actual_input_type, transcript_data, audio_file_obj = get_current_input_data()
    # Note: transcript_data might be None if PDF extraction failed, error is in session state.

    # --- Validation ---
    if actual_input_type == "Paste Text" and not transcript_data: st.session_state.error_message = "⚠️ Text area is empty."
    elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_message = "⚠️ No PDF uploaded or failed to extract text." # Check both
    elif actual_input_type == "Upload Audio" and not audio_file_obj: st.session_state.error_message = "⚠️ No audio file uploaded."
    # Custom prompt validation
    if meeting_type == "Custom" and not user_prompt_text.strip():
        st.session_state.error_message = "⚠️ Custom Prompt cannot be empty when 'Custom' meeting type is selected."

    # Get Context (if enabled)
    final_context = None
    if st.session_state.add_context_enabled:
        final_context = st.session_state.get("context_input", "").strip()

    # --- Determine Final Prompt ---
    final_prompt_for_api = None
    prompt_was_edited_or_custom = False # Flag for audio handling

    if not st.session_state.error_message:
        if meeting_type == "Custom":
            final_prompt_for_api = user_prompt_text # Use exactly what user entered
            prompt_was_edited_or_custom = True
        elif meeting_type in ["Expert Meeting", "Earnings Call"]:
            if view_edit_enabled:
                final_prompt_for_api = user_prompt_text # Use potentially edited prompt
                prompt_was_edited_or_custom = True
            else:
                # Generate prompt freshly if view/edit was not enabled
                prompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earnings_call_prompt
                # Need transcript data here if not audio
                if actual_input_type != "Upload Audio":
                    final_prompt_for_api = prompt_function(transcript_data, final_context)
                else:
                    # Generate base prompt for audio (no transcript needed for function call)
                    base_prompt = prompt_function(transcript=None, context=final_context)
                    # Add wrapper (only if prompt wasn't edited)
                    final_prompt_for_api = (
                        "1. First, accurately transcribe the provided audio file.\n"
                        "2. Then, using the transcription, create notes based on:\n---\n"
                        f"{base_prompt}"
                    )
        else: # Should not happen
            st.session_state.error_message = "Internal Error: Invalid meeting type during prompt determination."

    # --- API Call ---
    if not st.session_state.error_message and final_prompt_for_api and (transcript_data or audio_file_obj):
        try:
            st.toast(f"🧠 Generating notes with {st.session_state.selected_model_display_name}...", icon="✨")
            model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=generation_config)
            response = None
            api_payload = None

            if actual_input_type == "Upload Audio":
                 if not audio_file_obj: raise ValueError("Audio file object missing.") # Should be caught earlier
                 # Process audio file first (upload to Google Cloud)
                 st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️")
                 audio_bytes = audio_file_obj.getvalue()
                 audio_file_for_api = genai.upload_file(content=audio_bytes, display_name=f"audio_{int(time.time())}", mime_type=audio_file_obj.type)
                 st.session_state.uploaded_audio_info = audio_file_for_api # Store for cleanup
                 # Polling for readiness
                 polling_start_time = time.time()
                 while audio_file_for_api.state.name == "PROCESSING":
                     if time.time() - polling_start_time > 300: raise TimeoutError("Audio processing timed out.")
                     st.toast(f"🎧 Processing '{audio_file_obj.name}'...", icon="⏳")
                     time.sleep(5); audio_file_for_api = genai.get_file(audio_file_for_api.name)
                 if audio_file_for_api.state.name == "FAILED": raise Exception(f"Audio processing failed: {audio_file_for_api.name}")
                 if audio_file_for_api.state.name != "ACTIVE": raise Exception(f"Unexpected audio state: {audio_file_for_api.state.name}")
                 st.toast(f"🎧 Audio ready!", icon="✅")

                 # Prepare payload for API
                 api_payload = [final_prompt_for_api, audio_file_for_api] # Pass prompt text + audio object

            else: # Text or PDF input
                if not transcript_data: raise ValueError("Transcript data missing.") # Should be caught earlier
                # Payload is just the final prompt text
                api_payload = final_prompt_for_api

            # --- Generate Content Call ---
            if api_payload:
                response = model.generate_content(api_payload)
            else:
                 raise ValueError("API Payload could not be constructed.")

            # --- Handle Response ---
            if response and response.text:
                st.session_state.generated_notes = response.text
                st.toast("🎉 Notes generated!", icon="✅")
            elif response: st.session_state.error_message = "🤔 AI returned an empty response."
            else: st.session_state.error_message = "😥 AI generation failed (No response)."

            # --- Cleanup Audio (if applicable) ---
            if actual_input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                try:
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
                    st.toast("☁️ Temp audio file cleaned up.", icon="🗑️")
                except Exception as delete_err: st.warning(f"Could not delete temp audio: {delete_err}", icon="⚠️")

        except Exception as e:
            st.session_state.error_message = f"❌ API Error: {e}"
            # Attempt audio cleanup on error
            try:
                if actual_input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
            except Exception: pass

    # --- Finish processing ---
    st.session_state.processing = False
    # Don't rerun if there was no error and notes were generated
    # Only rerun if there WAS an error to display it correctly
    if st.session_state.error_message:
        st.rerun()
    # If successful, the UI updates automatically without full rerun


# --- Footer ---
st.divider()
st.caption("Powered by [Google Gemini](https://deepmind.google.technologies/gemini/) | App by SynthNotes AI")

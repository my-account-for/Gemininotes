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
import math
# Removed pydub import

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI", # Simplified title
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
# (CSS remains the same - omitted for brevity)
st.markdown(""" <style> ... </style> """, unsafe_allow_html=True)


# --- Define Available Models & Meeting Types ---
# ... (Model definitions remain the same) ...
AVAILABLE_MODELS = {
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03-25",
}
DEFAULT_NOTES_MODEL_NAME = "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)"
if DEFAULT_NOTES_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_NOTES_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"
DEFAULT_TRANSCRIPTION_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
if DEFAULT_TRANSCRIPTION_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_TRANSCRIPTION_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]
DEFAULT_REFINEMENT_MODEL_NAME = "Gemini 1.5 Pro (Complex Reasoning)"
if DEFAULT_REFINEMENT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_REFINEMENT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]

MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]

# --- Sector-Specific Topics ---
# ... (Sector definitions remain the same) ...
SECTOR_OPTIONS = ["Other / Manual Topics", "IT Services", "QSR"]
DEFAULT_SECTOR = SECTOR_OPTIONS[0]
SECTOR_TOPICS = {
    "IT Services": """...""", # Keep full topic strings
    "QSR": """..."""
}

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_KEY)
    # Base Generation Configs
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    # Config for notes step - NO max_output_tokens
    notes_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    # Config for transcription - Keep 8k limit as safety/default
    transcription_gen_config = {"temperature": 0.1, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    # Config for refinement chunk - Keep 8k limit PER CHUNK
    refinement_chunk_gen_config = {"temperature": 0.3, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, # Will store the single cloud file reference
    'add_context_enabled': False,
    'selected_notes_model_display_name': DEFAULT_NOTES_MODEL_NAME,
    'selected_transcription_model_display_name': DEFAULT_TRANSCRIPTION_MODEL_NAME,
    'selected_refinement_model_display_name': DEFAULT_REFINEMENT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'selected_sector': DEFAULT_SECTOR,
    'earnings_call_topics': '',
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None,
    'refined_transcript': None,
    # 'chunking_occurred_transcription': False, # No longer needed
    'chunking_occurred_refinement': False, # Keep this one
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream): ... # (Keep implementation)
def update_topic_template(): ... # (Keep implementation)
def create_expert_meeting_prompt(transcript, context=None): ... # (Keep implementation)
def create_earnings_call_prompt(transcript, user_topics_text=None, context=None): ... # (Keep implementation)
def create_docx(text): ... # (Keep implementation)
def get_current_input_data(): ... # (Keep implementation)
def get_prompt_display_text(): ... # (Keep implementation)

def clear_all_state():
    # Reset selections and inputs
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
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
    st.session_state.uploaded_audio_info = None # Reset cloud ref
    st.session_state.history = []
    st.session_state.raw_transcript = None
    st.session_state.refined_transcript = None
    # st.session_state.chunking_occurred_transcription = False # Removed
    st.session_state.chunking_occurred_refinement = False # Reset flag
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type): ... # (Keep implementation)
def add_to_history(notes): ... # (Keep implementation)
def restore_note_from_history(index): ... # (Keep implementation)

# --- RENAMED: Text Refinement Helper Function (Handles Text Chunking) ---
def refine_transcript_possibly_chunked(raw_transcript_text, refinement_model, general_context, status):
    """
    Refines transcript TEXT. Chunks TEXT input if it's too long.
    Warns about potential context loss if chunking occurs.
    Returns the concatenated refined transcript text and a flag indicating if chunking occurred.
    """
    MAX_INPUT_CHUNK_CHARS = 35000 # Heuristic for *text* input length
    MIN_TRANSCRIPT_LEN_FOR_CHUNK = 40000 # Only chunk if text is reasonably long
    refined_outputs = []
    chunking_performed = False

    # Check length of the input *text*
    if len(raw_transcript_text) > MIN_TRANSCRIPT_LEN_FOR_CHUNK:
        chunking_performed = True
        st.warning(
            "⚠️ Raw transcript text is long. Performing refinement in chunks. "
            "Speaker labels and context may be inconsistent across chunks.", icon="❗"
        )
        status.update(label="🧹 Step 2: Transcript text long, preparing chunks...")

        # Split text by paragraph, group into chunks
        segments = raw_transcript_text.split('\n\n')
        current_chunk_segments = []
        current_chunk_len = 0
        input_text_chunks = []

        for segment in segments:
            segment_len = len(segment)
            if current_chunk_len > 0 and (current_chunk_len + segment_len + 2) > MAX_INPUT_CHUNK_CHARS:
                input_text_chunks.append("\n\n".join(current_chunk_segments))
                current_chunk_segments = [segment]
                current_chunk_len = segment_len
            else:
                current_chunk_segments.append(segment)
                current_chunk_len += segment_len + 2

        if current_chunk_segments:
            input_text_chunks.append("\n\n".join(current_chunk_segments))

        num_chunks = len(input_text_chunks)
        status.update(label=f"🧹 Step 2: Refining transcript text in {num_chunks} chunks...")

        # Loop through text chunks and refine
        for i, chunk_text in enumerate(input_text_chunks):
            status.update(label=f"🧹 Step 2: Refining text chunk {i+1}/{num_chunks}...")
            # (Refinement prompt for chunk remains the same as before)
            refinement_prompt = f"""Please refine the following raw audio transcript chunk: ... **Refined Transcript Chunk:** """ # Keep prompt content
            refinement_prompt = f"""Please refine the following raw audio transcript chunk:

            **Raw Transcript Chunk:**
            ```
            {chunk_text}
            ```

            **Instructions:**
            1.  **Identify Speakers:** Assign consistent labels (e.g., Speaker 1, Speaker 2). Place the label on a new line before the speaker's turn. ***Note: Speaker labels may restart within this chunk.***
            2.  **Translate to English:** Convert any non-English speech found within the transcript to English, ensuring it fits naturally within the conversation.
            3.  **Correct Errors:** Fix spelling mistakes and grammatical errors. Use the overall conversation context *within this chunk* to correct potentially misheard words or phrases.
            4.  **Format:** Ensure clear separation between speaker turns using the speaker labels. Maintain the original conversational flow and content *of this chunk*.
            5.  **Output:** Provide *only* the refined, speaker-diarized, translated, and corrected transcript text for this chunk. Do not add any introduction, summary, or commentary.

            **Additional Context (Optional - use for understanding terms, names, etc.):**
            {general_context if general_context else "None provided."}

            **Refined Transcript Chunk:**
            """
            try:
                r_response = refinement_model.generate_content(
                    refinement_prompt,
                    generation_config=refinement_chunk_gen_config, # Use config WITH output limit
                    safety_settings=safety_settings
                )
                if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                    refined_outputs.append(r_response.text.strip())
                # ... (Handle block/empty response for chunk, append raw chunk as fallback) ...
                elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                    st.warning(f"⚠️ Refinement blocked for chunk {i+1}: {r_response.prompt_feedback.block_reason}. Using raw chunk.", icon="⚠️")
                    refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Failed/Blocked) ---\n{chunk_text}")
                else:
                    st.warning(f"🤔 Refinement returned empty response for chunk {i+1}. Using raw chunk.", icon="⚠️")
                    refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Failed) ---\n{chunk_text}")

            except Exception as refine_chunk_err:
                 st.warning(f"❌ Error refining text chunk {i+1}: {refine_chunk_err}. Using raw chunk.", icon="⚠️")
                 refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Error) ---\n{chunk_text}")

        # Combine refined text chunks
        final_refined_text = "\n\n".join(refined_outputs)
        status.update(label="🧹 Step 2: Finished refining all text chunks.")
        return final_refined_text, chunking_performed

    else:
        # Transcript TEXT is short enough, refine in one go
        status.update(label=f"🧹 Step 2: Refining transcript text (single pass)...")
        # (Original full refinement prompt remains the same)
        refinement_prompt = f"""Please refine the following raw audio transcript: ... **Refined Transcript:** """ # Keep prompt content
        refinement_prompt = f"""Please refine the following raw audio transcript:

        **Raw Transcript:**
        ```
        {raw_transcript_text}
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
        try:
            r_response = refinement_model.generate_content(
                refinement_prompt,
                generation_config=refinement_chunk_gen_config, # Still use config with output limit
                safety_settings=safety_settings
            )
            if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                status.update(label="🧹 Step 2: Refinement complete!")
                return r_response.text.strip(), chunking_performed
            # ... (Handle block/empty/error for single pass, return raw text as fallback) ...
            elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                st.warning(f"⚠️ Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript.", icon="⚠️")
                status.update(label="⚠️ Refinement blocked. Using raw transcript.")
                return raw_transcript_text, chunking_performed
            else:
                st.warning("🤔 Refinement returned empty response. Using raw transcript.", icon="⚠️")
                status.update(label="⚠️ Refinement failed. Using raw transcript.")
                return raw_transcript_text, chunking_performed
        except Exception as refine_err:
            st.warning(f"❌ Error during Step 2 (Refinement): {refine_err}. Using raw transcript.", icon="⚠️")
            status.update(label="⚠️ Refinement error. Using raw transcript.")
            return raw_transcript_text, chunking_performed


# --- Streamlit App UI ---
# ... (UI definition remains the same) ...
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
# ... Input Section (including model selectors) ...
# ... Prompt Area ...
# ... Generate Button ...

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    # (Output display logic)
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        # Display intermediate transcripts + warnings
        if st.session_state.raw_transcript:
            with st.expander("View Raw Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
                # Removed transcription chunk warning
        if st.session_state.refined_transcript:
             with st.expander("View Refined Transcript (Step 2 Output)"):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)
                # Keep refinement chunk warning
                if 'chunking_occurred_refinement' in st.session_state and st.session_state.chunking_occurred_refinement:
                    st.warning("Note: Refinement was performed in chunks due to transcript length. Speaker labels/context may be inconsistent.", icon="⚠️")
        # ... (Notes display, edit, download buttons) ...
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        # ...

    elif not st.session_state.processing:
        st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
# ... (remains the same) ...
with st.expander("📜 Recent Notes History (Last 3)", expanded=False): ...

# --- Processing Logic ---
if generate_button:
    # Reset state
    st.session_state.processing = True
    # ... (reset other state variables) ...
    # st.session_state.chunking_occurred_transcription = False # Removed
    st.session_state.chunking_occurred_refinement = False # Reset
    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    # Use uploaded_audio_info to store the SINGLE cloud file reference
    st.session_state.uploaded_audio_info = None
    with st.status("🚀 Initializing process...", expanded=True) as status:
        try: # Outer try-finally
            status.update(label="⚙️ Reading inputs and settings...")
            # ... (Retrieve state: meeting_type, model IDs, context, topics, etc.) ...
            meeting_type = st.session_state.selected_meeting_type
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            # ... (rest of state retrieval)
            general_context = ...
            earnings_call_topics_text = ...
            actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

            # Reset flags
            st.session_state.chunking_occurred_refinement = False

            status.update(label="🧠 Initializing AI models...")
            # Initialize models
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            final_transcript_for_notes = transcript_data # Start with text/PDF data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None
            uploaded_file_ref = None # Temporary variable for the single uploaded file

            status.update(label="✔️ Validating inputs...")
            # ... (Input Validation) ...

            # ==============================================
            # --- STEP 1: Transcription (Single Audio Upload) ---
            # ==============================================
            if actual_input_type == "Upload Audio":
                temp_file_path = None # Define for potential cleanup
                try:
                    status.update(label=f"☁️ Uploading '{audio_file_obj.name}'...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj.name)[1]) as tf:
                         audio_file_obj.seek(0)
                         tf.write(audio_file_obj.getvalue()); temp_file_path = tf.name
                    if not temp_file_path: raise Exception("Failed to create temporary file for audio.")

                    # Upload the single file
                    uploaded_file_ref = genai.upload_file(
                        path=temp_file_path,
                        display_name=f"audio_{int(time.time())}_{audio_file_obj.name}"
                    )
                    # Store the single reference in session state for potential later cleanup if needed
                    st.session_state.uploaded_audio_info = uploaded_file_ref

                    status.update(label="🎧 Processing uploaded audio...")
                    # Poll for the single upload completion
                    polling_start = time.time(); MAX_POLLING_TIME_SEC = 600; POLLING_INTERVAL_SEC = 10
                    while uploaded_file_ref.state.name == "PROCESSING":
                        if time.time() - polling_start > MAX_POLLING_TIME_SEC: raise TimeoutError("Audio processing timed out.")
                        time.sleep(POLLING_INTERVAL_SEC); uploaded_file_ref = genai.get_file(uploaded_file_ref.name)
                    if uploaded_file_ref.state.name != "ACTIVE": raise Exception(f"Audio file processing failed. State: {uploaded_file_ref.state.name}")
                    status.update(label="🎧 Audio ready for transcription!")

                    # --- Perform single transcription call ---
                    status.update(label=f"✍️ Step 1: Transcribing audio...")
                    t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                    # Use transcription_gen_config (with 8k output limit)
                    t_response = transcription_model.generate_content(
                        [t_prompt, uploaded_file_ref],
                        generation_config=transcription_gen_config
                    )

                    if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                        st.session_state.raw_transcript = t_response.text.strip()
                        status.update(label="✍️ Step 1: Transcription complete!")
                        final_transcript_for_notes = st.session_state.raw_transcript
                    # ... (Handle transcription error/block) ...
                    elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                        raise Exception(f"Transcription blocked: {t_response.prompt_feedback.block_reason}")
                    else: raise Exception("Transcription failed: AI returned empty response.")

                except Exception as step1_err:
                     raise Exception(f"Error during Audio Upload/Transcription: {step1_err}") from step1_err
                finally:
                     # Clean up local temp file
                     if temp_file_path and os.path.exists(temp_file_path):
                         try: os.remove(temp_file_path)
                         except OSError as e: st.warning(f"Could not delete temp file {temp_file_path}: {e}")
                     # Clean up the single cloud file IMMEDIATELY after transcription attempt
                     if uploaded_file_ref:
                         try:
                             status.update(label=f"🗑️ Cleaning up cloud audio file...")
                             genai.delete_file(uploaded_file_ref.name)
                             st.session_state.uploaded_audio_info = None # Clear ref after deletion
                             status.update(label=f"🗑️ Cloud audio file cleaned up.")
                         except Exception as cleanup_err:
                             st.warning(f"⚠️ Failed to delete cloud audio file ({uploaded_file_ref.name}): {cleanup_err}", icon="❗")
                             # Keep ref in session state if deletion failed, for final cleanup attempt

            # ==============================================
            # --- STEP 2: Refinement (Possibly Text Chunked) ---
            # ==============================================
            if final_transcript_for_notes: # Check if we have a transcript
                 # Refinement always uses the raw transcript if available (most direct from audio)
                 # Otherwise uses the text/PDF input
                 transcript_to_refine = st.session_state.raw_transcript if st.session_state.raw_transcript else final_transcript_for_notes

                 if transcript_to_refine:
                      # Call the TEXT chunking/refinement helper
                      refined_text_result, chunking_happened_ref = refine_transcript_possibly_chunked(
                          transcript_to_refine,
                          refinement_model,
                          general_context,
                          status
                      )
                      st.session_state.refined_transcript = refined_text_result
                      st.session_state.chunking_occurred_refinement = chunking_happened_ref
                      final_transcript_for_notes = st.session_state.refined_transcript # Use refined for notes
                 else:
                      status.update(label="⚠️ Step 2: Skipped Refinement (No transcript text found).")

            # =============================
            # --- STEP 3: Generate Notes ---
            # =============================
            if not final_transcript_for_notes:
                 raise ValueError("No transcript available to generate notes.")

            status.update(label="📝 Preparing final prompt...")
            # ... (Determine final_prompt_for_api and api_payload_parts) ...

            # --- Generate Notes API Call ---
            try:
                status.update(label=f"✨ Step 3: Generating notes...")
                # Use notes_gen_config (WITHOUT max_output_tokens)
                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=notes_gen_config
                    # Safety settings already part of model init
                )
                # ... (Handle Notes Response) ...
                if response and hasattr(response, 'text') and response.text.strip():
                    st.session_state.generated_notes = response.text.strip()
                    st.session_state.edited_notes_text = st.session_state.generated_notes
                    add_to_history(st.session_state.generated_notes)
                    st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                    status.update(label="✅ Notes generated successfully!", state="complete")
                # ... (Handle block/empty/no response, update status to error) ...
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


        except Exception as e: # Catch errors from validation, audio, steps 1/2 etc.
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        finally: # Runs after try/except within the 'with status' block
            st.session_state.processing = False
            # --- Final Cloud Audio Cleanup Attempt (if Step 1 cleanup failed) ---
            if st.session_state.uploaded_audio_info:
                file_ref_to_clean = st.session_state.uploaded_audio_info
                if file_ref_to_clean and hasattr(file_ref_to_clean, 'name'):
                    try:
                        st.toast(f"☁️ Performing final cleanup for cloud file {file_ref_to_clean.name}...", icon="🗑️")
                        genai.delete_file(file_ref_to_clean.name)
                        st.session_state.uploaded_audio_info = None # Clear ref if successful
                    except Exception as final_cleanup_error:
                        st.warning(f"Final cloud file cleanup failed for {file_ref_to_clean.name}: {final_cleanup_error}", icon="⚠️")

            st.rerun()


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

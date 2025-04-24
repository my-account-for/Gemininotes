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
from pydub import AudioSegment # !pip install pydub
# Make sure ffmpeg is installed in your environment!

# --- Page Configuration ---
# ... (remains the same) ...
st.set_page_config(
    page_title="SynthNotes AI", # Removed emoji
    page_icon="✨", # page_icon is usually fine with emojis
    layout="wide",
    initial_sidebar_state="collapsed"
)
# --- Custom CSS Injection ---
# ... (remains the same) ...
st.markdown(""" <style> ... </style> """, unsafe_allow_html=True)


# --- Define Available Models & Meeting Types ---
# ... (remains the same) ...
AVAILABLE_MODELS = { ... }
DEFAULT_NOTES_MODEL_NAME = ...
DEFAULT_TRANSCRIPTION_MODEL_NAME = ...
DEFAULT_REFINEMENT_MODEL_NAME = ...
MEETING_TYPES = ...
DEFAULT_MEETING_TYPE = ...

# --- Sector-Specific Topics ---
# ... (remains the same) ...
SECTOR_OPTIONS = ...
DEFAULT_SECTOR = ...
SECTOR_TOPICS = { ... }

# --- Load API Key and Configure Gemini Client ---
# ... (remains the same) ...
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error(...); st.stop()
try:
    genai.configure(...)
    # ... (Generation Configs remain the same) ...
    filename_gen_config = ...
    notes_gen_config = ...
    transcription_gen_config = ... # Still useful for single calls
    refinement_chunk_gen_config = ...
    safety_settings = ...
except Exception as e: st.error(...); st.stop()

# --- Initialize Session State ---
default_state = {
    # ... (Existing state variables) ...
    'chunking_occurred_transcription': False, # NEW flag
    'chunking_occurred_refinement': False,
}
# ... (Initialize state loop) ...
for key, value in default_state.items(): ...

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream): ...
def update_topic_template(): ...
def create_expert_meeting_prompt(transcript, context=None): ...
def create_earnings_call_prompt(transcript, user_topics_text=None, context=None): ...
def create_docx(text): ...
def get_current_input_data(): ...
def get_prompt_display_text(): ...
def clear_all_state():
    # ... (Existing resets) ...
    st.session_state.chunking_occurred_transcription = False # Reset NEW flag
    st.session_state.chunking_occurred_refinement = False
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type): ...
def add_to_history(notes): ...
def restore_note_from_history(index): ...
def chunk_and_refine_transcript(raw_transcript, refinement_model, general_context, status): ... # (Keep this function)

# --- NEW: Audio Transcription Helper with Chunking ---
def transcribe_audio_possibly_chunked(audio_file_obj, transcription_model, status):
    """
    Transcribes audio. Chunks input audio if duration exceeds threshold.
    Returns concatenated transcript and a flag indicating if chunking occurred.
    Requires pydub and ffmpeg.
    """
    MAX_CHUNK_DURATION_MINS = 30 # Target duration for each audio chunk
    # Threshold slightly higher than chunk duration to avoid chunking for borderline cases
    DURATION_THRESHOLD_MINS = 35
    MAX_POLLING_TIME_SEC = 600 # Max time to wait for one chunk upload/process
    POLLING_INTERVAL_SEC = 10

    transcript_chunks_text = []
    gemini_file_references = [] # To store cloud refs for cleanup
    chunking_performed = False
    original_filename = audio_file_obj.name

    status.update(label="🔊 Analyzing audio file...")
    try:
        # Load audio using pydub - requires ffmpeg
        audio_file_obj.seek(0) # Ensure stream position is at the beginning
        # Determine format (important for pydub)
        file_extension = os.path.splitext(original_filename)[1].lower().replace('.', '')
        if not file_extension:
             # Try common formats if no extension
             try: audio = AudioSegment.from_file(io.BytesIO(audio_file_obj.getvalue()), format="mp3")
             except:
                  try: audio = AudioSegment.from_file(io.BytesIO(audio_file_obj.getvalue()), format="wav")
                  except:
                       try: audio = AudioSegment.from_file(io.BytesIO(audio_file_obj.getvalue()), format="m4a")
                       except: raise ValueError("Could not determine audio format. Please use standard extensions (mp3, wav, m4a, etc.).")
        else:
             audio = AudioSegment.from_file(io.BytesIO(audio_file_obj.getvalue()), format=file_extension)

        duration_ms = len(audio)
        duration_mins = duration_ms / (1000 * 60)
        status.update(label=f"🔊 Audio duration: {duration_mins:.1f} minutes.")

        if duration_mins > DURATION_THRESHOLD_MINS:
            chunking_performed = True
            chunk_duration_ms = MAX_CHUNK_DURATION_MINS * 60 * 1000
            num_chunks = math.ceil(duration_ms / chunk_duration_ms)
            st.warning(f"⚠️ Audio duration ({duration_mins:.1f} min) exceeds threshold. Splitting into {num_chunks} chunks for transcription.", icon="❗")
            status.update(label=f"🔊 Splitting audio into {num_chunks} chunks...")

            for i in range(num_chunks):
                start_ms = i * chunk_duration_ms
                end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
                audio_chunk = audio[start_ms:end_ms]
                status.update(label=f"☁️ Uploading audio chunk {i+1}/{num_chunks}...")

                # Export chunk to a temporary file (use wav for broad compatibility)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                    try:
                        audio_chunk.export(temp_chunk_file.name, format="wav")
                        temp_chunk_path = temp_chunk_file.name

                        # Upload the temporary chunk file
                        processed_chunk_ref = genai.upload_file(
                            path=temp_chunk_path,
                            display_name=f"chunk_{i+1}_{int(time.time())}_{original_filename}"
                        )
                        gemini_file_references.append(processed_chunk_ref) # Add ref for later use & cleanup

                        # Poll for upload completion
                        polling_start = time.time()
                        while processed_chunk_ref.state.name == "PROCESSING":
                           if time.time() - polling_start > MAX_POLLING_TIME_SEC: raise TimeoutError(f"Audio chunk {i+1} upload timed out.")
                           time.sleep(POLLING_INTERVAL_SEC); processed_chunk_ref = genai.get_file(processed_chunk_ref.name)
                        if processed_chunk_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {i+1} processing failed. State: {processed_chunk_ref.state.name}")
                        status.update(label=f"☁️ Audio chunk {i+1}/{num_chunks} ready.")

                    finally:
                        # Ensure local temp file is deleted
                        if temp_chunk_path and os.path.exists(temp_chunk_path):
                            os.remove(temp_chunk_path)

            # --- Transcription Loop (after all chunks uploaded) ---
            status.update(label=f"✍️ Transcribing {num_chunks} audio chunks...")
            for i, file_ref in enumerate(gemini_file_references):
                status.update(label=f"✍️ Transcribing chunk {i+1}/{num_chunks}...")
                transcript_text = ""
                try:
                    t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                    # Use transcription_gen_config (with output limit per chunk)
                    t_response = transcription_model.generate_content(
                        [t_prompt, file_ref],
                        generation_config=transcription_gen_config
                    )
                    if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                        transcript_text = t_response.text.strip()
                        transcript_chunks_text.append(transcript_text)
                    elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                        st.warning(f"⚠️ Transcription blocked for chunk {i+1}: {t_response.prompt_feedback.block_reason}", icon="⚠️")
                        transcript_chunks_text.append(f"\n\n--- TRANSCRIPTION FAILED/BLOCKED FOR CHUNK {i+1} ---\n\n")
                    else:
                        st.warning(f"🤔 Transcription returned empty for chunk {i+1}", icon="⚠️")
                        transcript_chunks_text.append(f"\n\n--- TRANSCRIPTION FAILED FOR CHUNK {i+1} ---\n\n")
                except Exception as trans_chunk_err:
                    st.warning(f"❌ Error transcribing chunk {i+1}: {trans_chunk_err}", icon="⚠️")
                    transcript_chunks_text.append(f"\n\n--- TRANSCRIPTION ERROR FOR CHUNK {i+1} ---\n\n")
                finally:
                    # --- CRITICAL: Clean up cloud file immediately after use ---
                    try:
                        status.update(label=f"🗑️ Cleaning up cloud file for chunk {i+1}...")
                        genai.delete_file(file_ref.name)
                        # Remove from list to avoid double deletion attempt if outer finally runs
                        # Be cautious if modifying list while iterating - maybe copy list first
                    except Exception as cleanup_err:
                         st.warning(f"⚠️ Failed to delete cloud file for chunk {i+1}: {cleanup_err}", icon="❗")

            status.update(label="✍️ Finished transcribing all chunks.")


        else:
             # Audio is short enough, process as single file
             status.update(label=f"☁️ Uploading audio file '{original_filename}'...")
             processed_audio_file_ref = None # Define for finally block
             try:
                  with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_filename)[1]) as tf:
                       audio_file_obj.seek(0)
                       tf.write(audio_file_obj.getvalue()); temp_file_path = tf.name
                  if not temp_file_path: raise Exception("Failed to create temporary file for audio.")
                  processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{original_filename}")
                  gemini_file_references.append(processed_audio_file_ref) # Add the single ref for cleanup

                  # Poll for upload completion
                  status.update(label="🎧 Processing uploaded audio...")
                  polling_start = time.time()
                  while processed_audio_file_ref.state.name == "PROCESSING":
                       if time.time() - polling_start > MAX_POLLING_TIME_SEC: raise TimeoutError("Audio processing timed out.")
                       time.sleep(POLLING_INTERVAL_SEC); processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                  if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio file processing failed. Final state: {processed_audio_file_ref.state.name}")
                  status.update(label="🎧 Audio ready!")

                  # Transcribe the single file
                  status.update(label="✍️ Step 1: Transcribing audio (single pass)...")
                  t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                  t_response = transcription_model.generate_content(
                       [t_prompt, processed_audio_file_ref],
                       generation_config=transcription_gen_config
                  )
                  if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                       transcript_chunks_text.append(t_response.text.strip())
                       status.update(label="✍️ Step 1: Transcription complete!")
                  elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                       raise Exception(f"Transcription blocked: {t_response.prompt_feedback.block_reason}")
                  else: raise Exception("Transcription failed: AI returned empty response.")

             finally:
                 # Clean up local temp file
                 if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
                     os.remove(temp_file_path)
                 # Clean up the single cloud file immediately after transcription attempt
                 if processed_audio_file_ref:
                     try:
                         status.update(label=f"🗑️ Cleaning up cloud file...")
                         genai.delete_file(processed_audio_file_ref.name)
                     except Exception as cleanup_err:
                         st.warning(f"⚠️ Failed to delete cloud file: {cleanup_err}", icon="❗")


        # Combine transcript chunks
        final_transcript = "\n\n".join(transcript_chunks_text)
        return final_transcript, chunking_performed

    except ImportError:
         st.error("Audio processing requires 'pydub'. Please install it (`pip install pydub`).", icon="🚨")
         raise # Re-raise to stop processing
    except FileNotFoundError:
         st.error("Audio processing requires 'ffmpeg'. Please ensure it's installed and accessible in your system PATH.", icon="🚨")
         raise # Re-raise to stop processing
    except Exception as audio_err:
        st.error(f"Error during audio processing/transcription: {audio_err}", icon="🚨")
        raise # Re-raise to stop processing

# --- Streamlit App UI ---
# ... (UI Definition remains the same) ...
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
# ... Input Section ...
# ... Prompt Area ...
# ... Generate Button ...

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    # (Output display logic remains the same, but add transcription chunk warning)
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        # Display intermediate transcripts + warnings
        if st.session_state.raw_transcript:
            with st.expander("View Raw Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
                # Add warning if transcription chunking happened
                if 'chunking_occurred_transcription' in st.session_state and st.session_state.chunking_occurred_transcription:
                     st.warning("Note: Transcription was performed in chunks due to audio length. Minor discontinuities might exist.", icon="⚠️")
        if st.session_state.refined_transcript:
             with st.expander("View Refined Transcript (Step 2 Output)"):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)
                if 'chunking_occurred_refinement' in st.session_state and st.session_state.chunking_occurred_refinement:
                    st.warning("Note: Refinement was performed in chunks due to transcript length. Speaker labels/context may be inconsistent.", icon="⚠️")
        # ... (Notes display, edit, download buttons) ...
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        # ... (rest of notes display/download)

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
    st.session_state.chunking_occurred_transcription = False # Reset flag
    st.session_state.chunking_occurred_refinement = False
    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    # Use a single list to track ALL cloud files for final cleanup attempt
    cloud_files_to_cleanup = []
    with st.status("🚀 Initializing process...", expanded=True) as status:
        try: # Outer try-finally
            status.update(label="⚙️ Reading inputs and settings...")
            # ... (Retrieve state) ...
            st.session_state.chunking_occurred_transcription = False # Reset flags at start
            st.session_state.chunking_occurred_refinement = False

            status.update(label="🧠 Initializing AI models...")
            # ... (Initialize models) ...
            transcription_model = genai.GenerativeModel(...)
            refinement_model = genai.GenerativeModel(...)
            notes_model = genai.GenerativeModel(...)


            final_transcript_for_notes = transcript_data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            status.update(label="✔️ Validating inputs...")
            # ... (Input Validation) ...

            # ==============================================
            # --- STEP 1: Transcription (Possibly Chunked) ---
            # ==============================================
            if actual_input_type == "Upload Audio":
                 # Call the new helper function
                 raw_text_result, chunking_happened_trans = transcribe_audio_possibly_chunked(
                     audio_file_obj,
                     transcription_model,
                     status # Pass status object
                 )
                 st.session_state.raw_transcript = raw_text_result
                 st.session_state.chunking_occurred_transcription = chunking_happened_trans
                 final_transcript_for_notes = st.session_state.raw_transcript
                 # Note: Cloud file cleanup for transcription chunks now happens *inside* the helper function

            # ==============================================
            # --- STEP 2: Refinement (Possibly Chunked) ---
            # ==============================================
            if final_transcript_for_notes: # Check if we have a transcript (from text, PDF, or Step 1)
                 # Refinement needs the raw transcript regardless of input type if available
                 transcript_to_refine = st.session_state.raw_transcript if st.session_state.raw_transcript else final_transcript_for_notes

                 if transcript_to_refine: # Check if there's actually text to refine
                      # Call the refinement chunking helper
                      refined_text_result, chunking_happened_ref = chunk_and_refine_transcript(
                          transcript_to_refine,
                          refinement_model,
                          general_context,
                          status
                      )
                      st.session_state.refined_transcript = refined_text_result
                      st.session_state.chunking_occurred_refinement = chunking_happened_ref
                      final_transcript_for_notes = st.session_state.refined_transcript # Update for notes step
                 else:
                      status.update(label="⚠️ Step 2: Skipped Refinement (No transcript text available).")


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
                )
                # ... (Handle Notes Response) ...
                if response and hasattr(response, 'text') and response.text.strip(): ... # Success
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: ... # Block
                elif response: ... # Empty
                else: ... # No response
                status.update(label="✅ Notes generated successfully!", state="complete") # Or error state

            except Exception as api_call_err: ... # Handle API call error


        except Exception as e: # Catch errors from validation, audio, steps 1/2 etc.
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        finally: # Runs after try/except within the 'with status' block
            st.session_state.processing = False
            # --- Cloud Audio Cleanup (Attempt any *remaining* - should be none if helper funcs worked) ---
            # The helper functions should handle their own cleanup. This is a final safeguard.
            # Copy list before iterating if modifying it inside the loop
            remaining_files = st.session_state.get('uploaded_audio_info', []) # Assuming this is populated correctly now
            if isinstance(remaining_files, genai.types.File): # Handle single file case
                 remaining_files = [remaining_files]
            elif not isinstance(remaining_files, list):
                 remaining_files = []

            if remaining_files:
                st.toast(f"☁️ Performing final cleanup check for {len(remaining_files)} cloud file(s)...", icon="🗑️")
                for file_ref in remaining_files:
                     if file_ref and hasattr(file_ref, 'name'):
                          try: genai.delete_file(file_ref.name)
                          except Exception as final_cleanup_error: st.warning(f"Final cloud file cleanup failed for {file_ref.name}: {final_cleanup_error}", icon="⚠️")
                st.session_state.uploaded_audio_info = None # Clear the list/ref

            st.rerun()


# --- Footer ---
# ... (remains the same) ...
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

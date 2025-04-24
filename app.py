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
import math # For ceiling function in chunking estimate

# --- Page Configuration ---
st.set_page_config(
    page_title="SynthNotes AI ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
# (CSS remains the same - omitted for brevity)
st.markdown(""" <style> ... </style> """, unsafe_allow_html=True)


# --- Define Available Models & Meeting Types ---
# (Model definitions remain the same)
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
# (Sector definitions remain the same)
SECTOR_OPTIONS = ["Other / Manual Topics", "IT Services", "QSR"]
DEFAULT_SECTOR = SECTOR_OPTIONS[0]
SECTOR_TOPICS = { "IT Services": "...", "QSR": "..." } # Keep topics definition

# --- Load API Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨"); st.stop()
try:
    genai.configure(api_key=API_KEY)
    # Base Generation Configs (can be overridden)
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    # Config for notes step - NO max_output_tokens for potentially higher capacity model
    notes_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "response_mime_type": "text/plain"}
    # Config for transcription - keep output limit as text unlikely to exceed this unless audio is extreme
    transcription_gen_config = {"temperature": 0.1, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    # Config for refinement - Explicitly limit output PER CHUNK
    refinement_chunk_gen_config = {"temperature": 0.3, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
# (Session state definition remains the same)
default_state = { ... } # Keep state definition
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    # (No changes needed)
    try: ...
    except Exception as e: ...

def update_topic_template():
    # (No changes needed)
    selected_sector = st.session_state.selected_sector
    if selected_sector in SECTOR_TOPICS: ...
    else: ...

# --- Prompts ---
def create_expert_meeting_prompt(transcript, context=None):
    # (No changes needed)
    core_prompt = """..."""
    final_prompt_elements = [...]
    return "\n".join(final_prompt_elements)

def create_earnings_call_prompt(transcript, user_topics_text=None, context=None):
    # (No changes needed)
    topic_instructions = ""
    if user_topics_text and user_topics_text.strip(): ...
    else: ...
    prompt_parts = [...]
    return "\n".join(filter(None, prompt_parts))

def create_docx(text):
    # (No changes needed)
    document = docx.Document(); ...; return buffer.getvalue()

def get_current_input_data():
    # (No changes needed)
    input_type = st.session_state.input_method_radio
    ...
    return input_type, transcript, audio_file

def get_prompt_display_text():
    # (No changes needed)
    ...
    return display_text

def clear_all_state():
    # (No changes needed)
    ...
    st.toast("Inputs and outputs cleared!", icon="🧹")

def generate_suggested_filename(notes_content, meeting_type):
    # (No changes needed)
    ...
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    # (No changes needed)
    ...

def restore_note_from_history(index):
    # (No changes needed)
    ...

# --- NEW: Chunking and Refinement Helper Function ---
def chunk_and_refine_transcript(raw_transcript, refinement_model, general_context, status):
    """
    Splits the raw transcript and refines it chunk by chunk if it's too long.
    Warns about potential context loss. Returns the concatenated refined transcript.
    """
    MAX_INPUT_CHUNK_CHARS = 35000 # Heuristic: Aim for inputs likely producing <8k output tokens
    MIN_TRANSCRIPT_LEN_FOR_CHUNK = 40000 # Only chunk if transcript is reasonably long
    refined_outputs = []
    chunking_performed = False

    if len(raw_transcript) > MIN_TRANSCRIPT_LEN_FOR_CHUNK:
        chunking_performed = True
        st.warning(
            "⚠️ Raw transcript is long. Performing refinement in chunks. "
            "Speaker labels and context may be inconsistent across chunks.", icon="❗"
        )
        status.update(label="🧹 Step 2: Transcript long, preparing chunks...")

        # Split by paragraph first, then handle segments
        segments = raw_transcript.split('\n\n')
        current_chunk_segments = []
        current_chunk_len = 0
        input_chunks = []

        for segment in segments:
            segment_len = len(segment)
            # Check if adding the next segment would exceed the limit
            if current_chunk_len > 0 and (current_chunk_len + segment_len + 2) > MAX_INPUT_CHUNK_CHARS:
                # Finalize the current chunk
                input_chunks.append("\n\n".join(current_chunk_segments))
                # Start a new chunk
                current_chunk_segments = [segment]
                current_chunk_len = segment_len
            else:
                # Add segment to the current chunk
                current_chunk_segments.append(segment)
                current_chunk_len += segment_len + 2 # Add 2 for the potential "\n\n"

        # Add the last chunk
        if current_chunk_segments:
            input_chunks.append("\n\n".join(current_chunk_segments))

        num_chunks = len(input_chunks)
        status.update(label=f"🧹 Step 2: Refining transcript in {num_chunks} chunks...")

        for i, chunk in enumerate(input_chunks):
            status.update(label=f"🧹 Step 2: Refining chunk {i+1}/{num_chunks}...")
            refinement_prompt = f"""Please refine the following raw audio transcript chunk: ... [Refinement Prompt Content] ... **Refined Transcript:** """ # Keep prompt content
            refinement_prompt = f"""Please refine the following raw audio transcript chunk:

            **Raw Transcript Chunk:**
            ```
            {chunk}
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
            """ # Slightly modified prompt for chunk awareness

            try:
                r_response = refinement_model.generate_content(
                    refinement_prompt,
                    generation_config=refinement_chunk_gen_config, # Use config WITH output limit
                    safety_settings=safety_settings
                )
                if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                    refined_outputs.append(r_response.text.strip())
                elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                    st.warning(f"⚠️ Refinement blocked for chunk {i+1}: {r_response.prompt_feedback.block_reason}. Using raw chunk.", icon="⚠️")
                    refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Failed/Blocked) ---\n{chunk}") # Fallback
                else:
                    st.warning(f"🤔 Refinement returned empty response for chunk {i+1}. Using raw chunk.", icon="⚠️")
                    refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Failed) ---\n{chunk}") # Fallback
            except Exception as refine_chunk_err:
                 st.warning(f"❌ Error refining chunk {i+1}: {refine_chunk_err}. Using raw chunk.", icon="⚠️")
                 refined_outputs.append(f"--- RAW CHUNK {i+1} (Refinement Error) ---\n{chunk}") # Fallback

        # Combine refined chunks
        final_refined_text = "\n\n".join(refined_outputs)
        status.update(label="🧹 Step 2: Finished refining all chunks.")
        return final_refined_text, chunking_performed

    else:
        # Transcript is short enough, refine in one go
        status.update(label=f"🧹 Step 2: Refining transcript (single pass)...")
        refinement_prompt = f"""Please refine the following raw audio transcript: ... [Original Full Refinement Prompt Content] ... **Refined Transcript:** """ # Keep prompt content
        refinement_prompt = f"""Please refine the following raw audio transcript:

        **Raw Transcript:**
        ```
        {raw_transcript}
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
                generation_config=refinement_chunk_gen_config, # Still use config with output limit for safety
                safety_settings=safety_settings
            )
            if r_response and hasattr(r_response, 'text') and r_response.text.strip():
                status.update(label="🧹 Step 2: Refinement complete!")
                return r_response.text.strip(), chunking_performed # Return text and False for chunking_performed
            elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                st.warning(f"⚠️ Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript.", icon="⚠️")
                status.update(label="⚠️ Refinement blocked. Using raw transcript.")
                return raw_transcript, chunking_performed # Fallback
            else:
                st.warning("🤔 Refinement returned empty response. Using raw transcript.", icon="⚠️")
                status.update(label="⚠️ Refinement failed. Using raw transcript.")
                return raw_transcript, chunking_performed # Fallback
        except Exception as refine_err:
            st.warning(f"❌ Error during Step 2 (Refinement): {refine_err}. Using raw transcript.", icon="⚠️")
            status.update(label="⚠️ Refinement error. Using raw transcript.")
            return raw_transcript, chunking_performed # Fallback


# --- Streamlit App UI ---
# (UI definition remains largely the same, ensure model selectors are correct)
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            st.radio("Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True)
        with col1b:
            st.subheader("AI Model Selection")
            st.selectbox("Notes Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_notes_model_display_name", help="Model for final note generation (Step 3).")
            st.selectbox("Transcription Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_transcription_model_display_name", help="Model for Audio Transcription (Step 1).")
            st.selectbox("Refinement Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_refinement_model_display_name", help="Model for Audio Refinement (Step 2).")

    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

    st.divider(); st.subheader("Source Input")
    # ... (Rest of Input UI: radio, file uploaders) ...
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed")
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")


    st.divider(); col3a, col3b = st.columns(2); selected_mt = st.session_state.selected_meeting_type
    # ... (Rest of Input UI: Topics, Context, View/Edit Prompt Checkbox) ...
    with col3a: # Topics / Context
        if selected_mt == "Earnings Call":
            st.selectbox("Select Sector (for Topic Template):", options=SECTOR_OPTIONS, key="selected_sector", on_change=update_topic_template)
            st.text_area("Earnings Call Topics (Edit below):", key="earnings_call_topics", height=150, placeholder="Enter topics manually or edit loaded template...")
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.add_context_enabled: st.text_area("Context Details:", height=75, key="context_input", placeholder="Company Name, Ticker...")
    with col3b: # View/Edit Prompt Checkbox
        if selected_mt != "Custom": st.checkbox("View/Edit Final Notes Prompt", key="view_edit_prompt_enabled")


# Prompt Area (Conditional)
# (No changes needed)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and selected_mt != "Custom") or (selected_mt == "Custom")
if show_prompt_area:
    # ... (Prompt Area UI) ...
    with st.container(border=True):
        prompt_title = "Final Notes Prompt Preview/Editor" if selected_mt != "Custom" else "Custom Final Notes Prompt (Required)"
        st.subheader(prompt_title)
        display_prompt_text = get_prompt_display_text()
        prompt_value = st.session_state.current_prompt_text if st.session_state.current_prompt_text else display_prompt_text
        st.text_area(label="Prompt Text:", value=prompt_value, key="current_prompt_text", height=350, label_visibility="collapsed", help="This prompt is used for the *final* step of generating notes.")


# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# Output Section
output_container = st.container(border=True)
with output_container:
    # (Output display logic remains the same)
    if st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        # ... (Display intermediate transcripts, notes, edit checkbox, download buttons) ...
        if st.session_state.raw_transcript:
            with st.expander("View Raw Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True)
        if st.session_state.refined_transcript:
             with st.expander("View Refined Transcript (Step 2 Output)"):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True)
                # Add warning if chunking happened during refinement
                if 'chunking_occurred_refinement' in st.session_state and st.session_state.chunking_occurred_refinement:
                    st.warning("Note: Refinement was performed in chunks due to transcript length. Speaker labels/context may be inconsistent.", icon="⚠️")

        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_to_use)

        st.write("") # Spacer
        dl_cols = st.columns(3)
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"
        fname_base = st.session_state.suggested_filename or default_fname
        with dl_cols[0]: st.download_button(label="⬇️ Notes (.txt)",data=notes_content_to_use,file_name=f"{fname_base}.txt",mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]: st.download_button(label="⬇️ Notes (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            if st.session_state.refined_transcript:
                refined_fname_base = fname_base.replace("_notes", "_refined_transcript") if "_notes" in fname_base else f"{fname_base}_refined_transcript"
                st.download_button(label="⬇️ Refined Tx (.txt)", data=st.session_state.refined_transcript, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-txt', use_container_width=True, help="Download the speaker-diarized and corrected transcript from Step 2.")
            else: st.button("Refined Tx N/A", disabled=True, use_container_width=True, help="Refined transcript is only available after successful audio processing.")


    elif not st.session_state.processing:
        st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
# (History display logic remains the same)
with st.expander("📜 Recent Notes History (Last 3)", expanded=False): ...

# --- Processing Logic ---
if generate_button:
    # Reset state before starting
    st.session_state.processing = True
    # ... (reset other state variables) ...
    st.session_state.generating_filename = False
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.suggested_filename = None
    st.session_state.raw_transcript = None
    st.session_state.refined_transcript = None
    st.session_state.chunking_occurred_refinement = False # Add state to track if chunking happened

    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    processed_audio_file_ref = None
    with st.status("🚀 Initializing process...", expanded=True) as status:
        try: # Outer try-finally
            status.update(label="⚙️ Reading inputs and settings...")
            # ... (Retrieve state: meeting_type, model names, context, topics, etc.) ...
            meeting_type = st.session_state.selected_meeting_type
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            user_prompt_text = st.session_state.current_prompt_text
            general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
            earnings_call_topics_text = st.session_state.earnings_call_topics.strip() if meeting_type == "Earnings Call" else ""
            actual_input_type, transcript_data, audio_file_obj = get_current_input_data()
            st.session_state.chunking_occurred_refinement = False # Reset chunking flag


            status.update(label="🧠 Initializing AI models...")
            # Initialize models (safety settings included here)
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            final_transcript_for_notes = transcript_data # Start with text/PDF data
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None

            status.update(label="✔️ Validating inputs...")
            # ... (Input Validation) ...
            if actual_input_type == "Paste Text" and not final_transcript_for_notes: raise ValueError("Text input is empty.")
            elif actual_input_type == "Upload PDF" and not final_transcript_for_notes: raise ValueError("PDF processing failed or returned empty text.")
            elif actual_input_type == "Upload Audio" and not audio_file_obj: raise ValueError("No audio file uploaded.")
            if meeting_type == "Custom" and not user_prompt_text.strip(): raise ValueError("Custom Prompt is required but empty.")


            # ==============================================
            # --- STEP 1 & 2: Audio Processing Pipeline ---
            # ==============================================
            if actual_input_type == "Upload Audio":
                # --- Upload Audio ---
                status.update(label=f"☁️ Uploading '{audio_file_obj.name}'...")
                # ... (Audio Upload logic using tempfile) ...
                try:
                    # ... tempfile logic ...
                    processed_audio_file_ref = genai.upload_file(...)
                    st.session_state.uploaded_audio_info = processed_audio_file_ref
                finally:
                    # ... tempfile cleanup ...
                    pass

                status.update(label="🎧 Processing uploaded audio...")
                # ... (Audio Polling logic) ...
                while processed_audio_file_ref.state.name == "PROCESSING": ...
                if processed_audio_file_ref.state.name != "ACTIVE": ...
                status.update(label="🎧 Audio ready!")


                # --- Step 1: Transcription ---
                try:
                    status.update(label=f"✍️ Step 1: Transcribing audio...")
                    t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                    # Use transcription_gen_config (which includes 8192 limit for now)
                    t_response = transcription_model.generate_content(
                        [t_prompt, processed_audio_file_ref],
                        generation_config=transcription_gen_config
                    )
                    if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                        st.session_state.raw_transcript = t_response.text.strip()
                        status.update(label="✍️ Step 1: Transcription complete!")
                        final_transcript_for_notes = st.session_state.raw_transcript
                    # ... (Handle transcription errors/blocks) ...
                    elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason: ...
                    else: raise Exception("Transcription failed: AI returned empty response.")
                except Exception as trans_err: ...


                # --- Step 2: Refinement (Potentially Chunked) ---
                if st.session_state.raw_transcript:
                     # Call the helper function
                     refined_text_result, chunking_happened = chunk_and_refine_transcript(
                         st.session_state.raw_transcript,
                         refinement_model,
                         general_context,
                         status # Pass status object for updates within the function
                     )
                     st.session_state.refined_transcript = refined_text_result
                     st.session_state.chunking_occurred_refinement = chunking_happened # Store if chunking occurred
                     final_transcript_for_notes = st.session_state.refined_transcript # Use refined (or raw fallback from func)
                else:
                    status.update(label="⚠️ Step 2: Skipped Refinement (No raw transcript).")
                    final_transcript_for_notes = None # Or handle appropriately

            # =============================
            # --- STEP 3: Generate Notes ---
            # =============================
            if not final_transcript_for_notes:
                 # Handle case where transcript is missing (e.g., audio failed, text empty)
                 if actual_input_type == "Upload Audio" and not st.session_state.raw_transcript:
                      raise ValueError("Transcription failed, cannot generate notes.")
                 else: # Text/PDF input was empty
                      raise ValueError("No transcript available to generate notes.")

            status.update(label="📝 Preparing final prompt...")
            # ... (Determine final_prompt_for_api and api_payload_parts based on meeting type) ...
            final_prompt_for_api = None
            api_payload_parts = []
            if meeting_type == "Custom": ...
            elif meeting_type == "Expert Meeting": ...
            elif meeting_type == "Earnings Call": ...
            else: raise ValueError(...)

            if not final_prompt_for_api: raise ValueError(...)

            # --- Generate Notes API Call ---
            try:
                status.update(label=f"✨ Step 3: Generating notes...")
                # Use notes_gen_config (WITHOUT max_output_tokens)
                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=notes_gen_config
                )
                # ... (Handle Notes Response: success, block, empty, add to history, suggest filename) ...
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    st.session_state.generated_notes = response.text.strip()
                    # ... rest of success logic ...
                    add_to_history(...)
                    st.session_state.suggested_filename = generate_suggested_filename(...)
                    status.update(label="✅ Notes generated successfully!", state="complete")
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: ... # Handle block
                elif response: ... # Handle empty
                else: ... # Handle no response

            except Exception as api_call_err: ... # Handle API call error


        except Exception as e: # Catch errors from validation, audio, steps 1/2 etc.
            st.session_state.error_message = f"❌ Processing Error: {e}"
            status.update(label=f"❌ Error: {e}", state="error")

        finally: # Runs after try/except within the 'with status' block
            st.session_state.processing = False
            # --- Cloud Audio Cleanup (use toast as status is finished) ---
            if st.session_state.uploaded_audio_info:
                try:
                    st.toast("☁️ Cleaning up uploaded audio...", icon="🗑️")
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
                except Exception as final_cleanup_error:
                    st.warning(f"Final cloud audio cleanup failed: {final_cleanup_error}", icon="⚠️")

            # Rerun happens automatically when exiting 'with status' if state changed triggering it,
            # but explicit rerun ensures UI update after cleanup attempt if needed.
            st.rerun()


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

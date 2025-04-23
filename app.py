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
    div[data-enabled", on_change=update_prompt_display_text)
        # Correct indentation for the text_area below
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=100, key="context_input", on_change=update_prompt_display_text, placeholder="Attendees...")
    with col3b: # View/Edit Prompt Checkbox
        # Correct indentation for the checkbox below
        if st.session_state.selected_meeting_type != "Custom":
            st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled", on_change=update_prompt_display_text)

# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
                   (st.session_state.selected_meeting_type == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_title); caption = "Edit prompt..." if st.session_state.selected_meeting_type != "Custom" else "Enter prompt..."
        st.caption(caption); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: status_message = st.session_state.processing_step or "⏳ Processing..."; st.info(status_message, icon="🧠")
    elif st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_to_use)
        default_fname = f"{st.session_state.selected_meeting_type.lower().replace(' ', '_')}_notes"; fname_base = st.session_state.suggested_filename or default_fname
        st.write(""); col_btn_dl1, col_btn_dl2 = st.columns(2)
        with col_btn_dl1: st.download_button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with col_btn_dl2: st.download_button(label="⬇️ Markdown", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html=True)

# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**"); st.markdown(f"```\n{entry['notes'][:200]}...\n```")
                st.button(f"View/Use Notes #{i+1}", key=f"restore_{imt_cleaned}. Output ONLY filename.\n\nNOTES:\n{notes_content[:1500]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if re.match(r"\d{8}_[\w\testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"][style*="border"] {
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
    .stButton > button[kind="primary"] { background-color: #007AFF; color: white; box-shadow: 0 4px 6px rgba(0, 122, 255, 0.1), 0 1px 3px rgba(0,}", on_click=restore_note_from_history, args=(i,))
                if i < len(st.session_state.history) - 1: st.divider()


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generating_filename = False; st.session_state.processing_step = None
    st.session_state.generated_notes = None; st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None; st.session_state.suggested_filename = None
    st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    try: # Outer try
        # Retrieve state & inputs
        meeting_type = st.session_state.selected_meeting_type
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled = st.session_state.view_edit_prompt_enabled
        user_prompt_text = st.session_state.current_prompt_text
        final_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # Validation
        if actual_input_type == "Paste Text" and not transcript_data: st.session_state.error_message = "⚠️ Text area empty."
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_message = "⚠️ PDF error."
        elif actual_input_type == "Upload Audio" and not audio_file_obj: st.session_state.error_message = "⚠️ No audio uploaded."
        -\.]+_\w+", s_name): st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename suggestion '{s_name}' bad format.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False

def add_to_history(notes):
    if not notes: return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"timestamp": timestamp, "notes": notes}
    current_history = st.session_state.get('history', [])
    current_history.insert(0, new_entry)
    st.session_state.history = current_history[:3]

def restore_note_from_history(index):
    if 0 <= index < len(st.session_state.history):
        entry = st.session_state.history[index]
        st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None
        st.session_state.error_message = None; st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a: st.subheader("Meeting Details"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True, on_change=update_prompt_display_text)
        with col1b: st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary")
    st.divider(); st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed", on_change=update_prompt_display_text)
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")
    st.divider(); col3a, col3b = st.columns(2) # Optional Elements
    with col3a: # Context
        st.checkbox("Add Context", key="add_context_enabled", on_change=update_prompt_display_text)
        if st.session_state.add_context_enabled:
             # Use proper indentation for the conditional element
             st.text_area("Context Details:", height 0, 0, 0.08); }
    .stButton > button[kind="primary"]:hover { background-color: #005ECB; box-shadow: 0 7px 14px rgba(0, 122, 255, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08); transform: translateY(-1px); }
    .stButton > button[kind="primary"]:focus { box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.4); outline: none; }
    .stButton > button:disabled, .stButton > button[kind="primary"]:disabled { background-color: #D1D5DB; color: #6B7280; box-shadow: none; transform: none; cursor: not-allowed; }
     /* Secondary Button styling for Clear */
    .stButton>button[type="secondary"], .stButton>button.secondary-button { background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB;
        width: auto; padding: 0.5rem 1rem; margin-right: 0.5rem; font-weight: 500; }
    .stButton>button[type="secondary"]:hover, .stButton>button.secondary-button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
     /* Download Buttons */
    .stDownloadButton > button { border-radius: 0.5rem; padding: 0.6rem 1.2rem; font-weight: 500; background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB; transition: background-color 0.2s ease-in-out; widthif meeting_type == "Custom" and not user_prompt_text.strip(): st.session_state.error_message = "⚠️ Custom Prompt empty."

        # Determine Final Prompt / Transcript for API
        final_prompt_for_api = None
        api_payload_parts = [] # Use a list to build the payload
        processed_audio_file_ref = None # Cloud reference
        prompt_was_edited_or_custom = (meeting_type == "Custom" or view_edit_enabled)

        if not st.session_state.error_message:
            # --- Inner Try for pre-processing, prompt-building, audio handling ---
            try:
                # --- Handle Audio Upload & Transcription (if applicable) ---
                if actual_input_type == "Upload Audio":
                    if not audio_file_obj: raise ValueError("Audio file missing.")
                    st.session_state.processing_step = f"☁️ Uploading '{audio_file_obj.name}'..."
                    st.rerun() # Show status

                    audio_bytes = audio_file_obj.getvalue(); temp_file_path = None
                    try: # Tempfile handling
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os=100, key="context_input", on_change=update_prompt_display_text, placeholder="Attendees...")
    with col3b: # View/Edit Prompt Checkbox
        if st.session_state.selected_meeting_type != "Custom":
            # Use proper indentation for the conditional element
            st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled", on_change=update_prompt_display_text)

# Prompt Area (Conditional)
show_prompt_area = (st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom") or \
                   (st.session_state.selected_meeting_type == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if st.session_state.selected_meeting_type != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_title); caption = "Edit prompt..." if st.session_state.selected_meeting_type != "Custom" else "Enter prompt..."
        st.caption(caption); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# Generate Button
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: status_message = st.session_state.processing_step or "⏳ Processing: auto; margin-top: 0; margin-right: 0.5rem;} /* Add margin-right */
    .stDownloadButton > button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }
    /* Output Area Styling */
    .output-container { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1.5rem; margin-top: 1.5rem; min-height: 150px; }
    .output-container .stMarkdown { background-color: transparent; border: none; padding: 0; color: #374151; font-size: 1rem; line-height: 1.6; }
    .output-container .stMarkdown h3, .output-container .stMarkdown h4, .output-container .stMarkdown strong { color: #111827; font-weight: 600; }
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem; }
    .output-container .initial-prompt { color: #6B7280; font-style: italic; text-align: center; padding-top: 2rem; }
    /* Prompt.path.splitext(audio_file_obj.name)[1]) as temp_file:
                            temp_file.write(audio_bytes); temp_file_path = temp_file.name
                        if temp_file_path: processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}")
                        else: raise Exception("Temp file creation failed.")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path) # Disk cleanup

..."; st.info(status_message, icon="🧠")
    elif st.session_state.generating_filename: st.info("⏳ Generating filename...", icon="💡")
    elif st.session_state.error_message: st.error(st.session_state.error_message, icon="🚨"); st.session_state.error_message = None
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.checkbox("Edit Notes", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        if st.session_state.edit_notes_enabled: st.text_area("Editable Notes:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else: st.markdown(notes_content_to_use)
        default_fname = f"{st.session Edit Area */
    #prompt-edit-area textarea { font-family: monospace; font-size: 0.9rem; line-height: 1.4; background-color: #FDFDFD; }
    /* History Styling */
    .history-entry { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
    .history-entry:last-child { border-bottom: none; }
    .history-entry pre { background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.25rem; max-height: 150px; overflow-y: auto; }
    /* Footer */
    footer { text-align: center; color: #9CA3AF; font-size: 0.8rem                    st.session_state.uploaded_audio_info = processed_audio_file_ref # Store cloud ref for potential later cleanup
                    # Polling...
                    st.session_state.processing_step = f"🎧 Processing audio_state.selected_meeting_type.lower().replace(' ', '_')}_notes"; fname_base = st.session_state.suggested_filename or default_fname
        st.write(""); col_btn_dl1, col; padding: 2rem 0 1rem 0; }
    footer a { color: #6B7280; text-decoration: none; }
    footer a:hover { color: #007AFF..."
                    st.rerun() # Show status
                    polling_start = time.time()
                    while_btn_dl2 = st.columns(2)
        with col_btn_dl1: st.download; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# processed_audio_file_ref.state.name == "PROCESSING":
                        if time.time() - polling_start > 300: raise TimeoutError("Audio timeout.")
                        time.sleep(5); processed_button(label="⬇️ TXT", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_ --- Define Available Models & Meeting Types (Keep as is) ---
AVAILABLE_MODELS = {
    "Gemini _audio_file_ref = genai.get_file(processed_audio_file_ref.name)width=True)
        with col_btn_dl2: st.download_button(label="⬇️1.5 Flash (Fast & Versatile)": "gemini-1.5-flash", "Gemini 1 # Refresh state
                    if processed_audio_file_ref.state.name != "ACTIVE": raise Exception( Markdown", data=notes_content_to_use, file_name=f"{fname_base}.md",.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume)": "gemini-1.5-flash-8bf"Audio state: {processed_audio_file_ref.state.name}")
                    st.toast( mime="text/markdown", key='download-md', use_container_width=True)
    else: st.markdown("<p class='initial-prompt'>Generated notes will appear here.</p>", unsafe_allow_html", "Gemini 2.0 Flash (Next Gen Speed)": "gemini-2.0-flashf"🎧 Audio ready!", icon="✅")

                    # --- TWO-STEP LOGIC Check: Default Prompt for=True)

# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)",",
    "Gemini 2.0 Flash-Lite (Low Latency)": "gemini-2 Audio? ---
                    if not prompt_was_edited_or_custom:
                        st.session_state.processing_step = f"✍️ Transcribing audio..."
                        st.rerun() # Show status
                        transcription_prompt = expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else.0-flash-lite", "Gemini 2.5 Flash Preview (Adaptive)": "gemini-:
        for i, entry in enumerate(st.session_state.history):
            with st.container "Transcribe the provided audio file accurately. Output only the raw transcribed text."
                        transcription_model = gen2.5-flash-preview-04-17",
    "Gemini 2.5 Pro Exp. Preview (Enhanced Reasoning)": "models/gemini-2.5-pro-exp-03():
                st.markdown(f"**#{i+1} - {entry['timestamp']}**"); st.markdown(ai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings,-25",
}
DEFAULT_MODEL_NAME = "Gemini 2.5 Pro Exp. Previewf"```\n{entry['notes'][:200]}...\n```")
                st.button( generation_config=transcription_gen_config)
                        transcript_response = transcription_model.generate_content (Enhanced Reasoning)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS: DEFAULT_MODEL_NAME =f"View/Use Notes #{i+1}", key=f"restore_{i}", on_click=restore([transcription_prompt, processed_audio_file_ref])

                        if transcript_response and hasattr(transcript_response, "Gemini 1.5 Flash (Fast & Versatile)"
MEETING_TYPES = ["Expert Meeting",_note_from_history, args=(i,))
                if i < len(st.session_state. 'text') and transcript_response.text.strip():
                            transcript_data = transcript_response.text.strip() # "Earnings Call", "Custom"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]

# --- Load APIhistory) - 1: st.divider()


# --- Processing Logic ---
if generate_button:
     OVERWRITE with transcript
                            st.toast("✍️ Transcription complete!", icon="✅")
                            # NOW we Key and Configure Gemini Client ---
load_dotenv(); API_KEY = os.getenv("GEMINI_API_st.session_state.processing = True; st.session_state.generating_filename = False; st.KEY")
if not API_KEY: st.error("### 🔑 API Key Not Found!", icon="🚨 have transcript data, proceed as if text input
                            actual_input_type = "Generated Transcript" # Mark as textsession_state.processing_step = None
    st.session_state.generated_notes = None; st"); st.stop()
try:
    genai.configure(api_key=API_KEY)
.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled =    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 5-based now
                            # No longer need the audio reference for the *notes* call
                            # We can delete the cloud False
    st.session_state.error_message = None; st.session_state.suggested_0, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature file now if desired, or wait until very end
                        else:
                            raise Exception("Failed to transcribe audio.")
                    filename = None; st.rerun()

if st.session_state.processing and not st.session_state.generating": 0.7, "top_p": 1.0, "top_k": 32_filename:
    try: # Outer try
        meeting_type = st.session_state.selected_meeting_type, "max_output_tokens": 8192, "response_mime_type": "text/# --- End Two-Step Logic ---
                # --- End Audio Pre-processing ---

                # --- Determine Final
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_displayplain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime__name]
        view_edit_enabled = st.session_state.view_edit_prompt_enabled Prompt Text (if not already custom) ---
                if not prompt_was_edited_or_custom:
                    type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_
        user_prompt_text = st.session_state.current_prompt_text
        final_contextprompt_function = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else create_earningsMEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_ = st.session_state.context_input.strip() if st.session_state.add_context__call_prompt
                    if transcript_data: # This will be true for Text, PDF, and successfully transcribed Audio
                        CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "Henabled else None
        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # Validation
        if actual_input_type == "Paste Text" and notfinal_prompt_for_api = prompt_function(transcript_data, final_context)
                    else:ARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session transcript_data: st.session_state.error_message = "⚠️ Text area empty."
        elif actual_input_
                         st.session_state.error_message = "Error: Transcript data unavailable for note generation."
                else: # Custom or Edited prompt
                    final_prompt_for_api = user_prompt_text.split("################ State ---
default_state = {
    'processing': False, 'processing_step': None, 'generating_filename':type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st####################\n\n")[-1] # Clean potential note

                # --- Prepare API Payload ---
                if False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info.session_state.error_message = "⚠️ PDF error."
        elif actual_input_type == "Upload Audio" and not audio_file_obj: st.session_state.error_message = "⚠️ No not st.session_state.error_message and final_prompt_for_api:
                    api_payload_parts': None, 'add_context_enabled': False,
    'selected_model_display_name': DEFAULT audio uploaded."
        if meeting_type == "Custom" and not user_prompt_text.strip(): st.session_MODEL_NAME, 'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio.append(final_prompt_for_api) # Add the prompt text first
                    # Add audio object ONLY_state.error_message = "⚠️ Custom Prompt empty."

        # Determine Final Prompt
        final_prompt': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None_for_api = None
        prompt_was_edited_or_custom = (meeting_type == ",
    'context_input': '', 'edit_notes_enabled': False, 'edited_notes_text': "", ' if it's audio input AND it's a custom/edited prompt
                    # (because default audio prompts use the generated transcriptCustom" or view_edit_enabled)
        if not st.session_state.error_message:
            ifsuggested_filename': None,
    'history': [],
}
for key, value in default_state now)
                    if actual_input_type == "Upload Audio" and prompt_was_edited_or_custom prompt_was_edited_or_custom:
                final_prompt_for_api = user_prompt_.items():
    if key not in st.session_state: st.session_state[key] =:
                        if processed_audio_file_ref:
                            api_payload_parts.append(processed_audio_text.split("####################################\n\n")[-1] # Clean note
            else: # Generate value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        pdf_file_stream.seek(0); pdf_reader = PyPDF2.Pdffile_ref)
                        else:
                            st.session_state.error_message = "Error: Audio default prompt (will be handled differently below for audio)
                if actual_input_type != "Upload Audio":
                    if transcript_data:
                        prompt_function = create_expert_meeting_prompt if meeting_typeReader(pdf_file_stream)
        text_parts = [page.extract_text() for page in reference missing for custom audio prompt."
                    elif actual_input_type not in ["Upload Audio", "Generated Transcript", "Paste == "Expert Meeting" else create_earnings_call_prompt
                        final_prompt_for_api = prompt pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts); return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e Text", "Upload PDF"]:
                        st.session_state.error_message = f"Error: Unexpected input type '{_function(transcript_data, final_context)
                    else: st.session_state.error_message: st.session_state.error_message = f"📄 PDF Read Error: {e}"; return None
actual_input_type}' for final payload."

            except Exception as pre_process_err:
                st = "Error: Transcript missing."
                # For default audio, prompt is determined later after transcription

        # API    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error:.session_state.error_message = f"❌ Pre-processing Error: {pre_process_err}"
 Call Section
        if not st.session_state.error_message and (final_prompt_for_api or (actual_input_type == "Upload Audio" and not prompt_was_edited_or_custom)) and ( {e}"; return None

def create_expert_meeting_prompt(transcript, context=None):
    prompt                # Attempt cleanup if audio ref exists from pre-processing failure
                if st.session_state.uploaded_audiotranscript_data or audio_file_obj):
            try: # Inner try for API/processing
                st.session__parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:", "\n**Structure:**",
        "- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda,_info:
                     try: genai.delete_file(st.session_state.uploaded_audio_infostate.processing_step = "🧠 Initializing..."
                st.rerun() # Show initial status

                model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=main_gen_config)
                response = None; api_payload = None; processed_ or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.",
        "- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.",
.name); st.session_state.uploaded_audio_info = None
                     except Exception: pass


            # --- Generate Notes API Call (only if no errors and payload is ready) ---
            if not st.session_audio_file_ref = None; temp_file_path = None
                transcribed_text_for_notes = transcript_data # Use transcript_data initially

                if actual_input_type == "Upload Audio":
                    # Upload & Process Audio File (common for both default and custom audio)
                    if not audio_file_obj: raise ValueError        "  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).",
        "  - **Answers:** Use bullet points directly below the question. Each bullet MUST be a complete sentence with one distinct fact. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or actionstate.error_message and api_payload_parts:
                try:
                    st.session_state.processing_step = f"📝 Generating notes..."
                    st.rerun() # Show status
                    model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=main_gen_config)
                    response = model.generate_content(api_payload_parts)("Audio missing.")
                    st.session_state.processing_step = f"☁️ Uploading '{audio_file_obj.name}'..."
                    st.rerun()
                    audio_bytes = audio_file_obj.getvalue()
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj.name)[1]) as temp_file:
                            temp_file.write(audio_bytes); temp_file_path = temp_file.name
                        if temp_file_path: processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj items.",
        "\n**Additional Instructions:**",
        "- Accuracy is paramount. Capture ALL facts precisely.", "- Be clear and concise.",
        "- Include ONLY information present in the transcript.", "- If a section (like Opening Overview) isn't present, OMIT it.",
        "\n---", (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT:**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES:**\n"); return "\n".join(filter(None, prompt_parts))

                    # Handle Response
                    if response and hasattr(response, 'text') and response.text and response.text.strip():
                        st.session_state.generated_notes = response.text.strip()
                        st.session_state.edited_notes_text = st.session_state.generated_notes
                        add_to_history(st.session_state.generated_notes)
                        st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                        st.toast("🎉 Notes generated!", icon="✅")
                    elif response and hasattr(response, 'prompt_feedback.name}")
                        else: raise Exception("Temp file creation failed.")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
                    st.session_state.uploaded_audio_info = processed_audio_file_ref
                    # Polling...
                    st.session_state.processing_step = f"🎧 Processing audio..."
                    st

def create_earnings_call_prompt(transcript, context=None):
    prompt_parts = [
        "You are a financial analyst tasked with summarizing an earnings call transcript. Your output MUST be structured notes.",
        "Analyze the entire transcript and extract key information, numerical data, guidance, strategic comments, and management sentiment.",
        "Present the information using the EXACT headings and subheadings provided below. You MUST categorize all relevant comments under the correct heading') and response.prompt_feedback.block_reason:
                        st.session_state.error_message = f"⚠️ Response blocked: {response.prompt_feedback.block_reason}."
                    elif response: st.session_state.error_message = "🤔 AI returned empty response."
                    else: st.session_state.error_message = "😥 Generation failed (No response)."

                except Exception as api_call_err: # Catch API call errors
                    st.session_state.error_message = f"❌ API Call Error: {api.rerun()
                    polling_start = time.time()
                    while processed_audio_file_ref.state.name == "PROCESSING":
                        if time.time() - polling_start > 300: raise Timeout.",
        "\n**Mandatory Structure:**",
        "- **Call Participants:** (List names and titles mentioned. If none mentioned, state 'Not specified')",
        "- **Opening Remarks/CEO Statement:** (Summarize key themes, vision, achievements/challenges mentioned.)",
        "- **Financial Highlights:** (List specific Revenue, Profitability, EPS, Margins, etc. Include numbers and comparisons (YoY/QoQ) EXACTLY as stated.)",
_call_err}"

            # --- Cloud Audio File Cleanup (if reference still exists) ---
            if st.session_state.uploaded_audio_info:
                try:
                    st.toast("☁️ CleaningError("Audio timeout.")
                        time.sleep(5); processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                    if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio state: {processed_audio_file_ref.state        "- **Segment Performance:** (If discussed, detail performance by business unit, geography, or product line.)",
 up cloud audio...", icon="🗑️")
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
                except Exception as e: st.warning(f"Cloud cleanup failed: {e}", icon="⚠️")


        # --- End of.name}")
                    st.toast(f"🎧 Audio ready!", icon="✅")

                    # --- TWO-STEP vs ONE-STEP logic for AUDIO ---
                    if not prompt_was_edited_or_custom: #        "- **Key Business Updates/Strategy:** (Summarize new initiatives, partnerships, market position, M&A activity discussed.)",
        "\n**Industry-Specific Categorization (Apply ONLY ONE section based on company type identified from the transcript):**",
        "\n  **>>> If IT Services Topics Discussed <<<**",
        "    *(Scan inner try...except ---
    # --- Outer FINALLY block: Always runs ---
    finally:
        st. the transcript for these specific topics and categorize comments STRICTLY under these subheadings)*",
        "    - **Futuresession_state.processing = False
        st.session_state.processing_step = None # Clear step
 Default Audio -> Two Step
                        st.session_state.processing_step = f"✍️ Transcribing audio..."
 Investments / Capital Allocation:** (List all mentions of R&D, technology spend, acquisitions, buybacks, dividends.)        st.rerun() # Rerun to display final results or errors


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

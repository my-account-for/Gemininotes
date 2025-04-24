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


# --- Define Available Models & Meeting Types ---
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
    filename_gen_config = {"temperature": 0.2, "max_output_tokens": 50, "response_mime_type": "text/plain"}
    main_gen_config = {"temperature": 0.7, "top_p": 1.0, "top_k": 32, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
    transcription_gen_config = {"temperature": 0.1, "response_mime_type": "text/plain"}
    safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
except Exception as e: st.error(f"### 💥 Error Configuring Google AI Client: {e}", icon="🚨"); st.stop()

# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None, 'add_ ✨", page_icon="✨", layout="wide", initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
    /* ... Keep previous CSS ... */
    footer {context_enabled': False,
    'selected_model_display_name': DEFAULT_MODEL_NAME, 'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'view_edit_prompt_enabled': text-align: center; color: #9CA3AF; font-size: 0.8rem; padding: 2rem 0 1rem 0; }
    footer a { color: #6B False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '', # General context
    'earnings_call_topics': '', # <-- ADDED INITIALIZATION
    'edit7280; text-decoration: none; }
    footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# --- Define Available Models & Meeting Types ---
AVAILABLE_MODELS = {
    "Gemini 1_notes_enabled': False, 'edited_notes_text': "", 'suggested_filename': None,
    'history': [],
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    try: pdf_file_stream.seek(0); pdf_reader = PyPDF2.PdfReader(pdf_file_stream); text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()]); return text.strip() if text else None
    except Exception as e: st.session_state.error_message = f"⚙️ PDF Extraction Error: {e}"; return None

# --- Restore ORIGINAL Expert Meeting Prompt ---
def create_expert_meeting_prompt(transcript, context=None):
    """Creates the prompt for 'Expert Meeting' type (Q&A format)."""
    prompt_parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:", "\n**Structure:**",
        "- **Opening overview or Expert background (Optional):** If the transcript begins with an overview, agenda, or expert intro, include it FIRST as bullet points. Capture ALL details (names, dates, numbers, etc.). Use simple language. DO NOT summarize.",
        "- **Q&A format:** Structure the main body STRICTLY in Question/Answer format.",
        "  - **Questions:** Extract clear questions. Rephrase slightly ONLY for clarity if needed. Format clearly (e.g., 'Q:' or bold).",
        "  - **Answers:** Use bullet points directly below the question. Each bullet MUST be a complete sentence with one distinct fact. Capture ALL specifics (data, names, examples, $, %, etc.). DO NOT use sub-bullets or section headers within answers. DO NOT add interpretations, summaries, conclusions, or action items.",
        "\n**Additional Instructions:**",
        "- Accuracy is paramount. Capture ALL facts precisely.", "- Be clear and concise.",
        "- Include ONLY information present in the transcript.", "- If a section (like Opening Overview) isn't present, OMIT it.",
        "\n---", (f"\n**MEETING TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    # General context can still be useful
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT (Use for understanding):**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED NOTES (Q&A Format):**\n")
    return "\n".join(filter(None, prompt_parts))
# --- END Restored Prompt ---

# --- MODIFIED Earnings Call Prompt ---
def create_earnings_call_prompt(transcript, user_topics=None, context=None):
    """Creates the prompt for 'Earnings Call' type, tailored for investment firms, allowing user topics."""
    topic_instructions = ""
    if user_topics:
        topic_list_str = "\n".join(f"- **{topic.strip()}**" for topic in user_topics)
        topic_instructions = (
            f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n"
            f"{topic_list_str}\n"
            f"- **Other Key Points** (Use this MANDATORY heading for any important information NOT covered under the topics above)\n\n"
            f"Place all relevant details, data points, and quotes from the transcript under the most appropriate heading (either user-specified or 'Other Key Points'). If a user-specified topic isn't discussed, state 'Not discussed' under that heading."
        )
    else: # No topics provided by user
        topic_instructions = (
            f"Since no specific topics were provided by the user, first identify the logical main themes or key sections discussed in the earnings call transcript (e.g., Financials, Segment Performance, Strategy, Outlook, Q&A). "
            f"Use these identified themes as **bold headings** for your notes.\n"
            f"After sections for the main themes, include a final mandatory section:\n"
            f"- **Other Key Points** (Use this heading for any important information not covered under the main themes you identified)\n\n"
            f"Place all relevant details, data points, and quotes from the transcript under the most appropriate heading you created or 'Other Key Points'."
        )

    prompt_parts = [
        "You are an expert AI assistant creating DETAILED notes from an earnings call transcript for an investment firm.",
        "The output MUST be comprehensive, factual notes, capturing all critical financial and strategic information.",
        "**Formatting Requirements (Mandatory):**",
        "- Use standard financial notation: US$ for dollars (e.g., US$2.5M), % for percentages.",
        "- Clearly state comparison periods when mentioned (e.g., +5% YoY, -2% QoQ).",
        "- Represent fiscal periods accurately if mentioned (e.g., Q3 FY25, H1 2024).",
        "- Use common business abbreviations where appropriate (e.g., CEO, CFO, KPI, ROI, SSSG).",
        "- Present information under headings as bullet points.",
        "- Each bullet point should be a complete sentence containing a distinct piece of information.",
        "- Capture ALL specific numbers, data points, examples, company/person names accurately.",
        "- Extract direct quotes for significant statements using quotation marks.",
        "- DO NOT add summaries, interpretations, opinions, or action items unless part of the explicitly stated structure below (e.g., Q&A summary).",

        "\n**Note Structure:**",
        "- **Call Participants:** (List names and titles mentioned. If none mentioned, state 'Not specified')", # Keep this standard section
        topic_instructions, # Inserts instructions based on whether user provided topics OR asks model to create sections + Other Key Points

        "\n**CRITICAL:** Ensure all numerical data and specific details are captured precisely as stated in the transcript and formatted according to the requirements above. Adhere strictly to the requested heading structure.",
        "\n---",
        (f"\n**EARNINGS CALL TRANSCRIPT:**\n{transcript}\n---" if transcript else ""),
    ]
    # General context is still useful (Company name, ticker etc)
    if context: prompt_parts.extend(["\n**ADDITIONAL CONTEXT (Use for understanding):**\n", context, "\n---"])
    prompt_parts.append("\n**GENERATED EARNINGS CALL NOTES:**\n")
    return "\n".join(filter(None, prompt_parts))
# --- END MODIFIED PROMPT ---


# --- Keep Other Helpers (docx, get_current_input_data, filename, history, clear_all) ---
def create_docx(text): document = docx.Document(); [document.add_paragraph(line) for line in text.split('\n')]; buffer = io.BytesIO(); document.save(buffer); buffer.seek(0); return buffer.getvalue()
def get_current_input_data():
    input_type = st.session_state.input_method_radio; transcript = None; audio_file = None
    if input_type == "Paste Text": transcript = st.session_state.text_input.strip()
    elif input_type == "Upload PDF": pdf_file = st.session_state.pdf_uploader; \
        if pdf_file is not None: try: transcript = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue())) except Exception as e: st.session_state.error_message = f"Error processing PDF: {e}"; transcript = None
    elif input_type == "Upload Audio": audio_file = st.session_state.audio_uploader
    return input_type, transcript, audio_file
def update_prompt_display_text():
    meeting_type = st.session_state.selected_meeting_type
    if st.session_state.view_edit_prompt_enabled and meeting_type != "Custom":
        temp_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        input_type = st.session_state.input_method_radio
        # Determine topics based on meeting type
        user_topics = None
        if meeting_type == "Earnings Call":
             topics_str = st.session_state.earnings_call_topics.strip()
             if topics_str: user_topics = [topic.strip() for topic in topics_str.split(',')]

        prompt_func = create_expert_meeting_prompt if meeting_type == "Expert Meeting" else lambda t, c: create_earnings_call_prompt(t, user_topics, c)

        placeholder = "[TRANSCRIPT ...]" if input_type != "Upload Audio" else None
        if input_type == "Upload Audio": base_prompt = prompt_func(transcript=None, context=temp_context); st.session_state.current_prompt_text = ("# NOTE FOR AUDIO...\n#######\n\n" + base_prompt)
        else: st.session_state.current_prompt_text = prompt_func(transcript=placeholder, context=temp_context)
    elif meeting_type == "Custom":
         if not st.session_state.current_prompt_text: st.session_state.current_prompt_text = "# Enter custom prompt..."
    elif not st.session_state.view_edit_prompt_enabled and meeting_type != "Custom": st.session_state.current_prompt_text = ""
def clear_all_state():
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE; st.session_state.selected_model_display_name = DEFAULT_MODEL_NAME
    st.session_state.input_method_radio = 'Paste Text'; st.session_state.text_input = ""; st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None; st.session_state.context_input = ""; st.session_state.add_context_enabled = False
    st.session_state.earnings_call_topics = ""; # Clear specific topics
    st.session_state.current_prompt_text = ""; st.session_state.view_edit_prompt_enabled = False; st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""; st.session_state.edit_notes_enabled = False; st.session_state.error_message = None
    st.session_state.processing = False; st.session_state.suggested_filename = None; st.session_state.uploaded_audio_info = None
    st.session_state.history = []; update_prompt_display_text(); st.toast("Inputs/outputs cleared!", icon="🧹")
def generate_suggested_filename(notes_content, meeting_type):
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True; st.toast("💡 Generating filename...", icon="⏳")
        filename_model = genai.GenerativeModel("gemini-1.5-flash")
        today_date = datetime.now().strftime("%Y%m%d"); mt_cleaned = meeting_type.replace(" ", "")
        filename_prompt = (f"Analyze notes. Suggest filename: YYYYMMDD_ClientOrTopic_MeetingType. Use {today_date}. Extract main client/topic. Use CamelCase/underscores. Type: '{mt_cleaned}'. Max 3 words topic. Output ONLY filename.\n\nNOTES:\n{notes_content[:1500]}")
        response = filename_model.generate_content(filename_prompt, generation_config=filename_gen_config, safety_settings=safety_settings)
        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())[:100]
            if re.match(r"\d{8}_[\w\-\.]+_\w+", s_name): st.toast("💡 Filename suggested!", icon="✅"); return s_name
            else: st.warning(f"Filename sugg. '{s_name}' bad format.", icon="⚠️"); return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.warning(f"Filename blocked: {response.prompt_feedback.block_reason}", icon="⚠️"); return None
        else: st.warning("Could not gen filename.", icon="⚠️"); return None
    except Exception as e: st.warning(f"Filename gen error: {e}", icon="⚠️"); return None
    finally: st.session_state.generating_filename = False
def add_to_history(notes):
    if not notes: return; timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); new_entry = {"timestamp": timestamp, "notes": notes};
    current_history = st.session_state.get('history', []); current_history.insert(0, new_entry); st.session_state.history = current_history[:3]
def restore_note_from_history(index):
    if 0 <= index < len(st.session_state.history):
        entry = st.session_state.history[index]; st.session_state.generated_notes = entry["notes"]; st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False; st.session_state.suggested_filename = None; st.session_state.error_message = None
        st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI"); st.markdown("Instantly transform meeting recordings into structured, factual notes.")
with st.container(border=True): # Input Section
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a: st.subheader("Meeting Details"); st.radio(label="Meeting Type:", options=MEETING_TYPES, key="selected_meeting_type", horizontal=True, on_change=update_prompt_display_text)
        with col1b: st.subheader("AI Model"); st.selectbox(label="Model:", options=list(AVAILABLE_MODELS.keys()), key="selected_model_display_name", label_visibility="collapsed")
    with col_main_2: st.subheader(""); st.button("🧹 Clear All", on_click=clear_all_state, use_container_width=True, type="secondary")

    st.divider()
    # --- Conditional Inputs based on Meeting Type ---
    selected_mt = st.session_state.selected_meeting_type

    # Input Source (Always visible)
    st.subheader("Source Input")
    st.radio(label="Input type:", options=("Paste Text", "Upload PDF", "Upload Audio"), key="input_method_radio", horizontal=True, label_visibility="collapsed", on_change=update_prompt_display_text)
    input_type_ui = st.session_state.input_method_radio
    if input_type_ui == "Paste Text": st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste transcript...")
    elif input_type_ui == "Upload PDF": st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader")
    else: st.file_uploader("Upload Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.divider()

    # Optional Elements (Topics for Earnings Call, Context for All, Prompt Edit for non-Custom)
    col3a, col3b = st.columns(2)
    with col3a:
        # Show specific topics input ONLY for Earnings Call
        if selected_mt == "Earnings Call":
            st.subheader("Earnings Call Topics (Optional)")
            st.text_area("Enter comma-separated topics for note structure:",
                         key="earnings_call_topics", # New state key
                         height=100,
                         on_change=update_prompt_display_text, # Update prompt preview if topics change
                         placeholder="E.g., Guidance, Segment Performance, Capital Allocation",
                         help="If provided, notes will be structured under these topics + 'Other Key Points'. If blank, AI will determine main themes.")
        # Show general context for all types (useful for company name etc. in earnings)
        st.checkbox("Add General Context", key="add_context_enabled", on_change=update_prompt_display_text)
        if st.session_state.add_context_enabled:
            st.text_area("Context Details:", height=75, key="context_input", on_change=update_prompt_display_text, placeholder="E.g., Company: ACME Corp (ACME), Q2 FY24 Results")

    with col3b:
        # View/Edit Prompt Checkbox (NOT for Custom)
        if selected_mt != "Custom":
            st.checkbox("View/Edit Prompt", key="view_edit_prompt_enabled", on_change=update_prompt_display_text)

# --- Prompt Display/Input Area (Conditional) ---
show_prompt_area = (st.session_state.view_edit_prompt_enabled and selected_mt != "Custom") or (selected_mt == "Custom")
if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Prompt Preview/Editor" if selected_mt != "Custom" else "Custom Prompt (Required)"
        st.subheader(prompt_title); caption = "Edit prompt..." if selected_mt != "Custom" else "Enter prompt..."
        st.caption(caption); st.text_area(label="Prompt Text:", value=st.session_state.current_prompt_text, key="current_prompt_text", height=350, label_visibility="collapsed")

# --- Generate Button ---
st.write(""); generate_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True, disabled=st.session_state.processing or st.session_state.generating_filename)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    # (Output display logic remains the same)
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)
    if st.session_state.processing: st.info("⏳ Processing request...", icon="🧠")
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

# --- History Section (Keep as is) ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    if not st.session_state.history: st.caption("No history yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.container(): st.markdown(f"**#{i+1} - {entry['timestamp']}**"); st.markdown(f"```\n{entry['notes'][:200]}...\n```"); st.button(f"View/Use #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,)); \
                               if i < len(st.session_state.history) - 1: st.divider()


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True; st.session_state.generating_filename = False
    st.session_state.generated_notes = None; st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False; st.session_state.error_message = None
    st.session_state.suggested_filename = None; st.rerun()

if st.session_state.processing and not st.session_state.generating_filename:
    try: # Outer try
        # Retrieve state & inputs
        meeting_type = st.session_state.selected_meeting_type
        selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model_display_name]
        view_edit_enabled = st.session_state.view_edit_prompt_enabled
        user_prompt_text = st.session_state.current_prompt_text
        general_context = st.session_state.context_input.strip() if st.session_state.add_context_enabled else None
        # Get specific topics only if Earnings Call is selected
        user_topics_str = st.session_state.earnings_call_topics.strip() if meeting_type == "Earnings Call" else ""
        user_topic_list = [topic.strip() for topic in user_topics_str.split(',')] if user_topics_str else None

        actual_input_type, transcript_data, audio_file_obj = get_current_input_data()

        # Validation (remains the same)
        if actual_input_type == "Paste Text" and not transcript_data: st.session_state.error_message = "⚠️ Text area empty."
        elif actual_input_type == "Upload PDF" and not transcript_data and not st.session_state.error_message: st.session_state.error_message = "⚠️ PDF error."
        elif actual_input_type == "Upload Audio" and not audio_file_obj: st.session_state.error_message = "⚠️ No audio uploaded."
        if meeting_type == "Custom" and not user_prompt_text.strip(): st.session_state.error_message = "⚠️ Custom Prompt empty."

        # Initialize variables
        final_prompt_for_api = None; processed_audio_file_ref = None; api_payload_parts = []
        prompt_was_edited_or_custom = (meeting_type == "Custom" or view_edit_enabled)
        notes_model_id = selected_model_id; transcribed_text_for_notes = transcript_data

        if not st.session_state.error_message:
            # Inner Try for pre-processing, prompt-building, audio handling
            try:
                # --- Handle Audio Upload & Transcription (if applicable) ---
                if actual_input_type == "Upload Audio":
                    # (Audio upload and polling logic remains the same - using tempfile)
                    if not audio_file_obj: raise ValueError("Audio missing.")
                    st.toast(f"☁️ Uploading '{audio_file_obj.name}'...", icon="⬆️"); audio_bytes = audio_file_obj.getvalue(); temp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file_obj.name)[1]) as tf: tf.write(audio_bytes); temp_file_path = tf.name
                        if temp_file_path: processed_audio_file_ref = genai.upload_file(path=temp_file_path, display_name=f"audio_{int(time.time())}_{audio_file_obj.name}")
                        else: raise Exception("Temp file failed.")
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
                    st.session_state.uploaded_audio_info = processed_audio_file_ref
                    st.toast(f"🎧 Processing audio...", icon="⏳"); polling_start = time.time()
                    while processed_audio_file_ref.state.name == "PROCESSING":
                        if time.time() - polling_start > 300: raise TimeoutError("Audio timeout.")
                        time.sleep(5); processed_audio_file_ref = genai.get_file(processed_audio_file_ref.name)
                    if processed_audio_file_ref.state.name != "ACTIVE": raise Exception(f"Audio state: {processed_audio_file_ref.state.name}")
                    st.toast(f"🎧 Audio ready!", icon="✅")

                    # --- TWO-STEP LOGIC ---
                    if not prompt_was_edited_or_custom:
                        st.toast(f"✍️ Transcribing audio...", icon="⏳")
                        transcription_prompt = "Transcribe the audio accurately."; t_model = genai.GenerativeModel(selected_model_id);
                        t_response = t_model.generate_content([transcription_prompt, processed_audio_file_ref], generation_config=transcription_gen_config, safety_settings=safety_settings)
                        if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                            transcribed_text_for_notes = t_response.text.strip() # OVERWRITE
                            st.toast("✍️ Transcription complete!", icon="✅"); actual_input_type = "Generated Transcript"
                        else: raise Exception("Audio transcription failed.")
                # --- End Audio Pre-processing ---

                # --- Determine Final Prompt Text ---
                if not st.session_state.error_message:
                    if prompt_was_edited_or_custom:
                        final_prompt_for_api = user_prompt_text.split("####################################\n\n")[-1] # Clean note
                    else: # Generate default prompt
                        if meeting_type == "Expert Meeting":
                             if transcribed_text_for_notes: final_prompt_for_api = create_expert_meeting_prompt(transcribed_text_for_notes, general_context)
                             else: st.session_state.error_message = "Error: Source text missing for Expert Meeting."
                        elif meeting_type == "Earnings Call":
                             if transcribed_text_for_notes: final_prompt_for_api = create_earnings_call_prompt(transcribed_text_for_notes, user_topic_list, general_context)
                             else: st.session_state.error_message = "Error: Source text missing for Earnings Call."

                if not final_prompt_for_api and not st.session_state.error_message:
                     st.session_state.error_message = "Error: Failed to determine prompt."

                # --- Prepare API Payload ---
                if not st.session_state.error_message and final_prompt_for_api:
                    api_payload_parts.append(final_prompt_for_api)
                    if actual_input_type == "Upload Audio" and prompt_was_edited_or_custom:
                        if processed_audio_file_ref: api_payload_parts.append(processed_audio_file_ref)
                        else: st.session_state.error_message = "Error: Audio ref missing for custom audio."
                elif not st.session_state.error_message: st.session_state.error_message = "Error: Prompt not ready for API."

            except Exception as pre_process_err:
                st.session_state.error_message = f"❌ Pre-processing Error: {pre_process_err}"
                if st.session_state.uploaded_audio_info: try: genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None; except Exception: pass


            # --- Generate Notes API Call ---
            if not st.session_state.error_message and api_payload_parts:
                try: # Inner try for API call
                    st.toast(f"📝 Generating notes...", icon="✨")
                    model = genai.GenerativeModel(model_name=notes_model_id, safety_settings=safety_settings, generation_config=main_gen_config)
                    response = model.generate_content(api_payload_parts)

                    # Handle Response
                    if response and hasattr(response, 'text') and response.text and response.text.strip():
                        st.session_state.generated_notes = response.text.strip(); st.session_state.edited_notes_text = st.session_state.generated_notes
                        add_to_history(st.session_state.generated_notes); st.session_state.suggested_filename = generate_suggested_filename(st.session_state.generated_notes, meeting_type)
                        st.toast("🎉 Notes generated!", icon="✅")
                    elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: st.session_state.error_message = f"⚠️ Response blocked: {response.prompt_feedback.block_reason}."
                    elif response: st.session_state.error_message = "🤔 AI returned empty response."
                    else: st.session_state.error_message = "😥 Generation failed (No response)."

                except Exception as api_call_err: st.session_state.error_message = f"❌ API Call Error: {api_call_err}"

            # --- Cloud Audio Cleanup (always attempt if ref exists) ---
            if st.session_state.uploaded_audio_info:
                try: st.toast("☁️ Cleaning cloud audio...", icon="🗑️"); genai.delete_file(st.session_state.uploaded_audio_info.name); st.session_state.uploaded_audio_info = None
                except Exception as e: st.warning(f"Cloud cleanup failed: {e}", icon="⚠️")
        # End of inner try...except

    # --- Outer FINALLY block ---
    finally:
        st.session_state.processing = False
        # Rerun to display results/errors/filename update
        st.rerun()

# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

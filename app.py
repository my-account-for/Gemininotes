
**Instructions:**
1.  **Translate to English:** Convert any substantial non-English text found to English, ensuring accuracy and natural flow. Preserve original names or technical terms if translation is uncertain.
2.  **Correct Errors & Improve Readability:** Fix obvious spelling mistakes and grammatical errors. Use the overall context to correct potentially incorrect words or phrases where confident. Preserve technical terms or names if unsure. Remove excessive filler words ONLY if they severely hinder readability, but retain the natural voice where possible. Do not paraphrase or summarize extensively. Aim for clarity and conciseness while preserving meaning.
3.  **Format:** Ensure clear paragraph structure where appropriate. Avoid adding speaker labels unless they were already present and meaningful in the raw text.
4.  **Output:** Provide *only* the refined, translated (where applicable), and corrected text. Do not add any introduction, summary, or commentary before or after the text itself.

**Additional Context (Optional - use for understanding terms, names, etc.):**
{context}

**Refined Text:**
"""


# --- Initialize Session State ---
default_state = {
    'processing': False, 'generating_filename': False, 'generated_notes': None, 'error_message': None,
    'uploaded_audio_info': None,
    'add_context_enabled': False,
    'selected_notes_model_display_name': DEFAULT_NOTES_MODEL_NAME,
    'selected_transcription_model_display_name': DEFAULT_TRANSCRIPTION_MODEL_NAME,
    'selected_refinement_model_display_name': DEFAULT_REFINEMENT_MODEL_NAME,
    'selected_meeting_type': DEFAULT_MEETING_TYPE,
    'expert_meeting_prompt_option': DEFAULT_EXPERT_MEETING_OPTION,
    'view_edit_prompt_enabled': False, 'current_prompt_text': "",
    'input_method_radio': 'Paste Text', 'text_input': '', 'pdf_uploader': None, 'audio_uploader': None,
    'context_input': '',
    'selected_sector': DEFAULT_SECTOR,
    'previous_selected_sector': DEFAULT_SECTOR, # Tracks previous sector
    'earnings_call_topics': '', # Unified key for topics
    'earnings_call_mode': DEFAULT_EARNINGS_CALL_MODE,
    'existing_notes_input': "",
    'edit_notes_enabled': False,
    'edited_notes_text': "", 'suggested_filename': None, 'history': [],
    'raw_transcript': None, 'refined_transcript': None, # For audio
    'raw_text_input': None, 'refined_text_input': None, # For pasted text
    'processed_audio_chunk_references': [],
    'earnings_call_topics_initialized': False # Initialize flag
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Helper Functions ---
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
    """
    Updates the earnings_call_topics state based on the selected sector.
    Only updates if the sector has changed and is a template sector (not 'Other').
    """
    selected_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if selected_sector in SECTOR_TOPICS and selected_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[selected_sector]
        st.toast(f"Loaded topic template for {selected_sector}", icon="📝")
    # Clear the edited prompt text when loading a new template automatically
    st.session_state.current_prompt_text = ""


# --- Initialize topics on first run ---
if not st.session_state.earnings_call_topics_initialized:
    initial_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
    if initial_sector in SECTOR_TOPICS and initial_sector != "Other / Manual Topics":
        st.session_state.earnings_call_topics = SECTOR_TOPICS[initial_sector]
    else:
        st.session_state.earnings_call_topics = ""
    st.session_state.earnings_call_topics_initialized = True

# --- Detect Sector Change and Update Topics ---
current_sector = st.session_state.get('selected_sector', DEFAULT_SECTOR)
prev_sector = st.session_state.get('previous_selected_sector', DEFAULT_SECTOR)

if current_sector != prev_sector:
    update_topic_template()
    st.session_state.previous_selected_sector = current_sector
    # No rerun needed here, changes reflected in UI rendering

def format_prompt_safe(prompt_template, **kwargs):
    """Safely formats a prompt string, replacing placeholders."""
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
    """Creates a DOCX file in memory from text."""
    document = docx.Document()
    for line in text.split('\n'):
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

def get_current_input_data():
    """Gets transcript/text data based on selected input method."""
    input_type = st.session_state.input_method_radio
    source_text = None
    audio_file = None
    if input_type == "Paste Text":
        source_text = st.session_state.text_input.strip()
    elif input_type == "Upload PDF":
        pdf_file = st.session_state.pdf_uploader
        if pdf_file is not None:
            try:
                source_text = extract_text_from_pdf(io.BytesIO(pdf_file.getvalue()))
            except Exception as e:
                st.session_state.error_message = f"Error processing PDF: {e}"
                source_text = None
    elif input_type == "Upload Audio":
        audio_file = st.session_state.audio_uploader
    return input_type, source_text, audio_file

def validate_inputs():
    """Validates required inputs based on selected options."""
    input_method = st.session_state.get('input_method_radio', 'Paste Text')
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
    custom_prompt = st.session_state.get('current_prompt_text', "").strip()
    view_edit_enabled = st.session_state.get('view_edit_prompt_enabled', False)
    is_enrich_mode = (meeting_type == "Earnings Call" and
                      st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")

    # Check source input
    if input_method == "Paste Text" and not st.session_state.get('text_input', "").strip():
        return False, "Please paste the source text."
    if input_method == "Upload PDF" and st.session_state.get('pdf_uploader') is None:
        return False, "Please upload a source PDF file."
    if input_method == "Upload Audio" and st.session_state.get('audio_uploader') is None:
        return False, "Please upload a source audio file."

    # Check meeting type specific requirements
    if meeting_type == "Custom":
         if not custom_prompt:
             return False, "Custom prompt cannot be empty for 'Custom' meeting type."
    elif meeting_type == "Earnings Call":
        if is_enrich_mode:
            if not st.session_state.get('existing_notes_input',"").strip():
                return False, "Please provide your existing notes for enrichment (in the dedicated input box)."
        # Check edited prompt for required placeholders if edit is enabled
        if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                 return False, "Edited prompt is missing the required {transcript} placeholder."
            if "{topic_instructions}" not in custom_prompt and meeting_type == "Earnings Call" and st.session_state.get('earnings_call_mode') == "Generate New Notes":
                 # topic_instructions is specific to Generate New Notes mode
                 return False, "Edited Earnings Call prompt is missing the required {topic_instructions} placeholder."
    elif meeting_type == "Expert Meeting":
         # Check edited prompt for required placeholders if edit is enabled
         if view_edit_enabled and custom_prompt:
            if "{transcript}" not in custom_prompt:
                return False, "Edited Expert Meeting prompt is missing the required {transcript} placeholder."

    return True, ""

def handle_edit_toggle():
    """Clears custom prompt edits if 'View/Edit Prompt' is toggled off for non-Custom types."""
    if not st.session_state.view_edit_prompt_enabled and st.session_state.selected_meeting_type != "Custom":
        st.session_state.current_prompt_text = ""

def get_prompt_display_text(for_display_only=False):
    """
    Generates the prompt text for display/editing, ensuring it uses the
    correct structure and placeholders identical to the processing logic.
    """
    meeting_type = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)

    # If editing is enabled AND there's text in the editor state, show that text.
    # Exception: If 'for_display_only' is True, force regeneration from template.
    if not for_display_only \
       and st.session_state.get('view_edit_prompt_enabled', False) \
       and meeting_type != "Custom" \
       and st.session_state.get('current_prompt_text', "").strip():
        return st.session_state.current_prompt_text

    # Generate the default prompt view from the base template
    display_text = ""
    temp_context = st.session_state.get('context_input',"").strip() if st.session_state.get('add_context_enabled') else None
    input_type = st.session_state.get('input_method_radio', 'Paste Text')
    transcript_placeholder = "{transcript}"
    context_placeholder_section = f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{temp_context}\n---" if temp_context else ""

    format_kwargs = {
        'transcript': transcript_placeholder,
        'context_section': context_placeholder_section
    }
    prompt_template_to_display = None
    processing_note = "" # Note about refinement steps

    if input_type == "Upload Audio":
        processing_note = ("# NOTE FOR AUDIO: 3-step process (Chunked Transcribe -> Refine Transcript -> Notes).\n"
                           "# This prompt (or your edited version) is used in Step 3 with the *refined transcript*.\n"
                           "####################################\n\n")
    elif input_type == "Paste Text":
        processing_note = ("# NOTE FOR TEXT INPUT: 2-step process (Refine Text -> Notes).\n"
                           "# This prompt (or your edited version) is used in Step 2 with the *refined input text* (translation, corrections etc. applied).\n"
                           "####################################\n\n")

    try:
        if meeting_type == "Expert Meeting":
            expert_option = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
            if expert_option == "Option 1: Existing (Detailed & Strict)":
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
            else: # Option 2 and 3 use Option 2 template
                prompt_template_to_display = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]

            if prompt_template_to_display:
                 display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)
                 if expert_option == "Option 3: Option 2 + Executive Summary":
                     summary_prompt_preview = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY, "Summary prompt not found.").split("---")[0]
                     display_text += f"\n\n# NOTE: Option 3 includes an additional Executive Summary step generated *after* these notes, using a separate prompt starting like:\n'''\n{summary_prompt_preview.strip()}\n'''"
            else:
                display_text = "# Error: Could not find prompt template for Expert Meeting display."

        elif meeting_type == "Earnings Call":
             prompt_template_to_display = PROMPTS["Earnings Call"]["Generate New Notes"]
             earnings_call_topics_text = st.session_state.get('earnings_call_topics', "")

             # Replicate the topic formatting logic from the main processing block
             topic_instructions = ""
             if earnings_call_topics_text and earnings_call_topics_text.strip():
                 formatted_topics = []
                 for line in earnings_call_topics_text.split('\n'):
                     trimmed_line = line.strip()
                     if trimmed_line and not trimmed_line.startswith(('-', '*', '#')):
                         formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                     elif trimmed_line:
                         formatted_topics.append(trimmed_line)
                 topic_list_str = "\n".join(formatted_topics)
                 topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                       f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                       f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
             else: # No topics provided
                 topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                       f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                       f"Place all relevant details under the most appropriate heading.")
             format_kwargs["topic_instructions"] = topic_instructions
             display_text = format_prompt_safe(prompt_template_to_display, **format_kwargs)

        elif meeting_type == "Custom":
             custom_audio_note = ("\n# NOTE FOR AUDIO: If using audio, the system will first chunk and transcribe,\n"
                                  "# then *refine* the full transcript (speaker ID, translation, corrections).\n"
                                  "# Your custom prompt below will receive this *refined transcript* as the {transcript}.\n"
                                  "# Design your prompt accordingly.\n")
             custom_text_note = ("\n# NOTE FOR TEXT INPUT: If using pasted text, the system will first *refine* it\n"
                                 "# (translation, corrections, readability improvements).\n"
                                 "# Your custom prompt below will receive this *refined text* as the {transcript}.\n"
                                 "# Design your prompt accordingly.\n")
             default_custom = "# Enter your custom prompt here...\n# Use {transcript} and {context_section} placeholders.\n# Example: Summarize this meeting:\n# {transcript}\n# {context_section}"
             # For Custom mode, the text area's value IS the prompt, so we just display it or the default
             display_text = st.session_state.get('current_prompt_text', default_custom)
             if input_type == 'Upload Audio': display_text += custom_audio_note
             elif input_type == 'Paste Text': display_text += custom_text_note
             return display_text # Return directly

        else:
             st.error(f"Internal Error: Invalid meeting type '{meeting_type}' encountered for prompt preview.")
             return "Error generating prompt preview."

        # Prepend the processing note (if applicable) for non-custom types
        display_text = processing_note + display_text

    except Exception as e:
         st.error(f"Error generating prompt preview: {e}")
         display_text = f"# Error generating preview: Review inputs/prompt structure.\nDetails: {e}"

    return display_text

def clear_all_state():
    """Resets most session state variables to their defaults."""
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE
    st.session_state.selected_notes_model_display_name = DEFAULT_NOTES_MODEL_NAME
    st.session_state.selected_transcription_model_display_name = DEFAULT_TRANSCRIPTION_MODEL_NAME
    st.session_state.selected_refinement_model_display_name = DEFAULT_REFINEMENT_MODEL_NAME
    st.session_state.expert_meeting_prompt_option = DEFAULT_EXPERT_MEETING_OPTION
    st.session_state.input_method_radio = 'Paste Text'
    st.session_state.text_input = ""
    st.session_state.pdf_uploader = None
    st.session_state.audio_uploader = None
    st.session_state.context_input = ""
    st.session_state.add_context_enabled = False
    st.session_state.selected_sector = DEFAULT_SECTOR
    st.session_state.previous_selected_sector = DEFAULT_SECTOR # Reset tracker
    # Reset earnings call topics based on the default sector
    st.session_state.earnings_call_topics = SECTOR_TOPICS.get(DEFAULT_SECTOR, "") if DEFAULT_SECTOR != "Other / Manual Topics" else ""
    st.session_state.earnings_call_topics_initialized = True # Mark as initialized here too
    st.session_state.current_prompt_text = "" # Clear edited prompt
    st.session_state.view_edit_prompt_enabled = False
    st.session_state.earnings_call_mode = DEFAULT_EARNINGS_CALL_MODE
    st.session_state.existing_notes_input = ""
    st.session_state.generated_notes = None
    st.session_state.edited_notes_text = ""
    st.session_state.edit_notes_enabled = False
    st.session_state.error_message = None
    st.session_state.processing = False
    st.session_state.generating_filename = False
    st.session_state.suggested_filename = None
    st.session_state.uploaded_audio_info = None
    st.session_state.history = []
    st.session_state.raw_transcript = None # Audio
    st.session_state.refined_transcript = None # Audio
    st.session_state.raw_text_input = None # Text
    st.session_state.refined_text_input = None # Text
    st.session_state.processed_audio_chunk_references = []
    st.toast("Inputs and outputs cleared!", icon="🧹")
    st.rerun()

def generate_suggested_filename(notes_content, meeting_type):
    """Suggests a filename using a fast LLM based on notes content."""
    if not notes_content: return None
    try:
        st.session_state.generating_filename = True # Indicate processing start
        filename_model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        today_date = datetime.now().strftime("%Y%m%d")
        mt_cleaned = meeting_type.replace(" ", "_").lower()

        notes_preview = notes_content
        summary_marker = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
        if summary_marker in notes_content:
            notes_preview = notes_content.split(summary_marker)[0]

        filename_prompt = (f"Suggest a concise filename (max 5 words including type, use underscores_not_spaces). Start with Date={today_date}, Type='{mt_cleaned}'. Base suggestion on key topics/company names from the start of these notes. Output ONLY the filename string (e.g., {today_date}_{mt_cleaned}_topic_name.txt). NOTES_PREVIEW:\n{notes_preview[:1000]}")

        response = filename_model.generate_content(
            filename_prompt,
            generation_config=filename_gen_config,
            safety_settings=safety_settings
        )

        if response and hasattr(response, 'text') and response.text:
            s_name = re.sub(r'[^\w\-.]', '_', response.text.strip())
            s_name = re.sub(r'_+', '_', s_name)
            s_name = s_name.strip('_')
            s_name = s_name[:100]

            if not s_name.startswith(today_date):
                s_name = f"{today_date}_{s_name}"

            if s_name:
                st.toast("💡 Filename suggested!", icon="✅")
                return s_name
            else:
                st.warning(f"Filename suggestion was empty or invalid after cleaning: '{response.text}'", icon="⚠️")
                return None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            st.warning(f"Filename suggestion blocked: {response.prompt_feedback.block_reason}", icon="⚠️")
            return None
        else:
            st.warning("Could not generate filename suggestion (empty response).", icon="⚠️")
            return None
    except Exception as e:
        st.warning(f"Filename generation error: {e}", icon="⚠️")
        return None
    finally:
        st.session_state.generating_filename = False

def add_to_history(notes):
    """Adds the generated notes to the history in session state."""
    if not notes: return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {"timestamp": timestamp, "notes": notes}
        current_history = st.session_state.get('history', [])
        if not isinstance(current_history, list):
            st.warning("History state was not a list, resetting.", icon="⚠️")
            current_history = []
        current_history.insert(0, new_entry)
        st.session_state.history = current_history[:3]
    except Exception as e:
        st.warning(f"⚠️ Error updating note history: {e}", icon="❗")

def restore_note_from_history(index):
    """Restores selected notes from history to the main output area."""
    if 0 <= index < len(st.session_state.history):
        entry = st.session_state.history[index]
        st.session_state.generated_notes = entry["notes"]
        st.session_state.edited_notes_text = entry["notes"]
        st.session_state.edit_notes_enabled = False
        st.session_state.suggested_filename = None
        st.session_state.error_message = None
        # Clear transcript/text states when restoring
        st.session_state.raw_transcript = None
        st.session_state.refined_transcript = None
        st.session_state.raw_text_input = None
        st.session_state.refined_text_input = None
        st.toast(f"Restored notes from {entry['timestamp']}", icon="📜")
        st.rerun()

# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings or text into structured, factual notes.")

# --- Settings Container ---
with st.container(border=True):
    col_main_1, col_main_2 = st.columns([3, 1])
    with col_main_1:
        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Meeting Details")
            mt_options = list(MEETING_TYPES)
            mt_default = st.session_state.get('selected_meeting_type', DEFAULT_MEETING_TYPE)
            mt_index = mt_options.index(mt_default) if mt_default in mt_options else 0
            # Clear edits/toggle when meeting type changes
            st.radio("Meeting Type:", options=mt_options, key="selected_meeting_type", horizontal=True,
                     index=mt_index,
                     on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False))

            # Conditional options
            if st.session_state.get('selected_meeting_type') == "Expert Meeting":
                 em_options = list(EXPERT_MEETING_OPTIONS)
                 em_default = st.session_state.get('expert_meeting_prompt_option', DEFAULT_EXPERT_MEETING_OPTION)
                 em_index = em_options.index(em_default) if em_default in em_options else 0
                 st.radio(
                    "Expert Meeting Note Style:",
                    options=em_options, key="expert_meeting_prompt_option", index=em_index,
                    help="Choose output format: Strict Q&A, Natural Q&A, or Natural Q&A + Summary.",
                    on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False) # Clear edits/toggle
                )
            elif st.session_state.get('selected_meeting_type') == "Earnings Call":
                 ec_options = list(EARNINGS_CALL_MODES)
                 ec_default = st.session_state.get('earnings_call_mode', DEFAULT_EARNINGS_CALL_MODE)
                 ec_index = ec_options.index(ec_default) if ec_default in ec_options else 0
                 # Clear edits/toggle when mode changes
                 st.radio(
                    "Mode:", options=ec_options, key="earnings_call_mode", horizontal=True, index=ec_index,
                    help="Generate notes from scratch or enrich existing ones.",
                    on_change=lambda: st.session_state.update(current_prompt_text="", view_edit_prompt_enabled=False)
                )

        with col1b:
            st.subheader("AI Model Selection")
            trans_options = list(AVAILABLE_MODELS.keys())
            trans_default = st.session_state.get('selected_transcription_model_display_name', DEFAULT_TRANSCRIPTION_MODEL_NAME)
            trans_index = trans_options.index(trans_default) if trans_default in trans_options else 0
            st.selectbox("Transcription Model:", options=trans_options, key="selected_transcription_model_display_name", index=trans_index, help="Model for audio transcription (Step 1). Used only if audio is uploaded.")

            refine_options = list(AVAILABLE_MODELS.keys())
            refine_default = st.session_state.get('selected_refinement_model_display_name', DEFAULT_REFINEMENT_MODEL_NAME)
            refine_index = refine_options.index(refine_default) if refine_default in refine_options else 0
            st.selectbox("Refinement Model:", options=refine_options, key="selected_refinement_model_display_name", index=refine_index, help="Model for refining audio transcripts OR pasted text (Step 2 - Translation, Corrections, Readability).") # Updated help text

            notes_options = list(AVAILABLE_MODELS.keys())
            notes_default = st.session_state.get('selected_notes_model_display_name', DEFAULT_NOTES_MODEL_NAME)
            notes_index = notes_options.index(notes_default) if notes_default in notes_options else 0
            st.selectbox("Notes/Enrichment Model:", options=notes_options, key="selected_notes_model_display_name", index=notes_index, help="Model for final output generation (Step 3 for Audio, Step 2 for PDF, Step 3 for Text).")

    with col_main_2:
        st.subheader("") # Spacer
        st.button("🧹 Clear All Inputs & Outputs", on_click=clear_all_state, use_container_width=True, type="secondary", key="clear_button")

st.divider()

# --- Input Sections ---
with st.container(border=True):
    is_enrich_mode = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                      st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")

    if is_enrich_mode:
        st.subheader("Existing Notes Input (Required for Enrichment)")
        st.text_area("Paste your existing notes here:", height=200, key="existing_notes_input",
                     placeholder="Paste the notes you want to enrich...",
                     help="These notes will be used as the base for enrichment.",
                     value=st.session_state.get("existing_notes_input", ""))
        st.markdown("---")
        st.subheader("Source Input (Text, PDF, or Audio)")
    else:
        st.subheader("Source Input (Text, PDF, or Audio)")

    # Source Input Widgets
    input_options = ("Paste Text", "Upload PDF", "Upload Audio")
    input_default = st.session_state.get('input_method_radio', 'Paste Text')
    input_index = input_options.index(input_default) if input_default in input_options else 0
    st.radio(label="Source input type:", options=input_options, key="input_method_radio", horizontal=True, label_visibility="collapsed", index=input_index)
    input_type_ui = st.session_state.get('input_method_radio', 'Paste Text')
    if input_type_ui == "Paste Text":
        st.text_area("Paste source text:", height=150, key="text_input",
                     placeholder="Paste transcript or other source text here...", value=st.session_state.get("text_input", ""))
    elif input_type_ui == "Upload PDF":
        st.file_uploader("Upload source PDF:", type="pdf", key="pdf_uploader")
    else: # Upload Audio
        st.file_uploader("Upload source Audio:", type=['wav','mp3','m4a','ogg','flac','aac'], key="audio_uploader")

    st.markdown("---")

    # Topics & Context Section
    st.subheader("Topics & Context")
    col3a, col3b = st.columns(2)

    with col3a: # Topics
        if st.session_state.get('selected_meeting_type') == "Earnings Call":
            sector_options = list(SECTOR_OPTIONS)
            sector_default = st.session_state.get('selected_sector', DEFAULT_SECTOR)
            sector_index = sector_options.index(sector_default) if sector_default in sector_options else 0
            st.selectbox("Select Sector (for Topic Template):",
                         options=sector_options, key="selected_sector", index=sector_index,
                         help="Loads a topic template. Select 'Other' to keep/edit manually.")

            st.text_area("Earnings Call Topics (Edit below):",
                         value=st.session_state.get("earnings_call_topics", ""), key="earnings_call_topics",
                         height=150, placeholder="Enter topics manually or select a sector to load a template...",
                         help="Guides structure for new notes or focuses enrichment. Edit freely.")
        else:
             st.caption("Topic selection/editing is available for Earnings Calls.")

    with col3b: # Context & Prompt Edit Toggle
        st.checkbox("Add General Context", key="add_context_enabled")
        if st.session_state.get('add_context_enabled'):
            st.text_area("Context Details:", height=75, key="context_input",
                         placeholder="E.g., Company Name, Ticker, Date, Key Competitors...",
                         value=st.session_state.get("context_input", ""))

        st.write("")
        selected_mt = st.session_state.get('selected_meeting_type')
        disable_edit_checkbox = is_enrich_mode or (selected_mt == "Custom")

        if selected_mt != "Custom":
             st.checkbox("View/Edit Final Prompt", key="view_edit_prompt_enabled",
                         disabled=disable_edit_checkbox,
                         on_change=handle_edit_toggle, # Clears edits if toggled OFF
                         help="View/edit the base prompt used for generation. Disabled in Enrichment mode & Custom type.")
             if is_enrich_mode:
                 st.caption("Prompt editing is disabled in Enrichment mode.")

# --- Prompt Area (Conditional Display) ---
show_prompt_area = (st.session_state.get('selected_meeting_type') == "Custom") or \
                   (st.session_state.get('view_edit_prompt_enabled') and
                    st.session_state.get('selected_meeting_type') != "Custom" and not is_enrich_mode)

if show_prompt_area:
    with st.container(border=True):
        prompt_title = "Final Prompt Editor" if st.session_state.get('selected_meeting_type') != "Custom" else "Custom Final Prompt (Required)"
        st.subheader(prompt_title)

        # Get the base template text (correctly formatted, including processing notes)
        base_template_text = get_prompt_display_text(for_display_only=True)
        # If editing is enabled and the current edit state is empty, populate it with the base template
        if st.session_state.view_edit_prompt_enabled and not st.session_state.current_prompt_text.strip():
             st.session_state.current_prompt_text = base_template_text

        # Display the content of current_prompt_text (either user edits or the populated base)
        st.text_area(
            label="Prompt Text:",
            value=st.session_state.current_prompt_text,
            key="current_prompt_text", # Let user edits update the state directly
            height=350,
            label_visibility="collapsed",
            help="Edit the prompt used for note generation. Ensure required placeholders like {transcript} are present. {transcript} will receive the REFINED text/transcript if applicable." if st.session_state.get('selected_meeting_type') != "Custom" else "Enter your full custom prompt. Use {transcript} and {context_section} placeholders. {transcript} will receive the REFINED text/transcript if applicable.",
            disabled=False,
            id="prompt-edit-area" # Added for potential CSS targeting if needed
        )

        if st.session_state.get('selected_meeting_type') != "Custom":
             st.caption("Editing enabled. Ensure required placeholders like `{transcript}` (and `{topic_instructions}` for Earnings Calls) are present. Placeholders like `{context_section}` will be filled if context is added. Note: `{transcript}` receives refined input if applicable (audio/text).")
        else:
             st.caption("Placeholders `{transcript}` and `{context_section}` will be automatically filled during processing. Note: `{transcript}` receives refined input if applicable (audio/text).")


# --- Generate Button ---
st.write("")
is_valid, error_msg = validate_inputs() # Validation now checks placeholders in edited prompts
generate_tooltip = error_msg if not is_valid else "Generate or enrich notes based on current inputs and settings."
generate_button_label = "🚀 Enrich Notes" if is_enrich_mode else "🚀 Generate Notes"

generate_button = st.button(generate_button_label,
                            type="primary",
                            use_container_width=True,
                            disabled=st.session_state.get('processing') or st.session_state.get('generating_filename') or not is_valid,
                            help=generate_tooltip)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    if st.session_state.get('processing'):
        op_desc = "Enriching notes" if is_enrich_mode else "Generating notes"
        st.info(f"⏳ Processing... Currently {op_desc}. Please wait.", icon="⏳")
    elif st.session_state.get('generating_filename'):
        st.info("⏳ Generating suggested filename...", icon="💡")
    elif st.session_state.get('error_message'):
        st.error(st.session_state.error_message, icon="🚨")
        # Clear error after displaying if desired:
        # st.session_state.error_message = None
    elif st.session_state.get('generated_notes'):
        output_title = "✅ Enriched Notes" if is_enrich_mode else "✅ Generated Notes"
        st.subheader(output_title)

        # --- Display Source Previews (Audio or Text) ---
        # Audio Transcript previews
        if st.session_state.get('raw_transcript'):
            with st.expander("View Raw Audio Transcript (Step 1 Output)"):
                st.text_area("Raw Transcript", st.session_state.raw_transcript, height=200, disabled=True, key="raw_transcript_view")
        if st.session_state.get('refined_transcript'):
             with st.expander("View Refined Audio Transcript (Step 2 Output)", expanded=True):
                st.text_area("Refined Transcript", st.session_state.refined_transcript, height=300, disabled=True, key="refined_transcript_view")

        # Pasted Text previews
        if st.session_state.get('raw_text_input'):
            with st.expander("View Raw Pasted Input Text"):
                st.text_area("Raw Input Text", st.session_state.raw_text_input, height=200, disabled=True, key="raw_text_input_view")
        if st.session_state.get('refined_text_input'):
            with st.expander("View Refined Input Text (Step 2 Output)", expanded=True):
                st.text_area("Refined Input Text", st.session_state.refined_text_input, height=300, disabled=True, key="refined_text_input_view")
        # --- End Source Previews ---

        # Edit and display notes
        st.checkbox("Edit Output", key="edit_notes_enabled")
        notes_content_to_use = st.session_state.edited_notes_text if st.session_state.edit_notes_enabled else st.session_state.generated_notes
        is_expert_meeting_summary = (st.session_state.get('selected_meeting_type') == "Expert Meeting" and
                                     st.session_state.get('expert_meeting_prompt_option') == "Option 3: Option 2 + Executive Summary" and
                                     "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n" in notes_content_to_use)
        if st.session_state.get('edit_notes_enabled'):
            st.text_area("Editable Output:", value=notes_content_to_use, key="edited_notes_text", height=400, label_visibility="collapsed")
        else:
            if is_expert_meeting_summary:
                 try:
                     notes_part, summary_part = notes_content_to_use.split("\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n", 1)
                     st.markdown("### Detailed Notes (Q&A Format)")
                     st.markdown(notes_part)
                     st.markdown("---")
                     st.markdown("### Executive Summary")
                     st.markdown(summary_part)
                 except ValueError:
                     # Fallback if separator isn't found unexpectedly
                     st.markdown(notes_content_to_use)
            else:
                 st.markdown(notes_content_to_use)

        # Download buttons
        st.write("")
        dl_cols = st.columns(3)
        output_type_label = "enriched_notes" if is_enrich_mode else "notes"
        default_fname_base = f"{st.session_state.get('selected_meeting_type', 'meeting').lower().replace(' ', '_')}_{output_type_label}"
        fname_base = st.session_state.get('suggested_filename', default_fname_base)
        with dl_cols[0]:
            st.download_button(label=f"⬇️ Output (.txt)", data=notes_content_to_use, file_name=f"{fname_base}.txt", mime="text/plain", key='download-txt', use_container_width=True)
        with dl_cols[1]:
            st.download_button(label=f"⬇️ Output (.md)", data=notes_content_to_use, file_name=f"{fname_base}.md", mime="text/markdown", key='download-md', use_container_width=True)
        with dl_cols[2]:
            # Offer download for refined transcript OR refined text, whichever is available
            refined_content = st.session_state.get('refined_transcript') or st.session_state.get('refined_text_input')
            if refined_content:
                dl_label = "⬇️ Refined Tx (.txt)" if st.session_state.get('refined_transcript') else "⬇️ Refined Txt (.txt)"
                help_text = "Download the speaker-diarized and corrected source transcript (if audio was processed)." if st.session_state.get('refined_transcript') else "Download the refined source text (if text was processed)."
                refined_fname_base = fname_base.replace(f"_{output_type_label}", "_refined_source") if f"_{output_type_label}" in fname_base else f"{fname_base}_refined_source"
                st.download_button(label=dl_label, data=refined_content, file_name=f"{refined_fname_base}.txt", mime="text/plain", key='download-refined-src', use_container_width=True, help=help_text)
            else:
                st.button("Refined Src N/A", disabled=True, use_container_width=True, help="Refined source is only available after successful audio or text input processing including refinement.")

    elif not st.session_state.get('processing'):
        st.markdown("<p class='initial-prompt'>Configure inputs above and click 'Generate / Enrich Notes' to start.</p>", unsafe_allow_html=True)


# --- History Section ---
with st.expander("📜 Recent Notes History (Last 3)", expanded=False):
    history = st.session_state.get('history', [])
    if not history:
        st.caption("No generated notes in history for this session.")
    else:
        for i, entry in enumerate(history):
             with st.container():
                st.markdown(f"**#{i+1} - {entry.get('timestamp', 'N/A')}**")
                display_note = entry.get('notes', '')
                summary_separator = "\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n"
                preview_text = ""
                if summary_separator in display_note:
                     notes_part = display_note.split(summary_separator, 1)[0]
                     preview_text = "\n".join(notes_part.strip().splitlines()[:3]) + "\n... (+ Executive Summary)"
                else:
                    preview_text = "\n".join(display_note.strip().splitlines()[:4]) + "..."
                st.text(preview_text[:300] + ("..." if len(preview_text) > 300 else ""))
                st.button(f"Restore Notes #{i+1}", key=f"restore_{i}", on_click=restore_note_from_history, args=(i,))
                if i < len(history) - 1: st.divider()

# --- Processing Logic ---
if generate_button:
    # Re-validate just before starting processing
    is_valid_on_click, error_msg_on_click = validate_inputs()
    if not is_valid_on_click:
        st.session_state.error_message = f"Validation Error: {error_msg_on_click}"
        st.session_state.processing = False # Ensure processing flag is off
        st.rerun() # Rerun to show validation error immediately
    else:
        # Set flags and clear previous results
        st.session_state.processing = True
        st.session_state.generating_filename = False
        st.session_state.generated_notes = None
        st.session_state.edited_notes_text = ""
        st.session_state.edit_notes_enabled = False
        st.session_state.error_message = None # Clear previous errors
        st.session_state.suggested_filename = None
        st.session_state.raw_transcript = None # Clear previous audio transcript
        st.session_state.refined_transcript = None # Clear previous audio transcript
        st.session_state.raw_text_input = None # Clear previous text input
        st.session_state.refined_text_input = None # Clear previous text input
        st.session_state.processed_audio_chunk_references = []
        st.rerun() # Rerun to show the status indicator and hide old results

# --- Processing Block ---
if st.session_state.get('processing') and not st.session_state.get('generating_filename') and not st.session_state.get('error_message'):

    processed_audio_chunk_references = []
    is_enrich_mode = (st.session_state.get('selected_meeting_type') == "Earnings Call" and
                      st.session_state.get('earnings_call_mode') == "Enrich Existing Notes")
    operation_desc = "Enriching Notes" if is_enrich_mode else "Generating Notes"
    step_counter = 1 # Initialize step counter

    with st.status(f"🚀 Initializing {operation_desc} process...", expanded=True) as status:
        try:
            # --- 1. Read Inputs and Settings ---
            status.update(label=f"⚙️ Step {step_counter}: Reading inputs and settings...")
            # Re-validate inside status (belt-and-suspenders)
            is_valid_process, error_msg_process = validate_inputs()
            if not is_valid_process:
                raise ValueError(f"Input validation failed just before processing: {error_msg_process}")

            # Get settings from session state
            meeting_type = st.session_state.selected_meeting_type
            expert_meeting_option = st.session_state.expert_meeting_prompt_option
            notes_model_id = AVAILABLE_MODELS[st.session_state.selected_notes_model_display_name]
            transcription_model_id = AVAILABLE_MODELS[st.session_state.selected_transcription_model_display_name]
            refinement_model_id = AVAILABLE_MODELS[st.session_state.selected_refinement_model_display_name]
            view_edit_enabled = st.session_state.get('view_edit_prompt_enabled', False)
            edited_prompt_text = st.session_state.current_prompt_text.strip()

            use_edited_prompt = view_edit_enabled and edited_prompt_text and meeting_type != "Custom" and not is_enrich_mode
            custom_prompt_text = edited_prompt_text if meeting_type == "Custom" else None
            if meeting_type == "Custom" and not custom_prompt_text:
                 raise ValueError("Custom prompt is required for 'Custom' meeting type but is empty.")

            general_context = st.session_state.get('context_input', "").strip() if st.session_state.get('add_context_enabled') else None
            earnings_mode = st.session_state.get('earnings_call_mode')
            user_existing_notes = st.session_state.get('existing_notes_input', "").strip() if is_enrich_mode else None
            actual_input_type, source_text_data, source_audio_file_obj = get_current_input_data()

            if meeting_type == "Earnings Call":
                earnings_call_topics_text = st.session_state.get("earnings_call_topics", "").strip()
            else:
                 earnings_call_topics_text = ""

            step_counter += 1

            # --- 2. Initialize AI Models ---
            status.update(label=f"🧠 Step {step_counter}: Initializing AI models...")
            transcription_model = genai.GenerativeModel(transcription_model_id, safety_settings=safety_settings)
            refinement_model = genai.GenerativeModel(refinement_model_id, safety_settings=safety_settings)
            notes_model = genai.GenerativeModel(notes_model_id, safety_settings=safety_settings)

            step_counter += 1

            # --- 3. Process Input Source (Text, PDF, or Audio) & Optional Refinement ---
            final_source_text_for_notes = None # This will hold the text used for the final notes step
            st.session_state.raw_transcript = None
            st.session_state.refined_transcript = None
            st.session_state.raw_text_input = None
            st.session_state.refined_text_input = None


            if actual_input_type == "Upload Audio":
                # --- Audio Processing ---
                status.update(label=f"🔊 Step {step_counter}a: Starting Audio Processing...")
                if source_audio_file_obj is None:
                     raise ValueError("Audio file selected but no file object found.")
                st.session_state.uploaded_audio_info = source_audio_file_obj
                status.update(label=f"🔊 Loading source audio file '{source_audio_file_obj.name}'...")

                audio_bytes = source_audio_file_obj.getvalue()
                file_extension = os.path.splitext(source_audio_file_obj.name)[1].lower().replace('.', '')
                audio_format = file_extension
                if audio_format == 'm4a': audio_format = 'mp4'
                elif audio_format in ['oga']: audio_format = 'ogg'
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                except Exception as audio_load_err:
                     if "ffmpeg" in str(audio_load_err).lower() or "Couldn't find ffmpeg or avconv" in str(audio_load_err):
                         raise ValueError(f"❌ **Error:** Could not load audio. `ffmpeg` or `libav` might be missing. Install it (check https://ffmpeg.org/download.html). (Original error: {audio_load_err})")
                     else:
                         raise ValueError(f"❌ Could not load audio file using pydub (format: '{audio_format}'). Ensure file is valid. Error: {audio_load_err}")

                chunk_length_ms = 50 * 60 * 1000 # 50 minutes per chunk
                chunks = make_chunks(audio, chunk_length_ms)
                num_chunks = len(chunks)
                status.update(label=f"🔪 Splitting source audio into {num_chunks} chunk(s)...")

                # --- Step 3a: Transcription per chunk ---
                all_transcripts = []
                processed_audio_chunk_references = []
                for i, chunk in enumerate(chunks):
                    chunk_num = i + 1
                    status.update(label=f"🔄 Processing Source Chunk {chunk_num}/{num_chunks}...")
                    temp_chunk_path = None
                    chunk_file_ref = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                            chunk.export(temp_chunk_file.name, format="wav")
                            temp_chunk_path = temp_chunk_file.name

                        status.update(label=f"☁️ Uploading Source Chunk {chunk_num}/{num_chunks}...")
                        chunk_display_name = f"chunk_{chunk_num}_of_{num_chunks}_{int(time.time())}_{source_audio_file_obj.name}"
                        chunk_file_ref = genai.upload_file(path=temp_chunk_path, display_name=chunk_display_name)
                        processed_audio_chunk_references.append(chunk_file_ref)

                        status.update(label=f"⏳ Waiting for Source Chunk {chunk_num}/{num_chunks} API processing...")
                        polling_start_time = time.time()
                        timeout_seconds = 600
                        while chunk_file_ref.state.name == "PROCESSING":
                            if time.time() - polling_start_time > timeout_seconds:
                                raise TimeoutError(f"Audio processing timed out for chunk {chunk_num} after {timeout_seconds}s.")
                            time.sleep(10)
                            chunk_file_ref = genai.get_file(chunk_file_ref.name)

                        if chunk_file_ref.state.name != "ACTIVE":
                            raise Exception(f"Audio chunk {chunk_num} processing failed. State: {chunk_file_ref.state.name}")

                        status.update(label=f"✍️ Transcribing Source Chunk {chunk_num}/{num_chunks} using {st.session_state.selected_transcription_model_display_name}...")
                        t_prompt = "Transcribe the audio accurately. Output only the raw transcript text."
                        t_response = transcription_model.generate_content(
                            [t_prompt, chunk_file_ref],
                            generation_config=transcription_gen_config, safety_settings=safety_settings
                        )

                        if t_response and hasattr(t_response, 'text') and t_response.text.strip():
                            all_transcripts.append(t_response.text.strip())
                        elif hasattr(t_response, 'prompt_feedback') and t_response.prompt_feedback.block_reason:
                            raise Exception(f"Transcription blocked for chunk {chunk_num}: {t_response.prompt_feedback.block_reason}")
                        else:
                            st.warning(f"⚠️ Source Transcription for chunk {chunk_num} returned empty. Skipping.", icon="🤔")
                            all_transcripts.append("")

                    except Exception as chunk_err:
                        raise Exception(f"❌ Error processing source audio chunk {chunk_num}: {chunk_err}") from chunk_err
                    finally:
                        if temp_chunk_path and os.path.exists(temp_chunk_path):
                            try: os.remove(temp_chunk_path)
                            except OSError as remove_err: st.warning(f"Could not remove temp chunk file {temp_chunk_path}: {remove_err}")

                status.update(label="🧩 Combining source chunk transcripts...")
                st.session_state.raw_transcript = "\n\n".join(all_transcripts).strip()
                raw_audio_transcript = st.session_state.raw_transcript
                final_source_text_for_notes = raw_audio_transcript # Initially use raw transcript

                status.update(label="✅ Step 3a: Full Source Transcription Complete!")
                step_counter += 1 # Increment step counter after transcription

                # --- Step 3b: Audio Refinement ---
                if raw_audio_transcript:
                    status.update(label=f"🧹 Step {step_counter}: Refining audio transcript using {st.session_state.selected_refinement_model_display_name}...")
                    try:
                        refinement_prompt = format_prompt_safe(
                            AUDIO_REFINEMENT_PROMPT,
                            raw_transcript=raw_audio_transcript,
                            context=general_context if general_context else "None provided."
                        )
                        r_response = refinement_model.generate_content(
                            refinement_prompt,
                            generation_config=refinement_gen_config, safety_settings=safety_settings
                        )

                        if r_response and hasattr(r_response, 'text') and r_response.text and r_response.text.strip():
                            st.session_state.refined_transcript = r_response.text.strip()
                            final_source_text_for_notes = st.session_state.refined_transcript # Update for Notes step
                            status.update(label="🧹 Step 3b: Audio Refinement complete!")
                        elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                            st.warning(f"⚠️ Audio Refinement blocked: {r_response.prompt_feedback.block_reason}. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Audio Refinement blocked. Proceeding with raw transcript.")
                        else:
                            st.warning("🤔 Audio Refinement step returned empty response. Using raw transcript for notes.", icon="⚠️")
                            status.update(label="⚠️ Audio Refinement failed. Proceeding with raw transcript.")
                    except Exception as refine_err:
                         st.warning(f"❌ Error during Step 3b (Audio Refinement): {refine_err}. Using raw transcript for notes.", icon="⚠️")
                         status.update(label="⚠️ Audio Refinement error. Proceeding with raw transcript.")
                else:
                    status.update(label="⚠️ Skipping Audio Refinement (Step 3b) as raw transcript is empty.")
                step_counter += 1 # Increment step counter after refinement attempt

            elif actual_input_type == "Paste Text":
                # --- Text Processing ---
                status.update(label=f"✍️ Step {step_counter}a: Processing Pasted Text Input...")
                st.session_state.raw_text_input = source_text_data # Save raw text for display
                raw_pasted_text = st.session_state.raw_text_input
                final_source_text_for_notes = raw_pasted_text # Initially use raw text

                if raw_pasted_text:
                    # --- Step 3b: Text Refinement ---
                    step_counter += 1 # Increment step counter before text refinement
                    status.update(label=f"🧹 Step {step_counter}: Refining input text using {st.session_state.selected_refinement_model_display_name}...")
                    try:
                        text_refinement_prompt = format_prompt_safe(
                            TEXT_REFINEMENT_PROMPT,
                            raw_text=raw_pasted_text,
                            context=general_context if general_context else "None provided."
                        )
                        r_response = refinement_model.generate_content(
                            text_refinement_prompt,
                            generation_config=refinement_gen_config, safety_settings=safety_settings
                        )

                        if r_response and hasattr(r_response, 'text') and r_response.text and r_response.text.strip():
                            st.session_state.refined_text_input = r_response.text.strip()
                            final_source_text_for_notes = st.session_state.refined_text_input # Update for Notes step
                            status.update(label="🧹 Step 3b: Text Refinement complete!")
                        elif hasattr(r_response, 'prompt_feedback') and r_response.prompt_feedback.block_reason:
                            st.warning(f"⚠️ Text Refinement blocked: {r_response.prompt_feedback.block_reason}. Using original pasted text for notes.", icon="⚠️")
                            status.update(label="⚠️ Text Refinement blocked. Proceeding with original text.")
                        else:
                            st.warning("🤔 Text Refinement step returned empty response. Using original pasted text for notes.", icon="⚠️")
                            status.update(label="⚠️ Text Refinement failed. Proceeding with original text.")
                    except Exception as refine_err:
                         st.warning(f"❌ Error during Step 3b (Text Refinement): {refine_err}. Using original pasted text for notes.", icon="⚠️")
                         status.update(label="⚠️ Text Refinement error. Proceeding with original text.")
                else:
                     status.update(label="⚠️ Skipping Text Refinement (Step 3b) as input text is empty.")
                step_counter += 1 # Increment step counter after text refinement attempt


            elif actual_input_type == "Upload PDF":
                # --- PDF Processing (No Refinement) ---
                status.update(label=f"📄 Step {step_counter}: Processing PDF Input...")
                final_source_text_for_notes = source_text_data
                if not final_source_text_for_notes:
                    raise ValueError("Failed to extract text from the uploaded PDF.")
                status.update(label="📄 Step 3: PDF Text Extracted.")
                step_counter += 2 # Skip the 'refinement' step number for consistency

            else:
                raise ValueError(f"Unknown input type encountered: {actual_input_type}")


            # --- 4. Prepare Final Prompt for Notes/Enrichment ---
            if not final_source_text_for_notes:
                 raise ValueError(f"No source text available for Step {step_counter} (from {actual_input_type} processing). Cannot generate or enrich notes.")

            status.update(label=f"📝 Step {step_counter}: Preparing final prompt for {operation_desc}...")
            final_api_prompt = None
            api_payload_parts = []
            prompt_template_base_text = None
            gen_config_to_use = main_gen_config

            format_kwargs = {
                'transcript': final_source_text_for_notes, # Use the potentially refined text/transcript
                'context_section': f"\n**ADDITIONAL CONTEXT (Use for understanding):**\n{general_context}\n---" if general_context else ""
            }

            # Determine which prompt to use based on priority
            if meeting_type == "Custom":
                prompt_template_base_text = custom_prompt_text
                final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                api_payload_parts = [final_api_prompt]
                status.update(label=f"📝 Using user's Custom prompt...")

            elif is_enrich_mode: # Enrichment mode
                prompt_template_base_text = PROMPTS["Earnings Call"]["Enrich Existing Notes"]
                gen_config_to_use = enrichment_gen_config
                # Format topic instructions for enrichment focus
                topic_instructions = ""
                if earnings_call_topics_text:
                    formatted_topics = [f"- {line.strip()}" for line in earnings_call_topics_text.split('\n') if line.strip()]
                    topic_list_str = "\n".join(formatted_topics)
                    topic_instructions = (f"Focus enrichment primarily on details related to the following user-specified topic structure:\n{topic_list_str}\n\n"
                                        f"Also incorporate any other highly significant financial or strategic points found in the transcript, potentially under an 'Other Key Points' section if they don't fit the provided structure.")
                else: # No specific topics provided
                    topic_instructions = (f"Since no specific topics were provided, identify the logical main themes in the transcript (e.g., Financials, Strategy, Outlook, Q&A) and enrich the user's existing notes based on significant information related to those themes.\n"
                                          f"Include any other highly significant points under an 'Other Key Points' section if relevant.")
                format_kwargs["topic_instructions"] = topic_instructions
                if user_existing_notes is None: raise ValueError("Existing notes required for Enrichment mode but not found.")
                format_kwargs["existing_notes"] = user_existing_notes
                final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                api_payload_parts = [final_api_prompt]
                status.update(label=f"📝 Using standard Enrich prompt...")

            elif use_edited_prompt: # Edit enabled and text area has content
                 prompt_template_base_text = edited_prompt_text # Use the edited text as the template
                 # Re-generate and add topic instructions if it's Generate New Notes Earnings Call
                 if meeting_type == "Earnings Call" and earnings_mode == "Generate New Notes":
                     topic_instructions = "" # Calculate as before
                     if earnings_call_topics_text:
                         formatted_topics = []
                         for line in earnings_call_topics_text.split('\n'):
                             trimmed_line = line.strip()
                             if trimmed_line and not trimmed_line.startswith(('-', '*', '#')): formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                             elif trimmed_line: formatted_topics.append(trimmed_line)
                         topic_list_str = "\n".join(formatted_topics)
                         topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                               f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                               f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
                     else: # No topics provided
                         topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                               f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                               f"Place all relevant details under the most appropriate heading.")
                     format_kwargs["topic_instructions"] = topic_instructions
                 # Format the *edited* prompt text
                 final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                 api_payload_parts = [final_api_prompt]
                 status.update(label=f"📝 Using edited prompt for {operation_desc}...")

            else: # Standard path (Not Custom, Not Enrich, Not Edited)
                 if meeting_type == "Expert Meeting":
                    if expert_meeting_option == "Option 1: Existing (Detailed & Strict)":
                        prompt_template_base_text = PROMPTS["Expert Meeting"]["Option 1: Existing (Detailed & Strict)"]
                    else: # Option 2 and 3 use Option 2 template
                        prompt_template_base_text = PROMPTS["Expert Meeting"]["Option 2: Less Verbose (Default)"]
                 elif meeting_type == "Earnings Call": # Must be Generate New Notes mode here
                     prompt_template_base_text = PROMPTS["Earnings Call"]["Generate New Notes"]
                     gen_config_to_use = main_gen_config
                     # Format topic instructions
                     topic_instructions = "" # Calculate as before
                     if earnings_call_topics_text:
                         formatted_topics = []
                         for line in earnings_call_topics_text.split('\n'):
                             trimmed_line = line.strip()
                             if trimmed_line and not trimmed_line.startswith(('-', '*', '#')): formatted_topics.append(f"- **{trimmed_line.strip(':')}**")
                             elif trimmed_line: formatted_topics.append(trimmed_line)
                         topic_list_str = "\n".join(formatted_topics)
                         topic_instructions = (f"Structure the main body of the notes under the following user-specified headings EXACTLY as provided:\n{topic_list_str}\n\n"
                                               f"- **Other Key Points** (Use this MANDATORY heading for important info NOT covered by the topics above)\n\n"
                                               f"Place all relevant details under the most appropriate heading. If a specific user topic isn't discussed in the transcript, state 'Not discussed' under that heading.")
                     else: # No topics provided
                         topic_instructions = (f"Since no specific topics were provided, first identify the logical main themes discussed in the call (e.g., Financial Highlights, Segment Performance, Strategic Initiatives, Outlook/Guidance, Q&A Key Points). Use these themes as **bold headings**.\n"
                                               f"Include a final mandatory section:\n- **Other Key Points** (Use this heading for any important information that doesn't fit neatly into the main themes you identified)\n\n"
                                               f"Place all relevant details under the most appropriate heading.")
                     format_kwargs["topic_instructions"] = topic_instructions
                 else:
                     raise ValueError(f"Unhandled meeting type '{meeting_type}' in standard prompt selection logic.")

                 if not prompt_template_base_text:
                     raise ValueError(f"Could not find standard prompt template (Meeting: {meeting_type}, Option/Mode: {expert_meeting_option if meeting_type=='Expert Meeting' else earnings_mode}).")

                 final_api_prompt = format_prompt_safe(prompt_template_base_text, **format_kwargs)
                 api_payload_parts = [final_api_prompt]
                 status.update(label=f"📝 Using standard prompt for {operation_desc}...")


            if not final_api_prompt or not api_payload_parts:
                raise ValueError("Failed to prepare the final prompt for the API call.")

            step_counter += 1 # Increment for the main generation step

            # --- 5. Execute API Call (Notes/Enrichment) ---
            try:
                status.update(label=f"✨ Step {step_counter}: {operation_desc} using {st.session_state.selected_notes_model_display_name}...")
                # Optional Debug: Print final prompt before sending
                # print("-" * 20 + " FINAL PROMPT TO API " + "-" * 20)
                # print(final_api_prompt)
                # print("-" * 60)

                response = notes_model.generate_content(
                    api_payload_parts,
                    generation_config=gen_config_to_use,
                    safety_settings=safety_settings
                )

                # --- 6. Process Response & Handle Summary Step ---
                generated_content = None
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    generated_content = response.text.strip()
                    status.update(label=f"✅ Initial {operation_desc} successful!")

                    # Check if Expert Meeting Option 3 (Summary step) is needed
                    is_expert_summary_step = (meeting_type == "Expert Meeting" and
                                              expert_meeting_option == "Option 3: Option 2 + Executive Summary" and
                                              not use_edited_prompt) # Don't summarize if prompt was edited

                    if is_expert_summary_step:
                        step_counter += 1 # Increment for summary step
                        status.update(label=f"✨ Step {step_counter}: Generating Executive Summary...")
                        summary_prompt_template = PROMPTS["Expert Meeting"].get(EXPERT_MEETING_SUMMARY_PROMPT_KEY)
                        if not summary_prompt_template:
                             st.warning("⚠️ Could not find summary prompt template. Skipping summary step.", icon="❗")
                             st.session_state.generated_notes = generated_content
                             status.update(label="⚠️ Summary Prompt Missing. Only detailed notes generated.", state="warning")
                        else:
                            summary_kwargs = {'generated_notes': generated_content}
                            summary_prompt = format_prompt_safe(summary_prompt_template, **summary_kwargs)
                            try:
                                summary_response = notes_model.generate_content(
                                    summary_prompt,
                                    generation_config=summary_gen_config,
                                    safety_settings=safety_settings
                                )
                                if summary_response and hasattr(summary_response, 'text') and summary_response.text.strip():
                                    summary_text = summary_response.text.strip()
                                    st.session_state.generated_notes = f"{generated_content}\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{summary_text}"
                                    status.update(label="✅ Notes and Summary generated successfully!", state="complete")
                                else:
                                    reason = "Blocked" if hasattr(summary_response, 'prompt_feedback') else "Empty Response"
                                    st.warning(f"⚠️ Summary generation failed or was blocked ({reason}). Only detailed notes provided.", icon="⚠️")
                                    st.session_state.generated_notes = generated_content
                                    status.update(label="⚠️ Summary Failed/Blocked. Only detailed notes generated.", state="warning")
                            except Exception as summary_err:
                                 st.warning(f"❌ Error during summary generation step: {summary_err}. Only detailed notes provided.", icon="⚠️")
                                 st.session_state.generated_notes = generated_content
                                 status.update(label="⚠️ Summary Error. Only detailed notes generated.", state="warning")
                    else:
                        st.session_state.generated_notes = generated_content
                        status.update(label=f"✅ {operation_desc} completed successfully!", state="complete")

                    # --- 7. Post-generation Steps ---
                    if st.session_state.generated_notes:
                        st.session_state.edited_notes_text = st.session_state.generated_notes # Populate editor
                        add_to_history(st.session_state.generated_notes) # Add to history

                        status.update(label="💡 Suggesting filename...")
                        fname_label = meeting_type.replace(" ","_")
                        if is_enrich_mode: fname_label = "Enriched_Earnings_Call"
                        suggested_fname = generate_suggested_filename(st.session_state.generated_notes, fname_label)
                        st.session_state.suggested_filename = suggested_fname

                # Handle API errors/blocks for the main notes/enrichment call
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    st.session_state.error_message = f"⚠️ {operation_desc} was blocked by the API. Reason: {response.prompt_feedback.block_reason}. Please modify inputs or prompt if applicable."
                    status.update(label=f"❌ Blocked: {response.prompt_feedback.block_reason}", state="error")
                elif response is not None:
                    st.session_state.error_message = f"🤔 The AI returned an empty response during {operation_desc}. Try adjusting the input or model."
                    status.update(label="❌ Error: AI returned empty response.", state="error")
                else: # No response object at all
                    st.session_state.error_message = f"😥 The {operation_desc} API call failed (No response received). Check connection or API status."
                    status.update(label="❌ Error: API call failed (No response).", state="error")

            except Exception as api_call_err:
                 st.session_state.error_message = f"❌ Error during Step {step_counter} ({operation_desc} API Call): {api_call_err}"
                 status.update(label=f"❌ Error during API call: {api_call_err}", state="error")

        except Exception as e:
             st.session_state.error_message = f"❌ Processing Error: {e}"
             status.update(label=f"❌ Error: {e}", state="error")

        finally:
            # --- 8. Cleanup ---
            st.session_state.processing = False # Mark processing as finished regardless of outcome
            if processed_audio_chunk_references:
                 st.toast(f"☁️ Performing final cleanup of {len(processed_audio_chunk_references)} cloud audio chunk(s)...", icon="🗑️")
                 refs_to_delete = list(processed_audio_chunk_references)
                 for file_ref in refs_to_delete:
                    try:
                        if file_ref and hasattr(file_ref, 'name'):
                           genai.delete_file(file_ref.name)
                           processed_audio_chunk_references.remove(file_ref)
                        else:
                            st.warning(f"Skipping cleanup for invalid file reference: {file_ref}", icon="⚠️")
                    except Exception as final_cleanup_error:
                        # Suppress detailed errors during final cleanup unless debugging
                        # st.warning(f"Final cloud audio chunk cleanup failed for {getattr(file_ref, 'name', 'Unknown File')}: {final_cleanup_error}", icon="⚠️")
                        print(f"Warning: Final cloud audio chunk cleanup failed for {getattr(file_ref, 'name', 'Unknown File')}: {final_cleanup_error}")
                 st.session_state.processed_audio_chunk_references = []
            st.rerun() # Rerun one last time to update UI


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini | App by SynthNotes AI")

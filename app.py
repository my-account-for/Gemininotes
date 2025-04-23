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
    /* Overall App Background */
    .stApp {
        background: linear-gradient(to bottom right, #F0F2F6, #FFFFFF); /* Subtle gradient */
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1000px;
        margin: auto;
    }

    /* General Container Styling (using st.container(border=True)) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"][style*="border"] {
         background-color: #FFFFFF;
         border: 1px solid #E5E7EB; /* Softer border */
         border-radius: 0.75rem; /* More rounded */
         padding: 1.5rem; /* Inner padding */
         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Softer shadow */
         margin-bottom: 1.5rem; /* Space between containers */
    }


    /* Headers */
    h1 {
        color: #111827; font-weight: 700; text-align: center; margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #1F2937; font-weight: 600; border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    /* App Subtitle */
    .main .block-container > div:nth-child(3) > div > div > div > p {
       text-align: center; color: #4B5563; font-size: 1.1rem; margin-bottom: 2rem;
    }


    /* Input Widgets */
    .stTextInput textarea, .stFileUploader div[data-testid="stFileUploaderDropzone"], .stTextArea textarea {
        border-radius: 0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05); transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput textarea:focus, .stFileUploader div[data-testid="stFileUploaderDropzone"]:focus-within, .stTextArea textarea:focus {
        border-color: #007AFF; box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05), 0 0 0 3px rgba(0, 122, 255, 0.2);
        background-color: #FFFFFF;
    }
    .stFileUploader p { font-size: 0.95rem; color: #4B5563; }

    /* Radio Buttons */
    div[role="radiogroup"] > label {
        background-color: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 0.5rem;
        padding: 0.6rem 1rem; margin-right: 0.5rem; transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        display: inline-block; /* Prevent stretching */
        margin-bottom: 0.5rem; /* Add space below radio */
    }
    div[role="radiogroup"] label:hover { border-color: #9CA3AF; }
    div[role="radiogroup"] input[type="radio"]:checked + div {
       background-color: #EFF6FF; border-color: #007AFF; color: #005ECB;
       font-weight: 500; box-shadow: 0 1px 3px rgba(0, 122, 255, 0.1);
    }

    /* Checkbox styling */
    .stCheckbox {
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
    }
    .stCheckbox label span { font-weight: 500; color: #374151; }

    /* Selectbox Styling */
    .stSelectbox > div {
        border-radius: 0.5rem; border: 1px solid #D1D5DB; background-color: #F9FAFB;
    }
    .stSelectbox > div:focus-within {
         border-color: #007AFF; box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.2);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: 600;
        transition: all 0.2s ease-in-out; border: none; width: 100%;
    }
    .stButton > button[kind="primary"] {
        background-color: #007AFF; color: white;
        box-shadow: 0 4px 6px rgba(0, 122, 255, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #005ECB;
        box-shadow: 0 7px 14px rgba(0, 122, 255, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
     .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.4); outline: none;
    }
     .stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
         background-color: #D1D5DB; color: #6B7280; box-shadow: none;
         transform: none; cursor: not-allowed;
     }

    /* Download Button */
    .stDownloadButton > button {
        border-radius: 0.5rem; padding: 0.6rem 1.2rem; font-weight: 500;
        background-color: #F3F4F6; color: #1F2937; border: 1px solid #D1D5DB;
        transition: background-color 0.2s ease-in-out; width: auto; margin-top: 1rem;
    }
    .stDownloadButton > button:hover { background-color: #E5E7EB; border-color: #9CA3AF; }

    /* Output Area Styling */
    .output-container {
        background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 0.75rem;
        padding: 1.5rem; margin-top: 1.5rem; min-height: 150px;
    }
    .output-container .stMarkdown {
        background-color: transparent; border: none; padding: 0;
        color: #374151; font-size: 1rem; line-height: 1.6;
    }
    .output-container .stMarkdown h3, .output-container .stMarkdown h4, .output-container .stMarkdown strong {
       color: #111827; font-weight: 600;
    }
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem; }
    .output-container .initial-prompt {
        color: #6B7280; font-style: italic; text-align: center; padding-top: 2rem;
    }

    /* Footer */
    footer {
        text-align: center; color: #9CA3AF; font-size: 0.8rem;
        padding-top: 2rem; padding-bottom: 1rem;
    }
    footer a { color: #6B7280; text-decoration: none; }
     footer a:hover { color: #007AFF; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# --- Define Available Models ---
AVAILABLE_MODELS = {
    # Stable Models
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume, Lower Intelligence Tasks)": "gemini-1.5-flash-8b",
    # Newer/Preview Models
    "Gemini 2.0 Flash (Next Gen Speed/Multimodal)": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite (Cost Efficiency & Low Latency)": "gemini-2.0-flash-lite",
    "Gemini 2.5 Flash Preview (Adaptive & Cost Efficient)": "gemini-2.5-flash-preview-04-17",
    "Gemini 2.5 Pro Preview (Enhanced Reasoning & Multimodal)": "gemini-2.5-pro-preview-03-25", # New Default
}
# --- Set New Default Model ---
DEFAULT_MODEL_NAME = "Gemini 2.5 Pro Preview (Enhanced Reasoning & Multimodal)"
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
     DEFAULT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0] # Fallback


# --- Define Meeting Types ---
MEETING_TYPES = ["Expert Meeting", "Earnings Call"]
DEFAULT_MEETING_TYPE = MEETING_TYPES[0]


# --- Load API Key and Configure Gemini Client ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("### 🔑 API Key Not Found!", icon="🚨")
    st.markdown("Please ensure your `GEMINI_API_KEY` is set in your environment variables or in a `.env` file.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    generation_config = {
        "temperature": 0.7, "top_p": 1.0, "top_k": 32,
        "max_output_tokens": 8192, "response_mime_type": "text/plain",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
except Exception as e:
     st.error(f"### 💥 Error Configuring Google AI Client", icon="🚨")
     st.error(f"Details: {e}")
     st.stop()


# --- Initialize Session State ---
if 'processing' not in st.session_state: st.session_state.processing = False
if 'generated_notes' not in st.session_state: st.session_state.generated_notes = None
if 'error_message' not in st.session_state: st.session_state.error_message = None
if 'uploaded_audio_info' not in st.session_state: st.session_state.uploaded_audio_info = None
if 'add_context_enabled' not in st.session_state: st.session_state.add_context_enabled = False
if 'selected_model_display_name' not in st.session_state:
    st.session_state.selected_model_display_name = DEFAULT_MODEL_NAME
# Add state for meeting type
if 'selected_meeting_type' not in st.session_state:
    st.session_state.selected_meeting_type = DEFAULT_MEETING_TYPE


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from a PDF file stream, updates session state on error."""
    try:
        pdf_file_stream.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        # Ensure text is extracted correctly, handle potential None returns per page
        text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        text = "\n".join(text_parts)
        return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e:
        st.session_state.error_message = f"📄 Error reading PDF: {e}. Is it password-protected or corrupted?"
        return None
    except Exception as e:
        st.session_state.error_message = f"⚙️ An unexpected error occurred during PDF extraction: {e}"
        return None

def create_expert_meeting_prompt(transcript, context=None):
    """Creates the prompt for the 'Expert Meeting' type."""
    # This is the original prompt structure
    prompt_parts = [
        "You are an expert meeting note-taker analyzing an expert consultation or similar focused meeting.",
        "Generate detailed, factual notes from the provided meeting transcript.",
        "Follow this specific structure EXACTLY:",
        "\n**Structure:**",
        "- **Opening overview or Expert background (Optional):**",
        "  - If the transcript begins with a meeting overview, agenda, or introduction of an expert's background, include it as the FIRST section.",
        "  - Present this information in bullet points.",
        "  - Capture ALL details mentioned (names, dates, numbers, objectives, context) accurately.",
        "  - Use easy-to-understand language.",
        "  - DO NOT add any summary, interpretation, or your own words.",
        "- **Q&A format:**",
        "  - Structure the main body of the notes strictly in a Question and Answer format.",
        "  - **Questions:**",
        "    - Extract the clear questions asked during the meeting.",
        "    - Rephrase slightly ONLY for clarity if absolutely necessary, keeping them short and focused.",
        "    - Ensure you correctly identify questions and do not mistake parts of answers for questions.",
        "    - Format each question clearly, e.g., starting with 'Q:' or bolding.",
        "  - **Answers:**",
        "    - Present the response to each question using bullet points directly below the question.",
        "    - Each bullet point must be a complete sentence containing one distinct piece of factual information.",
        "    - Capture ALL specific details mentioned in the answer: data points, statistics, dates, names, examples, monetary values, percentages, etc.",
        "    - DO NOT use sub-bullets or headings within an answer section.",
        "    - DO NOT include any interpretations, summaries, conclusions, action items, or to-dos. Only document what was explicitly stated.",
        "\n**Additional Instructions:**",
        "- Accuracy is paramount. Ensure every piece of factual information is captured precisely.",
        "- Maintain clarity and conciseness.",
        "- Avoid adding any personal insights, opinions, or information not present in the transcript.",
        "- If a section is not present in the transcript, omit it.",
        "\n---",
        "\n**MEETING TRANSCRIPT:**\n",
        transcript,
        "\n---"
    ]
    if context:
        prompt_parts.extend([
            "\n**ADDITIONAL CONTEXT (Use this to better understand the transcript):**\n",
            context,
            "\n---"
        ])
    prompt_parts.append("\n**GENERATED NOTES:**\n")
    return "\n".join(prompt_parts)

def create_earnings_call_prompt(transcript, context=None):
    """Creates the prompt for the 'Earnings Call' type."""
    prompt_parts = [
        "You are a financial analyst summarizing an earnings call transcript.",
        "Generate detailed, factual notes, extracting key information and statements.",
        "Structure the notes logically based on common earnings call flow.",
        "Pay close attention to numerical data, guidance, strategic comments, and management sentiment.",

        "\n**General Structure to Follow (Adapt as necessary based on content):**",
        "- **Call Participants:** (List names and titles if mentioned)",
        "- **Opening Remarks/CEO Statement:** (Key themes, company vision, major achievements/challenges)",
        "- **Financial Highlights:** (Revenue, Profitability, EPS, Margins - include specific numbers and YoY/QoQ comparisons if stated)",
        "- **Segment Performance:** (Breakdown by business unit, geography, or product line if detailed)",
        "- **Key Business Updates/Strategy:** (New initiatives, partnerships, market position, M&A activity)",

        "\n**Industry-Specific Sections (Include *only* if the company clearly belongs to one of these sectors and discusses these topics):**",
        "  - **If IT Services Company:**",
        "    - **Future Investments / Capital Allocation:** (Mention specifics: R&D, technology, acquisitions, buybacks, dividends)",
        "    - **Talent Supply Chain:** (Hiring trends, attrition, utilization, training, location strategy)",
        "    - **Org Structure Changes:** (Leadership changes, reorganizations)",
        "    - **Short-term Outlook & Demand:**",
        "      - **Guidance:** (Specific quarterly/annual revenue or margin targets)",
        "      - **Order Booking / Pipeline:** (Deal wins, TCV, book-to-bill ratio)",
        "      - **Macro Impact:** (Management commentary on economic slowdown, client spending)",
        "    - **Other Key IT Comments:** (Cloud, AI, digital transformation progress, etc.)",
        "  - **If QSR (Quick Service Restaurant) Company:**",
        "    - **Customer Proposition / Menu Strategy:** (New product launches, value offerings, marketing campaigns, loyalty programs)",
        "    - **Business Update (Operations):** (SSSG/Comps, Traffic, Average Check/Ticket, Price increases)",
        "    - **Unit Economics / Store Performance:** (Restaurant-level margins, cost pressures - food, labor)",
        "    - **Store Network:** (New store openings, closures, remodels, footprint strategy - domestic/international)",
        "    - **Other Key QSR Comments:** (Digital sales mix, delivery, technology adoption, etc.)",

        "\n- **Q&A Session Summary:**",
        "  - Summarize key questions asked by analysts and management's core responses.",
        "  - Structure as Q: [Analyst Question Summary] / A: [Management Response Summary - use bullets for key points]",
        "  - Focus on new information or clarifications provided during Q&A.",
        "- **Guidance Summary (Reiterate/Confirm):** (Consolidated view of forward-looking statements if not fully covered earlier)",
        "- **Closing Remarks:** (Final key message)",

        "\n**Additional Instructions:**",
        "- Extract direct quotes for impactful statements where appropriate, using quotation marks.",
        "- Be factual and objective. Avoid interpretation or adding external information.",
        "- If a section is not discussed, omit it.",
        "\n---",
        "\n**EARNINGS CALL TRANSCRIPT:**\n",
        transcript,
        "\n---"
    ]
    if context:
        prompt_parts.extend([
            "\n**ADDITIONAL CONTEXT (Company name, ticker, previous quarter info if known):**\n",
            context,
            "\n---"
        ])
    prompt_parts.append("\n**GENERATED EARNINGS CALL SUMMARY:**\n")
    return "\n".join(prompt_parts)


# --- Streamlit App UI ---
st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Input Section ---
with st.container(border=True):
    # Row 1: Meeting Type and Model Selection
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Meeting Details")
        # Update session state when radio changes
        st.session_state.selected_meeting_type = st.radio(
            "Select Meeting Type:",
            options=MEETING_TYPES,
            key="meeting_type_radio",
            index=MEETING_TYPES.index(st.session_state.selected_meeting_type), # Set default from state
            horizontal=True,
            help="Choose the type of meeting to tailor the note structure."
        )
    with col1b:
        st.subheader("AI Model")
        st.session_state.selected_model_display_name = st.selectbox(
            "Choose model:",
            options=list(AVAILABLE_MODELS.keys()),
            key="model_select",
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_display_name),
            help="Select the Gemini model. Preview models may offer newer features."
        )

    st.divider()

    # Row 2: Source Input
    st.subheader("Source Input")
    input_method = st.radio(
        "Input type:",
        ("Paste Text", "Upload PDF", "Upload Audio"),
        key="input_method_radio",
        horizontal=True,
        label_visibility="collapsed" # Hide label as subheader exists
    )

    text_input_area = None
    pdf_file_uploader = None
    audio_file_uploader = None

    if input_method == "Paste Text":
        text_input_area = st.text_area("Paste transcript:", height=150, key="text_input", placeholder="Paste the full meeting transcript here...")
    elif input_method == "Upload PDF":
        pdf_file_uploader = st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader", help="Upload a PDF file containing the meeting transcript.")
    else: # Upload Audio
        audio_file_uploader = st.file_uploader(
            "Upload Audio:",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'],
            key="audio_uploader",
            help="Upload an audio recording (WAV, MP3, M4A, etc.). Processing time depends on length."
        )

    # Row 3: Context Checkbox
    st.divider()
    add_context_enabled = st.checkbox(
        "Add Context (Optional)",
        key="add_context_cb",
        value=st.session_state.add_context_enabled,
        help="Check this box to provide additional background information."
    )
    st.session_state.add_context_enabled = add_context_enabled


# --- Conditional Context Input Section ---
if st.session_state.add_context_enabled:
    with st.container(border=True):
        st.subheader("Context Details")
        # Use context_input_area = ... if needed, but key access is primary
        st.text_area(
            "Provide background:",
            height=150,
            key="context_input",
            placeholder="Add context like attendees, company name, goals, project name...",
            help="Adding context helps the AI understand the conversation better."
        )


# --- Generate Button ---
st.write("")
generate_button = st.button(
    "🚀 Generate Notes",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.processing
)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True) # Marker for CSS

    if st.session_state.processing:
        st.info("⏳ Processing your request... Please wait.", icon="🧠")
    elif st.session_state.error_message:
        st.error(st.session_state.error_message, icon="🚨")
        st.session_state.error_message = None # Clear error after displaying
    elif st.session_state.generated_notes:
        st.subheader("✅ Generated Notes")
        st.markdown(st.session_state.generated_notes)
        st.download_button(
             label="⬇️ Download Notes (.txt)",
             data=st.session_state.generated_notes,
             file_name="meeting_notes.txt",
             mime="text/plain",
             key='download-txt'
         )
    else:
        st.markdown("<p class='initial-prompt'>Generated notes will appear here once processed.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    st.session_state.processing = True
    st.session_state.generated_notes = None
    st.session_state.error_message = None
    st.rerun()

if st.session_state.processing:
    transcript = None
    audio_file_input = None
    # Retrieve selections from session state
    input_type = st.session_state.get("input_method_radio", "Paste Text")
    meeting_type = st.session_state.get("selected_meeting_type", DEFAULT_MEETING_TYPE)
    selected_model_display_name = st.session_state.get("selected_model_display_name", DEFAULT_MODEL_NAME)
    selected_model_id = AVAILABLE_MODELS.get(selected_model_display_name, AVAILABLE_MODELS[DEFAULT_MODEL_NAME])

    # Retrieve input widget values from session state
    text_content = st.session_state.get("text_input", "")
    pdf_file = st.session_state.get("pdf_uploader")
    audio_file = st.session_state.get("audio_uploader")

    # Basic input validation
    if input_type == "Paste Text" and not text_content: st.session_state.error_message = "⚠️ Text area is empty."
    elif input_type == "Upload PDF" and not pdf_file: st.session_state.error_message = "⚠️ No PDF file uploaded."
    elif input_type == "Upload Audio" and not audio_file: st.session_state.error_message = "⚠️ No audio file uploaded."

    # Get Context (if enabled)
    final_context = None
    if st.session_state.get("add_context_enabled", False):
        final_context = st.session_state.get("context_input", "").strip()

    if not st.session_state.error_message:
        try:
            # --- Prepare Input Data (Transcript or Audio File Object) ---
            if input_type == "Paste Text":
                transcript = text_content.strip()
            elif input_type == "Upload PDF":
                st.toast("📄 Processing PDF...", icon="⏳")
                pdf_stream = io.BytesIO(pdf_file.getvalue())
                transcript = extract_text_from_pdf(pdf_stream)
                if transcript and not st.session_state.error_message: st.toast("📄 PDF processed!", icon="✅")
                elif not transcript and not st.session_state.error_message: st.session_state.error_message = "⚠️ PDF contains no extractable text."
            elif input_type == "Upload Audio":
                st.toast(f"☁️ Uploading '{audio_file.name}'...", icon="⬆️")
                audio_bytes = audio_file.getvalue()
                audio_file_for_api = genai.upload_file(content=audio_bytes, display_name=f"audio_{int(time.time())}", mime_type=audio_file.type)
                st.session_state.uploaded_audio_info = audio_file_for_api
                polling_start_time = time.time()
                while audio_file_for_api.state.name == "PROCESSING":
                    if time.time() - polling_start_time > 300: raise TimeoutError("Audio processing timed out.")
                    st.toast(f"🎧 Processing '{audio_file.name}'...", icon="⏳")
                    time.sleep(5)
                    audio_file_for_api = genai.get_file(audio_file_for_api.name)
                if audio_file_for_api.state.name == "FAILED":
                    st.session_state.error_message = f"😥 Audio processing failed: {audio_file_for_api.name}"
                    try: genai.delete_file(audio_file_for_api.name)
                    except Exception: pass
                elif audio_file_for_api.state.name == "ACTIVE":
                    st.toast(f"🎧 Audio ready!", icon="✅")
                    audio_file_input = audio_file_for_api # Assign the file object
                else: st.session_state.error_message = f"Unexpected audio state: {audio_file_for_api.state.name}"

            # --- Select Prompt Function based on Meeting Type ---
            if not st.session_state.error_message:
                prompt_function = None
                if meeting_type == "Expert Meeting":
                    prompt_function = create_expert_meeting_prompt
                elif meeting_type == "Earnings Call":
                    prompt_function = create_earnings_call_prompt
                else: # Fallback or error
                    st.session_state.error_message = f"Unknown meeting type selected: {meeting_type}"
                    prompt_function = create_expert_meeting_prompt # Default to expert?

                # --- Construct Final Prompt and Call API ---
                if prompt_function and (transcript or audio_file_input):
                    st.toast(f"🧠 Generating notes with {selected_model_display_name}...", icon="✨")
                    model = genai.GenerativeModel(model_name=selected_model_id, safety_settings=safety_settings, generation_config=generation_config)
                    response = None

                    if input_type == "Paste Text" or input_type == "Upload PDF":
                        # Generate prompt using the transcript directly
                        final_prompt = prompt_function(transcript, final_context)
                        response = model.generate_content(final_prompt)
                    elif input_type == "Upload Audio":
                        # Generate the base prompt structure (without transcript)
                        # Pass a placeholder or modify functions if needed, but simpler to pass None
                        base_prompt_text = prompt_function(transcript=None, context=final_context) # Pass None for transcript
                        # Remove the placeholder line if it exists
                        base_prompt_text = base_prompt_text.replace("\n**MEETING TRANSCRIPT:**\n\n---","\n---")
                        base_prompt_text = base_prompt_text.replace("\n**EARNINGS CALL TRANSCRIPT:**\n\n---","\n---")

                        # Wrap with audio instructions
                        audio_wrapper_prompt = (
                            "1. First, accurately transcribe the provided audio file.\n"
                            "2. Then, using the transcription you just generated, create detailed notes based on the following structure and instructions:\n"
                            "---\n"
                            f"{base_prompt_text}" # Insert the specific meeting type instructions
                            # No need for the final **GENERATED NOTES:** part here, the model handles it.
                        )
                        response = model.generate_content([audio_wrapper_prompt, audio_file_input])
                        # Clean up audio file
                        try:
                            if st.session_state.uploaded_audio_info:
                                genai.delete_file(st.session_state.uploaded_audio_info.name)
                                st.session_state.uploaded_audio_info = None
                                st.toast("☁️ Temp audio file cleaned up.", icon="🗑️")
                        except Exception as delete_err: st.warning(f"Could not delete temp audio file: {delete_err}", icon="⚠️")

                    # --- Handle Response ---
                    if response and response.text:
                        st.session_state.generated_notes = response.text
                        st.toast("🎉 Notes generated successfully!", icon="✅")
                    elif response: st.session_state.error_message = "🤔 AI returned an empty response."
                    else: st.session_state.error_message = "😥 AI generation failed."

        except Exception as e:
            st.session_state.error_message = f"❌ An error occurred: {e}"
            # Attempt audio cleanup on error
            try:
                if input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
            except Exception: pass

    # --- Finish processing ---
    st.session_state.processing = False
    st.rerun()


# --- Footer ---
st.divider()
st.caption("Powered by [Google Gemini](https://deepmind.google/technologies/gemini/) | App by SynthNotes AI")

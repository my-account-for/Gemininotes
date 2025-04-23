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

# --- Custom CSS Injection ---
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
    /* Targets containers used for input/output sections */
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
    /* Find the specific markdown element for subtitle - adjust index if layout changes */
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
    }
    div[role="radiogroup"] label:hover { border-color: #9CA3AF; }
    div[role="radiogroup"] input[type="radio"]:checked + div {
       background-color: #EFF6FF; border-color: #007AFF; color: #005ECB;
       font-weight: 500; box-shadow: 0 1px 3px rgba(0, 122, 255, 0.1);
    }

    /* Checkbox styling */
    .stCheckbox {
        margin-top: 1rem; /* Add some space above checkbox */
        padding: 0.5rem;
        background-color: #F9FAFB; /* Slight background */
        border-radius: 0.5rem;
    }
    .stCheckbox label span {
        font-weight: 500;
        color: #374151;
    }

    /* Selectbox Styling */
    .stSelectbox > div {
        border-radius: 0.5rem;
        border: 1px solid #D1D5DB;
        background-color: #F9FAFB;
    }
    .stSelectbox > div:focus-within {
         border-color: #007AFF;
         box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.2);
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
    .output-container { /* Target the container used for output */
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
    .output-container .stAlert { margin-top: 1rem; border-radius: 0.5rem;} /* Style alerts inside output */
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
# Using user-friendly names as keys and API IDs as values
AVAILABLE_MODELS = {
    # Stable Models
    "Gemini 1.5 Flash (Fast & Versatile)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Complex Reasoning)": "gemini-1.5-pro",
    "Gemini 1.5 Flash-8B (High Volume, Lower Intelligence Tasks)": "gemini-1.5-flash-8b",

    # Newer/Preview Models (May require specific API access or have different behaviour)
    "Gemini 2.0 Flash (Next Gen Speed/Multimodal)": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite (Cost Efficiency & Low Latency)": "gemini-2.0-flash-lite",
    "Gemini 2.5 Flash Preview (Adaptive & Cost Efficient)": "gemini-2.5-flash-preview-04-17",
    "Gemini 2.5 Pro Preview (Enhanced Reasoning & Multimodal)": "gemini-2.5-pro-preview-03-25",
}
# Keep 1.5 Flash as a stable default
DEFAULT_MODEL_NAME = "Gemini 1.5 Flash (Fast & Versatile)"
# Ensure the default exists in the list, fallback if necessary
if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
     DEFAULT_MODEL_NAME = list(AVAILABLE_MODELS.keys())[0]


# --- Load API Key and Configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("### 🔑 API Key Not Found!", icon="🚨")
    st.markdown("Please ensure your `GEMINI_API_KEY` is set in your environment variables or in a `.env` file.")
    st.stop()

# Configure the client library (but not the specific model yet)
try:
    genai.configure(api_key=API_KEY)
    # Define standard generation config and safety settings globally
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
# Add state for selected model display name (the key from AVAILABLE_MODELS)
if 'selected_model_display_name' not in st.session_state:
    st.session_state.selected_model_display_name = DEFAULT_MODEL_NAME


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from a PDF file stream, updates session state on error."""
    try:
        pdf_file_stream.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
        return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e:
        st.session_state.error_message = f"📄 Error reading PDF: {e}. Is it password-protected or corrupted?"
        return None
    except Exception as e:
        st.session_state.error_message = f"⚙️ An unexpected error occurred during PDF extraction: {e}"
        return None

def create_text_prompt(transcript, context=None):
    """Creates the prompt for text/PDF input."""
    prompt_parts = [
        "You are an expert meeting note-taker. Your task is to generate detailed, factual notes from the provided meeting transcript.",
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
        "- Accuracy is paramount. Ensure every piece of factual information (numbers, names, dates, examples) is captured precisely as stated in the transcript.",
        "- Maintain clarity and conciseness while preserving ALL shared details.",
        "- Avoid adding any personal insights, opinions, or information not present in the transcript.",
        "- If a section (like Opening Overview) is not present in the transcript, simply omit it from the notes.",
        "\n---",
        "\n**MEETING TRANSCRIPT:**\n",
        transcript,
        "\n---"
    ]
    if context:
        prompt_parts.extend([
            "\n**ADDITIONAL CONTEXT (Use this to better understand the transcript, but do not repeat it in the notes unless it's mentioned in the transcript itself):**\n",
            context,
            "\n---"
        ])
    prompt_parts.append("\n**GENERATED NOTES:**\n")
    return "\n".join(prompt_parts)

def create_audio_prompt(context=None):
    """Creates the prompt for audio input."""
    prompt_parts = [
        "You are an expert meeting note-taker.",
        "1. First, accurately transcribe the provided audio file.",
        "2. Then, using the transcription you just generated, create detailed, factual meeting notes.",
        "Follow this specific structure EXACTLY for the notes:",
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
        "\n**Additional Instructions for Note Generation:**",
        "- Base the notes *only* on the content of the audio transcription.",
        "- Accuracy is paramount. Ensure every piece of factual information (numbers, names, dates, examples) is captured precisely.",
        "- Maintain clarity and conciseness while preserving ALL shared details.",
        "- Avoid adding any personal insights, opinions, or information not present in the audio.",
        "- If a section (like Opening Overview) is not present, omit it.",
        "\n---"
    ]
    if context:
        prompt_parts.extend([
            "\n**ADDITIONAL CONTEXT (Use this to better understand the audio content, but do not repeat it in the notes unless it's mentioned in the audio itself):**\n",
            context,
            "\n---"
        ])
    prompt_parts.append("\n**AUDIO FILE FOR TRANSCRIPTION AND NOTE GENERATION IS PROVIDED SEPARATELY.**")
    prompt_parts.append("\n**GENERATED NOTES:**\n")
    return "\n".join(prompt_parts)

# --- Streamlit App UI ---

st.title("✨ SynthNotes AI")
st.markdown("Instantly transform meeting recordings into structured, factual notes.")

# --- Input Section ---
with st.container(border=True):
    col1, col2 = st.columns([2, 1]) # Allocate space for input type vs model selection

    with col1:
        st.subheader("Source Input")
        input_method = st.radio(
            "Input type:",
            ("Paste Text", "Upload PDF", "Upload Audio"),
            key="input_method_radio",
            horizontal=True,
        )

        text_input_area = None # Use different names for widget definition vs value access later
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

    with col2:
        st.subheader("AI Model")
        # Store the *selected display name* in session state when it changes
        st.session_state.selected_model_display_name = st.selectbox(
            "Choose model:",
            options=list(AVAILABLE_MODELS.keys()),
            key="model_select", # Key for the selectbox widget itself
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_display_name), # Set initial value from state
            help="Select the Gemini model. 'Pro' may offer higher quality but might be slower or have different usage limits."
        )

    # --- Context Checkbox ---
    st.divider() # Visual separator
    add_context_enabled = st.checkbox(
        "Add Context (Optional)",
        key="add_context_cb",
        value=st.session_state.add_context_enabled,
        help="Check this box to provide additional background information."
    )
    # Update session state when checkbox changes
    st.session_state.add_context_enabled = add_context_enabled


# --- Conditional Context Input Section ---
if st.session_state.add_context_enabled:
    with st.container(border=True):
        st.subheader("Context Details")
        context_input_area = st.text_area( # Use different name for widget definition
            "Provide background:",
            height=150,
            key="context_input", # Key to access value from session state
            placeholder="Add context like attendees, goals, project name...",
            help="Adding context helps the AI understand the conversation better."
        )


# --- Generate Button ---
st.write("") # Spacer
generate_button = st.button(
    "🚀 Generate Notes",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.processing # Disable button when processing
)

# --- Output Section ---
output_container = st.container(border=True)
with output_container:
    # Add CSS class marker - this helps target the container if CSS needs specificity
    st.markdown('<div class="output-container"></div>', unsafe_allow_html=True)

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
        # Initial state or after clearing
        st.markdown("<p class='initial-prompt'>Generated notes will appear here once processed.</p>", unsafe_allow_html=True)


# --- Processing Logic ---
if generate_button:
    # 1. Set processing state and clear previous output/errors
    st.session_state.processing = True
    st.session_state.generated_notes = None
    st.session_state.error_message = None
    st.rerun() # Rerun immediately to show spinner and disable button

# --- Actual processing happens after the rerun ---
if st.session_state.processing:
    transcript = None
    audio_file_input = None
    # Retrieve selected options from session state
    input_type = st.session_state.get("input_method_radio", "Paste Text")
    selected_model_display_name = st.session_state.get("selected_model_display_name", DEFAULT_MODEL_NAME)
    selected_model_id = AVAILABLE_MODELS.get(selected_model_display_name, AVAILABLE_MODELS[DEFAULT_MODEL_NAME]) # Get API ID

    # Retrieve input widget values from session state
    text_content = st.session_state.get("text_input", "")
    pdf_file = st.session_state.get("pdf_uploader")
    audio_file = st.session_state.get("audio_uploader")

    # 2. Input Validation
    if input_type == "Paste Text" and not text_content:
        st.session_state.error_message = "⚠️ Text area is empty."
    elif input_type == "Upload PDF" and not pdf_file:
        st.session_state.error_message = "⚠️ No PDF file uploaded."
    elif input_type == "Upload Audio" and not audio_file:
         st.session_state.error_message = "⚠️ No audio file uploaded."

    # 3. Get Context (if enabled)
    final_context = None
    if st.session_state.get("add_context_enabled", False):
        final_context = st.session_state.get("context_input", "").strip()

    # 4. Proceed if validation passed
    if not st.session_state.error_message:
        try:
            # --- Process Input (PDF/Audio Upload, Text Strip) ---
            if input_type == "Paste Text":
                transcript = text_content.strip()
                st.toast("📝 Using pasted text.", icon="✅")
            elif input_type == "Upload PDF":
                st.toast("📄 Processing PDF...", icon="⏳")
                pdf_stream = io.BytesIO(pdf_file.getvalue())
                transcript = extract_text_from_pdf(pdf_stream) # Sets error_message on failure
                if transcript and not st.session_state.error_message:
                    st.toast("📄 PDF processed!", icon="✅")
                elif not transcript and not st.session_state.error_message:
                     st.session_state.error_message = "⚠️ PDF contains no extractable text."
            elif input_type == "Upload Audio":
                st.toast(f"☁️ Uploading '{audio_file.name}'...", icon="⬆️")
                audio_bytes = audio_file.getvalue()
                audio_file_for_api = genai.upload_file(
                    content=audio_bytes, display_name=f"audio_{int(time.time())}", mime_type=audio_file.type
                )
                st.session_state.uploaded_audio_info = audio_file_for_api # Store info
                # Poll for readiness
                polling_start_time = time.time()
                while audio_file_for_api.state.name == "PROCESSING":
                    if time.time() - polling_start_time > 300: # 5 min timeout
                        raise TimeoutError("Audio processing timed out after 5 minutes.")
                    st.toast(f"🎧 Processing '{audio_file.name}' on server...", icon="⏳")
                    time.sleep(5) # Poll less frequently
                    audio_file_for_api = genai.get_file(audio_file_for_api.name)

                if audio_file_for_api.state.name == "FAILED":
                    st.session_state.error_message = f"😥 Audio processing failed for '{audio_file.name}'."
                    try: genai.delete_file(audio_file_for_api.name)
                    except Exception: pass
                    st.session_state.uploaded_audio_info = None
                elif audio_file_for_api.state.name == "ACTIVE":
                    st.toast(f"🎧 Audio '{audio_file.name}' ready!", icon="✅")
                    audio_file_input = audio_file_for_api
                else:
                    st.session_state.error_message = f"Unexpected audio state: {audio_file_for_api.state.name}"
                    try: genai.delete_file(audio_file_for_api.name)
                    except Exception: pass
                    st.session_state.uploaded_audio_info = None

            # 5. Call Gemini if input prepared successfully
            if not st.session_state.error_message and (transcript or audio_file_input):
                st.toast(f"🧠 Generating notes with {selected_model_display_name}...", icon="✨")

                # --- Initialize the specific model HERE ---
                try:
                    model = genai.GenerativeModel(
                        model_name=selected_model_id, # Use the selected model ID
                        safety_settings=safety_settings,
                        generation_config=generation_config,
                    )
                except Exception as model_init_error:
                    st.session_state.error_message = f"Failed to initialize model '{selected_model_id}': {model_init_error}"
                    st.session_state.processing = False
                    st.rerun() # Exit processing early

                # --- Generate Content ---
                response = None
                if input_type in ["Paste Text", "Upload PDF"]:
                    full_prompt = create_text_prompt(transcript, final_context)
                    response = model.generate_content(full_prompt)
                elif input_type == "Upload Audio":
                    full_prompt = create_audio_prompt(final_context)
                    response = model.generate_content([full_prompt, audio_file_input])
                    # Clean up audio file from Google Cloud *after successful use*
                    try:
                        if st.session_state.uploaded_audio_info:
                            genai.delete_file(st.session_state.uploaded_audio_info.name)
                            st.session_state.uploaded_audio_info = None
                            st.toast("☁️ Temporary audio file cleaned up.", icon="🗑️")
                    except Exception as delete_err:
                        st.warning(f"Could not delete temp audio file: {delete_err}", icon="⚠️") # Non-fatal

                # --- Handle Response ---
                if response and response.text:
                    st.session_state.generated_notes = response.text
                    st.toast("🎉 Notes generated successfully!", icon="✅")
                elif response: # Check if response exists but text is empty
                     st.session_state.error_message = "🤔 AI returned an empty response. Try adjusting context or input."
                else: # Handle cases where the API call itself might fail implicitly
                    st.session_state.error_message = "😥 AI generation failed. Please check API key and input."

        except Exception as e:
            st.session_state.error_message = f"❌ An error occurred during processing: {e}"
            # Attempt audio cleanup even on general error if info exists
            try:
                if input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
            except Exception: pass # Ignore cleanup error during main error

    # 6. Finish processing
    st.session_state.processing = False
    st.rerun() # Rerun again to update UI (remove spinner, show results/error)


# --- Footer ---
st.divider()
st.caption(
    "Powered by [Google Gemini](https://deepmind.google/technologies/gemini/) | App by SynthNotes AI"
)

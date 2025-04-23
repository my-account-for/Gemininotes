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
    initial_sidebar_state="collapsed", # Collapse sidebar if not used
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
    /* Overall App Background */
    .stApp {
        /* background-color: #F0F2F6; */ /* Removed to let containers define background */
        background: linear-gradient(to bottom right, #F0F2F6, #FFFFFF); /* Subtle gradient */
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem; /* More space at bottom */
        padding-left: 2rem;  /* Slightly less horizontal padding */
        padding-right: 2rem;
        max-width: 1000px; /* Limit max width for better readability */
        margin: auto;      /* Center the container */
    }

    /* General Container Styling (using st.container(border=True)) */
    div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { /* Target nested containers */
        /* Styles for containers with border=True */
         background-color: #FFFFFF;
         border: 1px solid #E5E7EB; /* Softer border */
         border-radius: 0.75rem; /* More rounded */
         padding: 1.5rem; /* Inner padding */
         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* Softer shadow */
         margin-bottom: 1.5rem; /* Space between containers */
    }

    /* Headers */
    h1 {
        color: #111827; /* Darker */
        font-weight: 700; /* Bolder */
        text-align: center; /* Center title */
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #1F2937;
        font-weight: 600;
        border-bottom: 1px solid #E5E7EB; /* Underline subheaders */
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    /* App Subtitle */
    .main .block-container > div:nth-child(2) > div > div > p { /* Target the subtitle markdown */
       text-align: center;
       color: #4B5563; /* Grey text */
       font-size: 1.1rem;
       margin-bottom: 2rem;
    }


    /* Input Widgets */
    .stTextInput textarea,
    .stFileUploader div[data-testid="stFileUploaderDropzone"],
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid #D1D5DB;
        background-color: #F9FAFB; /* Slightly off-white input background */
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput textarea:focus,
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:focus-within,
    .stTextArea textarea:focus {
        border-color: #007AFF;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05), 0 0 0 3px rgba(0, 122, 255, 0.2);
        background-color: #FFFFFF;
    }
    .stFileUploader p { /* Style file uploader text */
        font-size: 0.95rem;
        color: #4B5563;
    }

    /* Radio Buttons */
    div[role="radiogroup"] > label {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 0.5rem;
        padding: 0.6rem 1rem;
        margin-right: 0.5rem;
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    div[role="radiogroup"] label:hover {
        border-color: #9CA3AF;
    }
    div[role="radiogroup"] input[type="radio"]:checked + div {
       background-color: #EFF6FF; /* Light blue background for selected */
       border-color: #007AFF;
       color: #005ECB;
       font-weight: 500;
       box-shadow: 0 1px 3px rgba(0, 122, 255, 0.1);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem; /* Larger padding */
        font-weight: 600; /* Bolder text */
        transition: all 0.2s ease-in-out;
        border: none;
        width: 100%; /* Make buttons take full width */
    }
    .stButton > button[kind="primary"] {
        background-color: #007AFF;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 122, 255, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #005ECB;
        box-shadow: 0 7px 14px rgba(0, 122, 255, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
     .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.4);
        outline: none;
    }
     .stButton > button:disabled,
     .stButton > button[kind="primary"]:disabled {
         background-color: #D1D5DB; /* Grey background when disabled */
         color: #6B7280;
         box-shadow: none;
         transform: none;
         cursor: not-allowed;
     }

    /* Download Button */
    .stDownloadButton > button {
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        background-color: #F3F4F6;
        color: #1F2937;
        border: 1px solid #D1D5DB;
        transition: background-color 0.2s ease-in-out;
        width: auto; /* Don't force full width */
        margin-top: 1rem; /* Add space above download button */
    }
    .stDownloadButton > button:hover {
        background-color: #E5E7EB;
        border-color: #9CA3AF;
    }

    /* Output Area Styling */
    div[data-testid="stVerticalBlock"].output-container { /* Add a class later */
        background-color: #F9FAFB; /* Slightly different bg for output */
        border: 1px solid #E5E7EB;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
        min-height: 150px; /* Ensure it has some height */
    }
    .output-container .stMarkdown {
        background-color: transparent; /* Make markdown bg transparent inside */
        border: none;
        padding: 0;
        color: #374151; /* Standard text color for notes */
        font-size: 1rem;
        line-height: 1.6;
    }
    .output-container .stMarkdown h3,
    .output-container .stMarkdown h4,
    .output-container .stMarkdown strong { /* Style Q/A formatting */
       color: #111827;
       font-weight: 600;
    }
    .output-container .stAlert {
        margin-top: 1rem;
    }
    .output-container .initial-prompt { /* Style for placeholder text */
        color: #6B7280;
        font-style: italic;
        text-align: center;
        padding-top: 2rem;
    }


    /* Footer */
    footer {
        text-align: center;
        color: #9CA3AF; /* Lighter grey */
        font-size: 0.8rem; /* Smaller */
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    footer a {
        color: #6B7280;
        text-decoration: none;
    }
     footer a:hover {
        color: #007AFF;
        text-decoration: underline;
    }

</style>
""", unsafe_allow_html=True)

# --- Load API Key and Configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("### 🔑 API Key Not Found!", icon="🚨")
    st.markdown("""
        Please ensure your `GEMINI_API_KEY` is set in your environment variables
        or in a `.env` file in the project root.
        You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    """)
    st.stop()

# --- Initialize Session State ---
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'generated_notes' not in st.session_state:
    st.session_state.generated_notes = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'uploaded_audio_info' not in st.session_state:
    st.session_state.uploaded_audio_info = None

# --- Configure Gemini Model ---
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
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings, generation_config=generation_config,
    )
except Exception as e:
    st.error(f"### 🔌 Error Initializing AI Model", icon="🚨")
    st.error(f"Details: {e}")
    st.stop()

# --- Helper Functions (No changes needed in logic, added icons to errors) ---
def extract_text_from_pdf(pdf_file_stream):
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
    # Prompt content remains the same
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
    # Prompt content remains the same
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
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Input")
        input_method = st.radio(
            "Input type:",
            ("Paste Text", "Upload PDF", "Upload Audio"),
            key="input_method_radio",
            horizontal=True,
        )

        text_input = None
        uploaded_pdf_file = None
        uploaded_audio_file = None

        if input_method == "Paste Text":
            text_input = st.text_area("Paste transcript:", height=200, key="text_input", placeholder="Paste the full meeting transcript here...")
        elif input_method == "Upload PDF":
            uploaded_pdf_file = st.file_uploader("Upload PDF:", type="pdf", key="pdf_uploader", help="Upload a PDF file containing the meeting transcript.")
        else: # Upload Audio
            uploaded_audio_file = st.file_uploader(
                "Upload Audio:",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'],
                key="audio_uploader",
                help="Upload an audio recording (WAV, MP3, M4A, etc.). Processing time depends on length."
            )

    with col2:
        st.subheader("Context (Optional)")
        context_input = st.text_area(
            "Provide background:",
            height=200, # Match height
            key="context_input",
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
# Use st.container with a border for the output area
output_container = st.container(border=True)

# Add CSS class to the output container for specific styling
output_container.markdown('<div class="output-container-marker"></div>', unsafe_allow_html=True) # Dummy element to help target with CSS sibling selector if needed, or just style based on structure

with output_container:
    if st.session_state.processing:
        st.info("⏳ Processing your request... Please wait.", icon="🧠")
        # Maybe add a progress bar if you have discrete steps?
        # st.progress(50, text="Working...")

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


# --- Processing Logic (Triggered by Button Click) ---
if generate_button:
    # 1. Set processing state and clear previous output/errors
    st.session_state.processing = True
    st.session_state.generated_notes = None
    st.session_state.error_message = None
    st.rerun() # Rerun immediately to show the spinner and disable button

# --- Separate block for actual processing after rerun ---
if st.session_state.processing:
    # This block runs *after* the rerun triggered by the button click
    transcript = None
    audio_file_input = None
    input_type = st.session_state.get("input_method_radio", "Paste Text") # Get selected method

    # 2. Input Validation and Preparation
    # Need to access widget values via their keys after rerun
    text_content = st.session_state.get("text_input", "")
    pdf_file = st.session_state.get("pdf_uploader")
    audio_file = st.session_state.get("audio_uploader")

    num_inputs = sum([bool(text_content), bool(pdf_file), bool(audio_file)])

    # Check based on selected input_method
    if input_type == "Paste Text" and not text_content:
        st.session_state.error_message = "⚠️ Text area is empty. Please paste the transcript."
    elif input_type == "Upload PDF" and not pdf_file:
        st.session_state.error_message = "⚠️ No PDF file uploaded. Please upload a PDF."
    elif input_type == "Upload Audio" and not audio_file:
         st.session_state.error_message = "⚠️ No audio file uploaded. Please upload audio."
    # Note: No check for multiple inputs needed if radio button forces selection

    # 3. Process Valid Input
    if not st.session_state.error_message:
        try:
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
                    if time.time() - polling_start_time > 300: # 5 min timeout for processing
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

            # 4. Call Gemini if input prepared successfully
            if not st.session_state.error_message and (transcript or audio_file_input):
                st.toast("🧠 Generating notes with AI...", icon="✨")
                final_context = st.session_state.get("context_input", "").strip()
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

                if response and response.text:
                    st.session_state.generated_notes = response.text
                    st.toast("🎉 Notes generated successfully!", icon="✅")
                elif response: # Check if response exists but text is empty
                     st.session_state.error_message = "🤔 AI returned an empty response. Try adjusting context or input."
                else: # Handle cases where the API call itself might fail implicitly
                    st.session_state.error_message = "😥 AI generation failed. Please check API key and input."


        except Exception as e:
            st.session_state.error_message = f"❌ An error occurred: {e}"
            # Attempt audio cleanup even on general error if info exists
            try:
                if input_type == "Upload Audio" and st.session_state.uploaded_audio_info:
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
            except Exception: pass # Ignore cleanup error during main error


    # 5. Finish processing
    st.session_state.processing = False
    st.rerun() # Rerun again to update UI (remove spinner, show results/error)


# --- Footer ---
st.divider()
st.caption(
    "Powered by [Google Gemini 1.5](https://deepmind.google/technologies/gemini/) | App by SynthNotes AI"
)

import streamlit as st
import google.generativeai as genai
import os
import io
import time
from dotenv import load_dotenv
import PyPDF2

# --- Page Configuration (Apply theme defaults) ---
st.set_page_config(
    page_title="SynthNotes AI", # More startup-y name?
    page_icon="✨",         # Changed Icon
    layout="wide"
)

# --- Custom CSS Injection ---
# Inject custom CSS for polishing
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #F0F2F6; /* Match secondaryBackgroundColor for overall app background */
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #FFFFFF; /* White card for main content */
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Style Headers */
    h1 {
        color: #0F172A; /* Darker text */
        font-weight: 600; /* Semi-bold */
    }
     h2, h3 {
        color: #334155; /* Slightly lighter for subheaders */
        font-weight: 500;
    }

    /* Input Widgets */
    .stTextInput textarea, .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        border-radius: 0.375rem; /* Slightly rounded corners */
        border: 1px solid #CBD5E1; /* Subtle border */
        background-color: #FFFFFF;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stTextInput textarea:focus, .stFileUploader div[data-testid="stFileUploaderDropzone"]:focus-within {
        border-color: #007AFF; /* Use primary color for focus */
        box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.2); /* Focus ring */
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 0.375rem;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        border: none; /* Remove default border */
    }
    /* Primary Button specific */
     .stButton > button[kind="primary"] {
        background-color: #007AFF; /* Primary blue */
        color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #005ECB; /* Darker blue on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
     .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.4); /* Focus ring */
        outline: none;
    }
    /* Download Button Specific (often secondary style) */
    .stDownloadButton > button {
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        background-color: #E5E7EB; /* Light grey */
        color: #1F2937; /* Dark text */
        border: 1px solid #D1D5DB;
        transition: background-color 0.2s ease-in-out;
    }
    .stDownloadButton > button:hover {
        background-color: #D1D5DB; /* Darker grey */
        border-color: #9CA3AF;
    }

    /* Radio Buttons */
    div[role="radiogroup"] > label {
        padding: 0.5rem 0.75rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    div[role="radiogroup"] input[type="radio"]:checked + div {
       /* Style the selected radio button label background if needed */
       /* background-color: #DBEAFE; */ /* Example: Light blue background */
       /* border-color: #007AFF; */
    }

    /* Placeholders / Output Area */
    .stAlert { /* Style st.info, st.error etc. */
        border-radius: 0.375rem;
        border-left: 4px solid; /* Keep the left border for type indication */
        padding: 1rem;
    }
    /* Specific alert styles */
    .stAlert[data-baseweb="notification"][kind="info"] { border-left-color: #007AFF; background-color: #EFF6FF; }
    .stAlert[data-baseweb="notification"][kind="error"] { border-left-color: #EF4444; background-color: #FEF2F2; }
    .stAlert[data-baseweb="notification"][kind="warning"] { border-left-color: #F59E0B; background-color: #FFFBEB; }
    .stAlert[data-baseweb="notification"][kind="success"] { border-left-color: #10B981; background-color: #ECFDF5; }


    /* Style the markdown output area */
    .stMarkdown {
        padding: 1rem;
        background-color: #FAFAFA; /* Slightly off-white background */
        border-radius: 0.375rem;
        border: 1px solid #E5E7EB;
        margin-top: 1rem; /* Add space above the output */
    }

    /* Footer */
    footer {
        margin-top: 2rem;
        text-align: center;
        color: #6B7280; /* Grey text */
        font-size: 0.875rem;
    }

</style>
""", unsafe_allow_html=True)

# --- Load API Key and Configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Add a check and instruction if the API key is missing
if not API_KEY:
    # Use a more styled error message
    st.error("### 🛑 API Key Not Found!", icon="🔑")
    st.markdown("""
        Please add your `GEMINI_API_KEY` to a `.env` file in the same directory as the script.
        You can get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    """)
    st.stop() # Stop execution if no API key

# --- Global variable for uploaded file response ---
if 'uploaded_audio_info' not in st.session_state:
    st.session_state.uploaded_audio_info = None
if 'generated_notes' not in st.session_state: # Store notes in session state
    st.session_state.generated_notes = None

try:
    genai.configure(api_key=API_KEY)
    generation_config = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Or "gemini-pro"
        safety_settings=safety_settings,
        generation_config=generation_config,
    )
except Exception as e:
    st.error(f"### 😥 Error Initializing AI Model", icon="🔌")
    st.error(f"Details: {e}")
    st.stop()

# --- Helper Functions (No changes needed in logic) ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        pdf_file_stream.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else None
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading PDF: {e}. Is it password-protected or corrupted?", icon="📄")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF extraction: {e}", icon="⚙️")
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
st.markdown("Transform your meeting recordings (Text, PDF, Audio) into structured, factual notes instantly.")
st.divider()

# --- Input Section ---
with st.container(): # Group input elements
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Provide Meeting Source")
        input_method = st.radio(
            "Select input type:",
            ("Paste Text", "Upload PDF", "Upload Audio"),
            horizontal=True,
            # label_visibility="collapsed" # Keep label for clarity
        )

        text_input = None
        uploaded_pdf_file = None
        uploaded_audio_file = None

        if input_method == "Paste Text":
            text_input = st.text_area("Paste full transcript here:", height=250, key="text_input", placeholder="Enter meeting transcript...")
        elif input_method == "Upload PDF":
            uploaded_pdf_file = st.file_uploader("Upload PDF transcript:", type="pdf", key="pdf_uploader")
        else: # Upload Audio
            uploaded_audio_file = st.file_uploader(
                "Upload meeting audio:",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'],
                key="audio_uploader"
            )
            st.caption("Supports WAV, MP3, M4A, OGG, FLAC, AAC. Max size depends on API limits.")

    with col2:
        st.subheader("2. Add Optional Context")
        context_input = st.text_area(
            "Context (e.g., attendees, goals, project name):",
            height=250, # Match height
            key="context_input",
            placeholder="Provide any background info..."
        )

st.write("") # Add a bit of vertical space

# --- Processing Logic & Output Section ---
submit_button = st.button("🚀 Generate Notes", type="primary", use_container_width=True)
output_placeholder = st.container() # Use a container for output area

# Display existing notes if they are in session state (e.g., after a refresh)
if st.session_state.generated_notes:
     with output_placeholder:
        st.subheader("✅ Generated Notes")
        st.markdown(st.session_state.generated_notes)
        st.download_button(
             label="⬇️ Download Notes (.txt)",
             data=st.session_state.generated_notes,
             file_name="meeting_notes.txt",
             mime="text/plain",
             key='download-txt-persist'
         )

if submit_button:
    # Clear previous state before generating new notes
    st.session_state.generated_notes = None
    output_placeholder.empty() # Clear the output area

    transcript = None
    audio_file_input = None
    error_message = None
    input_type = None

    # --- Input Validation and Preparation ---
    num_inputs = sum([bool(text_input), bool(uploaded_pdf_file), bool(uploaded_audio_file)])

    if num_inputs > 1:
        error_message = "⚠️ Please provide input using only ONE method (Text, PDF, or Audio)."
    elif num_inputs == 0:
        error_message = "⚠️ Please provide input via text, PDF, or audio upload."
    else:
        processing_message = "" # Message for spinner
        if text_input:
            transcript = text_input.strip()
            if not transcript: error_message = "⚠️ Text area is empty. Please paste the transcript."
            else: input_type = "text"; processing_message = "Analyzing transcript..."
        elif uploaded_pdf_file:
            input_type = "pdf"; processing_message = "Extracting text from PDF..."
            with st.spinner(processing_message):
                pdf_stream = io.BytesIO(uploaded_pdf_file.getvalue())
                transcript = extract_text_from_pdf(pdf_stream) # Errors handled inside
                if transcript is None: error_message = " " # Space to prevent further processing, error already shown
                elif not transcript: error_message = "⚠️ The uploaded PDF appears to contain no extractable text."
                # Add success toast for PDF extraction
                if transcript and not error_message: st.toast("📄 PDF processed!", icon="✅")
        elif uploaded_audio_file:
             input_type = "audio"; processing_message = "Processing audio file..."
             with st.spinner(processing_message):
                try:
                    audio_bytes = uploaded_audio_file.getvalue()
                    st.toast(f"⬆️ Uploading '{uploaded_audio_file.name}'...", icon="☁️")
                    audio_file_for_api = genai.upload_file(
                        content=audio_bytes,
                        display_name=f"audio_{int(time.time())}",
                        mime_type=uploaded_audio_file.type
                    )
                    st.session_state.uploaded_audio_info = audio_file_for_api

                    while audio_file_for_api.state.name == "PROCESSING":
                         time.sleep(2)
                         audio_file_for_api = genai.get_file(audio_file_for_api.name)
                    if audio_file_for_api.state.name == "FAILED":
                        error_message = f"😥 Audio file processing failed for '{uploaded_audio_file.name}'."
                        try: genai.delete_file(audio_file_for_api.name)
                        except Exception: pass
                        st.session_state.uploaded_audio_info = None
                    elif audio_file_for_api.state.name == "ACTIVE":
                        st.toast(f"🎧 Audio '{uploaded_audio_file.name}' ready!", icon="✅")
                        audio_file_input = audio_file_for_api
                    else:
                        error_message = f"Audio file '{uploaded_audio_file.name}' is in an unexpected state: {audio_file_for_api.state.name}"
                        try: genai.delete_file(audio_file_for_api.name)
                        except Exception: pass
                        st.session_state.uploaded_audio_info = None
                except Exception as e:
                    error_message = f"😥 Error handling audio file: {e}"
                    st.session_state.uploaded_audio_info = None

    # --- Call Gemini if Input is Valid ---
    if not error_message and (transcript or audio_file_input):
        generation_spinner_message = "🧠 Generating notes with AI..."
        with st.spinner(generation_spinner_message):
            try:
                final_context = context_input.strip() if context_input else None
                response = None

                if input_type in ["text", "pdf"]:
                    full_prompt = create_text_prompt(transcript, final_context)
                    response = model.generate_content(full_prompt)
                elif input_type == "audio":
                    full_prompt = create_audio_prompt(final_context)
                    response = model.generate_content([full_prompt, audio_file_input])
                    # Clean up audio after use
                    try:
                        if st.session_state.uploaded_audio_info:
                            genai.delete_file(st.session_state.uploaded_audio_info.name)
                            st.session_state.uploaded_audio_info = None
                            # st.toast("☁️ Audio file cleaned up.", icon="🗑️") # Optional feedback
                    except Exception as delete_err:
                         st.warning(f"Could not delete temp audio file: {delete_err}", icon="⚠️")
                else:
                    raise ValueError("Invalid input_type.")

                if response and response.text:
                    st.session_state.generated_notes = response.text # Store in session state
                    st.toast("🎉 Notes generated successfully!", icon="✅")
                    # Display immediately after generation (will also be shown on reload)
                    with output_placeholder:
                        st.subheader("✅ Generated Notes")
                        st.markdown(st.session_state.generated_notes) # Display from session state
                        st.download_button(
                            label="⬇️ Download Notes (.txt)",
                            data=st.session_state.generated_notes,
                            file_name="meeting_notes.txt",
                            mime="text/plain",
                            key='download-txt-generate' # Different key from persistent one
                        )
                else:
                    error_message = " Gemini returned an empty response." # Treat empty response as error

            except Exception as e:
                 error_message = f"😥 AI generation failed: {e}"
                 # Attempt audio cleanup on error
                 try:
                     if input_type == "audio" and st.session_state.uploaded_audio_info:
                         genai.delete_file(st.session_state.uploaded_audio_info.name)
                         st.session_state.uploaded_audio_info = None
                 except Exception: pass

    # --- Display Errors ---
    if error_message:
        with output_placeholder: # Show errors in the output area
             st.error(error_message, icon="🚨")


# --- Footer ---
st.divider()
st.caption("Powered by Google Gemini 1.5 | Designed by SynthNotes AI")

import streamlit as st
import google.generativeai as genai
import os
import io
import time # For potential delays with file processing
from dotenv import load_dotenv
import PyPDF2

# --- Page Configuration ---
st.set_page_config(
    page_title="Meeting Notes Generator",
    page_icon="📝",
    layout="wide"
)

# --- Load API Key and Configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("🛑 GEMINI_API_KEY not found! Please add it to your .env file.")
    st.stop()

# --- Global variable for uploaded file response ---
# This helps manage the file lifecycle across reruns if needed,
# though for simple upload/process it might not be strictly necessary.
if 'uploaded_audio_info' not in st.session_state:
    st.session_state.uploaded_audio_info = None

try:
    genai.configure(api_key=API_KEY)
    generation_config = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 8192, # Keep this high for potentially long notes
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    # Use a model that explicitly supports audio (Gemini 1.5 Flash/Pro)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # or gemini-1.5-pro
        safety_settings=safety_settings,
        generation_config=generation_config,
    )
except Exception as e:
    st.error(f"😥 Error initializing Gemini model: {e}")
    st.stop()

# --- Helper Function: Extract Text from PDF ---
# (No changes needed)
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from a PDF file stream."""
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
        st.error(f"Error reading PDF: {e}. Is it password-protected or corrupted?")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF extraction: {e}")
        return None

# --- Helper Function: Create Prompt for Text/PDF Input ---
# This remains the same for text-based inputs
def create_text_prompt(transcript, context=None):
    """Creates the structured prompt for the Gemini model from TEXT transcript."""
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

# --- Helper Function: Create Prompt for Audio Input ---
# This prompt asks the model to first transcribe, then generate notes from the audio
def create_audio_prompt(context=None):
    """Creates the structured prompt for the Gemini model when input is AUDIO."""
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

st.title("📝 Meeting Notes Generator using Gemini")
st.markdown("Input a meeting transcript (text, PDF, or audio) and optional context to generate structured notes.")

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Source")
    # Use tabs for selecting input method
    input_method = st.radio(
        "Choose input method:",
        ("Paste Text", "Upload PDF", "Upload Audio"), # Added Audio option
        horizontal=True,
        label_visibility="collapsed"
    )

    text_input = None
    uploaded_pdf_file = None
    uploaded_audio_file = None # New variable for audio file

    if input_method == "Paste Text":
        text_input = st.text_area("Paste full transcript here:", height=300, key="text_input")
    elif input_method == "Upload PDF":
        uploaded_pdf_file = st.file_uploader("Upload PDF transcript:", type="pdf", key="pdf_uploader")
    else: # Upload Audio
        # Supported audio formats will depend on the Gemini API, common ones are usually supported.
        # Check Google's documentation for the specific model (1.5 Flash/Pro) for the latest list.
        # Common formats: WAV, MP3, AIFF, AAC, OGG, FLAC
        uploaded_audio_file = st.file_uploader(
            "Upload meeting audio:",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'], # Add relevant audio types
            key="audio_uploader"
        )
        st.caption("Supports formats like WAV, MP3, M4A, OGG, FLAC. Max file size depends on API limits (often generous for Gemini 1.5).")


with col2:
    st.subheader("Optional Context")
    context_input = st.text_area("Add context (attendees, goals, background info):", height=300, key="context_input")

st.divider() # Visual separator

# --- Processing Logic ---
submit_button = st.button("✨ Generate Notes", type="primary", use_container_width=True)
notes_placeholder = st.empty() # Create a placeholder for the notes output

if submit_button:
    transcript = None
    audio_file_input = None # To hold the processed audio file for the API
    error_message = None
    input_type = None # Track which input was used

    # --- Input Validation and Preparation ---
    num_inputs = sum([bool(text_input), bool(uploaded_pdf_file), bool(uploaded_audio_file)])

    if num_inputs > 1:
        error_message = "⚠️ Please provide input using only ONE method (Text, PDF, or Audio)."
    elif num_inputs == 0:
        error_message = "⚠️ Please provide input via text, PDF, or audio upload."
    else:
        # Process the valid single input
        if text_input:
            transcript = text_input.strip()
            if not transcript:
                error_message = "⚠️ Text area is empty. Please paste the transcript."
            else:
                input_type = "text"
        elif uploaded_pdf_file:
            input_type = "pdf"
            with st.spinner("Extracting text from PDF..."):
                pdf_stream = io.BytesIO(uploaded_pdf_file.getvalue())
                transcript = extract_text_from_pdf(pdf_stream)
                if transcript is None:
                    error_message = "Could not extract text from PDF (see error above)." # Error shown by function
                elif not transcript:
                    error_message = "⚠️ The uploaded PDF appears to contain no extractable text."
        elif uploaded_audio_file:
             input_type = "audio"
             notes_placeholder.info("⏳ Preparing audio file...")
             try:
                 # Upload the file to Google's service
                 # This is necessary for the API call with generate_content
                 audio_bytes = uploaded_audio_file.getvalue()
                 audio_file_for_api = genai.upload_file(
                     path=uploaded_audio_file.name, # Pass name for potential MIME type inference
                     display_name=uploaded_audio_file.name, # Optional display name
                 )
                 # Give feedback while upload/processing happens server-side
                 notes_placeholder.info(f"⏳ Uploading audio '{audio_file_for_api.display_name}'... This may take a moment depending on size.")

                 # Store the file info in session state if needed for potential reuse or cleanup
                 st.session_state.uploaded_audio_info = audio_file_for_api

                 # Check if the file is ready (optional, but good practice)
                 while audio_file_for_api.state.name == "PROCESSING":
                     time.sleep(2) # Wait before checking again
                     audio_file_for_api = genai.get_file(audio_file_for_api.name)
                     notes_placeholder.info(f"⏳ Processing audio '{audio_file_for_api.display_name}'...")

                 if audio_file_for_api.state.name == "FAILED":
                      error_message = f"😥 Audio file processing failed: {audio_file_for_api.name}"
                      # Clean up failed file object if desired (optional)
                      # genai.delete_file(audio_file_for_api.name)
                      st.session_state.uploaded_audio_info = None
                 else:
                     notes_placeholder.info(f"✅ Audio '{audio_file_for_api.display_name}' ready.")
                     audio_file_input = audio_file_for_api # Use this object for the API call


             except Exception as e:
                 error_message = f"😥 Error uploading/processing audio file: {e}"
                 st.session_state.uploaded_audio_info = None


    # --- Call Gemini if Input is Valid ---
    if not error_message and (transcript or audio_file_input):
        notes_placeholder.info("⏳ Generating notes... This might take longer for audio input.")
        try:
            final_context = context_input.strip() if context_input else None

            if input_type in ["text", "pdf"]:
                full_prompt = create_text_prompt(transcript, final_context)
                # Display prompt for debugging if needed (optional)
                # with st.expander("Show Prompt Sent to Gemini (Text/PDF)"):
                #    st.text(full_prompt)
                response = model.generate_content(full_prompt)
            elif input_type == "audio":
                full_prompt = create_audio_prompt(final_context)
                # Display prompt for debugging if needed (optional)
                # with st.expander("Show Prompt Sent to Gemini (Audio)"):
                #    st.text(full_prompt)

                # Send the prompt AND the audio file object
                response = model.generate_content([full_prompt, audio_file_input])

                # Optional: Clean up the uploaded file from Google's side after use
                # try:
                #    if st.session_state.uploaded_audio_info:
                #        genai.delete_file(st.session_state.uploaded_audio_info.name)
                #        st.session_state.uploaded_audio_info = None
                # except Exception as delete_err:
                #    st.warning(f"Could not delete uploaded audio file from cloud: {delete_err}")

            else:
                 # Should not happen if logic is correct, but good failsafe
                 raise ValueError("Invalid input_type determined.")

            # Use markdown to render potential formatting from the model
            notes_placeholder.markdown(response.text)

        except Exception as e:
            st.error(f"😥 An error occurred while generating notes with the Gemini API:")
            # Be careful about leaking too much detail in production errors
            st.error(f"{e}")
            # Try to clear the placeholder if there was an error after the spinner started
            notes_placeholder.empty()

            # Optional: Clean up audio file if generation failed mid-way
            # try:
            #    if st.session_state.uploaded_audio_info:
            #        genai.delete_file(st.session_state.uploaded_audio_info.name)
            #        st.session_state.uploaded_audio_info = None
            # except Exception: pass # Ignore cleanup error if main error occurred


    elif error_message:
        notes_placeholder.error(error_message)

# --- Add some footer information (optional) ---
st.divider()
st.caption("Powered by Google Gemini | Built with Streamlit")

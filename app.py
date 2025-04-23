import streamlit as st
import google.generativeai as genai
import os
import io
import time
from dotenv import load_dotenv
import PyPDF2

# --- [ Previous code remains the same: imports, config, helpers ] ---
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
# (No changes needed)
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
# (No changes needed)
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
        uploaded_audio_file = st.file_uploader(
            "Upload meeting audio:",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'], # Add relevant audio types
            key="audio_uploader"
        )
        st.caption("Supports formats like WAV, MP3, M4A, OGG, FLAC. Max file size depends on API limits.")


with col2:
    st.subheader("Optional Context")
    context_input = st.text_area("Add context (attendees, goals, background info):", height=300, key="context_input")

st.divider() # Visual separator

# --- Processing Logic ---
submit_button = st.button("✨ Generate Notes", type="primary", use_container_width=True)
notes_placeholder = st.empty() # Create a placeholder for the notes output
download_placeholder = st.empty() # Create a placeholder for the download button

if submit_button:
    # Clear previous results and download button
    notes_placeholder.empty()
    download_placeholder.empty()

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
                 audio_bytes = uploaded_audio_file.getvalue()
                 # Use a unique name for the upload if desired, or just the original name
                 upload_display_name = f"meeting_audio_{int(time.time())}_{uploaded_audio_file.name}"
                 audio_file_for_api = genai.upload_file(
                     # path=uploaded_audio_file.name, # Using path directly might fail in some environments
                     # Instead, provide content bytes and a name
                     content=audio_bytes,
                     display_name=upload_display_name,
                     mime_type=uploaded_audio_file.type # Pass the MIME type detected by Streamlit
                 )
                 notes_placeholder.info(f"⏳ Uploading '{uploaded_audio_file.name}' ({round(len(audio_bytes)/1024/1024, 2)} MB)...")

                 st.session_state.uploaded_audio_info = audio_file_for_api

                 while audio_file_for_api.state.name == "PROCESSING":
                     notes_placeholder.info(f"⏳ Processing audio '{uploaded_audio_file.name}' on server...")
                     time.sleep(3) # Check status less frequently for larger files
                     audio_file_for_api = genai.get_file(audio_file_for_api.name)


                 if audio_file_for_api.state.name == "FAILED":
                      error_message = f"😥 Audio file processing failed for '{uploaded_audio_file.name}'."
                      st.session_state.uploaded_audio_info = None
                      # Attempt to delete the failed file from Google's side
                      try: genai.delete_file(audio_file_for_api.name)
                      except Exception: pass
                 elif audio_file_for_api.state.name == "ACTIVE":
                     notes_placeholder.info(f"✅ Audio '{uploaded_audio_file.name}' ready.")
                     audio_file_input = audio_file_for_api # Use this object for the API call
                 else:
                     # Handle other unexpected states if necessary
                     error_message = f"Audio file '{uploaded_audio_file.name}' is in an unexpected state: {audio_file_for_api.state.name}"
                     st.session_state.uploaded_audio_info = None
                     try: genai.delete_file(audio_file_for_api.name)
                     except Exception: pass


             except Exception as e:
                 error_message = f"😥 Error uploading/processing audio file: {e}"
                 st.session_state.uploaded_audio_info = None


    # --- Call Gemini if Input is Valid ---
    if not error_message and (transcript or audio_file_input):
        notes_placeholder.info("⏳ Generating notes... This might take longer for audio input.")
        try:
            final_context = context_input.strip() if context_input else None
            generated_notes = None # Variable to store the final notes text

            if input_type in ["text", "pdf"]:
                full_prompt = create_text_prompt(transcript, final_context)
                response = model.generate_content(full_prompt)
                generated_notes = response.text # Store the result
            elif input_type == "audio":
                full_prompt = create_audio_prompt(final_context)
                # Send the prompt AND the audio file object
                response = model.generate_content([full_prompt, audio_file_input])
                generated_notes = response.text # Store the result

                # --- Optional: Clean up the uploaded audio file after successful use ---
                try:
                    if st.session_state.uploaded_audio_info:
                        # print(f"Attempting to delete file: {st.session_state.uploaded_audio_info.name}") # Debug print
                        genai.delete_file(st.session_state.uploaded_audio_info.name)
                        st.session_state.uploaded_audio_info = None
                        # print("File deleted successfully from cloud.") # Debug print
                except Exception as delete_err:
                    # Don't block the user, just log a warning or ignore
                    # print(f"Warning: Could not delete uploaded audio file from cloud: {delete_err}") # Debug print
                    pass # Ignore cleanup error
            else:
                 raise ValueError("Invalid input_type determined.")

            # --- Display Notes and Download Button ---
            if generated_notes:
                 notes_placeholder.markdown(generated_notes) # Display notes first

                 # Use the download_placeholder to add the button below the notes
                 download_placeholder.download_button(
                     label="⬇️ Download Notes (.txt)",
                     data=generated_notes, # The text content to download
                     file_name="meeting_notes.txt", # Default filename
                     mime="text/plain", # MIME type for text file
                     key='download-txt' # Unique key for the button
                 )
            else:
                notes_placeholder.warning(" Gemini returned an empty response.")


        except Exception as e:
            st.error(f"😥 An error occurred while generating notes with the Gemini API:")
            st.error(f"{e}")
            notes_placeholder.empty() # Clear any "Generating..." message on error
            download_placeholder.empty() # Ensure no download button on error

            # --- Optional: Attempt audio cleanup even if generation failed ---
            try:
                if input_type == "audio" and st.session_state.uploaded_audio_info:
                    # print(f"Attempting to delete file after error: {st.session_state.uploaded_audio_info.name}") # Debug print
                    genai.delete_file(st.session_state.uploaded_audio_info.name)
                    st.session_state.uploaded_audio_info = None
                    # print("File deleted successfully from cloud after error.") # Debug print
            except Exception:
                # print(f"Warning: Could not delete uploaded audio file after error: {delete_err}") # Debug print
                pass # Ignore cleanup error


    elif error_message:
        notes_placeholder.error(error_message)
        download_placeholder.empty() # Ensure no download button on input error


# --- Add some footer information (optional) ---
st.divider()
st.caption("Powered by Google Gemini | Built with Streamlit")

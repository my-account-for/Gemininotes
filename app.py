import streamlit as st
import google.generativeai as genai
import os
import io
from dotenv import load_dotenv
import PyPDF2

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Meeting Notes Generator",
    page_icon="📝",
    layout="wide"
)

# --- Load API Key and Configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Add a check and instruction if the API key is missing
if not API_KEY:
    st.error("🛑 GEMINI_API_KEY not found! Please add it to your .env file.")
    st.stop() # Stop execution if no API key

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
    st.error(f"😥 Error initializing Gemini model: {e}")
    st.stop()

# --- Helper Function: Extract Text from PDF ---
# (Same as Flask version)
def extract_text_from_pdf(pdf_file_stream):
    """Extracts text from a PDF file stream."""
    try:
        # Ensure the stream position is at the beginning
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

# --- Helper Function: Create Prompt ---
# (Same as Flask version)
def create_prompt(transcript, context=None):
    """Creates the structured prompt for the Gemini model."""
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

# --- Streamlit App UI ---

st.title("📝 Meeting Notes Generator using Gemini")
st.markdown("Input a meeting transcript (text or PDF) and optional context to generate structured notes.")

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Transcript")
    # Use tabs for selecting input method
    input_method = st.radio("Choose input method:", ("Paste Text", "Upload PDF"), horizontal=True, label_visibility="collapsed")

    text_input = None
    uploaded_file = None

    if input_method == "Paste Text":
        text_input = st.text_area("Paste full transcript here:", height=300, key="text_input")
    else:
        # Important: Give the file uploader a unique key if other widgets might rerun the script
        uploaded_file = st.file_uploader("Upload PDF transcript:", type="pdf", key="pdf_uploader")


with col2:
    st.subheader("Optional Context")
    context_input = st.text_area("Add context (attendees, goals, background info):", height=300, key="context_input")

st.divider() # Visual separator

# --- Processing Logic ---
submit_button = st.button("✨ Generate Notes", type="primary", use_container_width=True)
notes_placeholder = st.empty() # Create a placeholder for the notes output

if submit_button:
    transcript = None
    error_message = None

    # --- Input Validation ---
    if text_input and uploaded_file:
        error_message = "⚠️ Please provide input using EITHER 'Paste Text' OR 'Upload PDF', not both."
    elif text_input:
        transcript = text_input.strip()
        if not transcript:
            error_message = "⚠️ Text area is empty. Please paste the transcript."
    elif uploaded_file:
        # Process PDF
        pdf_stream = io.BytesIO(uploaded_file.getvalue()) # Read file content into memory
        transcript = extract_text_from_pdf(pdf_stream)
        if transcript is None:
            # Error message is handled within extract_text_from_pdf using st.error
             error_message = "Could not extract text from PDF (see error above)." # Set flag
        elif not transcript:
             error_message = "⚠️ The uploaded PDF appears to contain no extractable text."
    else:
        error_message = "⚠️ Please provide a transcript either by pasting text or uploading a PDF."

    # --- Call Gemini if Transcript is Valid ---
    if transcript and not error_message:
        notes_placeholder.info("⏳ Generating notes... Please wait.") # Show loading message in the placeholder
        try:
            full_prompt = create_prompt(transcript, context_input.strip() if context_input else None)

            # Display prompt for debugging if needed (optional)
            # with st.expander("Show Prompt Sent to Gemini"):
            #    st.text(full_prompt)

            response = model.generate_content(full_prompt)

            # Use markdown to render potential formatting from the model
            notes_placeholder.markdown(response.text)

        except Exception as e:
            st.error(f"😥 An error occurred while generating notes with the Gemini API:")
            # Be careful about leaking too much detail in production errors
            st.error(f"{e}")
            # Try to clear the placeholder if there was an error after the spinner started
            notes_placeholder.empty()


    elif error_message:
        notes_placeholder.error(error_message) # Show validation errors in the placeholder

# --- Add some footer information (optional) ---
st.divider()
st.caption("Powered by Google Gemini | Built with Streamlit")

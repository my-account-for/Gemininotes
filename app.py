import streamlit as st
import os
# Make sure to install openai: pip install openai
from openai import OpenAI # Use OpenAI library

# --- Configuration ---
# Fetch API key from environment variable or Streamlit secrets
try:
    # Attempt to get the key from Streamlit secrets first
    api_key = st.secrets["OPENAI_API_KEY"]
except (AttributeError, KeyError):
    # Fallback to environment variable if secrets aren't available/configured
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop() # Stop execution if key is missing

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
DEFAULT_MODEL = "gpt-4o" # Or your preferred model

# --- Session State Management ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "current_page": "Chat",
        "messages": [],
        "summary": None,
        "summary_input_text": "",
        "summary_processing": False,
        # Earnings Call Specific State
        "ec_transcript": None,
        "ec_mode": "Generate New Notes", # Default mode
        "ec_topics": "",
        "ec_context": "",
        "ec_existing_notes": "",
        "ec_generated_notes": None,
        "ec_enriched_notes": None,
        "ec_processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state(keys_to_clear):
    """Clears specific keys from session state."""
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize after clearing to set defaults for the new mode
    initialize_session_state()

# --- API Call Functions ---
def get_chat_completion(messages, model=DEFAULT_MODEL):
    """Gets completion from OpenAI Chat API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during the API call: {e}")
        return None

def generate_summary(text, model=DEFAULT_MODEL):
    """Generates a summary using OpenAI API."""
    system_prompt = "You are an expert assistant specializing in summarizing text concisely and accurately."
    user_prompt = f"Please summarize the following text:\n\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return get_chat_completion(messages, model)

def generate_ec_notes(transcript, topics, context, model=DEFAULT_MODEL):
    """Generates new earnings call notes using OpenAI API."""
    system_prompt = "You are an AI assistant specialized in analyzing earnings call transcripts and generating structured, insightful notes for investors or analysts."
    user_prompt = f"""Analyze the following earnings call transcript.
Focus on these specific topics if provided: {topics if topics else 'Key financial results, management outlook, Q&A insights'}.
Consider this context if provided: {context if context else 'General analysis for investment decision'}.

Generate detailed notes covering:
- Key Financial Highlights (Revenue, Profitability, EPS, Guidance)
- Management Commentary (Strategy, Product Updates, Market Position)
- Key Themes and Tone
- Q&A Session Insights (Significant questions, management responses)
- Potential Red Flags or Positive Signals

Transcript:
\"\"\"
{transcript}
\"\"\"

Provide the notes in a clear, well-organized format (e.g., using Markdown)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return get_chat_completion(messages, model)

def enrich_ec_notes(transcript, existing_notes, model=DEFAULT_MODEL):
    """Enriches existing earnings call notes using OpenAI API."""
    system_prompt = "You are an AI assistant specialized in analyzing earnings call transcripts. You will enrich existing notes by adding details, context, or insights found in the transcript but missing from the original notes."
    user_prompt = f"""Review the following earnings call transcript and the user's existing notes.
Identify key information, nuances, or data points from the transcript that are NOT adequately covered in the existing notes.
Enrich the existing notes by adding these missing pieces of information or providing deeper context based *only* on the transcript.
Do not simply repeat information already present. Focus on adding value and completeness.

Transcript:
\"\"\"
{transcript}
\"\"\"

Existing Notes:
\"\"\"
{existing_notes}
\"\"\"

Provide the enriched notes, clearly indicating or integrating the new information derived from the transcript."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return get_chat_completion(messages, model)

# --- Helper Functions ---
def process_transcript(uploaded_file):
    """Reads and returns the content of the uploaded transcript file."""
    if uploaded_file is not None:
        try:
            # Assuming text file, adjust if needed (e.g., for PDF, DOCX)
            return uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

# --- UI Functions ---

def chat_page():
    """Renders the Chat page."""
    st.header("💬 AI Chat")

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is up?"):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare messages for API call (usually includes history)
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # Add a system prompt if desired (optional, depends on use case)
        # api_messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        # Get assistant response
        with st.spinner("Thinking..."):
            assistant_response = get_chat_completion(api_messages)

        if assistant_response:
            # Add assistant message to state and display it
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            st.error("Failed to get response from AI.")


def summarization_page():
    """Renders the Summarization page."""
    st.header("📝 Text Summarizer")

    st.session_state.summary_input_text = st.text_area(
        "Paste text to summarize here:",
        height=250,
        key="summary_input_widget", # Use a different key for widget if needed
        value=st.session_state.summary_input_text # Ensure it uses the state value
    )

    if st.button("Generate Summary", key="summarize_button"):
        if st.session_state.summary_input_text:
            st.session_state.summary_processing = True
            st.session_state.summary = None # Clear previous summary
            with st.spinner("Generating summary..."):
                summary = generate_summary(st.session_state.summary_input_text)
                if summary:
                    st.session_state.summary = summary
                else:
                    st.error("Failed to generate summary.")
            st.session_state.summary_processing = False
        else:
            st.warning("Please paste some text to summarize.")

    if st.session_state.summary:
        st.subheader("Summary:")
        st.markdown(st.session_state.summary) # Use Markdown for better formatting


def earnings_call_notes_page():
    """Renders the Earnings Call Notes page."""
    st.header("📞 Earnings Call Notes Assistant")

    uploaded_file = st.file_uploader("Upload Earnings Call Transcript (Text File)", type=["txt"])

    if uploaded_file:
        # Process transcript only if it's newly uploaded or not yet processed
        # Check if the uploaded file is different from the one possibly in state
        # This simple check assumes file name is unique enough for this session
        # A more robust check might involve file size or hash if needed.
        # Let's simplify: process whenever a file is present in the uploader widget this run
        transcript_content = process_transcript(uploaded_file)
        if transcript_content:
            st.session_state.ec_transcript = transcript_content
            # Optional: Show a snippet or confirmation
            # st.success("Transcript uploaded successfully!")
            # st.text_area("Transcript Preview:", transcript_content[:500] + "...", height=100, disabled=True)
        else:
            # If processing failed, clear the transcript state
            st.session_state.ec_transcript = None


    # Only show options if a transcript is loaded into the session state
    if "ec_transcript" in st.session_state and st.session_state.ec_transcript:

        st.success("Transcript loaded. Choose an action:")

        # --- Mode Selection ---
        mode = st.radio(
            "Select Action:",
            ("Generate New Notes", "Enrich Existing Notes"),
            key="ec_mode", # Session state key to store the selected mode
            # The value from st.session_state.ec_mode will be used automatically
            horizontal=True,
        )

        # --- Generate New Notes Section ---
        if st.session_state.ec_mode == "Generate New Notes":
            st.subheader("Generate New Notes from Transcript")

            # Use session state to preserve input across reruns
            st.session_state.ec_topics = st.text_input(
                "Topics to focus on (optional, comma-separated):",
                value=st.session_state.get("ec_topics", ""), # Get value from state
                key="ec_topics_input" # Use a widget key if needed, but ensure state key is correct
            )
            st.session_state.ec_context = st.text_area(
                "Context for analysis (optional):",
                value=st.session_state.get("ec_context", ""), # Get value from state
                key="ec_context_input", # Use a widget key if needed
                height=100
            )

            if st.button("Generate Notes", key="generate_notes_button"):
                st.session_state.ec_processing = True
                st.session_state.ec_generated_notes = None # Clear previous notes
                st.session_state.ec_enriched_notes = None # Clear enriched notes too
                with st.spinner("Generating new notes..."):
                    # Pass the current values from session state
                    notes = generate_ec_notes(
                        st.session_state.ec_transcript,
                        st.session_state.ec_topics, # Use value from state
                        st.session_state.ec_context  # Use value from state
                    )
                    if notes:
                        st.session_state.ec_generated_notes = notes
                    else:
                        st.error("Failed to generate notes.")
                st.session_state.ec_processing = False

            if st.session_state.ec_generated_notes:
                st.subheader("Generated Notes:")
                st.markdown(st.session_state.ec_generated_notes)

        # --- Enrich Existing Notes Section ---
        elif st.session_state.ec_mode == "Enrich Existing Notes":
            st.subheader("Enrich Existing Notes")
            st.write("Paste your existing notes below and the AI will add details found in the transcript.")

            # Text area for user's manual notes
            # Use session state to preserve input across reruns
            st.session_state.ec_existing_notes = st.text_area(
                "Your Existing Notes:",
                height=300,
                value=st.session_state.get("ec_existing_notes", ""), # Get value from state
                key="ec_existing_notes_input" # Widget key
            )

            if st.button("Enrich Notes", key="enrich_notes_button"):
                if st.session_state.ec_existing_notes:
                    st.session_state.ec_processing = True
                    st.session_state.ec_enriched_notes = None # Clear previous result
                    st.session_state.ec_generated_notes = None # Clear generated notes too
                    with st.spinner("Enriching your notes..."):
                        # Pass transcript and existing notes from state
                        enriched_notes = enrich_ec_notes(
                            st.session_state.ec_transcript,
                            st.session_state.ec_existing_notes # Use value from state
                        )
                        if enriched_notes:
                            st.session_state.ec_enriched_notes = enriched_notes
                        else:
                            st.error("Failed to enrich notes.")
                    st.session_state.ec_processing = False
                else:
                    st.warning("Please paste your existing notes first.")

            if st.session_state.ec_enriched_notes:
                st.subheader("Enriched Notes:")
                st.markdown(st.session_state.ec_enriched_notes)

    else:
        # Show only the uploader if no transcript is loaded
         st.info("Please upload a transcript file to begin.")


# --- Main App Structure ---
st.set_page_config(layout="wide")
st.title("AI Assistant Suite")

# Initialize session state at the beginning
initialize_session_state()

# Sidebar for navigation
st.sidebar.title("Navigation")
page_options = ["Chat", "Summarizer", "Earnings Call Notes"]

# Store previous page to detect changes
prev_page = st.session_state.current_page

# Use radio buttons for navigation, updating session state
st.session_state.current_page = st.sidebar.radio(
    "Go to",
    page_options,
    key="navigation_radio",
    index=page_options.index(st.session_state.current_page) # Set default based on current state
)

# --- Page Routing ---
if st.session_state.current_page == "Chat":
    chat_page()
elif st.session_state.current_page == "Summarizer":
    summarization_page()
elif st.session_state.current_page == "Earnings Call Notes":
    earnings_call_notes_page()

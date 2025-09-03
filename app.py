# /--------------------------\
# |   START OF app.py FILE   |
# \--------------------------/

# --- 1. IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
import io
import time
from datetime import datetime
import uuid
import traceback
from dotenv import load_dotenv
import PyPDF2
from pydub import AudioSegment
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from streamlit_pills import pills
from streamlit_ace import st_ace
import re
import tempfile

# --- Local Imports ---
import database

# --- 2. CONSTANTS & CONFIG ---
load_dotenv()
try:
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"]:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    else:
        st.session_state.config_error = "🔴 GEMINI_API_KEY not found."
except Exception as e:
    st.session_state.config_error = f"🔴 Error configuring Google AI Client: {e}"

MAX_PDF_MB = 25
MAX_AUDIO_MB = 200
# MODIFIED: Chunk size increased as per your request.
CHUNK_WORD_SIZE = 6000
CHUNK_WORD_OVERLAP = 300 # Overlap is proportionally increased to maintain context (5%)

AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash", "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash-lite", "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite", "Gemini 2.5 Pro": "gemini-2.5-pro",
}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Custom"]
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]

EXPERT_MEETING_DETAILED_PROMPT = """### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles). Use simple, direct language.
- **DO NOT:** Summarize or include introductions about consulting firms.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format.

**(2.A) Questions:**
-   Extract the clear, primary question and format it in **bold**.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL SPECIFICS.** This includes all data, names, examples, monetary values (`$`), percentages (`%`), etc."""

EXPERT_MEETING_CONCISE_PROMPT = """### **PRIMARY DIRECTIVE: EFFICIENT & NUANCED**
Your goal is to be **efficient**, not just brief. Efficiency means removing conversational filler ("um," "you know," repetition) but **preserving all substantive information**. Your output should be concise yet information-dense.

### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles).
- **DO NOT:** Summarize.

**(2.) Q&A format:**
Structure the main body in Question/Answer format.

**(2.A) Questions:**
-   Extract the clear, primary question and format it in **bold**.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, and specific entities mentioned.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Do not over-summarize. You must retain the following:
    -   **Sentiment & Tone:** Note if the speaker is optimistic, hesitant, confident, or speculative (e.g., "The expert was cautiously optimistic about...", "He speculated that...").
    -   **Qualifiers:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "rarely," "a potential risk is...").
    -   **Key Examples & Analogies:** If the speaker uses a specific example to illustrate a point, capture it, even if it's a few sentences long.
    -   **Cause & Effect:** Retain any reasoning provided (e.g., "...because of the new regulations," "...which led to a decrease in...")."""

PROMPT_INITIAL = """You are a High-Fidelity Factual Extraction Engine. Your task is to analyze an expert consultation transcript chunk and generate detailed, factual notes.
Your primary directive is **100% completeness and accuracy**. You will process the transcript sequentially. For every Question/Answer pair you identify, you must generate notes following the structure below.
---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are a High-Fidelity Factual Extraction Engine continuing a note-taking task. 

⚠️ CRITICAL: You must ONLY extract factual information from the actual transcript provided. DO NOT invent, assume, or create any content that is not explicitly stated in the transcript.

### **CONTEXT FROM PREVIOUS PROCESSING**
{context_package}

### **CONTINUATION INSTRUCTIONS**
1.  **LOCATE YOUR STARTING POINT:** Carefully review the context. Your first task is to find where the "Last complete Q&A processed" ends within the new transcript chunk below.
2.  **RESUME PROCESSING:** Begin your work from the **first new question and answer** that immediately follows the context.
3.  **PROCESS NEW CONTENT ONLY:** Process only the new Q&A pairs that appear in the remainder of the transcript.
4.  **MAINTAIN FORMAT:** Maintain the same formatting style as established in previous chunks.

### **QUALITY CONTROL & ERROR HANDLING**
-   NEVER create fictional questions or answers.
-   NEVER expand on topics not explicitly covered in the transcript.
-   If a Q&A pair seems incomplete, note it but do not fabricate the missing parts.
-   **CRITICAL GUARDRAIL:** If you cannot reliably find your starting point or if the new chunk is too ambiguous to continue, you MUST output **only** the following text and stop: `Error: Could not resume processing from context.`

---
{base_instructions}
---

**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}

**REMINDER**: Extract ONLY what is actually said in the transcript above. Do not invent content. If you are uncertain where to begin, use the error message."""


# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Paste Text"
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    earnings_call_mode: str = "Generate New Notes"
    selected_sector: str = "IT Services"
    notes_model: str = "Gemini 2.5 Pro"
    refinement_model: str = "Gemini 2.5 Flash Lite"
    transcription_model: str = "Gemini 2.5 Flash"
    refinement_enabled: bool = True
    add_context_enabled: bool = False
    context_input: str = ""
    speaker_1: str = ""
    speaker_2: str = ""
    earnings_call_topics: str = ""
    existing_notes_input: str = ""
    text_input: str = ""
    uploaded_file: Optional[Any] = None
    processing: bool = False
    active_note_id: Optional[str] = None
    error_message: Optional[str] = None
    fallback_content: Optional[str] = None

# --- 4. CORE PROCESSING & UTILITY FUNCTIONS ---
def sanitize_input(text: str) -> str:
    """Removes characters and keywords commonly used in prompt injection attacks."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[{}<>`]', '', text)
    injection_patterns = [
        r'ignore all previous instructions',
        r'you are now in.*mode',
        r'stop being an ai'
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def safe_get_token_count(response):
    """Safely get the token count from a response, returning 0 if unavailable."""
    try:
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return getattr(response.usage_metadata, 'total_token_count', 0)
    except (AttributeError, ValueError):
        st.warning("Could not retrieve token count from API response.")
        pass
    return 0
    
@st.cache_data(ttl=3600)
def get_file_content(uploaded_file) -> Tuple[Optional[str], str]:
    name = uploaded_file.name
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    ext = os.path.splitext(name)[1].lower()
    
    try:
        if ext == ".pdf":
            reader = PyPDF2.PdfReader(file_bytes)
            if reader.is_encrypted: return "Error: PDF is encrypted.", name
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return (content, name) if content else ("Error: No text found in PDF.", name)
        elif ext in [".txt", ".md"]:
            return file_bytes.read().decode("utf-8"), name
        elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
            return "audio_file", name
    except Exception as e:
        return f"Error: Could not process file {name}. Details: {str(e)}", name
    return None, name

@st.cache_data
def db_get_sectors() -> dict:
    return database.get_sectors()

def create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks, start = [], 0
    while start < len(words):
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap to prevent infinite loops.")
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words): break
        start += (chunk_size - overlap)
    return chunks

def _create_enhanced_context_from_notes(notes_text, chunk_number=0):
    """Create richer context from previous notes"""
    if not notes_text or not notes_text.strip():
        return ""
    
    questions = re.findall(r"(\*\*.*?\*\*)", notes_text)
    
    if not questions:
        return ""
    
    context_questions = questions[-3:] if len(questions) >= 3 else questions
    
    context_parts = [
        f"**Chunk #{chunk_number} Context Summary:**",
        f"- Total questions processed so far: {len(questions)}",
        f"- Recent question topics: {', '.join(q.strip('*') for q in context_questions[-2:])}",
        f"- Last complete Q&A processed: {questions[-1]}"
    ]
    
    last_question = questions[-1]
    answer_match = re.search(
        re.escape(last_question) + r"(.*?)(?=\*\*|$)", 
        notes_text, 
        re.DOTALL
    )
    if answer_match:
        last_answer = answer_match.group(1).strip()
        context_parts.append(f"- Last answer content:\n{last_answer[:300]}...")
    
    return "\n".join(context_parts)

def get_dynamic_prompt(state: AppState, transcript_chunk: str) -> str:
    meeting_type = state.selected_meeting_type
    sanitized_context = sanitize_input(state.context_input)
    context_section = f"**ADDITIONAL CONTEXT:**\n{sanitized_context}" if state.add_context_enabled and sanitized_context else ""

    if meeting_type == "Expert Meeting":
        if state.selected_note_style == "Option 1: Detailed & Strict":
            prompt_base = EXPERT_MEETING_DETAILED_PROMPT
        else:
            prompt_base = EXPERT_MEETING_CONCISE_PROMPT
        return f"{prompt_base}\n\n{context_section}\n\n**MEETING TRANSCRIPT CHUNK:**\n{transcript_chunk}"
    elif meeting_type == "Earnings Call":
        topic_instructions = state.earnings_call_topics or "Identify logical themes and use them as bold headings."
        if state.earnings_call_mode == "Enrich Existing Notes":
            return f"Enrich the following existing notes based on the new transcript chunk. Focus on these topics: {topic_instructions}\n\n**EXISTING NOTES:**\n{state.existing_notes_input}\n\n**NEW TRANSCRIPT CHUNK:**\n{transcript_chunk}"
        else:
            return f"Generate detailed earnings call notes based on the transcript chunk. Structure your notes under these topics: {topic_instructions}\n\n{context_section}\n\n**TRANSCRIPT CHUNK:**\n{transcript_chunk}"
    elif meeting_type == "Custom":
        return f"CUSTOM PROMPT: Please follow user instructions.\n\n{context_section}\n\n**TRANSCRIPT:**\n{transcript_chunk}"
    return f"Error: Unknown meeting type '{meeting_type}'"

def validate_inputs(state: AppState) -> Optional[str]:
    if state.input_method == "Paste Text" and not state.text_input.strip():
        return "Please paste a transcript or switch to file upload."
    if state.input_method == "Upload File" and not state.uploaded_file:
        return "Please upload a file or switch to pasting text."
    
    if state.uploaded_file:
        size_mb = state.uploaded_file.size / (1024 * 1024)
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        if ext == ".pdf" and size_mb > MAX_PDF_MB:
            return f"PDF is too large ({size_mb:.1f}MB). Limit: {MAX_PDF_MB}MB."
        elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac'] and size_mb > MAX_AUDIO_MB:
            return f"Audio is too large ({size_mb:.1f}MB). Limit: {MAX_AUDIO_MB}MB."
            
    if state.selected_meeting_type == "Earnings Call" and state.earnings_call_mode == "Enrich Existing Notes" and not state.existing_notes_input:
        return "Please provide existing notes for enrichment mode."
    return None

def process_and_save_task(state: AppState, status_ui):
    start_time = time.time()
    notes_model = genai.GenerativeModel(AVAILABLE_MODELS[state.notes_model])
    refinement_model = genai.GenerativeModel(AVAILABLE_MODELS[state.refinement_model])
    transcription_model = genai.GenerativeModel(AVAILABLE_MODELS[state.transcription_model])
    
    status_ui.update(label="Step 1: Preparing Source Content...")
    raw_transcript, file_name = "", "Pasted Text"
    
    if state.input_method == "Upload File" and state.uploaded_file:
        file_type, name = get_file_content(state.uploaded_file)
        file_name = name
        if file_type == "audio_file":
            status_ui.update(label="Step 1.1: Processing Audio...")
            audio_bytes = state.uploaded_file.getvalue()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            chunk_length_ms = 5 * 60 * 1000
            audio_chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            all_transcripts, cloud_files, local_files = [], [], []
            try:
                for i, chunk in enumerate(audio_chunks):
                    try:
                        status_ui.update(label=f"Step 1.2: Transcribing audio chunk {i+1}/{len(audio_chunks)}...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                            chunk.export(temp_f.name, format="wav")
                            local_files.append(temp_f.name)
                            cloud_ref = genai.upload_file(path=temp_f.name)
                            cloud_files.append(cloud_ref.name)
                            while cloud_ref.state.name == "PROCESSING": time.sleep(2); cloud_ref = genai.get_file(cloud_ref.name)
                            if cloud_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {i+1} cloud processing failed.")
                            response = transcription_model.generate_content(["Transcribe this audio.", cloud_ref])
                            all_transcripts.append(response.text)
                    except Exception as e:
                        raise Exception(f"Transcription failed on chunk {i+1}/{len(audio_chunks)}. Reason: {e}")
                
                raw_transcript = "\n\n".join(all_transcripts).strip()
            finally:
                for path in local_files: os.remove(path)
                for cloud_name in cloud_files: 
                    try: genai.delete_file(cloud_name)
                    except Exception as e: st.warning(f"Could not delete cloud file {cloud_name}: {e}")
        
        elif file_type is None or file_type.startswith("Error:"):
            raise ValueError(file_type or "Failed to read file content.")
        else:
            raw_transcript = file_type
    elif state.input_method == "Paste Text":
        raw_transcript = state.text_input

    if not raw_transcript: raise ValueError("Source content is empty.")
    
    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0
    
    s1 = sanitize_input(state.speaker_1)
    s2 = sanitize_input(state.speaker_2)

    if state.refinement_enabled:
        status_ui.update(label="Step 2: Refining Transcript...")
        words = raw_transcript.split()
        
        if len(words) <= CHUNK_WORD_SIZE:
            speaker_info = f"Speakers are {s1} and {s2}." if s1 and s2 else ""
            refine_prompt = f"Refine the following transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info}\n\nTRANSCRIPT:\n{raw_transcript}"
            response = refinement_model.generate_content(refine_prompt)
            refined_transcript = response.text
            total_tokens += safe_get_token_count(response)
        else:
            chunks = create_chunks_with_overlap(raw_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
            all_refined_chunks = []
            
            for i, chunk in enumerate(chunks):
                status_ui.update(label=f"Step 2: Refining Transcript (Chunk {i+1}/{len(chunks)})...")
                
                context = ""
                if i > 0 and all_refined_chunks:
                    last_refined_chunk = all_refined_chunks[-1]
                    context_words = last_refined_chunk.split()
                    context = " ".join(context_words[-150:])
                
                speaker_info = f"Speakers are {s1} and {s2}." if s1 and s2 else ""
                
                if not context:
                    prompt = f"You are refining a transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info}\n\nTRANSCRIPT CHUNK TO REFINE:\n{chunk}"
                else:
                    # MODIFICATION 1: Simplified prompt. 
                    # We removed the confusing 'Do NOT repeat' instruction. We will now handle the overlap in our code.
                    prompt = f"""You are continuing to refine a long transcript. Below is the tail end of the previously refined section for context. Your task is to refine the new chunk provided, ensuring a seamless and natural transition.
{speaker_info}
---
CONTEXT FROM PREVIOUSLY REFINED CHUNK (FOR CONTINUITY ONLY):
...{context}
---
NEW TRANSCRIPT CHUNK TO REFINE:
{chunk}"""
                
                response = refinement_model.generate_content(prompt)
                all_refined_chunks.append(response.text)
                total_tokens += safe_get_token_count(response)
            
            # MODIFICATION 2: Intelligent stitching of refined chunks.
            # Instead of a simple join, we now handle the overlap properly to prevent text loss.
            if all_refined_chunks:
                final_refined_words = all_refined_chunks[0].split()
                for i in range(1, len(all_refined_chunks)):
                    # Get the original raw chunk to calculate the overlap proportion
                    original_chunk_words = chunks[i].split()
                    if not original_chunk_words:
                        continue
                        
                    # Calculate the proportion of overlap in the *input* text
                    overlap_proportion = CHUNK_WORD_OVERLAP / len(original_chunk_words)
                    
                    # Apply this proportion to the *output* text to find the stitch point
                    refined_chunk_words = all_refined_chunks[i].split()
                    estimated_overlap_in_refined = int(len(refined_chunk_words) * overlap_proportion)
                    
                    # Append only the new part of the refined chunk
                    final_refined_words.extend(refined_chunk_words[estimated_overlap_in_refined:])
                
                refined_transcript = " ".join(final_refined_words)
            else:
                refined_transcript = "" # Handle case where no chunks were processed
        
        final_transcript = refined_transcript

    status_ui.update(label="Step 3: Generating Notes...")
    words = final_transcript.split()
    final_notes_content = ""

    # NOTE: The CHUNK_WORD_SIZE now applies to note generation as well.
    if state.selected_meeting_type == "Expert Meeting" and len(words) > CHUNK_WORD_SIZE:
        chunks = create_chunks_with_overlap(final_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
        all_notes, context_package = [], ""
        
        if state.selected_note_style == "Option 1: Detailed & Strict":
            prompt_base = EXPERT_MEETING_DETAILED_PROMPT
        else:
            prompt_base = EXPERT_MEETING_CONCISE_PROMPT

        for i, chunk in enumerate(chunks):
            status_ui.update(label=f"Step 3: Generating Notes (Chunk {i+1}/{len(chunks)})...")
            prompt_template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION
            prompt = prompt_template.format(base_instructions=prompt_base, chunk_text=chunk, context_package=context_package)
            response = notes_model.generate_content(prompt)
            
            if "Error: Could not resume processing from context." in response.text:
                st.warning(f"Warning: Model could not process chunk {i+1} due to context ambiguity. It will be skipped.")
                continue

            all_notes.append(response.text)
            total_tokens += safe_get_token_count(response)
            context_package = _create_enhanced_context_from_notes("\n\n".join(all_notes), chunk_number=i + 1)
        
        final_notes_content = "\n\n".join(all_notes)

    else:
        prompt = get_dynamic_prompt(state, final_transcript)
        response = notes_model.generate_content(prompt)
        final_notes_content = response.text
        total_tokens += safe_get_token_count(response)

    if state.selected_note_style == "Option 3: Less Verbose + Summary" and state.selected_meeting_type == "Expert Meeting":
        status_ui.update(label="Step 4: Generating Executive Summary...")
        summary_prompt = f"Create a concise executive summary from these notes:\n\n{final_notes_content}"
        response = notes_model.generate_content(summary_prompt)
        final_notes_content += f"\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{response.text}"
        total_tokens += safe_get_token_count(response)

    status_ui.update(label="Step 5: Saving to Database...")
    note_data = {
        'id': str(uuid.uuid4()), 'created_at': datetime.now().isoformat(), 'meeting_type': state.selected_meeting_type,
        'file_name': file_name, 'content': final_notes_content, 'raw_transcript': raw_transcript,
        'refined_transcript': refined_transcript, 'token_usage': total_tokens, 'processing_time': time.time() - start_time
    }
    try:
        database.save_note(note_data)
    except Exception as db_error:
        st.session_state.app_state.fallback_content = final_notes_content
        raise Exception(f"Processing succeeded, but failed to save the note to the database. You can download the unsaved note below. Error: {db_error}")
    return note_data

# --- 5. UI RENDERING FUNCTIONS ---
def on_sector_change():
    state = st.session_state.app_state
    all_sectors = db_get_sectors()
    state.earnings_call_topics = all_sectors.get(state.selected_sector, "")

def render_input_and_processing_tab(state: AppState):
    state.input_method = pills("Input Method", ["Paste Text", "Upload File"], index=["Paste Text", "Upload File"].index(state.input_method))
    
    if state.input_method == "Paste Text":
        state.text_input = st.text_area("Paste source transcript here:", value=state.text_input, height=250, key="text_input_main")
        state.uploaded_file = None
    else:
        state.uploaded_file = st.file_uploader("Upload a File (PDF, TXT, MP3, etc.)", type=['pdf', 'txt', 'mp3', 'm4a', 'wav', 'ogg', 'flac'])

    st.subheader("Configuration")
    state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type))
    
    if state.selected_meeting_type == "Expert Meeting":
        state.selected_note_style = st.selectbox("Note Style", EXPERT_MEETING_OPTIONS, index=EXPERT_MEETING_OPTIONS.index(state.selected_note_style))
    elif state.selected_meeting_type == "Earnings Call":
        state.earnings_call_mode = st.radio("Mode", EARNINGS_CALL_MODES, horizontal=True, index=EARNINGS_CALL_MODES.index(state.earnings_call_mode))
        
        all_sectors = db_get_sectors()
        sector_options = ["Other / Manual Topics"] + sorted(list(all_sectors.keys()))
        
        try:
            current_sector_index = sector_options.index(state.selected_sector)
        except ValueError:
            current_sector_index = 0

        state.selected_sector = st.selectbox("Sector (for Topic Templates)", sector_options, index=current_sector_index, on_change=on_sector_change, key="sector_selector")
        state.earnings_call_topics = st.text_area("Topic Instructions", value=state.earnings_call_topics, height=150, placeholder="Select a sector to load a template, or enter topics manually.")

        with st.expander("✏️ Manage Sector Templates", expanded=False):
            st.write("Add, edit, or delete the sector templates used in the dropdown above.")
            st.markdown("**Edit or Delete an Existing Sector**")
            sector_to_edit = st.selectbox("Select Sector to Edit", sorted(list(all_sectors.keys())))
            
            if sector_to_edit:
                topics_for_edit = st.text_area("Sector Topics", value=all_sectors[sector_to_edit], key=f"topics_{sector_to_edit}")
                col1, col2 = st.columns([1,1])
                if col1.button("💾 Save Changes", key=f"save_{sector_to_edit}"):
                    database.save_sector(sector_to_edit, topics_for_edit); db_get_sectors.clear(); st.toast(f"✅ Sector '{sector_to_edit}' updated!", icon="💾"); st.rerun()
                if col2.button("❌ Delete Sector", type="primary", key=f"delete_{sector_to_edit}"):
                    database.delete_sector(sector_to_edit); db_get_sectors.clear(); state.selected_sector = "Other / Manual Topics"; on_sector_change(); st.toast(f"🗑️ Sector '{sector_to_edit}' deleted!", icon="🗑️"); st.rerun()

            st.divider()
            st.markdown("**Add a New Sector**")
            new_sector_name = st.text_input("New Sector Name")
            new_sector_topics = st.text_area("Topics for New Sector", key="new_sector_topics")
            
            if st.button("➕ Add New Sector"):
                if new_sector_name and new_sector_topics:
                    database.save_sector(new_sector_name, new_sector_topics); db_get_sectors.clear(); st.toast(f"✅ Sector '{new_sector_name}' added!", icon="➕"); st.rerun()
                else:
                    st.warning("Please provide both a name and topics for the new sector.")

        if state.earnings_call_mode == "Enrich Existing Notes":
            state.existing_notes_input = st.text_area("Paste Existing Notes to Enrich:", value=state.existing_notes_input)
    
    with st.expander("⚙️ Advanced Settings & Models"):
        state.refinement_enabled = st.toggle("Enable Transcript Refinement", value=state.refinement_enabled)
        state.add_context_enabled = st.toggle("Add General Context", value=state.add_context_enabled)
        if state.add_context_enabled: state.context_input = st.text_area("Context Details:", value=state.context_input, placeholder="e.g., Company Name, Date...")
        
        c1, c2 = st.columns(2)
        state.speaker_1 = c1.text_input("Speaker 1 Name (Optional)", value=state.speaker_1)
        state.speaker_2 = c2.text_input("Speaker 2 Name (Optional)", value=state.speaker_2)

        st.markdown("**Model Selection**")
        m1, m2, m3 = st.columns(3)
        state.notes_model = m1.selectbox("Notes Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.notes_model))
        state.refinement_model = m2.selectbox("Refinement Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.refinement_model))
        state.transcription_model = m3.selectbox("Transcription Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.transcription_model), help="Used for audio files.")
    
    st.subheader("Prompt Preview")
    prompt_preview = get_dynamic_prompt(state, "[...transcript content...]")
    st_ace(value=prompt_preview, language='markdown', theme='github', height=200, readonly=True, key="prompt_preview_ace")
    
    st.divider()
    st.subheader("🚀 Generate")
    validation_error = validate_inputs(state)
    
    if st.button("Generate Notes", type="primary", use_container_width=True, disabled=bool(validation_error)):
        state.processing = True; state.error_message = None; state.fallback_content = None; st.rerun()

    if validation_error: st.warning(f"⚠️ Please fix the following: {validation_error}")
        
    if state.processing:
        with st.status("Processing your request...", expanded=True) as status:
            try:
                final_note = process_and_save_task(state, status)
                state.active_note_id = final_note['id']
                status.update(label="✅ Success! View your note in the 'Output & History' tab.", state="complete")
            except Exception as e:
                state.error_message = f"An error occurred during processing:\n{e}"
                status.update(label=f"❌ Error: {e}", state="error")
        state.processing = False

    if state.error_message:
        st.error("Last run failed. See details below:")
        st.code(state.error_message)
        if state.fallback_content:
            st.download_button("⬇️ Download Unsaved Note (.txt)", state.fallback_content, "synthnotes_fallback.txt")
        if st.button("Clear Error"):
            state.error_message = None
            state.fallback_content = None
            st.rerun()

def render_output_and_history_tab(state: AppState):
    st.subheader("📄 Active Note")
    notes = database.get_all_notes()
    if not notes:
        st.info("No notes generated. Go to the 'Input & Generate' tab to create one.")
        return
        
    if not state.active_note_id or not any(n['id'] == state.active_note_id for n in notes):
        state.active_note_id = notes[0]['id']

    active_note = next((n for n in notes if n['id'] == state.active_note_id), notes[0])
    
    st.markdown(f"**Viewing Note for:** `{active_note['file_name']}`")
    st.caption(f"ID: {active_note['id']} | Generated: {datetime.fromisoformat(active_note['created_at']).strftime('%Y-%m-%d %H:%M')}")
    
    edited_content = st_ace(value=active_note['content'], language='markdown', theme='github', height=600, key=f"output_ace_{active_note['id']}")
    
    with st.expander("View Source Transcripts"):
        if active_note['refined_transcript']: st.text_area("Refined Transcript", value=active_note['refined_transcript'], height=200, disabled=True, key=f"refined_tx_{active_note['id']}")
        if active_note['raw_transcript']: st.text_area("Raw Source", value=active_note['raw_transcript'], height=200, disabled=True, key=f"raw_tx_{active_note['id']}")
            
    st.divider()
    st.subheader("📊 Analytics & History")
    summary = database.get_analytics_summary()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Notes in DB", summary['total_notes'])
    c2.metric("Avg. Time / Note", f"{summary['avg_time']:.1f}s")
    c3.metric("Total Tokens (Est.)", f"{summary['total_tokens']:,}")

    for note in notes:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**File:** `{note['file_name']}` ({note['meeting_type']})")
                st.caption(f"ID: {note['id']} | {datetime.fromisoformat(note['created_at']).strftime('%Y-%m-%d %H:%M')}")
            with col2:
                if st.button("Set as Active", key=f"view_{note['id']}", use_container_width=True):
                    state.active_note_id = note['id']; st.rerun()
            with st.expander("Preview"): st.markdown(note['content'])

# --- 6. MAIN APPLICATION RUNNER ---
def run_app():
    st.set_page_config(page_title="SynthNotes AI", layout="wide")
    st.title("SynthNotes AI")
    
    if "config_error" in st.session_state:
        st.error(st.session_state.config_error); st.stop()
        
    try:
        database.init_db()
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
            on_sector_change()

        tabs = st.tabs(["📝 Input & Generate", "📄 Output & History"])
        
        with tabs[0]: render_input_and_processing_tab(st.session_state.app_state)
        with tabs[1]: render_output_and_history_tab(st.session_state.app_state)
    
    except Exception as e:
        st.error("A critical application error occurred."); st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()

# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/

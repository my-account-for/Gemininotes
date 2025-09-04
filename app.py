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
from streamlit_pills import pills
from streamlit_ace import st_ace
import re
import tempfile
import json
import pandas as pd
import numpy as np

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
CHUNK_WORD_SIZE = 6000
CHUNK_WORD_OVERLAP = 300 

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
**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format.
**(2.A) Questions:**
-   Extract the clear, primary question and format it in **bold**.
**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL SPECIFICS.** This includes all data, names, examples, monetary values (`$`), percentages (`%`), etc.
-   **CRITICAL: For text extracted from PDFs, you MUST prepend each bullet point with the source page number, like this: `[p. 5] - The expert stated...` If the source page is not available or applicable, omit this tag.**"""

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
-   **CRITICAL: For text extracted from PDFs, you MUST prepend each bullet point with the source page number, like this: `[p. 12] - The expert was cautiously optimistic about...` If the source page is not available or applicable, omit this tag.**
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Do not over-summarize. You must retain sentiment, qualifiers, key examples, and cause & effect."""

PROMPT_INITIAL = """You are a High-Fidelity Factual Extraction Engine. Your task is to analyze an expert consultation transcript chunk and generate detailed, factual notes.
Your primary directive is **100% completeness and accuracy**. You will process the transcript sequentially. For every Question/Answer pair you identify, you must generate notes following the structure below.
---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are a High-Fidelity Factual Extraction Engine continuing a note-taking task from a long transcript.
### **CONTEXT FROM PREVIOUS PROCESSING**
Below is a summary of the notes generated from the previous transcript chunk. Use this to understand the flow of the conversation.
{context_package}
### **CONTINUATION INSTRUCTIONS**
1.  **PROCESS THE ENTIRE CHUNK:** Your task is to process the **entire** new transcript chunk provided below.
2.  **HANDLE OVERLAP:** The beginning of this new chunk overlaps with the end of the previous one. Process it naturally. Your output will be automatically de-duplicated later.
3.  **MAINTAIN FORMAT:** Continue to use the exact same Q&A formatting as established in the base instructions.
---
{base_instructions}
---
**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""

# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Upload File"
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
    processing_queue: List[Any] = field(default_factory=list)
    active_note_pdf_bytes: Optional[bytes] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    knowledge_explorer_selection: Dict = field(default_factory=dict)
    knowledge_explorer_search_results: List[Dict] = field(default_factory=list)

# --- 4. CORE PROCESSING & UTILITY FUNCTIONS ---
def sanitize_input(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'[{}<>`]', '', text); injection_patterns = [r'ignore all previous instructions', r'you are now in.*mode', r'stop being an ai']
    for pattern in injection_patterns: text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def safe_get_token_count(response):
    try:
        if hasattr(response, 'usage_metadata') and response.usage_metadata: return getattr(response.usage_metadata, 'total_token_count', 0)
    except (AttributeError, ValueError): pass
    return 0

@st.cache_data(ttl=3600)
def get_file_content(uploaded_file) -> Tuple[Optional[str], str]:
    name = uploaded_file.name; file_bytes_io = io.BytesIO(uploaded_file.getvalue()); ext = os.path.splitext(name)[1].lower()
    try:
        if ext == ".pdf":
            reader = PyPDF2.PdfReader(file_bytes_io)
            if reader.is_encrypted: return "Error: PDF is encrypted.", name
            content = "".join(f"--- START OF PAGE {i+1} ---\n{page.extract_text()}\n--- END OF PAGE {i+1} ---\n\n" for i, page in enumerate(reader.pages) if page.extract_text())
            return (content, name) if content else ("Error: No text found in PDF.", name)
        elif ext in [".txt", ".md"]: return file_bytes_io.read().decode("utf-8"), name
        elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]: return "audio_file", name
    except Exception as e: return f"Error: Could not process file {name}. Details: {str(e)}", name
    return None, name

@st.cache_data
def db_get_sectors() -> dict:
    return database.get_sectors()

def create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text: return []; words = text.split()
    if len(words) <= chunk_size: return [text]
    if chunk_size <= overlap: raise ValueError("Chunk size must be greater than overlap.")
    chunks = []; step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if (i + chunk_size) >= len(words): break
    return chunks

def _create_enhanced_context_from_notes(notes_text, chunk_number=0):
    if not notes_text or not notes_text.strip(): return ""
    questions = re.findall(r"(\*\*.*?\*\*)", notes_text)
    if not questions: return ""
    context_questions = questions[-3:]; last_question = questions[-1]
    context_parts = [f"**Chunk #{chunk_number} Context Summary:**", f"- Total questions processed so far: {len(questions)}", f"- Recent question topics: {', '.join(q.strip('*') for q in context_questions[-2:])}", f"- Last complete Q&A processed: {last_question}"]
    answer_match = re.search(re.escape(last_question) + r"(.*?)(?=\*\*|$)", notes_text, re.DOTALL)
    if answer_match: context_parts.append(f"- Last answer content:\n{answer_match.group(1).strip()[:300]}...")
    return "\n".join(context_parts)

def get_dynamic_prompt(state: AppState, transcript_chunk: str) -> str:
    meeting_type = state.selected_meeting_type; sanitized_context = sanitize_input(state.context_input); context_section = f"**ADDITIONAL CONTEXT:**\n{sanitized_context}" if state.add_context_enabled and sanitized_context else ""
    if meeting_type == "Expert Meeting":
        base_instructions = EXPERT_MEETING_DETAILED_PROMPT if state.selected_note_style == "Option 1: Detailed & Strict" else EXPERT_MEETING_CONCISE_PROMPT
        return f"{base_instructions}\n\n{context_section}\n\n**MEETING TRANSCRIPT CHUNK:**\n{transcript_chunk}"
    elif meeting_type == "Earnings Call":
        topic_instructions = state.earnings_call_topics or "Identify logical themes and use them as bold headings."
        if state.earnings_call_mode == "Enrich Existing Notes": return f"Enrich the following existing notes based on the new transcript chunk. Focus on these topics: {topic_instructions}\n\n**EXISTING NOTES:**\n{state.existing_notes_input}\n\n**NEW TRANSCRIPT CHUNK:**\n{transcript_chunk}"
        else: return f"Generate detailed earnings call notes based on the transcript chunk. Structure your notes under these topics: {topic_instructions}\n\n{context_section}\n\n**TRANSCRIPT CHUNK:**\n{transcript_chunk}"
    elif meeting_type == "Custom": return f"CUSTOM PROMPT: Please follow user instructions.\n\n{context_section}\n\n**TRANSCRIPT:**\n{transcript_chunk}"
    return f"Error: Unknown meeting type '{meeting_type}'"

def extract_entities(note_id: str, note_content: str, status_ui):
    status_ui.update(label="Step 6: Extracting Entities..."); model = genai.GenerativeModel(AVAILABLE_MODELS[st.session_state.app_state.notes_model])
    prompt = f"""Analyze the following meeting notes and extract key entities. Your output MUST be a valid JSON list of objects. Each object should represent a single entity and have: "entity", "type" (one of: "Company", "Person", "Product", "Technology", "Metric", "Other"), "sentiment" (one of: "Positive", "Negative", "Neutral"), and "context" (a brief quote).\n\nExample format:\n[ {{"entity": "InnovateCorp", "type": "Company", "sentiment": "Positive", "context": "The expert was very optimistic about InnovateCorp's market position."}} ]\n\nNotes:\n---\n{note_content}"""
    try:
        response = model.generate_content(prompt); json_text = response.text.strip().lstrip("```json").rstrip("```"); entities = json.loads(json_text)
        if isinstance(entities, list): database.save_entities(note_id, entities)
    except Exception as e: st.warning(f"Could not extract/save entities: {e}", icon="⚠️")

def process_and_save_task(state: AppState, status_ui):
    start_time = time.time()
    notes_model = genai.GenerativeModel(AVAILABLE_MODELS[state.notes_model])
    refinement_model = genai.GenerativeModel(AVAILABLE_MODELS[state.refinement_model])
    transcription_model = genai.GenerativeModel(AVAILABLE_MODELS[state.transcription_model])
    
    status_ui.update(label="Step 1: Preparing Source Content...")
    raw_transcript, file_name, pdf_bytes = "", "Pasted Text", None
    
    uploaded_file_obj = state.uploaded_file

    if state.input_method == "Upload File" and uploaded_file_obj:
        file_name = uploaded_file_obj.name
        if file_name.lower().endswith('.pdf'): pdf_bytes = uploaded_file_obj.getvalue()

        file_type, _ = get_file_content(uploaded_file_obj)
        if file_type == "audio_file":
            status_ui.update(label="Step 1.1: Processing Audio...")
            audio_bytes = uploaded_file_obj.getvalue()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            chunk_length_ms = 5 * 60 * 1000
            audio_chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            all_transcripts, cloud_files, local_files = [], [], []
            try:
                for i, chunk in enumerate(audio_chunks):
                    try:
                        status_ui.update(label=f"Step 1.2: Transcribing audio chunk {i+1}/{len(audio_chunks)}...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                            chunk.export(temp_f.name, format="wav"); local_files.append(temp_f.name); cloud_ref = genai.upload_file(path=temp_f.name); cloud_files.append(cloud_ref.name)
                            while cloud_ref.state.name == "PROCESSING": time.sleep(2); cloud_ref = genai.get_file(cloud_ref.name)
                            if cloud_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {i+1} cloud processing failed.")
                            response = transcription_model.generate_content(["Transcribe this audio.", cloud_ref]); all_transcripts.append(response.text)
                    except Exception as e: raise Exception(f"Transcription failed on chunk {i+1}/{len(audio_chunks)}. Reason: {e}")
                raw_transcript = "\n\n".join(all_transcripts).strip()
            finally:
                for path in local_files: os.remove(path)
                for cloud_name in cloud_files: 
                    try: genai.delete_file(cloud_name)
                    except Exception as e: st.warning(f"Could not delete cloud file {cloud_name}: {e}")
        elif file_type is None or file_type.startswith("Error:"): raise ValueError(file_type or "Failed to read file content.")
        else: raw_transcript = file_type
    elif state.input_method == "Paste Text": raw_transcript = state.text_input

    if not raw_transcript: raise ValueError("Source content is empty.")
    
    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0
    s1 = sanitize_input(state.speaker_1); s2 = sanitize_input(state.speaker_2)

    if state.refinement_enabled:
        status_ui.update(label="Step 2: Refining Transcript..."); words = raw_transcript.split()
        if len(words) <= CHUNK_WORD_SIZE:
            speaker_info = f"Speakers are {s1} and {s2}." if s1 and s2 else ""
            refine_prompt = f"Refine the following transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info}\n\nTRANSCRIPT:\n{raw_transcript}"
            response = refinement_model.generate_content(refine_prompt); refined_transcript = response.text; total_tokens += safe_get_token_count(response)
        else:
            chunks = create_chunks_with_overlap(raw_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP); all_refined_chunks = []
            for i, chunk in enumerate(chunks):
                status_ui.update(label=f"Step 2: Refining Transcript (Chunk {i+1}/{len(chunks)})...")
                context = " ".join(all_refined_chunks[-1].split()[-150:]) if i > 0 and all_refined_chunks else ""
                speaker_info = f"Speakers are {s1} and {s2}." if s1 and s2 else ""
                prompt = f"""You are continuing to refine a long transcript... CONTEXT: ...{context}\n---NEW TRANSCRIPT CHUNK TO REFINE:\n{chunk}""" if context else f"You are refining a transcript...TRANSCRIPT CHUNK TO REFINE:\n{chunk}"
                response = refinement_model.generate_content(prompt); all_refined_chunks.append(response.text); total_tokens += safe_get_token_count(response)
            if all_refined_chunks:
                final_refined_words = all_refined_chunks[0].split()
                for i in range(1, len(all_refined_chunks)):
                    original_chunk_words = chunks[i].split()
                    if not original_chunk_words: continue
                    overlap_proportion = CHUNK_WORD_OVERLAP / len(original_chunk_words); refined_chunk_words = all_refined_chunks[i].split(); estimated_overlap_in_refined = int(len(refined_chunk_words) * overlap_proportion)
                    final_refined_words.extend(refined_chunk_words[estimated_overlap_in_refined:])
                refined_transcript = " ".join(final_refined_words)
        final_transcript = refined_transcript if refined_transcript and refined_transcript.strip() else raw_transcript

    status_ui.update(label="Step 3: Generating Notes..."); words = final_transcript.split(); final_notes_content = ""
    if state.selected_meeting_type == "Expert Meeting" and len(words) > CHUNK_WORD_SIZE:
        chunks = create_chunks_with_overlap(final_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP); all_notes_chunks = []; context_package = ""
        prompt_base = EXPERT_MEETING_DETAILED_PROMPT if state.selected_note_style == "Option 1: Detailed & Strict" else EXPERT_MEETING_CONCISE_PROMPT
        for i, chunk in enumerate(chunks):
            status_ui.update(label=f"Step 3: Generating Notes (Chunk {i+1}/{len(chunks)})..."); prompt_template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION; prompt = prompt_template.format(base_instructions=prompt_base, chunk_text=chunk, context_package=context_package)
            response = notes_model.generate_content(prompt); total_tokens += safe_get_token_count(response); current_notes_text = response.text; all_notes_chunks.append(current_notes_text)
            context_package = _create_enhanced_context_from_notes("\n\n".join(all_notes_chunks), chunk_number=i + 1)
        if all_notes_chunks:
            final_notes_content = all_notes_chunks[0]
            for i in range(1, len(all_notes_chunks)):
                prev_notes = all_notes_chunks[i-1]; current_notes = all_notes_chunks[i]; last_q_match = list(re.finditer(r"(\*\*.*?\*\*)", prev_notes))
                if not last_q_match: final_notes_content += "\n\n" + current_notes; continue
                last_question = last_q_match[-1].group(1); stitch_point = current_notes.find(last_question)
                if stitch_point != -1:
                    next_q_match = re.search(r"(\*\*.*?\*\*)", current_notes[stitch_point + len(last_question):])
                    if next_q_match: final_notes_content += "\n\n" + current_notes[stitch_point + len(last_question) + next_q_match.start():]
                    else: final_notes_content += "\n\n" + current_notes[stitch_point + len(last_question):]
                else: final_notes_content += "\n\n" + current_notes
    else:
        prompt = get_dynamic_prompt(state, final_transcript); response = notes_model.generate_content(prompt); final_notes_content = response.text; total_tokens += safe_get_token_count(response)
    if state.selected_note_style == "Option 3: Less Verbose + Summary" and state.selected_meeting_type == "Expert Meeting":
        status_ui.update(label="Step 4: Generating Executive Summary..."); summary_prompt = f"Create a concise executive summary from these notes:\n\n{final_notes_content}"; response = notes_model.generate_content(summary_prompt); final_notes_content += f"\n\n---\n\n**EXECUTIVE SUMMARY:**\n\n{response.text}"; total_tokens += safe_get_token_count(response)
    
    status_ui.update(label="Step 5: Saving to Database..."); note_id = str(uuid.uuid4())
    note_data = {'id': note_id, 'created_at': datetime.now().isoformat(), 'meeting_type': state.selected_meeting_type, 'file_name': file_name, 'content': final_notes_content, 'raw_transcript': raw_transcript, 'refined_transcript': refined_transcript, 'token_usage': total_tokens, 'processing_time': time.time() - start_time, 'pdf_blob': pdf_bytes}; database.save_note(note_data)
    
    extract_entities(note_id, final_notes_content, status_ui)
    return note_data

def on_sector_change():
    state = st.session_state.app_state; all_sectors = db_get_sectors(); state.earnings_call_topics = all_sectors.get(state.selected_sector, "")

def render_input_and_processing_tab(state: AppState):
    state.input_method = pills("Input Method", ["Upload File", "Paste Text"], index=["Upload File", "Paste Text"].index(state.input_method))
    if state.input_method == "Upload File":
        st.subheader("Step 1: Upload Documents"); st.info("Upload multiple files to create a batch processing queue.")
        uploaded_files = st.file_uploader("Upload Files", type=['pdf', 'txt', 'mp3', 'm4a', 'wav', 'ogg', 'flac'], accept_multiple_files=True, key="file_uploader")
        if st.button("➕ Add to Processing Queue"):
            for f in uploaded_files:
                if not any(item['name'] == f.name for item in state.processing_queue if item['type'] == 'file'): state.processing_queue.append({'type': 'file', 'data': f, 'name': f.name})
            st.toast(f"Added {len(uploaded_files)} file(s) to the queue.", icon="✅")
        st.subheader("Step 2: Review Queue")
        if not state.processing_queue: st.caption("Your queue is empty.")
        else: 
            st.dataframe(pd.DataFrame({"File Name": [f['name'] for f in state.processing_queue if f['type'] == 'file']}), use_container_width=True, hide_index=True)
            if st.button("Clear Queue"): state.processing_queue = []; st.rerun()
    else: st.subheader("Input Text"); state.text_input = st.text_area("Paste source transcript here:", height=300, key="text_input_main")
    
    st.subheader("Configuration")
    with st.expander("⚙️ Processing Configuration", expanded=True):
        state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type))
        if state.selected_meeting_type == "Expert Meeting": state.selected_note_style = st.selectbox("Note Style", EXPERT_MEETING_OPTIONS, index=EXPERT_MEETING_OPTIONS.index(state.selected_note_style))
        elif state.selected_meeting_type == "Earnings Call":
            state.earnings_call_mode = st.radio("Mode", EARNINGS_CALL_MODES, horizontal=True, index=EARNINGS_CALL_MODES.index(state.earnings_call_mode)); all_sectors = db_get_sectors(); sector_options = ["Other / Manual Topics"] + sorted(list(all_sectors.keys()));
            try: current_sector_index = sector_options.index(state.selected_sector)
            except ValueError: current_sector_index = 0
            state.selected_sector = st.selectbox("Sector (for Topic Templates)", sector_options, index=current_sector_index, on_change=on_sector_change, key="sector_selector")
            state.earnings_call_topics = st.text_area("Topic Instructions", value=state.earnings_call_topics, height=150)
            with st.expander("✏️ Manage Sector Templates"):
                st.write("Add, edit, or delete the sector templates used in the dropdown above.")
                st.markdown("**Edit or Delete an Existing Sector**"); sector_to_edit = st.selectbox("Select Sector to Edit", sorted(list(all_sectors.keys())))
                if sector_to_edit:
                    topics_for_edit = st.text_area("Sector Topics", value=all_sectors[sector_to_edit], key=f"topics_{sector_to_edit}")
                    col1, col2 = st.columns([1,1])
                    if col1.button("💾 Save Changes", key=f"save_{sector_to_edit}"): database.save_sector(sector_to_edit, topics_for_edit); db_get_sectors.clear(); st.toast(f"✅ Sector '{sector_to_edit}' updated!"); st.rerun()
                    if col2.button("❌ Delete Sector", type="primary", key=f"delete_{sector_to_edit}"): database.delete_sector(sector_to_edit); db_get_sectors.clear(); state.selected_sector = "Other / Manual Topics"; on_sector_change(); st.toast(f"🗑️ Sector '{sector_to_edit}' deleted!"); st.rerun()
                st.divider(); st.markdown("**Add a New Sector**"); new_sector_name = st.text_input("New Sector Name"); new_sector_topics = st.text_area("Topics for New Sector", key="new_sector_topics")
                if st.button("➕ Add New Sector"):
                    if new_sector_name and new_sector_topics: database.save_sector(new_sector_name, new_sector_topics); db_get_sectors.clear(); st.toast(f"✅ Sector '{new_sector_name}' added!"); st.rerun()
                    else: st.warning("Please provide both a name and topics.")
            if state.earnings_call_mode == "Enrich Existing Notes": state.existing_notes_input = st.text_area("Paste Existing Notes to Enrich:", value=state.existing_notes_input, height=150)
        
        with st.expander("⚙️ Advanced Settings & Models"):
            state.refinement_enabled = st.toggle("Enable Transcript Refinement", value=state.refinement_enabled); state.add_context_enabled = st.toggle("Add General Context", value=state.add_context_enabled)
            if state.add_context_enabled: state.context_input = st.text_area("Context Details:", value=state.context_input)
            c1, c2 = st.columns(2); state.speaker_1 = c1.text_input("Speaker 1", value=state.speaker_1); state.speaker_2 = c2.text_input("Speaker 2", value=state.speaker_2)
            m1, m2, m3 = st.columns(3); state.notes_model = m1.selectbox("Notes Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.notes_model)); state.refinement_model = m2.selectbox("Refinement Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.refinement_model)); state.transcription_model = m3.selectbox("Transcription Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.transcription_model))

    st.subheader("Prompt Preview"); prompt_preview = get_dynamic_prompt(state, "[...transcript content...]"); st_ace(value=prompt_preview, language='markdown', theme='github', height=200, readonly=True, key="prompt_preview_ace"); st.divider(); st.subheader("🚀 Generate")
    
    if state.input_method == "Upload File":
        if st.button("Process Entire Queue", type="primary", use_container_width=True, disabled=not state.processing_queue): state.processing = True; state.error_message = None; st.rerun()
    else:
        if st.button("Generate Notes", type="primary", use_container_width=True, disabled=not state.text_input.strip()): state.processing = True; state.error_message = None; st.rerun()
    
    if state.processing:
        if state.input_method == "Paste Text": items_to_process = [{'type': 'text', 'name': f"Pasted Text @ {datetime.now().strftime('%H:%M:%S')}"}]
        else: items_to_process = state.processing_queue[:]
        
        total_items = len(items_to_process); progress_bar = st.progress(0, text=f"Starting processing for {total_items} item(s)...")
        for i, item in enumerate(items_to_process):
            progress_bar.progress((i) / total_items, text=f"Processing {i+1}/{total_items}: '{item['name']}'");
            with st.status(f"Processing: {item['name']}", expanded=True) as status:
                try:
                    if item['type'] == 'file': state.input_method = "Upload File"; state.uploaded_file = item['data']
                    else: state.input_method = "Paste Text"; state.uploaded_file = None
                    final_note = process_and_save_task(state, status)
                    state.active_note_id = final_note['id']
                    status.update(label="✅ Success!", state="complete")
                except Exception as e:
                    state.error_message = f"Error on '{item['name']}':\n{e}"; status.update(label=f"❌ Error", state="error"); st.error(state.error_message); break
        
        if not state.error_message: progress_bar.progress(1.0, text="✅ Processing complete!"); st.toast(f"Successfully processed {total_items} item(s).", icon="🎉", duration=5000)
        state.processing_queue = []; state.processing = False; time.sleep(1); st.rerun()

def render_cockpit_and_history_tab(state: AppState):
    notes = database.get_all_notes();
    if not notes: st.info("No notes generated. Go to '📝 Input & Process' tab."); return
    if not state.active_note_id or not any(n['id'] == state.active_note_id for n in notes): state.active_note_id = notes[0]['id']
    active_note = database.get_note_by_id(state.active_note_id); state.active_note_pdf_bytes = active_note.get('pdf_blob')
    st.subheader(" Gʀᴏᴜɴᴅᴇᴅ Aɴᴀʟʏsɪs Cᴏᴄᴋᴘɪᴛ"); st.markdown(f"**Viewing:** `{active_note['file_name']}`")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📄 Source Document")
        if state.active_note_pdf_bytes:
            st.info("Source document was a PDF. A viewer will be here in a future Streamlit version.")
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg", caption=f"Source: {active_note['file_name']}", width=150)
        else: st.text_area("Refined Transcript", value=active_note.get('refined_transcript', 'N/A'), height=600, disabled=True)
    with col2: st.markdown("##### 🧠 AI-Generated Notes"); edited_content = st_ace(value=active_note['content'], language='markdown', theme='tomorrow_night_eighties', height=700, key=f"output_ace_{active_note['id']}", readonly=False)
    st.divider()
    dl1, dl2, dl3 = st.columns(3); dl1.download_button("⬇️ Download Processed Output", edited_content, f"SynthNote_Output_{active_note['id']}.txt", use_container_width=True)
    if active_note.get('refined_transcript'): dl2.download_button("⬇️ Download Refined Transcript", active_note['refined_transcript'], f"Refined_{active_note['id']}.txt", use_container_width=True)
    if active_note.get('raw_transcript'): dl3.download_button("⬇️ Download Raw Transcript", active_note['raw_transcript'], f"Raw_{active_note['id']}.txt", use_container_width=True)
    with st.expander("View Source Transcripts"):
        if active_note.get('refined_transcript'): st.text_area("Refined Transcript", value=active_note['refined_transcript'], height=200, disabled=True, key=f"refined_tx_{active_note['id']}")
        if active_note.get('raw_transcript'): st.text_area("Raw Source", value=active_note['raw_transcript'], height=200, disabled=True, key=f"raw_tx_{active_note['id']}")
    st.subheader("💬 Chat with this Document")
    for msg in state.chat_history: st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask a question..."):
        state.chat_history.append({"role": "user", "content": prompt}); st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_prompt = f"Answer user's question based ONLY on the document. Refer to page numbers `(p. X)`.\n\nDOCUMENT:\n{active_note['content']}\n\nQUESTION: {prompt}"; response = genai.GenerativeModel(AVAILABLE_MODELS[state.notes_model]).generate_content(rag_prompt); full_response = response.text
                st.write(full_response); state.chat_history.append({"role": "assistant", "content": full_response})
    st.subheader("📂 History")
    for note in notes:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1]); c1.markdown(f"**File:** `{note['file_name']}` | {datetime.fromisoformat(note['created_at']).strftime('%Y-%m-%d %H:%M')}")
            if c2.button("Set as Active", key=f"view_{note['id']}", use_container_width=True): state.active_note_id = note['id']; state.chat_history = []; st.rerun()

def render_knowledge_explorer_tab(state: AppState):
    st.subheader("🔬 Human-in-the-Loop Knowledge Base"); st.info("View, edit, and connect insights across all documents.")
    active_note_id = state.active_note_id;
    if not active_note_id: st.warning("Select an active note from the 'Cockpit & History' tab."); return
    entities = database.get_entities_for_note(active_note_id)
    st.markdown(f"##### Entities for Active Note (`{database.get_note_by_id(active_note_id)['file_name']}`)")
    df = pd.DataFrame(entities); edited_df = st.data_editor(df, column_config={"id": None, "note_id": None}, use_container_width=True, num_rows="dynamic", key="entity_editor", hide_index=True)
    if st.button("💾 Save Entity Changes"):
        try: database.update_entities_for_note(active_note_id, edited_df.to_dict('records')); st.toast("Changes saved!", icon="💾")
        except Exception as e: st.error(f"Failed to save: {e}")
    st.subheader("🌐 Cross-Document Search"); st.write("Click a cell in the table above to search for that entity across all notes.")
    if 'entity_editor' in st.session_state:
        selection = st.session_state.entity_editor.get("selection", {"rows": [], "columns": []})
        if selection and selection["rows"] and not df.empty:
            try:
                search_term = df.iloc[selection["rows"][0]][df.columns[selection["columns"][0]]]
                if "last_search" not in st.session_state or st.session_state.last_search != search_term:
                    st.session_state.last_search = search_term
                    with st.spinner(f"Searching for '{search_term}'..."): state.knowledge_explorer_search_results = database.search_notes_by_entity(search_term, exclude_note_id=active_note_id)
                    st.rerun()
            except IndexError: pass
    if "last_search" in st.session_state and st.session_state.last_search:
        results = state.knowledge_explorer_search_results; st.markdown(f"Found **{len(results)}** other note(s) mentioning **'{st.session_state.last_search}'**:")
        for note in results:
            with st.expander(f"`{note['file_name']}` ({datetime.fromisoformat(note['created_at']).strftime('%Y-%m-%d')})"): st.markdown(note['content'])
    st.subheader("📊 Analytics"); summary, daily_counts = database.get_analytics_summary(); c1, c2, c3 = st.columns(3); c1.metric("Total Notes", summary['total_notes']); c2.metric("Avg. Time / Note", f"{summary['avg_time']:.1f}s"); c3.metric("Total Tokens", f"{summary['total_tokens']:,}")
    st.markdown("##### Notes Processed (Last 14 Days)")
    if daily_counts: st.bar_chart(pd.DataFrame(daily_counts.items(), columns=['Date', 'Count']), x='Date', y='Count', use_container_width=True)

# --- 6. MAIN APPLICATION RUNNER ---
def run_app():
    st.set_page_config(page_title="SynthNotes AI", layout="wide"); st.title("SynthNotes AI 🧠")
    if "config_error" in st.session_state: st.error(st.session_state.config_error); st.stop()
    try:
        database.init_db()
        if "app_state" not in st.session_state: st.session_state.app_state = AppState(); on_sector_change()
        tabs = st.tabs(["📝 Input & Process", "📄 Cockpit & History", "🔎 Knowledge Explorer"]);
        with tabs[0]: render_input_and_processing_tab(st.session_state.app_state)
        with tabs[1]: render_cockpit_and_history_tab(st.session_state.app_state)
        with tabs[2]: render_knowledge_explorer_tab(st.session_state.app_state)
    except Exception as e: st.error("A critical application error occurred."); st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()
# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/

# /--------------------------\
# |   START OF app.py FILE   |
# \--------------------------/

# --- 1. IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
import io
import json
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
import re
import tempfile
import html as html_module
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import copy

# --- Local Imports ---
import database
from chunking import create_chunks_with_context, estimate_chunk_count, cleanup_stitched_notes
from progress import (
    ProgressTracker,
    build_processing_plan,
    build_speaker_id_plan,
    build_notes_only_plan,
    parallel_batches,
)
# --- Prompt templates live in prompts.py ---
from prompts import (
    EXPERT_MEETING_DETAILED_PROMPT,
    EXPERT_MEETING_CONCISE_PROMPT,
    EARNINGS_CALL_PROMPT,
    MANAGEMENT_MEETING_PROMPT,
    INTERNAL_DISCUSSION_PROMPT,
    PROMPT_INITIAL,
    PROMPT_CONTINUATION,
    VALIDATION_DETAILED_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    REFINEMENT_INSTRUCTIONS,
    SPEAKER_ID_PROMPT_INITIAL,
    SPEAKER_ID_PROMPT_CONTINUATION,
    SPEAKER_NAME_MAP_PROMPT,
    OTG_EXTRACT_PROMPT,
    OTG_CONVERT_PROMPT,
    OTG_REFINE_CHUNK_PROMPT,
    IA_MANAGEMENT_KTA_PROMPT,
    IA_EXPERT_KTA_PROMPT,
    IA_REFINE_CHUNK_PROMPT,
    IA_TONE_INSTRUCTIONS,
    EC_TOPIC_DISCOVERY_PROMPT,
    EC_MULTI_FILE_NOTES_PROMPT,
    EC_MULTI_FILE_STITCH_HEADER,
    RC_DIMENSION_DISCOVERY_PROMPT,
    RC_PER_REPORT_EXTRACTION_PROMPT,
    RC_COMPARISON_PROMPT,
    RC_STITCH_HEADER,
)


# --- App-wide CSS ---
APP_CSS = """
<style>
/* ── Active navigation tab highlight ── */
[data-testid="stNavigation"] button[aria-selected="true"] {
    border-bottom: 3px solid var(--primary-color) !important;
    font-weight: 600 !important;
    color: var(--primary-color) !important;
}
[data-testid="stNavigation"] button {
    transition: border-bottom 0.15s ease, color 0.15s ease;
}

/* ── Reduce top padding for a tighter header ── */
.main .block-container {
    padding-top: 1.5rem !important;
}

/* ── Ultra-wide: cap content width for readability ── */
@media (min-width: 1800px) {
    .main .block-container {
        max-width: 1600px !important;
        margin: 0 auto !important;
    }
}

/* ── Section dividers: lighter, more breathing room ── */
hr {
    margin-top: 1.2rem !important;
    margin-bottom: 1.2rem !important;
    opacity: 0.3;
}

/* ── Note cards in history list: subtle hover lift ── */
[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
    box-shadow: 0 2px 8px rgba(128,128,128,0.15);
    transform: translateY(-1px);
}

/* ── Tighten metric blocks ── */
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}

/* ── Focus visibility for keyboard navigation (WCAG 2.1 AA) ── */
button:focus-visible,
[data-testid="stSelectbox"] select:focus-visible,
textarea:focus-visible,
input:focus-visible,
[role="tab"]:focus-visible {
    outline: 2px solid var(--primary-color) !important;
    outline-offset: 2px !important;
}

/* ── Text overflow: prevent long filenames from breaking layout ── */
[data-testid="stMarkdownContainer"] p {
    overflow-wrap: break-word;
    word-break: break-word;
}

/* ── Responsive: tablets (stack 4-col action bars into 2x2) ── */
@media (max-width: 1024px) and (min-width: 769px) {
    .main .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
}

/* ── Responsive: mobile ── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 0.5rem;
    }
    textarea {
        min-width: 100% !important;
    }
    /* 44px minimum touch target (WCAG 2.5.5) */
    button {
        min-height: 44px !important;
        padding: 0.5rem 1rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Stack the note header on mobile */
    h3 {
        font-size: 1.1rem !important;
    }
}

/* ── Copy button iframe ── */
iframe {
    min-height: 45px !important;
}

/* ── Print: hide navigation and interactive elements ── */
@media print {
    [data-testid="stNavigation"],
    [data-testid="stSidebar"],
    button,
    iframe,
    .stProgress {
        display: none !important;
    }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
}

/* ── Smooth transitions globally ── */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
</style>
"""

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
# Chunks are non-overlapping and aligned to speaker-turn boundaries (see
# chunking.py). The binding constraint on chunk size is the model's output
# budget, not its input window: detailed notes run well under transcript
# length, so 10k-word chunks fit comfortably inside MAX_OUTPUT_TOKENS while
# producing ~3x fewer seams than the old 4k-word overlapping chunks.
CHUNK_WORD_SIZE = 10000  # default; user-adjustable per session in Settings & Models
CHUNK_SIZE_OPTIONS = [4000, 6000, 8000, 10000, 15000, 20000]
# Tail of the previous chunk passed to the model as read-only continuity
# context. It is never processed into notes, so it cannot create duplicates.
CONTEXT_TAIL_WORDS = 800
# Rough ratio of generated-notes words to transcript words, used only to
# estimate within-chunk progress while streaming.
NOTES_OUTPUT_RATIO = 0.6
# High output token ceiling for notes generation.
# Without this, Gemini defaults to ~8192 output tokens and silently
# truncates long, detailed notes — especially on later chunks. Also applied
# to refinement and speaker tagging, whose outputs are roughly transcript-
# length and would exceed an 8192-token default at the 10k-word chunk size.
MAX_OUTPUT_TOKENS = 65536
GENERATION_CONFIG = {"max_output_tokens": MAX_OUTPUT_TOKENS}
# Transcript context for "Chat with this Note". Gemini 2.5/3 models take
# ~1M input tokens, so a generous cap keeps verbatim lookups working on
# long calls (the old 30k-char cap silently dropped most of them).
CHAT_TRANSCRIPT_CHAR_LIMIT = 400_000

AVAILABLE_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 3.0 Flash": "gemini-3-flash-preview",
    "Gemini 3.0 Pro": "gemini-3-pro-preview",
    "Gemini 3 Pro Preview": "gemini-3-pro-preview",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Management Meeting", "Internal Discussion", "Custom"]
MAX_TOPIC_DISCOVERY_FILES = 4  # Number of PDFs to scan for topic discovery
SPEAKER_ID_FLOW_OPTION = "Option 4: Speaker ID Flow"
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary", SPEAKER_ID_FLOW_OPTION]
SPEAKER_ID_DOWNSTREAM_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]

TONE_OPTIONS = ["As Is", "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
NUMBER_FOCUS_OPTIONS = ["No Numbers", "Light", "Moderate", "Data-Heavy"]
OTG_WORD_COUNT_OPTIONS = {
    "Short (~150 words)": "Approximately 150 words. Keep it very concise — only the most essential points.",
    "Medium (~300 words)": "Approximately 300 words. Short and direct but cover all key findings.",
    "Long (~500 words)": "Approximately 500 words. Cover all findings with enough detail for context.",
    "Detailed (~750 words)": "Approximately 750 words. Provide thorough coverage with supporting detail and nuance.",
}
NUMBER_FOCUS_INSTRUCTIONS = {
    "No Numbers": "Do NOT include any specific numbers, percentages, monetary values, or metrics. Describe trends and findings qualitatively using words like 'significant,' 'substantial,' 'modest,' etc.",
    "Light": "Include only the most critical 2-3 numbers that are essential to the narrative. Describe most findings qualitatively.",
    "Moderate": "Include key numbers, percentages, and metrics where they support your points. Balance data with narrative flow.",
    "Data-Heavy": "Include ALL specific numbers, percentages, monetary values, metrics, and data points from the notes. The output should be dense with quantitative evidence supporting every claim.",
}

MEETING_TYPE_HELP = {
    "Expert Meeting": "Q&A format with detailed factual extraction from expert consultations. Option 4 enables a speaker-tagging review step before notes are generated.",
    "Earnings Call": "Financial data, management commentary, guidance, and analyst Q&A",
    "Management Meeting": "Decisions, action items, owners, and key discussion points",
    "Internal Discussion": "Perspectives, ideas, reasoning, conclusions, and next steps",
    "Custom": "Provide your own formatting instructions via the context field",
}

# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Paste Text"
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    earnings_call_mode: str = "Generate New Notes"
    selected_sector: str = "IT Services"
    notes_model: str = "Gemini 2.5 Pro"
    refinement_model: str =  "Gemini 2.5 Flash"
    speaker_id_model: str = "Gemini 3 Pro Preview"
    transcription_model: str =  "Gemini 3.0 Flash"
    chat_model: str = "Gemini 2.5 Pro"
    refinement_enabled: bool = True
    chunk_word_size: int = CHUNK_WORD_SIZE
    add_context_enabled: bool = False
    context_input: str = ""
    speakers: str = ""
    earnings_call_topics: str = ""
    existing_notes_input: str = ""
    text_input: str = ""
    uploaded_file: Optional[Any] = None
    audio_recording: Optional[Any] = None
    processing: bool = False
    active_note_id: Optional[str] = None
    error_message: Optional[str] = None
    fallback_content: Optional[str] = None

# --- 4. CORE PROCESSING & UTILITY FUNCTIONS ---
def sanitize_input(text: str) -> str:
    """Removes keywords commonly used in prompt injection attacks."""
    if not isinstance(text, str):
        return ""
    injection_patterns = [
        r'ignore all previous instructions',
        r'you are now in.*mode',
        r'stop being an ai',
        r'disregard.*(?:above|prior|previous)',
        r'new instructions:',
        r'system:\s*override',
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
        pass
    return 0

def generate_with_retry(model, prompt_or_contents, max_retries=3, stream=False, generation_config=None):
    """Wrapper around generate_content with exponential backoff for transient API failures."""
    kwargs = {"stream": stream}
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt_or_contents, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(kw in error_str for kw in [
                '429', '503', '500', 'deadline', 'timeout', 'unavailable', 'resource_exhausted'
            ])
            if is_transient and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

def stream_and_collect(response, placeholder=None, on_progress=None, live_preview=None):
    """Consume a streaming response, optionally displaying progress.

    `on_progress(word_count)` is called as text streams in, so callers can
    drive a real progress bar instead of a detached word counter.
    `live_preview` is an st.empty() (ideally inside an autoscroll container)
    that gets the accumulated markdown as it streams.
    Returns (full_text, token_count).
    """
    full_text = ""
    update_counter = 0
    for chunk in response:
        if chunk.parts:
            full_text += chunk.text
            update_counter += 1
            # Throttle UI updates to every 5 chunks to reduce flickering
            if update_counter % 5 == 0:
                word_count = len(full_text.split())
                if on_progress:
                    on_progress(word_count)
                if placeholder:
                    placeholder.caption(f"Streaming... {word_count:,} words generated")
                if live_preview is not None:
                    live_preview.markdown(full_text)
    if placeholder:
        placeholder.empty()
    if live_preview is not None:
        live_preview.markdown(full_text)
    token_count = safe_get_token_count(response)
    return full_text, token_count

def copy_to_clipboard_button(text: str, button_label: str = "Copy Notes"):
    """Render a button that copies text to the clipboard using the browser Clipboard API."""
    # Adapt button colors to the current theme (light vs dark mode)
    theme = st.context.theme
    bg_color = theme.get("primaryColor", "#FF4B4B")
    text_color = theme.get("backgroundColor", "#FFFFFF")

    # Use JSON encoding to safely embed arbitrary text in a JS string literal
    json_encoded = json.dumps(text)
    safe_label = html_module.escape(button_label)
    st.iframe(
        f"""
        <button onclick="copyText()" aria-label="{safe_label}" role="button" tabindex="0"
            onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();copyText();}}"
            style="
                background-color:{bg_color}; color:{text_color}; border:none; padding:0.4rem 1rem;
                border-radius:0.3rem; cursor:pointer; font-size:0.875rem; width:100%;
                transition: opacity 0.15s ease, box-shadow 0.15s ease;
            "
            onmouseover="this.style.opacity='0.85'"
            onmouseout="this.style.opacity='1'"
            onfocus="this.style.boxShadow='0 0 0 2px {bg_color}40'"
            onblur="this.style.boxShadow='none'"
        >{safe_label}</button>
        <script>
        function copyText() {{
            const text = {json_encoded};
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.querySelector('button');
                btn.textContent = 'Copied!';
                btn.setAttribute('aria-label', 'Copied to clipboard');
                setTimeout(() => {{
                    btn.textContent = {json.dumps(button_label)};
                    btn.setAttribute('aria-label', {json.dumps(button_label)});
                }}, 2000);
            }}).catch(() => {{
                const btn = document.querySelector('button');
                btn.textContent = 'Failed';
                setTimeout(() => btn.textContent = {json.dumps(button_label)}, 2000);
            }});
        }}
        </script>
        """,
        height=45,
    )

@st.cache_data(ttl=3600)
def get_file_content(uploaded_file, audio_recording=None) -> Tuple[Optional[str], str, Optional[bytes]]:
    """
    Returns: (text_content_or_indicator, file_name, pdf_bytes_if_pdf)
    """
    # Priority 1: Audio Recording
    if audio_recording:
        return "audio_file", "Microphone Recording.wav", None

    # Priority 2: Uploaded File
    if uploaded_file:
        name = uploaded_file.name
        file_bytes_io = io.BytesIO(uploaded_file.getvalue())
        ext = os.path.splitext(name)[1].lower()

        try:
            if ext == ".pdf":
                reader = PyPDF2.PdfReader(file_bytes_io)
                if reader.is_encrypted: return "Error: PDF is encrypted.", name, None
                content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                return (content, name, uploaded_file.getvalue()) if content else ("Error: No text found in PDF.", name, None)

            elif ext in [".txt", ".md"]:
                return file_bytes_io.read().decode("utf-8"), name, None

            elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
                return "audio_file", name, None

        except Exception as e:
            return f"Error: Could not process file {name}. Details: {str(e)}", name, None

    return None, "Unknown", None

@st.cache_data
def db_get_sectors() -> dict:
    return database.get_sectors()

SKIP_TAG = "Skip"


def _parse_tagged_transcript(text: str) -> List[Dict[str, str]]:
    """Parse a transcript with `**Speaker N:**` or `**Skip:**` prefixes into segments.

    Returns a list of {"speaker": "Speaker N" | "Skip", "text": "..."} dicts.
    Tolerant of the marker appearing inline or on its own line.
    """
    if not text:
        return []
    pattern = re.compile(r"\*\*\s*(Speaker\s*\d+|Skip)\s*:\s*\*\*\s*", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    segments: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        raw = m.group(1).strip()
        if raw.lower().startswith("skip"):
            speaker = SKIP_TAG
        else:
            digit = re.search(r"\d+", raw).group(0)
            speaker = f"Speaker {digit}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg_text = text[start:end].strip()
        if seg_text:
            segments.append({"speaker": speaker, "text": seg_text})
    return segments


def _serialize_tagged_segments(
    segments: List[Dict[str, str]],
    display_names: Optional[Dict[str, str]] = None,
    exclude_tags: Optional[List[str]] = None,
) -> str:
    """Serialize segments back to tagged transcript text.

    If `display_names` is provided, replaces the generic "Speaker N" label with
    the user-supplied display name (e.g. "Jane Doe (CEO)") in the output.
    If `exclude_tags` is provided, segments whose speaker is in that list are
    dropped — used to filter out `Skip` (logistics) segments before notes gen.
    """
    if not segments:
        return ""
    excluded = set(exclude_tags or [])
    lines: List[str] = []
    for seg in segments:
        if seg["speaker"] in excluded:
            continue
        tag = seg["speaker"]
        if display_names and display_names.get(tag):
            tag = display_names[tag]
        lines.append(f"**{tag}:**\n{seg['text']}")
    return "\n\n".join(lines)


def _merge_consecutive_segments(segments: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive segments with the same speaker into one block.

    The model is told to merge these itself, but chunk boundaries and long
    answers still produce split turns. Merging deterministically here means
    fewer segments for the human to review and reassign."""
    merged: List[Dict[str, str]] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += "\n\n" + seg["text"]
        else:
            merged.append(dict(seg))
    return merged


def _infer_speaker_names(sid_model, tagged_transcript: str, participants: str) -> Tuple[Dict[str, str], int]:
    """Map generic Speaker N labels to the user-provided participant names so
    the rename fields come pre-filled. Best-effort: returns ({}, tokens) on
    any failure — the user can always rename manually."""
    try:
        prompt = SPEAKER_NAME_MAP_PROMPT.format(
            participants=participants,
            transcript_sample=tagged_transcript[:12000],
        )
        response = generate_with_retry(sid_model, prompt)
        raw_json = response.text.strip()
        if raw_json.startswith("```"):
            lines = raw_json.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_json = "\n".join(lines).strip()
        json_match = re.search(r'\{[\s\S]*\}', raw_json)
        if json_match:
            raw_json = json_match.group(0)
        parsed = json.loads(raw_json)
        names = {
            k: v.strip()
            for k, v in parsed.items()
            if isinstance(k, str) and k.startswith("Speaker") and isinstance(v, str) and v.strip()
        }
        return names, safe_get_token_count(response)
    except Exception:
        return {}, 0


def _detect_speakers_in_segments(segments: List[Dict[str, str]]) -> List[str]:
    """Return ordered, deduplicated list of speaker labels found in segments."""
    seen: List[str] = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])
    return seen


def generate_notes_from_transcript(
    state: AppState,
    final_transcript: str,
    notes_model,
    progress,
    *,
    skip_chunking: bool = False,
) -> Tuple[str, int]:
    """Generate notes for a (possibly long) transcript. Returns (notes, tokens).

    Long transcripts are split into NON-overlapping, speaker-turn-aligned
    chunks (see chunking.py). Each continuation prompt embeds the tail of the
    previous chunk's *transcript* as read-only context, so chunk outputs never
    overlap and are simply concatenated — no stitching or de-duplication step.
    Because every prompt is known upfront, chunks are generated in parallel.
    """
    words = final_transcript.split()
    chunk_size = getattr(state, "chunk_word_size", CHUNK_WORD_SIZE) or CHUNK_WORD_SIZE

    # --- Single-shot path (short transcript, or chunking disabled) ---
    if skip_chunking or len(words) <= chunk_size:
        progress.update("generate", 0, f"{len(words):,} words, single pass")
        prompt = get_dynamic_prompt(state, final_transcript)
        # Live preview of the notes as they stream — the most honest progress
        # indicator there is. Autoscroll keeps the latest text in view.
        with st.container(height=240, border=True, autoscroll=True):
            live_preview = st.empty()
        response = generate_with_retry(notes_model, prompt, stream=True, generation_config=GENERATION_CONFIG)
        expected_words = max(200, int(len(words) * NOTES_OUTPUT_RATIO))

        def _on_stream(words_so_far):
            progress.update(
                "generate",
                min(words_so_far / expected_words, 0.95),
                f"{words_so_far:,} words generated",
            )

        notes, tokens = stream_and_collect(response, on_progress=_on_stream, live_preview=live_preview)
        return notes, tokens

    # --- Chunked path ---
    chunks = create_chunks_with_context(final_transcript, chunk_size, CONTEXT_TAIL_WORDS)
    n = len(chunks)
    progress.set_units("generate", parallel_batches(n) * 5.0)
    progress.update("generate", 0, f"{len(words):,} words, {n} sections in parallel")

    prompt_base = _get_base_prompt_for_type(state)
    prompts = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompts.append(PROMPT_INITIAL.format(base_instructions=prompt_base, chunk_text=chunk["text"]))
        else:
            prompts.append(PROMPT_CONTINUATION.format(
                base_instructions=prompt_base,
                context_tail=chunk["context"],
                chunk_text=chunk["text"],
            ))

    expected_words = [max(200, int(len(c["text"].split()) * NOTES_OUTPUT_RATIO)) for c in chunks]
    # Shared per-chunk word counters; single-item assignments are GIL-atomic.
    # Worker threads must not touch st.* — all UI updates happen on the main
    # thread in the polling loop below.
    words_generated = [0] * n
    results: List[Optional[str]] = [None] * n
    chunk_tokens = [0] * n

    def _generate_chunk(idx: int, prompt: str):
        response = generate_with_retry(notes_model, prompt, stream=True, generation_config=GENERATION_CONFIG)
        text = ""
        for part in response:
            if part.parts:
                text += part.text
                words_generated[idx] = len(text.split())
        return idx, text, safe_get_token_count(response)

    executor = ThreadPoolExecutor(max_workers=min(3, n))
    try:
        futures = {executor.submit(_generate_chunk, i, p): i for i, p in enumerate(prompts)}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                idx, text, tokens = future.result()  # re-raises worker errors
                results[idx] = text
                chunk_tokens[idx] = tokens
                words_generated[idx] = len(text.split())
            done_count = sum(1 for r in results if r is not None)
            frac = sum(
                1.0 if results[i] is not None else min(words_generated[i] / expected_words[i], 0.95)
                for i in range(n)
            ) / n
            progress.update(
                "generate", frac,
                f"{done_count}/{n} sections · {sum(words_generated):,} words generated",
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if not any(r and r.strip() for r in results):
        raise ValueError("Failed to generate notes from any section. Please try again or use a different model.")

    final_notes = "\n\n".join(r.strip() for r in results if r and r.strip())
    return final_notes, sum(chunk_tokens)


def _get_base_prompt_for_type(state):
    """Returns the base prompt instructions for the selected meeting type."""
    mt = state.selected_meeting_type
    if mt == "Expert Meeting":
        # Speaker ID Flow uses a downstream Option 1/2/3 picker inside the
        # speaker review panel. When the user clicks "Generate Notes" there we
        # temporarily override selected_note_style to that downstream pick.
        # If something invokes this with the raw Option 4 selected, fall back
        # to the concise prompt — same default as Option 2.
        if state.selected_note_style == "Option 1: Detailed & Strict":
            return EXPERT_MEETING_DETAILED_PROMPT
        else:
            return EXPERT_MEETING_CONCISE_PROMPT
    elif mt == "Earnings Call":
        topic_instructions = state.earnings_call_topics or "Identify logical themes and use them as bold headings."
        return EARNINGS_CALL_PROMPT.format(topic_instructions=topic_instructions)
    elif mt == "Management Meeting":
        return MANAGEMENT_MEETING_PROMPT
    elif mt == "Internal Discussion":
        return INTERNAL_DISCUSSION_PROMPT
    elif mt == "Custom":
        sanitized = sanitize_input(state.context_input)
        if sanitized:
            return f"Follow the user's instructions to generate meeting notes.\n\n**USER INSTRUCTIONS:**\n{sanitized}"
        return "Generate comprehensive meeting notes capturing all key points, decisions, data, and action items. Use **bold headings** to organize by topic and bullet points for details."
    return ""

def get_dynamic_prompt(state: AppState, transcript_chunk: str) -> str:
    base = _get_base_prompt_for_type(state)
    sanitized_context = sanitize_input(state.context_input)
    context_section = f"**ADDITIONAL CONTEXT:**\n{sanitized_context}" if state.add_context_enabled and sanitized_context else ""

    if state.selected_meeting_type == "Earnings Call" and state.earnings_call_mode == "Enrich Existing Notes":
        return f"Enrich the following existing notes based on the new transcript. Maintain the same structure and format.\n\n{base}\n\n**EXISTING NOTES:**\n{state.existing_notes_input}\n\n**NEW TRANSCRIPT:**\n{transcript_chunk}"

    return f"{base}\n\n{context_section}\n\n**MEETING TRANSCRIPT:**\n{transcript_chunk}"

def validate_inputs(state: AppState) -> Optional[str]:
    if state.input_method == "Paste Text" and not state.text_input.strip():
        return "Please paste a transcript."

    if state.input_method == "Upload / Record":
        if not state.uploaded_file and not state.audio_recording:
             return "Please upload a file or record audio."

        if state.uploaded_file and not state.audio_recording:
            size_mb = state.uploaded_file.size / (1024 * 1024)
            ext = os.path.splitext(state.uploaded_file.name)[1].lower()
            if ext == ".pdf" and size_mb > MAX_PDF_MB:
                return f"PDF is too large ({size_mb:.1f}MB). Limit: {MAX_PDF_MB}MB."
            elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac'] and size_mb > MAX_AUDIO_MB:
                return f"Audio is too large ({size_mb:.1f}MB). Limit: {MAX_AUDIO_MB}MB."

    if state.selected_meeting_type == "Earnings Call" and state.earnings_call_mode == "Enrich Existing Notes" and not state.existing_notes_input:
        return "Please provide existing notes for enrichment mode."
    return None

def _get_cached_model(model_display_name: str) -> genai.GenerativeModel:
    """Return a cached GenerativeModel instance, creating it only if the model name changed."""
    cache_key = "_model_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    # Defensive: handle invalid model names gracefully
    if model_display_name not in AVAILABLE_MODELS:
        model_display_name = list(AVAILABLE_MODELS.keys())[0]  # fallback to first model
    model_id = AVAILABLE_MODELS[model_display_name]
    if model_id not in st.session_state[cache_key]:
        st.session_state[cache_key][model_id] = genai.GenerativeModel(model_id)
    return st.session_state[cache_key][model_id]

def is_mobile_device() -> bool:
    """Check if likely a mobile device based on viewport. Returns False on server-side."""
    # Note: This is a heuristic. Streamlit doesn't expose device info directly.
    # We'll use a session state flag that can be set via JS, defaulting to False.
    return st.session_state.get("_is_mobile", False)

AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']

def _input_is_audio(state: AppState) -> bool:
    """Whether the configured input will require audio transcription.
    Used to decide if the progress plan includes a transcription step —
    must mirror the audio detection in get_file_content/_load_source_text."""
    if state.input_method != "Upload / Record":
        return False
    if state.audio_recording is not None:
        return True
    if state.uploaded_file is not None:
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        return ext in AUDIO_EXTENSIONS
    return False

def send_browser_notification(title: str, body: str):
    """Send a browser notification using the Notifications API."""
    safe_title = json.dumps(title)
    safe_body = json.dumps(body)

    st.iframe(
        f"""
        <script>
        (function() {{
            if (!("Notification" in window)) return;
            var opts = {{body: {safe_body}, icon: "https://placehold.co/64x64?text=SN", tag: "synthnotes-complete"}};
            if (Notification.permission === "granted") {{
                new Notification({safe_title}, opts);
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(permission) {{
                    if (permission === "granted") new Notification({safe_title}, opts);
                }});
            }}
        }})();
        </script>
        """,
        height=1,  # st.iframe requires a positive height; 1px keeps it invisible
    )

def _load_source_text(state: AppState, status_ui, progress: ProgressTracker) -> Tuple[str, str, Optional[bytes]]:
    """Load raw transcript from the configured input source (text/upload/recording).

    Performs audio transcription when the input is audio. Returns
    (raw_transcript, file_name, pdf_bytes). Mirrors the logic at the top of
    process_and_save_task so the speaker-ID flow can reuse it without
    duplicating subtle behaviour (whitespace normalisation, cloud cleanup, etc.).
    """
    transcription_model = _get_cached_model(state.transcription_model)
    progress.update("prepare", 0, "Loading input...")

    raw_transcript, file_name = "", "Pasted Text"
    pdf_bytes_data = None

    if state.input_method == "Upload / Record":
        file_type, name, pdf_bytes = get_file_content(state.uploaded_file, state.audio_recording)
        file_name = name
        pdf_bytes_data = pdf_bytes

        if file_type == "audio_file":
            progress.update("transcribe", 0, "Processing audio file...")
            audio_bytes = state.audio_recording.getvalue() if state.audio_recording else state.uploaded_file.getvalue()
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as audio_err:
                raise ValueError(f"Failed to process audio file. It may be corrupted or in an unsupported format. Details: {audio_err}")

            chunk_length_ms = 5 * 60 * 1000
            audio_chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            # Re-scale the transcription step now that the real workload is known.
            progress.set_units("transcribe", max(2.0, len(audio_chunks) * 2.0))

            all_transcripts, cloud_files, local_files = [], [], []
            try:
                for i, chunk in enumerate(audio_chunks):
                    try:
                        progress.update("transcribe", i / len(audio_chunks), f"Chunk {i+1}/{len(audio_chunks)}")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                            chunk.export(temp_f.name, format="wav")
                            local_files.append(temp_f.name)
                            cloud_ref = genai.upload_file(path=temp_f.name)
                            cloud_files.append(cloud_ref.name)
                            while cloud_ref.state.name == "PROCESSING":
                                time.sleep(2)
                                cloud_ref = genai.get_file(cloud_ref.name)
                            if cloud_ref.state.name != "ACTIVE":
                                raise Exception(f"Audio chunk {i+1} cloud processing failed.")
                            response = generate_with_retry(transcription_model, ["Transcribe this audio.", cloud_ref])
                            all_transcripts.append(response.text)
                    except Exception as e:
                        raise Exception(f"Transcription failed on chunk {i+1}/{len(audio_chunks)}. Reason: {e}")

                raw_transcript = "\n\n".join(all_transcripts).strip()
                progress.complete_step("transcribe")
            finally:
                for path in local_files:
                    try: os.remove(path)
                    except Exception: pass
                for cloud_name in cloud_files:
                    try: genai.delete_file(cloud_name)
                    except Exception as e: st.warning(f"Could not delete cloud file {cloud_name}: {e}")

        elif file_type is None or (isinstance(file_type, str) and file_type.startswith("Error:")):
            raise ValueError(file_type or "Failed to read file content.")
        else:
            raw_transcript = file_type
    elif state.input_method == "Paste Text":
        raw_transcript = state.text_input

    if not raw_transcript or not raw_transcript.strip():
        raise ValueError("Source content is empty or contains only whitespace.")

    raw_transcript = re.sub(r'\n{3,}', '\n\n', raw_transcript.strip())
    return raw_transcript, file_name, pdf_bytes_data


def run_speaker_identification_task(state: AppState, status_ui, progress: ProgressTracker) -> Dict[str, Any]:
    """Refine transcript AND tag speakers (2 or 3, auto-detected).

    Returns a dict with: raw_transcript, file_name, pdf_bytes, tagged_transcript,
    segments, speakers, token_usage. The result is stored in session_state so
    the user can edit speaker tags before generating notes.
    """
    start_time = time.time()
    # Speaker ID uses its own model (defaults to a stronger one than refinement)
    # because distinguishing voices across a long transcript benefits from the
    # bigger context model. Fall back to refinement_model if unset.
    sid_model_name = getattr(state, "speaker_id_model", None) or state.refinement_model
    sid_model = _get_cached_model(sid_model_name)

    raw_transcript, file_name, pdf_bytes_data = _load_source_text(state, status_ui, progress)

    # Save checkpoints in case the user reloads
    st.session_state["_checkpoint_raw_transcript"] = raw_transcript
    st.session_state["_checkpoint_file_name"] = file_name

    progress.complete_step("prepare")

    speakers = sanitize_input(state.speakers)
    speaker_info = f"Known participants (use as hint only, but still tag as Speaker 1/2/3): {speakers}." if speakers else ""
    refinement_extra = REFINEMENT_INSTRUCTIONS.get("Expert Meeting", "")

    progress.update("refine", 0, "Refining and tagging speakers...")
    words = raw_transcript.split()
    chunk_size = getattr(state, "chunk_word_size", CHUNK_WORD_SIZE) or CHUNK_WORD_SIZE
    total_tokens = 0
    tagged_chunks: List[str] = []

    if len(words) <= chunk_size:
        prompt = SPEAKER_ID_PROMPT_INITIAL.format(
            speaker_info=speaker_info,
            refinement_extra=refinement_extra,
            transcript=raw_transcript,
        )
        response = generate_with_retry(sid_model, prompt, generation_config=GENERATION_CONFIG)
        tagged_chunks.append(response.text)
        total_tokens += safe_get_token_count(response)
        progress.update("refine", 1.0, "Done")
    else:
        # Non-overlapping, turn-aligned chunks: each transcript region is
        # tagged exactly once, so the joined tagged output has no duplicated
        # turns. Speaker continuity comes from prev_tagged_tail below, which
        # carries the previous chunk's *tagged output* across the boundary.
        chunks = [c["text"] for c in create_chunks_with_context(raw_transcript, chunk_size, 0)]
        speakers_seen: List[str] = []
        prev_tagged_tail = ""  # last few tagged segments from previous chunk
        for i, chunk in enumerate(chunks):
            progress.update("refine", i / len(chunks), f"Section {i+1}/{len(chunks)}")
            if i == 0:
                prompt = SPEAKER_ID_PROMPT_INITIAL.format(
                    speaker_info=speaker_info,
                    refinement_extra=refinement_extra,
                    transcript=chunk,
                )
            else:
                # Pass the last 2 tagged segments from the previous chunk's
                # output (not raw words) so the model sees who Speaker 1 vs
                # Speaker 2 actually is and keeps the mapping stable.
                speakers_label = ", ".join(s for s in speakers_seen if s != SKIP_TAG) or "Speaker 1, Speaker 2"
                prompt = SPEAKER_ID_PROMPT_CONTINUATION.format(
                    speakers_so_far=speakers_label,
                    context=prev_tagged_tail or "(no prior tagged context)",
                    speaker_info=speaker_info,
                    refinement_extra=refinement_extra,
                    chunk=chunk,
                )
            response = generate_with_retry(sid_model, prompt, generation_config=GENERATION_CONFIG)
            chunk_tagged = response.text
            tagged_chunks.append(chunk_tagged)
            total_tokens += safe_get_token_count(response)

            # Track speakers observed so far and remember the tail for next chunk.
            # Four segments of tagged context (vs the old two) give the model a
            # clearer picture of the established voice-to-label mapping, which
            # is the main defence against labels flipping at chunk boundaries.
            chunk_segments = _parse_tagged_transcript(chunk_tagged)
            for seg in chunk_segments:
                if seg["speaker"] not in speakers_seen:
                    speakers_seen.append(seg["speaker"])
            prev_tagged_tail = _serialize_tagged_segments(chunk_segments[-4:]) if chunk_segments else ""

    tagged_transcript = "\n\n".join(c.strip() for c in tagged_chunks if c and c.strip())
    segments = _parse_tagged_transcript(tagged_transcript)
    if not segments:
        raise ValueError("Speaker identification did not return any tagged segments. Try re-running, or switch to Option 1/2/3.")

    # Merge consecutive same-speaker blocks (chunk boundaries and long answers
    # produce splits) so the user has fewer segments to review.
    segments = _merge_consecutive_segments(segments)

    # Re-serialize from parsed segments so the stored tagged transcript has a
    # canonical format (consistent spacing, blank lines, no model artefacts).
    canonical_tagged = _serialize_tagged_segments(segments)
    speakers_list = _detect_speakers_in_segments(segments)

    # Pre-fill speaker display names from the user-provided participants list
    # (best-effort) so renaming is usually already done.
    inferred_names: Dict[str, str] = {}
    if speakers:
        progress.update("refine", 1.0, "Matching speakers to participants...")
        inferred_names, name_tokens = _infer_speaker_names(sid_model, canonical_tagged, speakers)
        total_tokens += name_tokens

    progress.complete_step("refine")

    # Durable checkpoint: persist the tagged transcript as a draft note so the
    # refinement/tagging work survives a reload while the user reviews tags.
    # process_tagged_to_notes_task fills in the content on the same row later;
    # "Discard & Start Over" deletes it if no notes were generated.
    draft_note_id = str(uuid.uuid4())
    try:
        database.save_note({
            'id': draft_note_id, 'created_at': datetime.now().isoformat(),
            'meeting_type': "Expert Meeting", 'file_name': file_name,
            'content': "", 'raw_transcript': raw_transcript,
            'refined_transcript': canonical_tagged, 'token_usage': total_tokens,
            'processing_time': 0, 'pdf_blob': pdf_bytes_data,
        })
    except Exception:
        draft_note_id = None

    return {
        "raw_transcript": raw_transcript,
        "file_name": file_name,
        "pdf_bytes": pdf_bytes_data,
        "tagged_transcript": canonical_tagged,
        "segments": segments,
        "speakers": speakers_list,
        "token_usage": total_tokens,
        "elapsed": time.time() - start_time,
        "note_id": draft_note_id,
        "inferred_names": inferred_names,
    }


def process_tagged_to_notes_task(
    state: AppState,
    status_ui,
    progress: ProgressTracker,
    *,
    tagged_transcript: str,
    raw_transcript: str,
    file_name: str,
    pdf_bytes: Optional[bytes],
    downstream_style: str,
    prior_tokens: int = 0,
    draft_note_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate notes from a user-edited, speaker-tagged transcript.

    Mirrors the notes-generation half of process_and_save_task but skips the
    transcription and refinement steps (those already ran in the speaker-ID
    step). The tagged transcript IS the refined transcript for this flow.
    """
    start_time = time.time()
    notes_model = _get_cached_model(state.notes_model)

    # Temporarily override note style so _get_base_prompt_for_type returns the
    # downstream pick (Option 1/2/3). Restore on exit.
    original_style = state.selected_note_style
    state.selected_note_style = downstream_style
    total_tokens = prior_tokens
    final_notes_content = ""
    try:
        final_transcript = tagged_transcript
        final_notes_content, tokens = generate_notes_from_transcript(
            state, final_transcript, notes_model, progress
        )
        total_tokens += tokens

        if not final_notes_content or not final_notes_content.strip():
            raise ValueError("The model returned empty notes. Please try again or use a different model.")

        progress.complete_step("generate")
        was_chunked = len(final_transcript.split()) > (getattr(state, "chunk_word_size", CHUNK_WORD_SIZE) or CHUNK_WORD_SIZE)
        if was_chunked:
            # Deterministic cleanup — no LLM call, zero content-loss risk.
            final_notes_content = cleanup_stitched_notes(final_notes_content)

        # Executive summary if downstream style is Option 3
        if downstream_style == "Option 3: Less Verbose + Summary":
            progress.update("summary", 0.3, "Working...")
            summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(notes=final_notes_content)
            response = generate_with_retry(notes_model, summary_prompt)
            final_notes_content += f"\n\n---\n\n{response.text}"
            total_tokens += safe_get_token_count(response)
            progress.complete_step("summary")

        progress.update("save", 0.5, "Writing to database...")
        note_data = {
            'id': draft_note_id or str(uuid.uuid4()), 'created_at': datetime.now().isoformat(),
            'meeting_type': "Expert Meeting",
            'file_name': file_name, 'content': final_notes_content,
            'raw_transcript': raw_transcript,
            'refined_transcript': tagged_transcript,
            'token_usage': total_tokens,
            'processing_time': time.time() - start_time,
            'pdf_blob': pdf_bytes,
        }
        try:
            if draft_note_id:
                # Fill in the draft row saved by the speaker-ID step, with the
                # user-edited tagged transcript as the refined transcript.
                database.update_note(draft_note_id, {
                    'content': final_notes_content,
                    'refined_transcript': tagged_transcript,
                    'token_usage': total_tokens,
                    'processing_time': note_data['processing_time'],
                })
            else:
                database.save_note(note_data)
        except Exception as db_error:
            st.session_state.app_state.fallback_content = final_notes_content
            raise Exception(f"Processing succeeded, but failed to save the note to the database. You can download the unsaved note below. Error: {db_error}")
        return note_data
    finally:
        state.selected_note_style = original_style


def process_and_save_task(state: AppState, status_ui, progress: ProgressTracker):
    start_time = time.time()
    note_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    notes_model = _get_cached_model(state.notes_model)
    refinement_model = _get_cached_model(state.refinement_model)

    # Load input (file, recording, or pasted text) — includes audio transcription.
    raw_transcript, file_name, pdf_bytes_data = _load_source_text(state, status_ui, progress)

    # Checkpoint: save raw transcript so it survives connection drops
    st.session_state["_checkpoint_raw_transcript"] = raw_transcript
    st.session_state["_checkpoint_file_name"] = file_name

    # Durable checkpoint: persist a draft note NOW so the transcript (which may
    # have cost minutes of audio transcription) survives a crash, reload, or a
    # failure in any later stage. The same row is updated as stages complete.
    draft_saved = False
    try:
        database.save_note({
            'id': note_id, 'created_at': created_at,
            'meeting_type': state.selected_meeting_type, 'file_name': file_name,
            'content': "", 'raw_transcript': raw_transcript,
            'refined_transcript': None, 'token_usage': 0,
            'processing_time': 0, 'pdf_blob': pdf_bytes_data,
        })
        draft_saved = True
    except Exception:
        # Non-fatal: processing continues; the final save below will retry.
        pass

    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0

    speakers = sanitize_input(state.speakers)
    speaker_info = f"Participants: {speakers}." if speakers else ""
    refinement_extra = REFINEMENT_INSTRUCTIONS.get(state.selected_meeting_type, "")

    # --- Step 2: Refinement ---
    if state.refinement_enabled:
        progress.complete_step("prepare")
        progress.update("refine", 0, "Starting refinement...")
        words = raw_transcript.split()

        lang_instruction = "IMPORTANT: Your entire output MUST be in English. If the transcript contains Hindi, Hinglish, or any other non-English language, translate all content into clear, natural English while preserving the original meaning, nuance, and speaker intent."

        chunk_size = getattr(state, "chunk_word_size", CHUNK_WORD_SIZE) or CHUNK_WORD_SIZE
        if len(words) <= chunk_size:
            refine_prompt = f"Refine the following transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n{lang_instruction}\n\nTRANSCRIPT:\n{raw_transcript}"
            response = generate_with_retry(refinement_model, refine_prompt, generation_config=GENERATION_CONFIG)
            refined_transcript = response.text
            total_tokens += safe_get_token_count(response)
        else:
            # Non-overlapping, turn-aligned chunks: each region is refined
            # exactly once, so joining the refined chunks cannot duplicate
            # content. The previous chunk's tail is passed as read-only
            # context for continuity (known upfront, enables parallelism).
            chunks = create_chunks_with_context(raw_transcript, chunk_size, CONTEXT_TAIL_WORDS)

            prompts = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    prompts.append(f"You are refining a transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n{lang_instruction}\n\nTRANSCRIPT CHUNK TO REFINE:\n{chunk['text']}")
                else:
                    prompts.append(f"""You are continuing to refine a long transcript. Below is the tail end of the previous section for context. Your task is to refine the new chunk provided, ensuring a seamless and natural transition. Do NOT include the context itself in your output — refine and output ONLY the new chunk.
{speaker_info} {refinement_extra}
{lang_instruction}
---
CONTEXT FROM PREVIOUS CHUNK (FOR CONTINUITY ONLY — DO NOT REFINE OR OUTPUT):
...{chunk['context']}
---
NEW TRANSCRIPT CHUNK TO REFINE:
{chunk['text']}""")

            progress.set_units("refine", parallel_batches(len(chunks)) * 1.5)
            progress.update("refine", 0.1, f"{len(chunks)} sections in parallel")

            # Process chunks in parallel (max 3 concurrent to respect API rate limits)
            all_refined_chunks = [None] * len(chunks)
            chunk_tokens = [0] * len(chunks)

            def refine_chunk(idx, prompt):
                resp = generate_with_retry(refinement_model, prompt, generation_config=GENERATION_CONFIG)
                return idx, resp.text, safe_get_token_count(resp)

            with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as executor:
                futures = {executor.submit(refine_chunk, i, p): i for i, p in enumerate(prompts)}
                for future in as_completed(futures):
                    idx, text, tokens = future.result()
                    all_refined_chunks[idx] = text
                    chunk_tokens[idx] = tokens
                    done_count = sum(1 for c in all_refined_chunks if c is not None)
                    progress.update("refine", 0.1 + (0.9 * done_count / len(chunks)), f"{done_count}/{len(chunks)} sections")

            total_tokens += sum(chunk_tokens)
            refined_transcript = "\n\n".join(c for c in all_refined_chunks if c) if any(all_refined_chunks) else ""

        final_transcript = refined_transcript
        progress.complete_step("refine")
    else:
        # Refinement disabled — the plan contains no refine step.
        progress.complete_step("prepare")

    # Checkpoint: save refined transcript
    st.session_state["_checkpoint_refined_transcript"] = refined_transcript
    if draft_saved and refined_transcript:
        try:
            database.update_note(note_id, {"refined_transcript": refined_transcript, "token_usage": total_tokens})
        except Exception:
            pass

    # --- Step 3: Generate Notes ---
    # Earnings calls should not be chunked: their topic-based structure causes
    # repeated sections when the same headings appear across multiple chunks.
    skip_chunking = state.selected_meeting_type == "Earnings Call"
    final_notes_content, gen_tokens = generate_notes_from_transcript(
        state, final_transcript, notes_model, progress, skip_chunking=skip_chunking
    )
    total_tokens += gen_tokens

    # Defensive: ensure we have content
    if not final_notes_content or not final_notes_content.strip():
        raise ValueError("The model returned empty notes. Please try again or use a different model.")

    # --- Step 4: Deterministic cleanup (no LLM call, zero content-loss risk) ---
    progress.complete_step("generate")
    was_chunked = not skip_chunking and len(final_transcript.split()) > (getattr(state, "chunk_word_size", CHUNK_WORD_SIZE) or CHUNK_WORD_SIZE)
    if was_chunked:
        final_notes_content = cleanup_stitched_notes(final_notes_content)

    # --- Step 5: Executive Summary (Expert Meeting Option 3 only) ---
    if state.selected_note_style == "Option 3: Less Verbose + Summary" and state.selected_meeting_type == "Expert Meeting":
        progress.update("summary", 0.3, "Working...")
        summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(notes=final_notes_content)
        response = generate_with_retry(notes_model, summary_prompt)
        final_notes_content += f"\n\n---\n\n{response.text}"
        total_tokens += safe_get_token_count(response)
        progress.complete_step("summary")

    # --- Step 6: Save ---
    progress.update("save", 0.5, "Writing to database...")

    note_data = {
        'id': note_id, 'created_at': created_at, 'meeting_type': state.selected_meeting_type,
        'file_name': file_name, 'content': final_notes_content, 'raw_transcript': raw_transcript,
        'refined_transcript': refined_transcript, 'token_usage': total_tokens,
        'processing_time': time.time() - start_time,
        'pdf_blob': pdf_bytes_data
    }
    try:
        if draft_saved:
            database.update_note(note_id, {
                'content': final_notes_content,
                'refined_transcript': refined_transcript,
                'token_usage': total_tokens,
                'processing_time': note_data['processing_time'],
            })
        else:
            database.save_note(note_data)
    except Exception as db_error:
        st.session_state.app_state.fallback_content = final_notes_content
        raise Exception(f"Processing succeeded, but failed to save the note to the database. You can download the unsaved note below. Error: {db_error}")
    return note_data

# --- 5. UI RENDERING FUNCTIONS ---
def on_sector_change():
    """Callback for sector selectbox. Reads new value from the widget key
    (st.session_state.sector_selector) because on_change fires BEFORE the
    selectbox return value updates state.selected_sector."""
    state = st.session_state.app_state
    new_sector = st.session_state.get("sector_selector", state.selected_sector)
    state.selected_sector = new_sector
    all_sectors = db_get_sectors()
    state.earnings_call_topics = all_sectors.get(new_sector, "")

def render_input_and_processing_tab(state: AppState):
    # --- Source Input ---
    state.input_method = st.pills("Input Method", ["Paste Text", "Upload / Record"], default=state.input_method, key="input_method_pills")

    if state.input_method == "Paste Text":
        state.text_input = st.text_area("Paste source transcript here:", value=state.text_input, height=250, key="text_input_main")
        state.uploaded_file = None
        state.audio_recording = None
    else:
        col_upload, col_record = st.columns(2)
        with col_upload:
            # "audio" is a MIME shortcut (audio/*) covering mp3/m4a/wav/ogg/flac.
            state.uploaded_file = st.file_uploader("Upload a File", type=['pdf', 'txt', 'md', 'audio'], help="PDF, TXT, MD, or any audio file")
        with col_record:
            state.audio_recording = st.audio_input("Record Microphone")

    # --- Word count preview ---
    preview_text = ""
    if state.input_method == "Paste Text" and state.text_input:
        preview_text = state.text_input
    elif state.input_method == "Upload / Record" and state.uploaded_file:
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        if ext not in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            content, _, _ = get_file_content(state.uploaded_file, None)
            if content and not str(content).startswith("Error:") and content != "audio_file":
                preview_text = content

    if preview_text:
        wc = len(preview_text.split())
        num_chunks = estimate_chunk_count(wc, state.chunk_word_size)
        info = f"**{wc:,}** words"
        if num_chunks > 1:
            info += f" | ~**{num_chunks}** sections (processed in parallel)"
        st.caption(info)
    elif state.input_method == "Upload / Record" and state.uploaded_file:
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        if ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            st.caption("Audio file — word count available after transcription")

    st.divider()

    # --- Configuration ---
    st.subheader("Configuration")

    # Meeting type + style on one row for Expert Meeting
    if state.selected_meeting_type == "Expert Meeting":
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type), help=MEETING_TYPE_HELP.get(state.selected_meeting_type, ""))
        with cfg_col2:
            state.selected_note_style = st.selectbox("Note Style", EXPERT_MEETING_OPTIONS, index=EXPERT_MEETING_OPTIONS.index(state.selected_note_style))
    else:
        state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type), help=MEETING_TYPE_HELP.get(state.selected_meeting_type, ""))

    if state.selected_meeting_type == "Earnings Call":
        state.earnings_call_mode = st.radio("Mode", EARNINGS_CALL_MODES, horizontal=True, index=EARNINGS_CALL_MODES.index(state.earnings_call_mode))

        all_sectors = db_get_sectors()
        sector_options = ["Other / Manual Topics"] + sorted(list(all_sectors.keys()))

        try:
            current_sector_index = sector_options.index(state.selected_sector)
        except ValueError:
            current_sector_index = 0

        sector_col, manage_col = st.columns([3, 1])
        with sector_col:
            state.selected_sector = st.selectbox("Sector (for Topic Templates)", sector_options, index=current_sector_index, on_change=on_sector_change, key="sector_selector")
        with manage_col:
            st.container(height=28, border=False)  # vertical spacer to align with selectbox
            with st.popover("Manage Sectors", use_container_width=True):
                st.markdown("**Edit or Delete Sector**")
                sector_to_edit = st.selectbox("Select Sector", sorted(list(all_sectors.keys())))

                if sector_to_edit:
                    topics_for_edit = st.text_area("Sector Topics", value=all_sectors[sector_to_edit], key=f"topics_{sector_to_edit}")
                    col1, col2 = st.columns([1,1])
                    if col1.button("Save Changes", key=f"save_{sector_to_edit}"):
                        database.save_sector(sector_to_edit, topics_for_edit); db_get_sectors.clear();
                        st.toast(f"Sector '{sector_to_edit}' updated!"); st.rerun()
                    if col2.button("Delete Sector", type="primary", key=f"delete_{sector_to_edit}"):
                        database.delete_sector(sector_to_edit); db_get_sectors.clear(); state.selected_sector = "Other / Manual Topics"; on_sector_change();
                        st.toast(f"Sector '{sector_to_edit}' deleted!"); st.rerun()

                st.divider()
                st.markdown("**Add a New Sector**")
                new_sector_name = st.text_input("New Sector Name")
                new_sector_topics = st.text_area("Topics for New Sector", key="new_sector_topics")

                if st.button("Add New Sector"):
                    if new_sector_name and new_sector_topics:
                        database.save_sector(new_sector_name, new_sector_topics); db_get_sectors.clear();
                        st.toast(f"Sector '{new_sector_name}' added!"); st.rerun()
                    else:
                        st.warning("Please provide both a name and topics for the new sector.")

        state.earnings_call_topics = st.text_area("Topic Instructions", value=state.earnings_call_topics, height=150, placeholder="Select a sector to load a template, or enter topics manually.")

        if state.earnings_call_mode == "Enrich Existing Notes":
            state.existing_notes_input = st.text_area("Paste Existing Notes to Enrich:", value=state.existing_notes_input)

    elif state.selected_meeting_type == "Custom":
        state.context_input = st.text_area("Custom Instructions", value=state.context_input, height=120, placeholder="Describe how you want the notes structured...")

    # --- Settings & Participants row ---
    col_settings, col_participants = st.columns(2)
    with col_settings:
        with st.popover("Settings & Models", use_container_width=True):
            state.refinement_enabled = st.toggle("Transcript Refinement", value=state.refinement_enabled)
            _chunk_options = CHUNK_SIZE_OPTIONS if state.chunk_word_size in CHUNK_SIZE_OPTIONS else sorted(set(CHUNK_SIZE_OPTIONS + [state.chunk_word_size]))
            state.chunk_word_size = st.select_slider(
                "Section size (words)",
                options=_chunk_options,
                value=state.chunk_word_size,
                format_func=lambda v: f"{v:,}",
                help="Long transcripts are split into sections of roughly this many words "
                     "(aligned to speaker turns) and processed in parallel. Larger sections "
                     "mean fewer seams and faster runs; smaller sections give the model less "
                     "to digest per call — try lowering this if notes feel thin on detail.",
            )
            if state.selected_meeting_type != "Custom":
                state.add_context_enabled = st.toggle("Add General Context", value=state.add_context_enabled)
                if state.add_context_enabled: state.context_input = st.text_area("Context Details:", value=state.context_input, placeholder="e.g., Company Name, Date...")

            st.divider()
            state.notes_model = st.selectbox("Notes Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.notes_model))
            state.refinement_model = st.selectbox("Refinement Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.refinement_model))
            _sid_default = state.speaker_id_model if state.speaker_id_model in AVAILABLE_MODELS else list(AVAILABLE_MODELS.keys())[0]
            state.speaker_id_model = st.selectbox(
                "Speaker ID Model",
                list(AVAILABLE_MODELS.keys()),
                index=list(AVAILABLE_MODELS.keys()).index(_sid_default),
                help="Used only for the Speaker ID Flow (Expert Meeting Option 4). A stronger model produces better speaker separation and tag continuity across long transcripts.",
            )
            state.transcription_model = st.selectbox("Transcription Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.transcription_model), help="Used for audio files.")
            state.chat_model = st.selectbox("Chat Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.chat_model), help="Used for chatting with the final output.")

            st.divider()
            st.caption("Browser notifications for processing completion.")
            if st.button("Enable Notifications", key="enable_notif_btn", use_container_width=True):
                st.iframe(
                    """
                    <script>
                    if ("Notification" in window) {
                        Notification.requestPermission().then(function(permission) {
                            if (permission === "granted") {
                                new Notification("Notifications Enabled", {
                                    body: "You'll be notified when processing completes.",
                                    icon: "https://placehold.co/64x64?text=SN"
                                });
                            }
                        });
                    }
                    </script>
                    """,
                    height=1,  # st.iframe requires a positive height; 1px keeps it invisible
                )
    with col_participants:
        state.speakers = st.text_input("Participants (Optional)", value=state.speakers, placeholder="e.g., John Smith (Analyst), Jane Doe (CEO)")

    # --- Generate ---
    st.divider()
    validation_error = validate_inputs(state)

    # Detect Speaker ID Flow (Option 4 of Expert Meeting)
    is_speaker_flow = (
        state.selected_meeting_type == "Expert Meeting"
        and state.selected_note_style == SPEAKER_ID_FLOW_OPTION
    )

    if is_speaker_flow:
        st.info(
            "The transcript is refined and tagged with speaker labels first. "
            "You can rename speakers, reassign per-segment tags, add a new tag, and download the tagged transcript "
            "before generating the final notes.",
            icon="\U0001f399\ufe0f",
            title="Speaker ID Flow",
        )

    if validation_error:
        st.warning(validation_error)

    button_label = "Identify Speakers" if is_speaker_flow else "Generate Notes"
    if st.button(button_label, type="primary", use_container_width=True, disabled=bool(validation_error)):
        state.error_message = None
        state.fallback_content = None
        if is_speaker_flow:
            st.session_state.sn_processing_id = True
            # Reset prior speaker-flow state so a fresh run starts clean,
            # including per-segment dropdown and per-speaker rename widget keys.
            for k in ("sn_segments", "sn_speakers", "sn_speaker_names", "sn_tagged_transcript", "sn_raw_transcript", "sn_file_name", "sn_pdf_bytes", "sn_id_tokens", "sn_id_elapsed", "sn_downstream_style_locked", "sn_draft_note_id"):
                st.session_state.pop(k, None)
            for k in [k for k in list(st.session_state.keys()) if k.startswith("sn_seg_tag_") or k.startswith("sn_name_")]:
                st.session_state.pop(k, None)
        else:
            state.processing = True
        st.rerun()

    # --- Prompt Preview (collapsed) ---
    with st.expander("Prompt Preview", expanded=False):
        if is_speaker_flow:
            preview = SPEAKER_ID_PROMPT_INITIAL.format(
                speaker_info=f"Known participants: {sanitize_input(state.speakers)}." if state.speakers else "",
                refinement_extra=REFINEMENT_INSTRUCTIONS.get("Expert Meeting", ""),
                transcript="[...transcript content...]",
            )
        else:
            preview = get_dynamic_prompt(state, "[...transcript content...]")
        st.code(preview, language="markdown", height=200)

    # --- Standard Processing (non-speaker flows) ---
    if state.processing:
        with st.status("Processing your request...", expanded=True) as status:
            progress = ProgressTracker(build_processing_plan(
                is_audio=_input_is_audio(state),
                refinement_enabled=state.refinement_enabled,
                with_summary=(
                    state.selected_note_style == "Option 3: Less Verbose + Summary"
                    and state.selected_meeting_type == "Expert Meeting"
                ),
            ))
            try:
                final_note = process_and_save_task(state, status, progress)
                state.active_note_id = final_note['id']
                progress.finish()
                processing_time = final_note.get('processing_time', 0)
                word_count = len(final_note.get('content', '').split())
                status.update(
                    label=f"Done! {word_count:,} words generated in {processing_time:.1f}s. Switch to the **Output & History** tab to view your note.",
                    state="complete"
                )
                st.toast("Notes generated successfully!", icon="\u2705")
                send_browser_notification(
                    "SynthNotes AI - Complete",
                    f"Your notes are ready! Processing took {processing_time:.1f}s"
                )
            except Exception as e:
                state.error_message = f"An error occurred during processing:\n{e}"
                status.update(label=f"Error: {e}", state="error")
                send_browser_notification(
                    "SynthNotes AI - Error",
                    "Processing failed. Check the app for details."
                )
        state.processing = False

    # --- Speaker ID Processing (speaker-tag step) ---
    if st.session_state.get("sn_processing_id"):
        with st.status("Identifying speakers...", expanded=True) as status:
            progress = ProgressTracker(build_speaker_id_plan(is_audio=_input_is_audio(state)))
            try:
                result = run_speaker_identification_task(state, status, progress)
                st.session_state.sn_segments = result["segments"]
                st.session_state.sn_speakers = result["speakers"]
                # Pre-filled from the participants list where inference was
                # confident; the user can still rename in the review panel.
                st.session_state.sn_speaker_names = {
                    s: result["inferred_names"].get(s, "") for s in result["speakers"]
                }
                st.session_state.sn_tagged_transcript = result["tagged_transcript"]
                st.session_state.sn_raw_transcript = result["raw_transcript"]
                st.session_state.sn_file_name = result["file_name"]
                st.session_state.sn_pdf_bytes = result["pdf_bytes"]
                st.session_state.sn_id_tokens = result["token_usage"]
                st.session_state.sn_id_elapsed = result["elapsed"]
                st.session_state.sn_draft_note_id = result["note_id"]
                progress.finish()
                n_speakers = len(result["speakers"])
                n_segments = len(result["segments"])
                status.update(
                    label=f"Done! Detected {n_speakers} speaker{'s' if n_speakers != 1 else ''} across {n_segments} segments in {result['elapsed']:.1f}s. Review and edit tags below.",
                    state="complete",
                )
                st.toast(f"Detected {n_speakers} speakers", icon="\U0001f399\ufe0f")
                send_browser_notification(
                    "SynthNotes AI - Speakers Identified",
                    f"{n_speakers} speakers detected across {n_segments} segments."
                )
            except Exception as e:
                state.error_message = f"Speaker identification failed:\n{e}"
                status.update(label=f"Error: {e}", state="error")
                send_browser_notification(
                    "SynthNotes AI - Error",
                    "Speaker identification failed. Check the app for details."
                )
        st.session_state.sn_processing_id = False

    # --- Speaker Review Panel + Notes Generation (only when segments exist) ---
    if is_speaker_flow and st.session_state.get("sn_segments"):
        _render_speaker_review_panel(state)

    if state.error_message:
        st.error(state.error_message, title="Processing failed")
        # Recovery: transcripts checkpointed before the failure are saved to
        # History as an incomplete note and downloadable here.
        ckpt_raw = st.session_state.get("_checkpoint_raw_transcript")
        ckpt_refined = st.session_state.get("_checkpoint_refined_transcript")
        if ckpt_raw or ckpt_refined:
            st.caption(
                "Work completed before the failure is not lost: the transcript was "
                "checkpointed to **Output & History** and can also be downloaded below."
            )
        err_col1, err_col2, err_col3, err_col4 = st.columns(4)
        if state.fallback_content:
            err_col1.download_button("Unsaved Note (.txt)", state.fallback_content, "synthnotes_fallback.txt", use_container_width=True)
        if ckpt_raw:
            err_col2.download_button("Raw Transcript (.txt)", ckpt_raw, "raw_transcript_checkpoint.txt", use_container_width=True)
        if ckpt_refined:
            err_col3.download_button("Refined Transcript (.txt)", ckpt_refined, "refined_transcript_checkpoint.txt", use_container_width=True)
        if err_col4.button("Dismiss Error", use_container_width=True):
            state.error_message = None
            state.fallback_content = None
            st.rerun()

def _on_speaker_name_change(speaker_tag: str):
    """Push the latest text_input value for `speaker_tag` into the speaker_names
    dict immediately on commit, so dropdown labels reflect the new name on the
    next render without depending on stale-vs-current comparisons."""
    new_val = st.session_state.get(f"sn_name_{speaker_tag}", "")
    names = st.session_state.get("sn_speaker_names")
    if names is not None:
        names[speaker_tag] = new_val


def _render_speaker_review_panel(state: AppState):
    """Render the speaker-tagged transcript editor used by the Speaker ID Flow.

    Lets the user rename speakers, add a new (initially empty) speaker tag,
    reassign any segment's tag (including `Skip` for logistics), download the
    tagged transcript, and then run the standard Expert notes generation on
    the edited transcript. `Skip`-tagged segments are excluded from the
    transcript that feeds the notes prompt.
    """
    st.divider()
    st.subheader("Speaker Review & Tagging")

    segments: List[Dict[str, str]] = st.session_state.sn_segments
    speakers: List[str] = st.session_state.sn_speakers
    speaker_names: Dict[str, str] = st.session_state.sn_speaker_names

    # Ensure Skip is always available as a tag option, ordered last
    if SKIP_TAG not in speakers:
        speakers.append(SKIP_TAG)
    else:
        speakers[:] = [s for s in speakers if s != SKIP_TAG] + [SKIP_TAG]
    speaker_names.setdefault(SKIP_TAG, "")

    file_name = st.session_state.get("sn_file_name", "transcript")
    n_seg = len(segments)
    real_speakers = [s for s in speakers if s != SKIP_TAG]
    n_spk = len(real_speakers)
    n_skip = sum(1 for s in segments if s["speaker"] == SKIP_TAG)
    st.caption(
        f"{n_spk} speaker tag{'s' if n_spk != 1 else ''} · {n_seg} segments"
        + (f" · {n_skip} skipped (logistics)" if n_skip else "")
        + f" · source: {file_name}"
    )

    # --- Speaker labels: rename and add new (Skip is not renameable) ---
    st.markdown("**Speakers**")
    name_cols = st.columns(min(max(len(real_speakers), 1), 3))
    for i, sp in enumerate(real_speakers):
        with name_cols[i % len(name_cols)]:
            st.text_input(
                f"{sp} display name",
                value=speaker_names.get(sp, ""),
                placeholder="e.g. Jane Doe (CEO)",
                key=f"sn_name_{sp}",
                on_change=_on_speaker_name_change,
                args=(sp,),
            )

    add_col, swap_col, _ = st.columns([1, 1, 2])
    with add_col:
        if st.button("➕ Add Speaker Tag", key="sn_add_speaker_btn", use_container_width=True):
            # Find the next free Speaker N (ignore Skip)
            existing_nums = []
            for s in real_speakers:
                m = re.search(r"\d+", s)
                if m:
                    existing_nums.append(int(m.group(0)))
            next_num = max(existing_nums) + 1 if existing_nums else 1
            new_tag = f"Speaker {next_num}"
            # Insert before Skip so Skip stays last
            speakers.insert(len(speakers) - 1, new_tag)
            speaker_names[new_tag] = ""
            st.toast(f"Added {new_tag}. Reassign any segment to it via the dropdown.")
            st.rerun()
    if len(real_speakers) == 2:
        with swap_col:
            if st.button(
                "🔄 Swap Speaker 1 ↔ 2",
                key="sn_swap_speakers_btn",
                use_container_width=True,
                help="One click fixes the most common tagging error: the model assigning "
                     "the interviewer's turns to the expert and vice versa. Swaps every "
                     "segment's tag and the display names.",
            ):
                s1, s2 = real_speakers[0], real_speakers[1]
                for seg in segments:
                    if seg["speaker"] == s1:
                        seg["speaker"] = s2
                    elif seg["speaker"] == s2:
                        seg["speaker"] = s1
                speaker_names[s1], speaker_names[s2] = speaker_names.get(s2, ""), speaker_names.get(s1, "")
                # Clear widget state so dropdowns and name fields re-render
                # from the swapped segment data instead of stale selections.
                for k in [k for k in list(st.session_state.keys()) if k.startswith("sn_seg_tag_") or k.startswith("sn_name_")]:
                    st.session_state.pop(k, None)
                st.toast("Swapped all Speaker 1 ↔ Speaker 2 tags.")
                st.rerun()

    st.caption(
        "Rename speakers above (display names appear in the final notes). "
        "Use **Add Speaker Tag** to create a new tag — reassign any segment via the dropdown below. "
        "**Skip** is used by the model to mark logistical chatter (charger/food/breaks/tech checks); "
        "those segments are excluded from the final notes. Reassign any segment to a real speaker if needed."
    )

    st.divider()

    # --- Per-segment reassign + edit ---
    st.markdown("**Segments**")
    st.caption("Each segment shows the tag dropdown (left) and the refined text (right). Change the dropdown to reassign a segment.")

    # Show just the display name once set; fall back to the generic tag.
    def _display_for_tag(tag: str) -> str:
        if tag == SKIP_TAG:
            return "Skip (logistics)"
        return speaker_names.get(tag) or tag

    for idx, seg in enumerate(segments):
        is_skip = seg["speaker"] == SKIP_TAG
        with st.container(border=True):
            sel_col, text_col = st.columns([1, 4])
            with sel_col:
                try:
                    default_idx = speakers.index(seg["speaker"])
                except ValueError:
                    default_idx = 0
                    seg["speaker"] = speakers[0]
                picked_tag = st.selectbox(
                    f"Tag #{idx + 1}",
                    speakers,
                    index=default_idx,
                    format_func=_display_for_tag,
                    key=f"sn_seg_tag_{idx}",
                    label_visibility="collapsed",
                )
                if picked_tag != seg["speaker"]:
                    seg["speaker"] = picked_tag
            with text_col:
                preview = seg["text"] if len(seg["text"]) <= 600 else seg["text"][:600] + "…"
                if is_skip:
                    st.markdown(f":gray[*(excluded from notes)*  \n{preview}]")
                else:
                    st.markdown(preview)
                if len(seg["text"]) > 600:
                    with st.expander("Full text", expanded=False):
                        st.markdown(seg["text"])

    st.divider()

    # --- Download tagged transcript (includes Skip for the record) ---
    canonical_tagged = _serialize_tagged_segments(segments)  # generic Speaker N + Skip labels
    named_tagged = _serialize_tagged_segments(segments, display_names=speaker_names)  # with display names
    # The transcript that feeds the notes prompt excludes Skip segments.
    notes_input_tagged = _serialize_tagged_segments(
        segments, display_names=speaker_names, exclude_tags=[SKIP_TAG]
    )

    safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", os.path.splitext(file_name)[0] or "transcript")[:60]
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download Tagged Transcript (.txt)",
            named_tagged,
            file_name=f"{safe_stem}_tagged.txt",
            mime="text/plain",
            use_container_width=True,
            key="sn_download_named",
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download Raw Refined (no names) (.txt)",
            canonical_tagged,
            file_name=f"{safe_stem}_speakers.txt",
            mime="text/plain",
            use_container_width=True,
            key="sn_download_raw",
        )

    st.divider()

    # --- Downstream notes step ---
    st.markdown("**Generate Final Notes**")
    ds_col1, ds_col2 = st.columns([2, 1])
    with ds_col1:
        downstream_style = st.selectbox(
            "Final note style (uses the existing Expert Meeting prompts)",
            SPEAKER_ID_DOWNSTREAM_OPTIONS,
            index=1,  # default to Option 2: Less Verbose
            key="sn_downstream_style",
        )
    with ds_col2:
        st.write("")  # spacer for vertical alignment
        st.write("")
        if st.button("Generate Notes", type="primary", use_container_width=True, key="sn_generate_notes_btn"):
            st.session_state.sn_processing_notes = True
            st.session_state.sn_downstream_style_locked = downstream_style
            st.rerun()

    # --- Run notes generation ---
    if st.session_state.get("sn_processing_notes"):
        with st.status("Generating notes from tagged transcript...", expanded=True) as status:
            progress = ProgressTracker(build_notes_only_plan(
                with_summary=(
                    st.session_state.sn_downstream_style_locked == "Option 3: Less Verbose + Summary"
                ),
            ))
            try:
                final_note = process_tagged_to_notes_task(
                    state,
                    status,
                    progress,
                    tagged_transcript=notes_input_tagged,
                    raw_transcript=st.session_state.sn_raw_transcript,
                    file_name=st.session_state.sn_file_name,
                    pdf_bytes=st.session_state.sn_pdf_bytes,
                    downstream_style=st.session_state.sn_downstream_style_locked,
                    prior_tokens=st.session_state.get("sn_id_tokens", 0),
                    draft_note_id=st.session_state.get("sn_draft_note_id"),
                )
                state.active_note_id = final_note['id']
                progress.finish()
                processing_time = final_note.get('processing_time', 0)
                word_count = len(final_note.get('content', '').split())
                status.update(
                    label=f"Done! {word_count:,} words generated in {processing_time:.1f}s. Switch to the **Output & History** tab to view your note.",
                    state="complete",
                )
                st.toast("Notes generated from tagged transcript!", icon="✅")
                send_browser_notification(
                    "SynthNotes AI - Complete",
                    f"Your speaker-tagged notes are ready! Processing took {processing_time:.1f}s",
                )
            except Exception as e:
                state.error_message = f"An error occurred while generating notes:\n{e}"
                status.update(label=f"Error: {e}", state="error")
                send_browser_notification(
                    "SynthNotes AI - Error",
                    "Notes generation failed. Check the app for details.",
                )
        st.session_state.sn_processing_notes = False

    # --- Reset / discard ---
    reset_col, _ = st.columns([1, 3])
    with reset_col:
        if st.button("Discard & Start Over", key="sn_reset_btn", use_container_width=True):
            # Remove the draft checkpoint note if no notes were generated on it.
            draft_id = st.session_state.get("sn_draft_note_id")
            if draft_id:
                try:
                    draft = database.get_note_by_id(draft_id)
                    if draft and not (draft.get("content") or "").strip():
                        database.delete_note(draft_id)
                except Exception:
                    pass
            for k in ("sn_segments", "sn_speakers", "sn_speaker_names", "sn_tagged_transcript", "sn_raw_transcript", "sn_file_name", "sn_pdf_bytes", "sn_id_tokens", "sn_id_elapsed", "sn_downstream_style_locked", "sn_draft_note_id"):
                st.session_state.pop(k, None)
            for k in [k for k in list(st.session_state.keys()) if k.startswith("sn_seg_tag_") or k.startswith("sn_name_")]:
                st.session_state.pop(k, None)
            st.rerun()


@st.dialog("Delete Note")
def _confirm_delete_dialog(note_id: str, note_name: str):
    # Truncate long names in the dialog
    display_name = note_name if len(note_name) <= 50 else note_name[:47] + "..."
    st.markdown(f"Are you sure you want to delete **{display_name}**?")
    st.caption("This action cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Yes, Delete", use_container_width=True):
        database.delete_note(note_id)
        if st.session_state.app_state.active_note_id == note_id:
            st.session_state.app_state.active_note_id = None
        st.toast(f"Note '{display_name}' deleted.")
        st.rerun()

def run_validation_in_chunks(notes: str, transcript: str, model_name: str) -> list:
    """Run per-Q&A HTML-annotated validation.

    Always passes the FULL NOTES for context to both chunks so neither pass
    incorrectly flags content as missing that is actually captured in the other
    chunk. Splits only the PORTION TO ANNOTATE at Q&A boundaries.

    Returns a list of 1 or 2 annotated HTML strings.
    """
    model = genai.GenerativeModel(model_name)
    tx_limit = 40000  # characters of transcript per call

    # Find bold question lines — lines that start AND end with ** (markdown bold)
    note_lines = notes.split('\n')
    bold_indices = [
        i for i, line in enumerate(note_lines)
        if line.strip().startswith('**') and line.strip().endswith('**') and len(line.strip()) > 4
    ]

    # Single-pass when notes are short or have too few Q&As to justify splitting
    if len(bold_indices) < 4 or len(notes) < 8000:
        prompt = VALIDATION_DETAILED_PROMPT.format(
            chunk_info="Full Notes",
            full_notes=notes,
            chunk_to_annotate=notes,
            transcript=transcript[:tx_limit]
        )
        r = generate_with_retry(model, prompt)
        return [r.text]

    # Two-pass: split the PORTION TO ANNOTATE at the Q&A midpoint.
    # Both passes receive the FULL NOTES for context — this prevents Part 1
    # from flagging content as missing that is captured in Part 2 and vice versa.
    split_q = len(bold_indices) // 2
    split_line = bold_indices[split_q]
    chunk1_notes = '\n'.join(note_lines[:split_line]).strip()
    chunk2_notes = '\n'.join(note_lines[split_line:]).strip()

    # Transcript: transcript is sequential, so the first half maps to Part 1
    # Q&As and the second half maps to Part 2. Send the full slice to both so
    # neither is starved of context for edge cases.
    tx_slice = transcript[:tx_limit]

    prompt1 = VALIDATION_DETAILED_PROMPT.format(
        chunk_info="Part 1 of 2 — first half of Q&As",
        full_notes=notes,
        chunk_to_annotate=chunk1_notes,
        transcript=tx_slice
    )
    r1 = generate_with_retry(model, prompt1)

    prompt2 = VALIDATION_DETAILED_PROMPT.format(
        chunk_info="Part 2 of 2 — second half of Q&As",
        full_notes=notes,
        chunk_to_annotate=chunk2_notes,
        transcript=tx_slice
    )
    r2 = generate_with_retry(model, prompt2)

    return [r1.text, r2.text]


def _render_history_sidebar(state: AppState, notes: List[dict]):
    """Compact note history + analytics in the sidebar, so the main page
    stays focused on the active note."""
    with st.sidebar:
        st.subheader("History")

        raw_summary_data = database.get_analytics_summary()
        summary_dict = {}
        if isinstance(raw_summary_data, dict):
            summary_dict = raw_summary_data
        elif isinstance(raw_summary_data, tuple) and raw_summary_data:
            if isinstance(raw_summary_data[0], dict):
                summary_dict = raw_summary_data[0]
            else:
                summary_dict['total_notes'] = raw_summary_data[0] if len(raw_summary_data) > 0 else 0
                summary_dict['avg_time'] = raw_summary_data[1] if len(raw_summary_data) > 1 else 0.0
                summary_dict['total_tokens'] = raw_summary_data[2] if len(raw_summary_data) > 2 else 0
        st.caption(
            f"{summary_dict.get('total_notes', 0)} notes · "
            f"avg {summary_dict.get('avg_time', 0.0):.0f}s/note · "
            f"{summary_dict.get('total_tokens', 0):,} tokens"
        )

        search_query = st.text_input(
            "Search notes by file name", placeholder="Search notes...",
            label_visibility="collapsed", key="history_search",
        )
        type_filter = st.selectbox(
            "Filter by meeting type", ["All Types"] + MEETING_TYPES,
            label_visibility="collapsed", key="history_type_filter",
        )

        filtered_notes = notes
        if search_query:
            filtered_notes = [n for n in filtered_notes if search_query.lower() in n.get('file_name', '').lower()]
        if type_filter != "All Types":
            filtered_notes = [n for n in filtered_notes if n.get('meeting_type') == type_filter]

        if not filtered_notes:
            st.info("No notes match your search.", title="No results")
            return

        for note in filtered_notes:
            is_active = note['id'] == state.active_note_id
            with st.container(border=True):
                card_name = note['file_name']
                if len(card_name) > 42:
                    card_name = card_name[:39] + "..."
                name_col, menu_col = st.columns([5, 1])
                with name_col:
                    st.markdown(f"**{card_name}**" + (" &nbsp; `viewing`" if is_active else ""))
                with menu_col:
                    action = st.menu_button(
                        "⋮", ["View", "Delete"],
                        key=f"note_menu_{note['id']}", type="tertiary",
                    )
                st.caption(
                    f"{note['meeting_type']} · "
                    f"{datetime.fromisoformat(note['created_at']).strftime('%b %d, %Y %H:%M')}"
                )
                if not (note.get('content') or '').strip():
                    st.caption("⚠️ Incomplete — transcript saved, no notes yet")
                if action == "View":
                    if not is_active:
                        state.active_note_id = note['id']
                        st.rerun()
                elif action == "Delete":
                    _confirm_delete_dialog(note['id'], note['file_name'])


def render_output_and_history_tab(state: AppState):
    notes = database.get_all_notes()

    if not notes:
        st.markdown("""
### No notes yet

1. Switch to the **Input & Generate** tab
2. Paste text, upload a file (PDF, TXT), or record audio
3. Pick a meeting type and click **Generate Notes**

Your generated notes, transcripts, and chat history will appear here.
        """)
        return

    # --- Active Note ---
    if not state.active_note_id or not any(n['id'] == state.active_note_id for n in notes):
        state.active_note_id = notes[0]['id']

    _render_history_sidebar(state, notes)

    active_note = database.get_note_by_id(state.active_note_id)
    if not active_note:
        active_note = database.get_note_by_id(notes[0]['id'])

    # --- Note header with inline metadata ---
    hdr_left, hdr_right = st.columns([3, 2])
    with hdr_left:
        # Truncate very long file names to prevent layout breakage
        display_name = active_note['file_name']
        if len(display_name) > 80:
            display_name = display_name[:77] + "..."
        st.markdown(f"### {display_name}")
        st.badge(active_note['meeting_type'])
    with hdr_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Time", f"{active_note.get('processing_time', 0):.1f}s")
        m2.metric("Tokens", f"{active_note.get('token_usage', 0):,}")
        m3.metric("Date", datetime.fromisoformat(active_note['created_at']).strftime('%b %d'))

    if not (active_note.get('content') or '').strip():
        st.warning(
            "Notes generation didn't finish for this entry, but the transcript was "
            "checkpointed and is shown on the right. Copy it into the Input & Generate "
            "tab (refinement can be toggled off if it already ran) to retry.",
            title="Incomplete note — transcript recovered",
        )

    # --- Side-by-side Notes & Transcript ---
    final_transcript = active_note.get('refined_transcript') or active_note.get('raw_transcript')
    transcript_source = "Refined" if active_note.get('refined_transcript') else "Raw"

    col_notes, col_transcript = st.columns([3, 2])
    with col_notes:
        view_mode = st.pills("View", ["Editor", "Preview"], default="Editor", key=f"view_mode_{active_note['id']}")
        if view_mode == "Editor":
            edited_content = st.text_area("Notes", value=active_note['content'], height=600, key=f"output_editor_{active_note['id']}")
            # Word count feedback for the notes editor
            note_wc = len(edited_content.split()) if edited_content else 0
            st.caption(f"{note_wc:,} words")
        else:
            edited_content = active_note['content']
            with st.container(height=600, border=True):
                st.markdown(edited_content)
            note_wc = len(edited_content.split()) if edited_content else 0
            st.caption(f"{note_wc:,} words")
    with col_transcript:
        st.markdown(f"**{transcript_source} Transcript**")
        if final_transcript:
            st.text_area("Transcript", value=final_transcript, height=600, disabled=True, label_visibility="collapsed", key=f"side_tx_{active_note['id']}")
        else:
            st.info("No transcript available.")

    # --- Actions bar ---
    note_id = active_note['id']
    fname = active_note.get('file_name', 'note')
    raw_tx = active_note.get('raw_transcript')

    act_col, fb_col = st.columns([1, 3])
    with act_col:
        with st.popover("Export / Copy", icon=":material/download:", use_container_width=True):
            copy_to_clipboard_button(edited_content)
            st.download_button(
                label="Notes (.txt)",
                data=edited_content,
                file_name=f"SynthNote_{fname}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                label="Notes (.md)",
                data=edited_content,
                file_name=f"SynthNote_{fname}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            if final_transcript:
                st.download_button(
                    label=f"{transcript_source} Transcript (.txt)",
                    data=final_transcript,
                    file_name=f"{transcript_source}_Transcript_{fname}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            elif raw_tx:
                st.download_button(
                    label="Raw Transcript (.txt)",
                    data=raw_tx,
                    file_name=f"Raw_Transcript_{fname}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
    with fb_col:
        st.feedback("thumbs", key=f"fb_{active_note['id']}")

    # --- VALIDATE OUTPUT COMPLETENESS ---
    if final_transcript:
        st.divider()
        st.subheader("Validate Output Completeness")
        st.caption(
            "Performs a detailed per-Q&A audit: each question and every answer bullet is checked "
            "against the source transcript. Long notes are automatically split into two parts. "
            "Results are displayed inline with colour-coded annotations."
        )

        val_key = f"validation_result_{active_note['id']}"

        if st.button("Validate Output Completeness", key=f"validate_btn_{active_note['id']}", type="secondary", use_container_width=False):
            with st.spinner("Running detailed per-Q&A validation — this may take a moment..."):
                try:
                    val_model_name = AVAILABLE_MODELS.get(state.chat_model, "gemini-2.5-pro")
                    chunks = run_validation_in_chunks(edited_content, final_transcript, val_model_name)
                    st.session_state[val_key] = chunks
                except Exception as e:
                    st.session_state[val_key] = [f"**Validation failed:** {str(e)}"]

        if val_key in st.session_state:
            chunks = st.session_state[val_key]
            # Legend
            st.markdown(
                "<div style='font-size:0.82em;margin:4px 0 8px 0;line-height:2'>"
                "<span style='background:#fef9c3;color:#78350f;border-radius:3px;"
                "padding:2px 6px;margin-right:8px'>⚠️ yellow</span>"
                "Missing content &nbsp;|&nbsp; "
                "<span style='color:#dc2626;text-decoration:line-through;margin-right:2px'>"
                "strikethrough</span>"
                "<span style='color:#16a34a;margin-right:8px'> → green</span>"
                "Misrepresentation &nbsp;|&nbsp; "
                "<span style='background:#ede9fe;color:#5b21b6;border-radius:3px;"
                "padding:2px 6px;margin-right:8px'>🔁 purple</span>"
                "Repeated / duplicate Q&A"
                "</div>",
                unsafe_allow_html=True
            )
            if len(chunks) == 2:
                tab1, tab2 = st.tabs(["Part 1", "Part 2"])
                for tab, chunk_html in zip([tab1, tab2], chunks):
                    with tab:
                        with st.container(height=620, border=True):
                            st.markdown(chunk_html, unsafe_allow_html=True)
            else:
                with st.container(height=620, border=True):
                    st.markdown(chunks[0], unsafe_allow_html=True)

    # --- CHAT ---
    st.divider()
    st.subheader("Chat with this Note")
    st.caption("Ask questions about the content. The model has access to both the notes and the source transcript for verbatim lookups.")

    st.session_state.chat_histories.setdefault(active_note['id'], [])
    history = st.session_state.chat_histories[active_note['id']]

    chat_box = None
    if history:
        chat_box = st.container(height=420, autoscroll=True)
        with chat_box:
            for message in history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Pinned to the bottom of the viewport so the input is always reachable.
    with st.bottom:
        prompt = st.chat_input("Ask a question about this note...")

    if prompt:
        if chat_box is None:
            chat_box = st.container(height=420, autoscroll=True)
        history.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=":material/progress_activity:"):
                full_response = ""
                try:
                    transcript_context = final_transcript[:CHAT_TRANSCRIPT_CHAR_LIMIT] if final_transcript else "Not available."
                    truncation_note = ""
                    if final_transcript and len(final_transcript) > CHAT_TRANSCRIPT_CHAR_LIMIT:
                        truncation_note = f"\n\nNote: The transcript was truncated from {len(final_transcript):,} to {CHAT_TRANSCRIPT_CHAR_LIMIT:,} characters. Some content at the end may be missing from the TRANSCRIPT section. The NOTES section contains the full meeting content."
                    system_prompt = f"""You are an expert analyst. Your task is to answer questions based on the provided meeting notes and source transcript.
If the user asks for verbatim quotes or exact wording, refer to the TRANSCRIPT section. For analysis and summary questions, use the NOTES section.{truncation_note}

MEETING NOTES:
---
{edited_content}
---
SOURCE TRANSCRIPT:
---
{transcript_context}
---
"""
                    chat_model_name = AVAILABLE_MODELS.get(state.chat_model, "gemini-2.5-flash")
                    chat_model = genai.GenerativeModel(chat_model_name, system_instruction=system_prompt)
                    messages_for_api = [{'role': "model" if m["role"] == "assistant" else "user", 'parts': [m['content']]} for m in history]

                    chat = chat_model.start_chat(history=messages_for_api[:-1])
                    response = chat.send_message(messages_for_api[-1]['parts'], stream=True)

                    message_placeholder = st.empty()
                    try:
                        for chunk in response:
                            if not chunk.parts:
                                continue
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "\u258c")
                    except Exception as stream_err:
                        # Handle streaming interruption gracefully
                        if full_response:
                            full_response += f"\n\n*(Stream interrupted: {stream_err})*"
                        else:
                            raise stream_err
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"Sorry, an error occurred: {str(e)}"
                    st.error(full_response, title="Chat failed")
                    if history:
                        history.pop()

        if 'full_response' in locals() and not full_response.startswith("Sorry"):
            history.append({"role": "assistant", "content": full_response})

def _build_ia_prompt_template(meeting_type: str) -> str:
    """Return the IA prompt for the given meeting type; {transcript} left as placeholder."""
    return IA_MANAGEMENT_KTA_PROMPT if meeting_type == "management" else IA_EXPERT_KTA_PROMPT


def render_ia_processing(state: AppState):
    """Investment Analyst Processing: two-step transcript → dual output (KTAs + Rough Notes)."""

    # --- Session state init ---
    for key, default in [
        ("ia_meeting_type", "management"),
        ("ia_transcript", ""),
        ("ia_output", ""),
        ("ia_prompt_text", ""),
        ("ia_prompt_seed", ("", "")),
        ("ia_company_name", ""),
        ("ia_area", ""),
        ("ia_refine_enabled", False),
        ("ia_refined_transcript", ""),
        ("ia_tone", "Neutral"),
        ("ia_number_focus", "Moderate"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Step 1: Meeting type ---
    st.markdown("#### Step 1 — Meeting Type")
    meeting_type_opt = st.radio(
        "Meeting type",
        options=["1 — Company Management Meeting", "2 — Expert / Industry Expert / Channel Check Meeting"],
        index=0 if st.session_state.ia_meeting_type != "expert" else 1,
        label_visibility="collapsed",
        key="ia_meeting_type_radio",
    )
    st.session_state.ia_meeting_type = "management" if meeting_type_opt.startswith("1") else "expert"

    # Company / area fields
    _c1, _c2 = st.columns(2)
    with _c1:
        st.session_state.ia_company_name = st.text_input(
            "Company / Entity Name",
            value=st.session_state.ia_company_name,
            placeholder="e.g. Reliance, Hero MotoCorp",
            key="ia_company_name_input",
        )
    with _c2:
        if st.session_state.ia_meeting_type == "expert":
            st.session_state.ia_area = st.text_input(
                "Coverage Area / Sector",
                value=st.session_state.ia_area,
                placeholder="e.g. Two-Wheeler Dealerships, Quick Commerce",
                key="ia_area_input",
            )

    # --- Auto-reset prompt when meeting type or prompt version changes ---
    _IA_PROMPT_VERSION = "v7"  # bump when prompts are updated to force rebuild in existing sessions
    current_seed = (_IA_PROMPT_VERSION, st.session_state.ia_meeting_type)
    if st.session_state.ia_prompt_seed != current_seed or not st.session_state.ia_prompt_text:
        st.session_state.ia_prompt_text = _build_ia_prompt_template(
            st.session_state.ia_meeting_type
        )
        st.session_state.ia_prompt_seed = current_seed

    # --- Editable prompt ---
    with st.expander("Edit Prompt (optional)", expanded=False):
        reset_col, note_col = st.columns([1, 3])
        with reset_col:
            if st.button("Reset to Default", key="ia_reset_prompt", use_container_width=True):
                st.session_state.ia_prompt_text = _build_ia_prompt_template(
                    st.session_state.ia_meeting_type
                )
                st.rerun()
        with note_col:
            st.caption("`{transcript}` in the prompt will be replaced with your transcript at generation time.")
        st.session_state.ia_prompt_text = st.text_area(
            "Prompt template",
            value=st.session_state.ia_prompt_text,
            height=520,
            label_visibility="collapsed",
            key="ia_prompt_editor",
        )

    st.divider()

    # --- Step 2: Transcript input ---
    st.markdown("#### Step 2 — Paste the Transcript")
    st.session_state.ia_transcript = st.text_area(
        "Transcript",
        value=st.session_state.ia_transcript,
        height=320,
        placeholder="Paste the full meeting transcript here…",
        label_visibility="collapsed",
        key="ia_transcript_input",
    )

    if not st.session_state.ia_transcript.strip():
        st.info("Paste the transcript above to continue.")
        return

    wc = len(st.session_state.ia_transcript.split())
    st.caption(f"{wc:,} words")

    st.divider()

    # --- Tone and Data Emphasis ---
    ia_tone_col, ia_number_col = st.columns(2)
    with ia_tone_col:
        ia_tone = st.pills("Tone", TONE_OPTIONS, default=st.session_state.ia_tone, key="ia_tone_pills")
        if ia_tone:
            st.session_state.ia_tone = ia_tone
    with ia_number_col:
        ia_number_focus = st.pills("Data Emphasis", NUMBER_FOCUS_OPTIONS, default=st.session_state.ia_number_focus, key="ia_number_pills")
        if ia_number_focus:
            st.session_state.ia_number_focus = ia_number_focus

    st.divider()

    # --- Refinement toggle + model selector ---
    refine_col, model_col = st.columns([1, 1])
    with refine_col:
        ia_enable_refine = st.toggle(
            "Refine transcript before generating",
            value=st.session_state.ia_refine_enabled,
            key="ia_refine_toggle",
            help="Chunks the transcript and extracts structured Q&A from each chunk before generating. Improves output quality for long or messy transcripts.",
        )
    st.session_state.ia_refine_enabled = ia_enable_refine
    with model_col:
        _ia_model_keys = list(AVAILABLE_MODELS.keys())
        _ia_model_default = state.notes_model if state.notes_model in AVAILABLE_MODELS else _ia_model_keys[0]
        ia_model_name = st.selectbox(
            "Model",
            _ia_model_keys,
            index=_ia_model_keys.index(_ia_model_default),
            key="ia_model_select",
        )

    # --- Generate ---
    if st.button("Generate Investment Analysis", type="primary", use_container_width=True, key="ia_generate_btn"):
        try:
            model = _get_cached_model(ia_model_name)
            transcript_for_generation = st.session_state.ia_transcript
            st.session_state.ia_refined_transcript = ""

            # --- Optional refinement: chunk and extract Q&A ---
            if ia_enable_refine:
                chunks = [c["text"] for c in create_chunks_with_context(st.session_state.ia_transcript, state.chunk_word_size, 0)]
                total_chunks = len(chunks)
                refined_parts = [None] * total_chunks

                with st.spinner(f"Refining transcript ({total_chunks} chunk{'s' if total_chunks > 1 else ''})..."):
                    def _refine_ia_chunk(idx, chunk):
                        prompt = IA_REFINE_CHUNK_PROMPT.format(
                            chunk_num=idx + 1,
                            total_chunks=total_chunks,
                            chunk=chunk,
                        )
                        resp = generate_with_retry(model, prompt, generation_config=GENERATION_CONFIG)
                        return idx, resp.text

                    with ThreadPoolExecutor(max_workers=min(3, total_chunks)) as executor:
                        futures = {executor.submit(_refine_ia_chunk, i, c): i for i, c in enumerate(chunks)}
                        for future in as_completed(futures):
                            idx, text = future.result()
                            refined_parts[idx] = text

                transcript_for_generation = "\n\n---\n\n".join(p for p in refined_parts if p)
                st.session_state.ia_refined_transcript = transcript_for_generation

            with st.spinner("Generating key takeaways and rough notes…"):
                prompt_template = st.session_state.ia_prompt_text
                if "{transcript}" in prompt_template:
                    prompt = prompt_template.format(transcript=transcript_for_generation)
                else:
                    prompt = prompt_template + "\n\n---\nTRANSCRIPT:\n" + transcript_for_generation

                # Inject tone and data emphasis as additional instructions
                _ia_tone = st.session_state.get("ia_tone", "Neutral") or "Neutral"
                _ia_number_focus = st.session_state.get("ia_number_focus", "Moderate") or "Moderate"
                _ia_tone_descriptions = {
                    "As Is": "Present findings exactly as stated in the transcript. Do not add any positive or negative framing — reproduce the sentiment already present in the source material.",
                    "Very Positive": "Frame KEY TAKEAWAYS constructively — strengths, growth, advantages. Challenges are temporary or manageable.",
                    "Positive": "Frame KEY TAKEAWAYS positively. Risks acknowledged but opportunities emphasized.",
                    "Neutral": "Present KEY TAKEAWAYS objectively and balanced.",
                    "Negative": "Emphasize risks and structural problems in KEY TAKEAWAYS. Positive developments are insufficient.",
                    "Very Negative": "Frame KEY TAKEAWAYS around fundamental weaknesses and unsustainable practices. Deeply problematic framing.",
                }
                _addendum = []
                if _ia_tone != "Neutral":
                    _tone_desc = _ia_tone_descriptions.get(_ia_tone, "")
                    if _tone_desc:
                        _addendum.append(f"TONE FOR KEY TAKEAWAYS: {_tone_desc}")
                _number_instruction = NUMBER_FOCUS_INSTRUCTIONS.get(_ia_number_focus, "")
                if _ia_number_focus != "Moderate" and _number_instruction:
                    _addendum.append(f"DATA EMPHASIS FOR ROUGH NOTES: {_number_instruction}")
                if _addendum:
                    prompt += "\n\n---\nADDITIONAL FORMATTING INSTRUCTIONS:\n" + "\n".join(_addendum)

                response = generate_with_retry(model, prompt, generation_config=GENERATION_CONFIG)
                st.session_state.ia_output = response.text
                st.rerun()
        except Exception as e:
            st.error(f"Failed to generate analysis: {e}")

    # --- Display output ---
    if st.session_state.ia_output:
        st.divider()
        if st.session_state.ia_refined_transcript:
            with st.expander("View refined Q&A transcript (intermediate step)", expanded=False):
                st.markdown(st.session_state.ia_refined_transcript)

        raw = st.session_state.ia_output

        # Split into KTA and Rough Notes sections
        kta_text = raw
        rough_text = ""
        raw_upper = raw.upper()
        # Try known markers first (ordered from most specific to least)
        for marker in (
            "OUTPUT 2: ROUGH NOTES",
            "OUTPUT 2:",
            "ROUGH NOTES",
            "MEETING NOTES",
            "RAW NOTES",
            "DETAILED NOTES",
        ):
            idx = raw_upper.find(marker)
            if idx != -1:
                kta_text = raw[:idx].strip()
                rough_text = raw[idx:].strip()
                break

        # Fallback: if no named marker found, try splitting on a "---" divider
        # that appears after meaningful KTA content (skip the first --- in prompts)
        if not rough_text:
            divider = "---"
            search_start = 0
            while True:
                div_idx = raw.find(divider, search_start)
                if div_idx == -1:
                    break
                candidate_kta = raw[:div_idx].strip()
                candidate_rough = raw[div_idx + len(divider):].strip()
                # Accept split only if both halves have substantial content
                if len(candidate_kta) > 50 and len(candidate_rough) > 50:
                    kta_text = candidate_kta
                    rough_text = candidate_rough
                    break
                search_start = div_idx + len(divider)

        # Build dynamic headings
        _co = st.session_state.ia_company_name.strip()
        _ar = st.session_state.ia_area.strip()
        if st.session_state.ia_meeting_type == "management":
            _kta_heading = f"KTAs — Management of {_co}" if _co else "Key Investment Takeaways — Management Meeting"
            _rough_heading = f"Meeting Notes — {_co} Management" if _co else "Rough Notes — Management Meeting"
        else:
            _kta_heading = (
                f"KTAs — Expert on {_co}" + (f" | {_ar}" if _ar else "")
                if _co else
                "Key Investment Takeaways — Expert Meeting"
            )
            _rough_heading = (
                f"Meeting Notes — Expert on {_co}" + (f" ({_ar})" if _ar else "")
                if _co else
                "Rough Notes — Expert Meeting"
            )

        col_kta, col_rough = st.columns(2, gap="large")

        with col_kta:
            st.markdown(f"### {_kta_heading}")
            with st.container(border=True):
                st.markdown(kta_text)
            copy_to_clipboard_button(kta_text, "Copy KTAs")
            st.download_button(
                "Download KTAs (.txt)",
                data=kta_text,
                file_name="Key_Investment_Takeaways.txt",
                mime="text/plain",
                use_container_width=True,
                key="ia_dl_kta",
            )

        with col_rough:
            st.markdown(f"### {_rough_heading}")
            with st.container(border=True):
                st.markdown(rough_text if rough_text else raw)
            copy_to_clipboard_button(rough_text if rough_text else raw, "Copy Rough Notes")
            st.download_button(
                "Download Rough Notes (.txt)",
                data=rough_text if rough_text else raw,
                file_name="Rough_Notes.txt",
                mime="text/plain",
                use_container_width=True,
                key="ia_dl_rough",
            )

        st.divider()
        copy_to_clipboard_button(raw, "Copy Full Output")
        st.download_button(
            "Download Full Output (.txt)",
            data=raw,
            file_name="Investment_Analysis_Full.txt",
            mime="text/plain",
            use_container_width=True,
            key="ia_dl_full",
        )


def render_otg_notes_tab(state: AppState):
    st.subheader("OTG Notes")

    # --- Top-level mode selector ---
    otg_mode = st.pills(
        "Mode",
        ["Research Style", "Investment Analyst"],
        default="Research Style",
        key="otg_mode_pills",
    )

    st.divider()

    if otg_mode == "Investment Analyst":
        render_ia_processing(state)
        return

    st.caption("Paste detailed meeting notes to convert them into concise, narrative-style research notes. Select entities, topics, tone, and data emphasis to control the output.")

    # --- OTG State init ---
    if "otg_input" not in st.session_state:
        st.session_state.otg_input = ""
    if "otg_extracted" not in st.session_state:
        st.session_state.otg_extracted = None
    if "otg_output" not in st.session_state:
        st.session_state.otg_output = ""
    if "otg_selected_topics" not in st.session_state:
        st.session_state.otg_selected_topics = []
    if "otg_selected_entities" not in st.session_state:
        st.session_state.otg_selected_entities = []
    if "otg_refine_enabled" not in st.session_state:
        st.session_state.otg_refine_enabled = False
    if "otg_refined_notes" not in st.session_state:
        st.session_state.otg_refined_notes = ""

    # --- Input: paste notes or load from existing ---
    input_source = st.pills("Source", ["Paste Notes", "From Saved Note"], default="Paste Notes", key="otg_source_pills")

    if input_source == "Paste Notes":
        st.session_state.otg_input = st.text_area(
            "Paste your detailed notes here:",
            value=st.session_state.otg_input,
            height=300,
            key="otg_paste_input"
        )
    else:
        notes = database.get_all_notes()
        if not notes:
            st.info("No saved notes. Generate notes first in the Input & Generate tab.")
            return
        note_labels = []
        note_id_by_label = {}
        for n in notes:
            label = n['file_name']
            # Disambiguate duplicate filenames by appending the date
            if label in note_id_by_label:
                created = datetime.fromisoformat(n['created_at']).strftime('%b %d %H:%M')
                label = f"{label} ({created})"
            note_labels.append(label)
            note_id_by_label[label] = n['id']
        selected_name = st.selectbox("Select a saved note", note_labels, key="otg_note_selector")
        if selected_name:
            selected_note = database.get_note_by_id(note_id_by_label[selected_name])
            if selected_note:
                st.session_state.otg_input = selected_note.get('content', '')
                with st.expander("Preview loaded notes", expanded=False):
                    st.markdown(st.session_state.otg_input[:2000] + ("..." if len(st.session_state.otg_input) > 2000 else ""))

    if not st.session_state.otg_input.strip():
        st.info("Paste notes above or load a saved note to get started.")
        return

    # Word count for OTG input
    otg_wc = len(st.session_state.otg_input.split())
    st.caption(f"{otg_wc:,} words in source notes")

    # --- Step 1: Extract entities, sector, topics ---
    st.divider()

    if st.button("Analyze Notes", use_container_width=True, key="otg_analyze_btn"):
        with st.spinner("Extracting entities, sector, and topics..."):
            try:
                extract_model = _get_cached_model(state.notes_model)
                prompt = OTG_EXTRACT_PROMPT.format(notes=st.session_state.otg_input)
                response = generate_with_retry(extract_model, prompt)
                raw_json = response.text.strip()
                # Strip markdown code fences if present (handle ```json and ``` variants)
                if raw_json.startswith("```"):
                    lines = raw_json.split("\n")
                    # Remove first line (```json or ```)
                    lines = lines[1:]
                    # Remove last line if it's just ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_json = "\n".join(lines).strip()

                # Try to find JSON object if there's extra text
                json_match = re.search(r'\{[\s\S]*\}', raw_json)
                if json_match:
                    raw_json = json_match.group(0)

                extracted = json.loads(raw_json)

                # Validate and normalize the extracted data
                if not isinstance(extracted, dict):
                    extracted = {}
                extracted.setdefault("entities", [])
                extracted.setdefault("people", [])
                extracted.setdefault("sector", "Unknown")
                extracted.setdefault("topics", [])

                # Ensure lists contain strings only
                extracted["entities"] = [str(e) for e in extracted["entities"] if e]
                extracted["people"] = [str(p) for p in extracted["people"] if p]
                extracted["topics"] = [str(t) for t in extracted["topics"] if t]

                st.session_state.otg_extracted = extracted
                st.session_state.otg_selected_topics = extracted.get("topics", [])
                st.session_state.otg_selected_entities = extracted.get("entities", [])
                st.session_state.otg_output = ""
                st.session_state.otg_refined_notes = ""
                st.rerun()
            except json.JSONDecodeError as je:
                st.error(f"Failed to parse analysis results. The model returned invalid JSON. Try again or use a different model.")
            except Exception as e:
                st.error(f"Failed to analyze notes: {e}")

    extracted = st.session_state.otg_extracted
    if not extracted:
        return

    # --- Display sector ---
    sector = extracted.get("sector", "Unknown")
    st.markdown(f"**Sector:** {sector}")

    st.divider()

    # --- Entity selection (use stable keys based on entity name hash) ---
    entities = extracted.get("entities", [])
    people = extracted.get("people", [])
    all_entity_names = list(dict.fromkeys(entities + people))  # Remove duplicates while preserving order
    if all_entity_names:
        st.markdown("**Select entities to focus on:**")
        selected_entities = st.multiselect(
            "Entities",
            options=all_entity_names,
            default=[e for e in st.session_state.otg_selected_entities if e in all_entity_names],
            key="otg_entity_multiselect",
            label_visibility="collapsed",
            accept_new_options=True,
            placeholder="Select or type to add new entities",
        )
        st.session_state.otg_selected_entities = selected_entities

    st.divider()

    # --- Topic selection ---
    topics = extracted.get("topics", [])
    if topics:
        st.markdown("**Select topics to focus on:**")
        selected_topics = st.multiselect(
            "Topics",
            options=topics,
            default=[t for t in st.session_state.otg_selected_topics if t in topics],
            key="otg_topic_multiselect",
            label_visibility="collapsed",
            accept_new_options=True,
            placeholder="Select or type to add new topics",
        )
        st.session_state.otg_selected_topics = selected_topics

    # --- Tone, Number Focus, and Length ---
    st.divider()
    tone_col, number_col = st.columns(2)
    with tone_col:
        tone = st.pills("Tone", TONE_OPTIONS, default="Neutral", key="otg_tone_pills")
    with number_col:
        number_focus = st.pills("Data Emphasis", NUMBER_FOCUS_OPTIONS, default="Moderate", key="otg_number_pills")

    word_count_options = list(OTG_WORD_COUNT_OPTIONS.keys())
    selected_word_count = st.select_slider(
        "Approximate Output Length",
        options=word_count_options,
        value=word_count_options[1],
        key="otg_word_count_slider",
    )

    # --- Custom instructions ---
    custom_instructions = st.text_area(
        "Additional Instructions (Optional)",
        placeholder="e.g., Emphasize competitive positioning vs Blinkit, keep the note under 200 words, mention the IPO timeline...",
        height=80,
        key="otg_custom_instructions"
    )

    # --- Generate OTG note ---
    st.divider()

    if not st.session_state.otg_selected_topics:
        st.warning("Select at least one topic to focus on.")
        return

    if not st.session_state.otg_selected_entities and all_entity_names:
        st.warning("Select at least one entity to focus on.")
        return

    # --- Refinement toggle ---
    refine_col, _ = st.columns([1, 2])
    with refine_col:
        enable_refine = st.toggle(
            "Refine notes before generating",
            value=st.session_state.otg_refine_enabled,
            key="otg_refine_toggle",
            help="Chunks the source notes and extracts structured Q&A from each chunk before generating the final note. Improves output quality for long or unstructured notes.",
        )
    st.session_state.otg_refine_enabled = enable_refine

    if st.button("Generate Research Note", type="primary", use_container_width=True, key="otg_generate_btn"):
        try:
            otg_model = _get_cached_model(state.notes_model)
            notes_for_generation = st.session_state.otg_input
            st.session_state.otg_refined_notes = ""

            # --- Optional refinement: chunk and extract Q&A ---
            if enable_refine:
                chunks = [c["text"] for c in create_chunks_with_context(st.session_state.otg_input, state.chunk_word_size, 0)]
                total_chunks = len(chunks)
                refined_parts = [None] * total_chunks

                with st.spinner(f"Refining notes ({total_chunks} chunk{'s' if total_chunks > 1 else ''})..."):
                    def _refine_otg_chunk(idx, chunk):
                        prompt = OTG_REFINE_CHUNK_PROMPT.format(
                            chunk_num=idx + 1,
                            total_chunks=total_chunks,
                            chunk=chunk,
                        )
                        resp = generate_with_retry(otg_model, prompt, generation_config=GENERATION_CONFIG)
                        return idx, resp.text

                    with ThreadPoolExecutor(max_workers=min(3, total_chunks)) as executor:
                        futures = {executor.submit(_refine_otg_chunk, i, c): i for i, c in enumerate(chunks)}
                        for future in as_completed(futures):
                            idx, text = future.result()
                            refined_parts[idx] = text

                notes_for_generation = "\n\n---\n\n".join(p for p in refined_parts if p)
                st.session_state.otg_refined_notes = notes_for_generation

            with st.spinner("Generating research note..."):
                topics_str = ", ".join(st.session_state.otg_selected_topics)
                entities_str = ", ".join(st.session_state.otg_selected_entities) if st.session_state.otg_selected_entities else "all entities mentioned"
                number_instruction = NUMBER_FOCUS_INSTRUCTIONS.get(number_focus, NUMBER_FOCUS_INSTRUCTIONS["Moderate"])
                length_instruction = OTG_WORD_COUNT_OPTIONS.get(selected_word_count, OTG_WORD_COUNT_OPTIONS["Medium (~300 words)"])
                custom_block = f"9. ADDITIONAL INSTRUCTIONS FROM THE ANALYST: {custom_instructions}" if custom_instructions.strip() else ""
                prompt = OTG_CONVERT_PROMPT.format(
                    tone=tone,
                    topics=topics_str,
                    entities=entities_str,
                    number_focus_instruction=number_instruction,
                    length_instruction=length_instruction,
                    custom_instructions_block=custom_block,
                    notes=notes_for_generation,
                )
                response = generate_with_retry(otg_model, prompt)
                st.session_state.otg_output = response.text
                st.rerun()
        except Exception as e:
            st.error(f"Failed to generate research note: {e}")

    # --- Display output ---
    if st.session_state.otg_output:
        st.divider()
        if st.session_state.otg_refined_notes:
            with st.expander("View refined Q&A notes (intermediate step)", expanded=False):
                st.markdown(st.session_state.otg_refined_notes)
        st.markdown("### Generated Research Note")
        with st.container(border=True):
            st.markdown(st.session_state.otg_output)

        otg_sector_slug = sector.replace(' ', '_')
        out1, out2, out3 = st.columns(3)
        with out1:
            copy_to_clipboard_button(st.session_state.otg_output, "Copy Research Note")
        out2.download_button(
            label="Download (.txt)",
            data=st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.txt",
            mime="text/plain",
            use_container_width=True
        )
        out3.download_button(
            label="Download (.md)",
            data=st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- EARNINGS CALL MULTI-FILE ANALYSIS ---

def _extract_pdf_texts(uploaded_files: list) -> List[Tuple[str, str]]:
    """Extract text from multiple uploaded PDF files.
    Returns list of (filename, text_content) tuples. Skips files with errors.
    """
    results = []
    for f in uploaded_files:
        name = f.name
        try:
            file_bytes_io = io.BytesIO(f.getvalue())
            reader = PyPDF2.PdfReader(file_bytes_io)
            if reader.is_encrypted:
                st.warning(f"Skipping encrypted PDF: {name}")
                continue
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            if not content or not content.strip():
                st.warning(f"No text found in: {name}")
                continue
            content = re.sub(r'\n{3,}', '\n\n', content.strip())
            results.append((name, content))
        except Exception as e:
            st.warning(f"Error reading {name}: {e}")
    return results


def _discover_topics(file_texts: List[Tuple[str, str]], model_name: str) -> dict:
    """Send first N transcripts to Gemini for topic discovery. Returns parsed JSON."""
    discovery_files = file_texts[:MAX_TOPIC_DISCOVERY_FILES]

    # Build combined transcript text with file labels
    transcript_parts = []
    for i, (fname, text) in enumerate(discovery_files, 1):
        # Limit each transcript to ~15k words to avoid context overflow
        words = text.split()
        truncated = " ".join(words[:15000])
        transcript_parts.append(f"--- TRANSCRIPT {i}: {fname} ---\n{truncated}")

    combined = "\n\n".join(transcript_parts)
    prompt = EC_TOPIC_DISCOVERY_PROMPT.format(transcripts=combined)

    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    raw_json = response.text.strip()

    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines).strip()

    json_match = re.search(r'\{[\s\S]*\}', raw_json)
    if json_match:
        raw_json = json_match.group(0)

    result = json.loads(raw_json)
    if not isinstance(result, dict):
        raise ValueError("Topic discovery returned invalid format.")

    result.setdefault("company_name", "Unknown Company")
    result.setdefault("primary_topics", [])
    result.setdefault("cross_cutting_topics", [])

    return result


def _build_topic_structure_text(selected_topics: dict) -> str:
    """Convert selected topics dict into a text structure for the notes prompt."""
    lines = []
    for primary in selected_topics.get("primary_topics", []):
        lines.append(f"**{primary['name']}**")
        if primary.get("description"):
            lines.append(f"  ({primary['description']})")
        for sub in primary.get("sub_topics", []):
            lines.append(f"  - {sub}")
        lines.append("")

    for cross in selected_topics.get("cross_cutting_topics", []):
        lines.append(f"**{cross['name']}**")
        if cross.get("description"):
            lines.append(f"  ({cross['description']})")
        lines.append("")

    return "\n".join(lines)


def _generate_notes_for_file(file_label: str, transcript: str, topic_structure_text: str,
                              model_name: str, on_progress=None) -> Tuple[str, int]:
    """Generate earnings call notes for a single file under the given topic structure.
    Returns (notes_text, token_count).
    """
    prompt = EC_MULTI_FILE_NOTES_PROMPT.format(
        topic_structure=topic_structure_text,
        file_label=file_label,
        transcript=transcript
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response, on_progress=on_progress)
    return full_text, token_count


def _stitch_multi_file_notes(company_name: str, file_notes: List[Tuple[str, str]]) -> str:
    """Stitch together notes from multiple files into a single output."""
    header = EC_MULTI_FILE_STITCH_HEADER.format(
        company_name=company_name,
        date=datetime.now().strftime("%B %d, %Y"),
        file_count=len(file_notes)
    )

    parts = [header]
    for i, (fname, notes) in enumerate(file_notes, 1):
        parts.append(f"## {i}. {fname}\n")
        parts.append(notes.strip())
        parts.append("\n\n---\n")

    return "\n".join(parts)


# --- REPORT COMPARISON HELPER FUNCTIONS ---

MAX_RC_DISCOVERY_FILES = 4  # Number of reports to scan for dimension discovery

def _discover_rc_dimensions(file_texts: List[Tuple[str, str]], model_name: str) -> dict:
    """Send reports to Gemini for qualitative dimension discovery. Returns parsed JSON."""
    discovery_files = file_texts[:MAX_RC_DISCOVERY_FILES]

    report_parts = []
    for i, (fname, text) in enumerate(discovery_files, 1):
        words = text.split()
        truncated = " ".join(words[:15000])
        report_parts.append(f"--- REPORT {i}: {fname} ---\n{truncated}")

    combined = "\n\n".join(report_parts)
    prompt = RC_DIMENSION_DISCOVERY_PROMPT.format(reports=combined)

    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    raw_json = response.text.strip()

    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines).strip()

    json_match = re.search(r'\{[\s\S]*\}', raw_json)
    if json_match:
        raw_json = json_match.group(0)

    result = json.loads(raw_json)
    if not isinstance(result, dict):
        raise ValueError("Dimension discovery returned invalid format.")

    result.setdefault("company_name", "Unknown Company")
    result.setdefault("report_years", [])
    result.setdefault("comparison_dimensions", [])

    return result


def _build_dimension_structure_text(selected_dimensions: dict) -> str:
    """Convert selected dimensions dict into a text structure for prompts."""
    lines = []
    for dim in selected_dimensions.get("comparison_dimensions", []):
        lines.append(f"**{dim['name']}**")
        if dim.get("description"):
            lines.append(f"  ({dim['description']})")
        for sub in dim.get("sub_dimensions", []):
            lines.append(f"  - {sub}")
        lines.append("")
    return "\n".join(lines)


def _extract_report_qualitative(file_label: str, report_text: str, dimension_structure_text: str,
                                 model_name: str, on_progress=None) -> Tuple[str, int]:
    """Extract qualitative data from a single report for the given dimensions.
    Returns (extraction_text, token_count).
    """
    prompt = RC_PER_REPORT_EXTRACTION_PROMPT.format(
        dimension_structure=dimension_structure_text,
        file_label=file_label,
        report_text=report_text
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response, on_progress=on_progress)
    return full_text, token_count


def _generate_rc_comparison(company_name: str, report_labels: str, dimension_structure_text: str,
                             per_report_extractions: str, model_name: str) -> Tuple[str, int]:
    """Generate the final year-over-year comparison from all per-report extractions.
    Returns (comparison_text, token_count).
    """
    prompt = RC_COMPARISON_PROMPT.format(
        company_name=company_name,
        report_labels=report_labels,
        dimension_structure=dimension_structure_text,
        per_report_extractions=per_report_extractions
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response)
    return full_text, token_count


def _stitch_rc_output(company_name: str, report_labels: str, comparison_text: str,
                       per_report_extractions: List[Tuple[str, str]]) -> str:
    """Stitch the comparison output with header and optional per-report appendix."""
    header = RC_STITCH_HEADER.format(
        company_name=company_name,
        date=datetime.now().strftime("%B %d, %Y"),
        report_labels=report_labels
    )
    parts = [header, comparison_text.strip()]

    # Add per-report extractions as appendix
    parts.append("\n\n---\n\n# Appendix: Per-Report Extractions\n")
    for i, (fname, extraction) in enumerate(per_report_extractions, 1):
        parts.append(f"## {i}. {fname}\n")
        parts.append(extraction.strip())
        parts.append("\n\n---\n")

    return "\n".join(parts)


# --- REPORT COMPARISON TAB ---

def render_report_comparison_tab(state: AppState):
    st.subheader("Annual Report Comparison")
    st.caption(
        "Upload annual reports (PDFs) from different years for the same company. "
        "The system will identify qualitative dimensions — management commentary, strategy, "
        "org structure, incentives, risk factors, etc. — and produce a year-over-year comparison. "
        "Numbers and financial data are intentionally de-emphasized; the focus is on narrative shifts."
    )

    # --- Session state initialization ---
    if "rc_files" not in st.session_state:
        st.session_state.rc_files = None
    if "rc_texts" not in st.session_state:
        st.session_state.rc_texts = []
    if "rc_discovered_dims" not in st.session_state:
        st.session_state.rc_discovered_dims = None
    if "rc_selected_dims" not in st.session_state:
        st.session_state.rc_selected_dims = None
    if "rc_comparison_output" not in st.session_state:
        st.session_state.rc_comparison_output = ""
    if "rc_processing" not in st.session_state:
        st.session_state.rc_processing = False
    if "rc_per_report_extractions" not in st.session_state:
        st.session_state.rc_per_report_extractions = []

    # --- Step 1: Upload PDFs ---
    st.markdown("### Step 1: Upload Annual Reports")
    uploaded_files = st.file_uploader(
        "Upload annual report PDFs (one per year)",
        type=["pdf"],
        accept_multiple_files=True,
        key="rc_pdf_uploader",
        help="Upload 2 or more annual report PDFs from different years for the same company."
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 annual report PDFs to begin. Name files clearly (e.g., 'CompanyName_AR_2023.pdf').")
        if not uploaded_files:
            st.session_state.rc_texts = []
            st.session_state.rc_discovered_dims = None
            st.session_state.rc_selected_dims = None
            st.session_state.rc_comparison_output = ""
            st.session_state.rc_per_report_extractions = []
        return

    file_names = [f.name for f in uploaded_files]
    st.caption(f"**{len(uploaded_files)}** reports uploaded: {', '.join(file_names)}")

    # --- Step 2: Extract text and discover dimensions ---
    st.divider()
    st.markdown("### Step 2: Identify Comparison Dimensions")
    st.caption(
        f"Analyzes the first {min(len(uploaded_files), MAX_RC_DISCOVERY_FILES)} reports "
        "to identify qualitative dimensions for comparison (strategy, governance, incentives, etc.)."
    )

    analysis_model = st.selectbox(
        "Model for Dimension Discovery",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="rc_analysis_model_select"
    )

    if st.button("Identify Dimensions", use_container_width=True, key="rc_discover_btn"):
        with st.spinner("Extracting text from PDFs and identifying comparison dimensions..."):
            try:
                file_texts = _extract_pdf_texts(uploaded_files)
                if len(file_texts) < 2:
                    st.error("Could not extract text from at least 2 PDFs. Check file quality.")
                    return
                st.session_state.rc_texts = file_texts

                discovered = _discover_rc_dimensions(file_texts, analysis_model)
                st.session_state.rc_discovered_dims = discovered
                st.session_state.rc_selected_dims = copy.deepcopy(discovered)
                st.session_state.rc_comparison_output = ""
                st.session_state.rc_per_report_extractions = []
                st.rerun()
            except json.JSONDecodeError:
                st.error("Failed to parse dimension discovery results. Try again or use a different model.")
            except Exception as e:
                st.error(f"Dimension discovery failed: {e}")

    # --- Step 3: Display and select dimensions ---
    discovered = st.session_state.rc_discovered_dims
    if not discovered:
        return

    company_name = discovered.get("company_name", "Unknown Company")
    report_years = discovered.get("report_years", [])
    years_display = ", ".join(report_years) if report_years else "multiple years"
    st.success(f"Dimensions identified for **{company_name}** ({years_display})")

    st.divider()
    st.markdown("### Step 3: Select & Customize Dimensions")
    st.caption("Check the dimensions you want compared across years. You can also add custom dimensions.")

    comparison_dims = discovered.get("comparison_dimensions", [])

    selected_dims = []
    for d_idx, dim in enumerate(comparison_dims):
        d_name = dim.get("name", f"Dimension {d_idx+1}")
        d_desc = dim.get("description", "")
        sub_dims = dim.get("sub_dimensions", [])

        d_key = f"rc_dim_{d_idx}"
        d_enabled = st.checkbox(
            f"**{d_name}**" + (f" — {d_desc}" if d_desc else ""),
            value=True,
            key=d_key
        )

        if d_enabled:
            selected_subs = []
            sub_cols = st.columns(min(len(sub_dims), 4)) if sub_dims else []
            for s_idx, sub in enumerate(sub_dims):
                col = sub_cols[s_idx % len(sub_cols)] if sub_cols else st
                s_key = f"rc_sub_{d_idx}_{s_idx}"
                if col.checkbox(sub, value=True, key=s_key):
                    selected_subs.append(sub)

            # Add custom sub-dimension
            custom_sub = st.text_input(
                "Add custom sub-dimension",
                key=f"rc_custom_sub_{d_idx}",
                placeholder="e.g., succession planning, digital strategy..."
            )
            if custom_sub and custom_sub.strip():
                for cs in [s.strip() for s in custom_sub.split(",") if s.strip()]:
                    if cs not in selected_subs:
                        selected_subs.append(cs)

            if selected_subs:
                selected_dims.append({
                    "name": d_name,
                    "description": d_desc,
                    "sub_dimensions": selected_subs
                })

        st.divider()

    # Add custom dimension
    with st.expander("Add Custom Dimension", expanded=False):
        custom_d_name = st.text_input("Dimension Name", key="rc_custom_dim_name",
                                       placeholder="e.g., Regulatory Environment")
        custom_d_desc = st.text_input("Description (optional)", key="rc_custom_dim_desc")
        custom_d_subs = st.text_input("Sub-dimensions (comma-separated)", key="rc_custom_dim_subs",
                                       placeholder="e.g., compliance changes, new regulations, policy shifts")
        if custom_d_name and custom_d_name.strip():
            subs = [s.strip() for s in custom_d_subs.split(",") if s.strip()] if custom_d_subs else []
            selected_dims.append({
                "name": custom_d_name.strip(),
                "description": custom_d_desc.strip() if custom_d_desc else "",
                "sub_dimensions": subs
            })

    # Store final selection
    final_selection = {
        "company_name": company_name,
        "report_years": report_years,
        "comparison_dimensions": selected_dims
    }
    st.session_state.rc_selected_dims = final_selection

    # Preview selected structure
    with st.expander("Preview Selected Dimension Structure", expanded=False):
        structure_text = _build_dimension_structure_text(final_selection)
        st.code(structure_text, language="markdown")

    if not selected_dims:
        st.warning("Select at least one dimension to generate comparison.")
        return

    # --- Step 4: Generate comparison ---
    st.divider()
    st.markdown("### Step 4: Generate Comparison")

    file_texts = st.session_state.rc_texts
    if not file_texts:
        st.warning("Report texts not available. Please re-run dimension identification.")
        return

    st.caption(
        f"Will extract qualitative data from **{len(file_texts)}** reports, "
        "then produce a year-over-year comparison across selected dimensions."
    )

    notes_model = st.selectbox(
        "Model for Analysis",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="rc_notes_model_select"
    )

    if st.button("Generate Comparison", type="primary", use_container_width=True, key="rc_generate_btn"):
        st.session_state.rc_processing = True
        st.rerun()

    if st.session_state.rc_processing:
        dimension_structure_text = _build_dimension_structure_text(final_selection)
        total_tokens = 0
        all_extractions = []
        start_time = time.time()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Phase 1: Extract qualitative data from each report
            num_files = len(file_texts)
            for i, (fname, text) in enumerate(file_texts):
                # Extraction is ~70% of the work, comparison is ~30%
                base = (i / num_files) * 0.7
                progress_bar.progress(base)
                status_text.markdown(
                    f"**{int(base*100)}%** — Extracting qualitative data from **{fname}** ({i+1}/{num_files})"
                )

                expected = max(200, int(len(text.split()) * NOTES_OUTPUT_RATIO))

                def _on_stream(words, base=base, fname=fname, i=i, expected=expected):
                    frac = base + (min(words / expected, 0.95) / num_files) * 0.7
                    progress_bar.progress(min(frac, 1.0))
                    status_text.markdown(
                        f"**{int(frac*100)}%** — Extracting qualitative data from **{fname}** "
                        f"({i+1}/{num_files}) · {words:,} words generated"
                    )

                extraction_text, tokens = _extract_report_qualitative(
                    fname, text, dimension_structure_text, notes_model, on_progress=_on_stream
                )
                total_tokens += tokens
                all_extractions.append((fname, extraction_text))

            # Phase 2: Generate comparison
            progress_bar.progress(0.75)
            status_text.markdown("**75%** — Generating year-over-year comparison...")

            report_labels = ", ".join(fname for fname, _ in file_texts)
            per_report_combined = "\n\n---\n\n".join(
                f"### Report: {fname}\n\n{extraction}"
                for fname, extraction in all_extractions
            )

            comparison_text, comp_tokens = _generate_rc_comparison(
                company_name, report_labels, dimension_structure_text,
                per_report_combined, notes_model
            )
            total_tokens += comp_tokens

            # Stitch final output
            progress_bar.progress(0.95)
            status_text.markdown("**95%** — Assembling final output...")

            stitched = _stitch_rc_output(company_name, report_labels, comparison_text, all_extractions)

            st.session_state.rc_comparison_output = stitched
            st.session_state.rc_per_report_extractions = all_extractions

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.markdown(
                f"**100%** — Done! {len(file_texts)} reports compared, "
                f"{total_tokens:,} tokens used, {elapsed:.1f}s elapsed."
            )
            st.toast("Report comparison complete!", icon="\u2705")

            send_browser_notification(
                "SynthNotes AI - Report Comparison Complete",
                f"{len(file_texts)} annual reports compared in {elapsed:.1f}s"
            )

        except Exception as e:
            status_text.markdown(f"**Error:** {e}")
            st.error(f"Comparison generation failed: {e}")
        finally:
            st.session_state.rc_processing = False

    # --- Step 5: Display output ---
    if st.session_state.rc_comparison_output:
        st.divider()
        st.markdown("### Output")

        view_mode = st.pills(
            "View", ["Comparison", "Per-Report Extractions"],
            default="Comparison", key="rc_view_mode"
        )

        if view_mode == "Comparison":
            with st.container(height=600, border=True):
                st.markdown(st.session_state.rc_comparison_output)
            note_wc = len(st.session_state.rc_comparison_output.split())
            st.caption(f"{note_wc:,} words")
        else:
            extractions = st.session_state.rc_per_report_extractions
            if extractions:
                tab_names = [fname for fname, _ in extractions]
                tabs = st.tabs(tab_names)
                for tab, (fname, extraction) in zip(tabs, extractions):
                    with tab:
                        with st.container(height=500, border=True):
                            st.markdown(extraction)
                        wc = len(extraction.split())
                        st.caption(f"{wc:,} words")

        # Actions bar
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            copy_to_clipboard_button(st.session_state.rc_comparison_output, "Copy Comparison")
        dl2.download_button(
            label="Download (.txt)",
            data=st.session_state.rc_comparison_output,
            file_name=f"Report_Comparison_{company_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        dl3.download_button(
            label="Download (.md)",
            data=st.session_state.rc_comparison_output,
            file_name=f"Report_Comparison_{company_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Save to database
        st.divider()
        if st.button("Save to Notes History", use_container_width=True, key="rc_save_btn"):
            try:
                note_data = {
                    'id': str(uuid.uuid4()),
                    'created_at': datetime.now().isoformat(),
                    'meeting_type': 'Report Comparison',
                    'file_name': f"Report Comparison — {company_name}",
                    'content': st.session_state.rc_comparison_output,
                    'raw_transcript': "\n\n---\n\n".join(
                        f"--- {fname} ---\n{text[:5000]}..." if len(text) > 5000 else f"--- {fname} ---\n{text}"
                        for fname, text in st.session_state.rc_texts
                    ),
                    'refined_transcript': None,
                    'token_usage': 0,
                    'processing_time': 0,
                    'pdf_blob': None
                }
                database.save_note(note_data)
                st.toast("Saved to Notes History!", icon="\u2705")
                st.session_state.app_state.active_note_id = note_data['id']
            except Exception as e:
                st.error(f"Failed to save: {e}")


def render_ec_analysis_tab(state: AppState):
    st.subheader("Multi-Transcript Earnings Call Analysis")
    st.caption(
        "Upload multiple earnings call PDFs. The system will identify key topics from the first "
        f"{MAX_TOPIC_DISCOVERY_FILES} transcripts, let you select and customize topics, "
        "then generate structured notes for every file."
    )

    # --- Session state initialization ---
    if "ec_analysis_files" not in st.session_state:
        st.session_state.ec_analysis_files = None
    if "ec_analysis_texts" not in st.session_state:
        st.session_state.ec_analysis_texts = []
    if "ec_discovered_topics" not in st.session_state:
        st.session_state.ec_discovered_topics = None
    if "ec_selected_topics" not in st.session_state:
        st.session_state.ec_selected_topics = None
    if "ec_analysis_output" not in st.session_state:
        st.session_state.ec_analysis_output = ""
    if "ec_analysis_processing" not in st.session_state:
        st.session_state.ec_analysis_processing = False
    if "ec_file_notes" not in st.session_state:
        st.session_state.ec_file_notes = []

    # --- Step 1: Upload multiple PDFs ---
    st.markdown("### Step 1: Upload Earnings Call Transcripts")
    uploaded_files = st.file_uploader(
        "Upload PDF transcripts",
        type=["pdf"],
        accept_multiple_files=True,
        key="ec_multi_pdf_uploader",
        help="Upload 2 or more earnings call transcript PDFs for the same company."
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 PDF transcripts to begin. For best topic discovery, upload 4 or more.")
        # Reset downstream state if files changed
        if not uploaded_files:
            st.session_state.ec_analysis_texts = []
            st.session_state.ec_discovered_topics = None
            st.session_state.ec_selected_topics = None
            st.session_state.ec_analysis_output = ""
            st.session_state.ec_file_notes = []
        return

    # Show file list
    file_names = [f.name for f in uploaded_files]
    st.caption(f"**{len(uploaded_files)}** files uploaded: {', '.join(file_names)}")

    # --- Step 2: Extract text and discover topics ---
    st.divider()
    st.markdown("### Step 2: Identify Topics")
    st.caption(
        f"Analyzes the first {min(len(uploaded_files), MAX_TOPIC_DISCOVERY_FILES)} transcripts "
        "to identify primary topics (brands, segments) and sub-topics (strategy, unit economics, etc.)."
    )

    # Model selection for analysis
    analysis_model = st.selectbox(
        "Model for Topic Discovery",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="ec_analysis_model_select"
    )

    if st.button("Identify Topics", use_container_width=True, key="ec_discover_btn"):
        with st.spinner("Extracting text from PDFs and identifying topics..."):
            try:
                # Extract texts
                file_texts = _extract_pdf_texts(uploaded_files)
                if len(file_texts) < 2:
                    st.error("Could not extract text from at least 2 PDFs. Check file quality.")
                    return
                st.session_state.ec_analysis_texts = file_texts

                # Discover topics
                discovered = _discover_topics(file_texts, analysis_model)
                st.session_state.ec_discovered_topics = discovered

                # Initialize selection with all topics selected
                st.session_state.ec_selected_topics = copy.deepcopy(discovered)
                st.session_state.ec_analysis_output = ""
                st.session_state.ec_file_notes = []
                st.rerun()
            except json.JSONDecodeError:
                st.error("Failed to parse topic discovery results. Try again or use a different model.")
            except Exception as e:
                st.error(f"Topic discovery failed: {e}")

    # --- Step 3: Display and select topics ---
    discovered = st.session_state.ec_discovered_topics
    if not discovered:
        return

    company_name = discovered.get("company_name", "Unknown Company")
    st.success(f"Topics identified for **{company_name}**")

    st.divider()
    st.markdown("### Step 3: Select & Customize Topics")
    st.caption("Check the topics you want included in the final notes. You can also add custom topics.")

    primary_topics = discovered.get("primary_topics", [])
    cross_cutting = discovered.get("cross_cutting_topics", [])

    # Build selected topics structure from user interaction
    selected_primary = []
    for p_idx, primary in enumerate(primary_topics):
        p_name = primary.get("name", f"Topic {p_idx+1}")
        p_desc = primary.get("description", "")
        sub_topics = primary.get("sub_topics", [])

        # Primary topic toggle
        p_key = f"ec_primary_{p_idx}"
        p_enabled = st.checkbox(
            f"**{p_name}**" + (f" — {p_desc}" if p_desc else ""),
            value=True,
            key=p_key
        )

        if p_enabled:
            # Sub-topic selection
            selected_subs = []
            sub_cols = st.columns(min(len(sub_topics), 4)) if sub_topics else []
            for s_idx, sub in enumerate(sub_topics):
                col = sub_cols[s_idx % len(sub_cols)] if sub_cols else st
                s_key = f"ec_sub_{p_idx}_{s_idx}"
                if col.checkbox(sub, value=True, key=s_key):
                    selected_subs.append(sub)

            # Add custom sub-topic
            custom_sub = st.text_input(
                "Add custom sub-topic",
                key=f"ec_custom_sub_{p_idx}",
                placeholder="e.g., digital transformation, new market entry..."
            )
            if custom_sub and custom_sub.strip():
                for cs in [s.strip() for s in custom_sub.split(",") if s.strip()]:
                    if cs not in selected_subs:
                        selected_subs.append(cs)

            if selected_subs:
                selected_primary.append({
                    "name": p_name,
                    "description": p_desc,
                    "sub_topics": selected_subs
                })

        st.divider()

    # Cross-cutting topics
    if cross_cutting:
        st.markdown("**Cross-Cutting Topics:**")
        selected_cross = []
        cross_cols = st.columns(min(len(cross_cutting), 4))
        for c_idx, cross in enumerate(cross_cutting):
            c_name = cross.get("name", f"Cross-topic {c_idx+1}")
            c_desc = cross.get("description", "")
            col = cross_cols[c_idx % len(cross_cols)]
            c_key = f"ec_cross_{c_idx}"
            if col.checkbox(c_name + (f" — {c_desc}" if c_desc else ""), value=True, key=c_key):
                selected_cross.append(cross)
        st.divider()
    else:
        selected_cross = []

    # Add custom primary topic
    with st.expander("Add Custom Primary Topic", expanded=False):
        custom_p_name = st.text_input("Primary Topic Name", key="ec_custom_primary_name",
                                       placeholder="e.g., New Business Segment")
        custom_p_desc = st.text_input("Description (optional)", key="ec_custom_primary_desc")
        custom_p_subs = st.text_input("Sub-topics (comma-separated)", key="ec_custom_primary_subs",
                                       placeholder="e.g., strategy, revenue, expansion")
        if custom_p_name and custom_p_name.strip():
            subs = [s.strip() for s in custom_p_subs.split(",") if s.strip()] if custom_p_subs else []
            selected_primary.append({
                "name": custom_p_name.strip(),
                "description": custom_p_desc.strip() if custom_p_desc else "",
                "sub_topics": subs
            })

    # Store final selection
    final_selection = {
        "company_name": company_name,
        "primary_topics": selected_primary,
        "cross_cutting_topics": selected_cross
    }
    st.session_state.ec_selected_topics = final_selection

    # Preview selected structure
    with st.expander("Preview Selected Topic Structure", expanded=False):
        structure_text = _build_topic_structure_text(final_selection)
        st.code(structure_text, language="markdown")

    if not selected_primary and not selected_cross:
        st.warning("Select at least one topic to generate notes.")
        return

    # --- Step 4: Generate notes for all files ---
    st.divider()
    st.markdown("### Step 4: Generate Notes")

    file_texts = st.session_state.ec_analysis_texts
    if not file_texts:
        st.warning("File texts not available. Please re-run topic identification.")
        return

    st.caption(f"Will generate structured notes for **{len(file_texts)}** transcripts under the selected topics.")

    # Model selection for note generation
    notes_model = st.selectbox(
        "Model for Notes Generation",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="ec_notes_model_select"
    )

    if st.button("Generate All Notes", type="primary", use_container_width=True, key="ec_generate_all_btn"):
        st.session_state.ec_analysis_processing = True
        st.rerun()

    if st.session_state.ec_analysis_processing:
        topic_structure_text = _build_topic_structure_text(final_selection)
        total_tokens = 0
        all_file_notes = []
        start_time = time.time()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            num_files = len(file_texts)
            for i, (fname, text) in enumerate(file_texts):
                base = i / num_files
                progress_bar.progress(base)
                status_text.markdown(f"**{int(base*100)}%** — Processing **{fname}** ({i+1}/{num_files})")

                expected = max(200, int(len(text.split()) * NOTES_OUTPUT_RATIO))

                def _on_stream(words, base=base, fname=fname, i=i, expected=expected):
                    frac = base + min(words / expected, 0.95) / num_files
                    progress_bar.progress(min(frac, 1.0))
                    status_text.markdown(
                        f"**{int(frac*100)}%** — Processing **{fname}** ({i+1}/{num_files}) · {words:,} words generated"
                    )

                notes_text, tokens = _generate_notes_for_file(
                    fname, text, topic_structure_text, notes_model, on_progress=_on_stream
                )
                total_tokens += tokens
                all_file_notes.append((fname, notes_text))

            progress_bar.progress(1.0)
            status_text.markdown("**100%** — Stitching notes together...")

            # Stitch
            stitched = _stitch_multi_file_notes(company_name, all_file_notes)

            st.session_state.ec_analysis_output = stitched
            st.session_state.ec_file_notes = all_file_notes

            elapsed = time.time() - start_time
            status_text.markdown(
                f"**100%** — Done! {len(file_texts)} files processed, "
                f"{total_tokens:,} tokens used, {elapsed:.1f}s elapsed."
            )
            st.toast("Earnings call analysis complete!", icon="\u2705")

            # Browser notification
            send_browser_notification(
                "SynthNotes AI - EC Analysis Complete",
                f"{len(file_texts)} transcripts analyzed in {elapsed:.1f}s"
            )

        except Exception as e:
            status_text.markdown(f"**Error:** {e}")
            st.error(f"Notes generation failed: {e}")
        finally:
            st.session_state.ec_analysis_processing = False

    # --- Step 5: Display output ---
    if st.session_state.ec_analysis_output:
        st.divider()
        st.markdown("### Output")

        # View mode
        view_mode = st.pills("View", ["Combined", "Per-File"], default="Combined", key="ec_view_mode")

        if view_mode == "Combined":
            with st.container(height=600, border=True):
                st.markdown(st.session_state.ec_analysis_output)
            note_wc = len(st.session_state.ec_analysis_output.split())
            st.caption(f"{note_wc:,} words")
        else:
            # Per-file tabs
            file_notes = st.session_state.ec_file_notes
            if file_notes:
                tab_names = [fname for fname, _ in file_notes]
                tabs = st.tabs(tab_names)
                for tab, (fname, notes) in zip(tabs, file_notes):
                    with tab:
                        with st.container(height=500, border=True):
                            st.markdown(notes)
                        wc = len(notes.split())
                        st.caption(f"{wc:,} words")

        # Actions bar
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            copy_to_clipboard_button(st.session_state.ec_analysis_output, "Copy All Notes")
        dl2.download_button(
            label="Download (.txt)",
            data=st.session_state.ec_analysis_output,
            file_name=f"EC_Analysis_{company_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        dl3.download_button(
            label="Download (.md)",
            data=st.session_state.ec_analysis_output,
            file_name=f"EC_Analysis_{company_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Save to database option
        st.divider()
        if st.button("Save to Notes History", use_container_width=True, key="ec_save_btn"):
            try:
                note_data = {
                    'id': str(uuid.uuid4()),
                    'created_at': datetime.now().isoformat(),
                    'meeting_type': 'Earnings Call',
                    'file_name': f"EC Analysis — {company_name}",
                    'content': st.session_state.ec_analysis_output,
                    'raw_transcript': "\n\n---\n\n".join(
                        f"--- {fname} ---\n{text[:5000]}..." if len(text) > 5000 else f"--- {fname} ---\n{text}"
                        for fname, text in st.session_state.ec_analysis_texts
                    ),
                    'refined_transcript': None,
                    'token_usage': 0,
                    'processing_time': 0,
                    'pdf_blob': None
                }
                database.save_note(note_data)
                st.toast("Saved to Notes History!", icon="\u2705")
                st.session_state.app_state.active_note_id = note_data['id']
            except Exception as e:
                st.error(f"Failed to save: {e}")


# --- 6. MAIN APPLICATION RUNNER ---
def _safe_utf8(s: Any) -> str:
    """Strip lone UTF-16 surrogates so the result can be UTF-8 encoded into
    Streamlit's protobuf string fields. Without this, error-display calls
    can themselves crash and hide the underlying exception."""
    return str(s).encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def run_app():
    st.set_page_config(page_title="SynthNotes AI", layout="wide", page_icon="🤖")

    # Inject app-wide CSS (navigation highlights, spacing, responsive)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.logo("https://placehold.co/64x64?text=SN", link="https://streamlit.io")

    # Theme switching is handled by Streamlit's built-in settings menu —
    # the old localStorage-rewrite toggle was brittle across versions.
    st.title("SynthNotes AI")

    if "config_error" in st.session_state:
        st.error(st.session_state.config_error); st.stop()

    try:
        database.init_db()
    except Exception as db_err:
        st.error(f"Failed to initialize database: {db_err}")
        st.info("The app may not be able to save notes. Check database permissions.")

    try:
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
            on_sector_change()
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}

        def _page_input():
            try:
                render_input_and_processing_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(_safe_utf8(f"Error in Input tab: {tab_err}"))

        def _page_output():
            try:
                render_output_and_history_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(_safe_utf8(f"Error in Output tab: {tab_err}"))

        def _page_otg():
            try:
                render_otg_notes_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(_safe_utf8(f"Error in OTG Notes tab: {tab_err}"))

        def _page_ec_analysis():
            try:
                render_ec_analysis_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(_safe_utf8(f"Error in EC Analysis tab: {tab_err}"))

        def _page_report_comparison():
            try:
                render_report_comparison_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(_safe_utf8(f"Error in Report Comparison tab: {tab_err}"))

        nav = st.navigation(
            [
                st.Page(_page_input, title="Input & Generate", icon=":material/edit_note:"),
                st.Page(_page_output, title="Output & History", icon=":material/history:"),
                st.Page(_page_ec_analysis, title="EC Analysis", icon=":material/analytics:"),
                st.Page(_page_report_comparison, title="Report Compare", icon=":material/compare:"),
                st.Page(_page_otg, title="OTG Notes", icon=":material/quick_phrases:"),
            ],
            position="top",
        )
        nav.run()

    except Exception as e:
        st.error("A critical application error occurred.")
        try:
            st.code(_safe_utf8(traceback.format_exc()))
        except Exception:
            st.exception(e)

if __name__ == "__main__":
    run_app()

# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/

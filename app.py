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
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

# --- Local Imports ---
import database

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

/* ── Section dividers: lighter, more breathing room ── */
hr {
    margin-top: 1.2rem !important;
    margin-bottom: 1.2rem !important;
    opacity: 0.3;
}

/* ── Note cards in history list: subtle hover lift ── */
[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    transition: box-shadow 0.15s ease;
}
[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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

/* ── Responsive columns - stack on small screens ── */
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
}

/* ── Copy button iframe ── */
iframe {
    min-height: 45px !important;
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
CHUNK_WORD_SIZE = 4000
CHUNK_WORD_OVERLAP = 400

AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash", "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash-lite", "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite", "Gemini 2.5 Pro": "gemini-2.5-pro","Gemini 3.0 Flash": "gemini-3-flash-preview", "Gemini 3.0 Pro": "gemini-3-pro-preview",
}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Management Meeting", "Internal Discussion", "Custom"]
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]

TONE_OPTIONS = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
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
    "Expert Meeting": "Q&A format with detailed factual extraction from expert consultations",
    "Earnings Call": "Financial data, management commentary, guidance, and analyst Q&A",
    "Management Meeting": "Decisions, action items, owners, and key discussion points",
    "Internal Discussion": "Perspectives, ideas, reasoning, conclusions, and next steps",
    "Custom": "Provide your own formatting instructions via the context field",
}

# --- PROMPT CONSTANTS ---

EXPERT_MEETING_DETAILED_PROMPT = """### **PRIMARY DIRECTIVE: MAXIMUM DETAIL & STRICT COMPLETENESS**
Your goal is to produce the most thorough, granular notes possible. Remove conversational filler ("um," "you know," repetition) but **nothing substantive should be omitted.** Every factual claim, example, explanation, aside, and data point in the transcript must appear in your notes. When in doubt, INCLUDE it. Err heavily on the side of over-inclusion. Longer, more detailed notes are always preferred over concise ones.

### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles, affiliations, years of experience, roles). Use simple, direct language.
- **DO NOT:** Summarize or include introductions about consulting firms.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format.

**(2.A) Questions:**
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy the question verbatim from the transcript — clean up filler, false starts, and rambling phrasing into a clear, well-formed question that preserves the original intent.
-   If the questioner provides context, framing, or a multi-part question, capture the full scope — do not reduce a multi-part question to a single line.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   Use **multiple bullet points** per answer — do NOT collapse a detailed response into a single bullet.
-   **ZERO SKIPPING RULE:** If the expert said it with substance, it must appear in your notes. Do NOT skip examples, anecdotes, specific sentences, or supporting details even if they seem minor or repetitive. Every distinct point gets its own bullet. If an answer contains 8 substantive points, you must produce at least 8 bullets — never condense them into 3-4.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, specific entities mentioned, time periods, market sizes, growth rates, company names, product names, and geographies.
-   **PRIORITY #2: CAPTURE ALL NUANCE & REASONING.** Do not over-summarize or reduce complex answers to surface-level statements. You must retain the following:
    -   **Sentiment & Tone:** Note if the expert is confident, uncertain, speculative, cautious, or enthusiastic (e.g., "The expert was highly confident that...," "He cautioned that...").
    -   **Qualifiers & Conditions:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "except in," "only when," "roughly," "approximately," "a potential risk is...").
    -   **Key Examples & Analogies:** If the expert uses a specific example, anecdote, case study, or analogy to illustrate a point, capture it in full, even if it spans multiple sentences — these are often the most valuable parts of an expert call.
    -   **Cause & Effect:** Retain any reasoning chains provided (e.g., "...because of regulatory changes," "...which led to a 15% decline in...").
    -   **Comparisons & Contrasts:** If the expert compares companies, products, approaches, or time periods, capture both sides of the comparison with the specific details for each.
    -   **Tangential but relevant points:** If the expert volunteers additional context, background, or related information beyond the direct question, include it — do NOT discard it as off-topic.
-   **PRIORITY #3: PRESERVE MULTI-STEP EXPLANATIONS.** If an answer involves a sequence of steps, a timeline, or a logical chain, preserve the full sequence rather than summarizing the conclusion only."""

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
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy verbatim from the transcript — clean up filler and rambling into a clear, well-formed question.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, and specific entities mentioned.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Do not over-summarize. You must retain the following:
    -   **Sentiment & Tone:** Note if the speaker is optimistic, hesitant, confident, or speculative (e.g., "The expert was cautiously optimistic about...", "He speculated that...").
    -   **Qualifiers:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "rarely," "a potential risk is...").
    -   **Key Examples & Analogies:** If the speaker uses a specific example to illustrate a point, capture it, even if it's a few sentences long.
    -   **Cause & Effect:** Retain any reasoning provided (e.g., "...because of the new regulations," "...which led to a decrease in...")."""

EARNINGS_CALL_PROMPT = """### **NOTES STRUCTURE: EARNINGS CALL**

Generate detailed earnings call notes based on the transcript. Structure your notes under the following topics, using **bold headings** and bullet points for each:

{topic_instructions}

**PRIORITY #1: CAPTURE ALL FINANCIAL DATA.** Revenue, margins, EPS, guidance ranges, growth rates, basis points, dollar amounts — every number matters.
**PRIORITY #2: CAPTURE FORWARD GUIDANCE.** Any forward-looking statements, guidance ranges, management expectations, or outlook commentary.
**PRIORITY #3: PRESERVE MANAGEMENT TONE.** Note confidence, caution, hedging language, or changes from prior quarter tone.
**PRIORITY #4: CAPTURE SEGMENT/VERTICAL DETAIL.** Business segment breakdowns, geographic splits, and vertical-specific commentary."""

MANAGEMENT_MEETING_PROMPT = """### **NOTES STRUCTURE: MANAGEMENT MEETING**

Structure the notes to capture decisions, action items, and key discussion points.

**(1.) Meeting Overview (Conditional):**
- If the transcript begins with an agenda or introductions, capture attendees, date, and agenda items as bullet points.

**(2.) Discussion Topics:**
Structure the body by topic/agenda item using **bold headings**.

For each topic:
- **Key Points:** Bullet-point the main arguments, data, and perspectives shared.
- **Decisions Made:** Clearly state any decisions reached, who made them, and the rationale.
- **Action Items:** List each action item with the responsible person and any stated deadline.
- **Open Questions:** Note unresolved issues or items deferred for follow-up.

**PRIORITY #1: CAPTURE ALL DECISIONS AND ACTION ITEMS.** These are the most critical outputs.
**PRIORITY #2: CAPTURE ALL DATA.** Names, numbers, dates, metrics, and specific references.
**PRIORITY #3: PRESERVE CONTEXT.** Include the reasoning behind decisions and any dissenting views."""

INTERNAL_DISCUSSION_PROMPT = """### **NOTES STRUCTURE: INTERNAL DISCUSSION**

Structure the notes to capture the flow of ideas, key arguments, and conclusions.

**(1.) Discussion Context (Conditional):**
- If the discussion has a stated purpose or background, capture it as bullet points at the top.

**(2.) Discussion Flow:**
Structure the body by topic or theme using **bold headings**.

For each topic:
- Capture each participant's key contributions and perspectives as bullet points.
- Note areas of agreement and disagreement.
- Highlight any data, examples, or evidence cited.
- Flag any concerns, risks, or caveats raised.

**(3.) Conclusions & Next Steps:**
- Summarize any conclusions reached.
- List follow-up items or next steps with owners if identified.

**PRIORITY #1: CAPTURE ALL PERSPECTIVES.** Include different viewpoints even if they disagree.
**PRIORITY #2: CAPTURE ALL DATA.** Names, numbers, references, and specific examples.
**PRIORITY #3: PRESERVE REASONING.** Include the "why" behind opinions and conclusions."""

PROMPT_INITIAL = """You are a High-Fidelity Factual Extraction Engine. Your task is to analyze a meeting transcript chunk and generate detailed, factual notes.
Your primary directive is **100% completeness and accuracy**. Process the transcript sequentially and generate notes following the structure below.
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
3.  **MAINTAIN FORMAT:** Continue to use the exact same formatting as established in the base instructions.
4.  **NO META-COMMENTARY:** NEVER produce statements about the transcript itself, such as "the transcript does not contain an answer," "no relevant information in this section," "the chunk starts mid-conversation," or similar. If a chunk begins mid-answer, capture that content as a continuation of the relevant section. Always extract and document whatever substantive content exists.
5.  **MID-CHUNK STARTS:** If the chunk starts in the middle of a response, begin your notes by capturing that content under the most relevant heading from context. Do not skip or discard partial content.

---
{base_instructions}
---

**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""

PROOFREAD_PROMPT = """You are an expert proofreader for meeting notes. You have the **original transcript** and the **generated notes**. Your task is to produce a corrected, polished final version of the notes.

### YOUR TASKS (in priority order):

1.  **Fix misinterpreted words**: Cross-reference the notes against the transcript. Correct any words that were misheard, misspelled, or misinterpreted — especially company names, proper nouns, technical terms, industry jargon, acronyms, and numbers.

2.  **Remove processing artifacts**: Delete any meta-commentary or error statements that are not part of the actual meeting content. Examples: "the transcript does not contain an answer," "no relevant information found," "this section appears incomplete," or any similar artifacts from chunked processing.

3.  **Verify completeness**: Check that every substantive point, data point, name, number, percentage, monetary value, example, and nuanced opinion from the transcript is reflected in the notes. If anything is missing, add it under the appropriate section following the same formatting.

4.  **Clean up stitching seams**: Fix any formatting inconsistencies, duplicated content, or awkward transitions that may have resulted from stitching multiple chunks together.

5.  **Preserve all correct content**: Do NOT summarize, shorten, condense, or remove any existing correct content. Your role is to fix and add, never to reduce. The final output must be at least as detailed as the input notes. Every bullet point, example, data point, and nuanced explanation from the input notes must appear in your output — do not merge multiple bullets into one or drop supporting details.

### OUTPUT:
Return ONLY the complete, corrected notes. Do not include any commentary about your changes or a preamble.

---
**ORIGINAL TRANSCRIPT:**
{transcript}
---
**GENERATED NOTES TO PROOFREAD:**
{notes}
"""

EXECUTIVE_SUMMARY_PROMPT = """Generate a structured executive summary from the following meeting notes.

### STRUCTURE:
1. **Key Takeaways** (3-5 bullet points): The most important findings, decisions, or insights from the meeting.
2. **Critical Data Points**: All significant numbers, metrics, percentages, and financial figures mentioned.
3. **Notable Quotes/Positions**: Any strong opinions, definitive statements, or notable positions taken by participants.
4. **Risks & Concerns**: Any risks, challenges, or concerns raised during the meeting.
5. **Action Items / Next Steps**: Any follow-ups, commitments, or next steps identified.

### RULES:
- Be specific — include actual numbers, names, and dates rather than vague references.
- Keep each section concise but complete.
- Do not introduce information not present in the notes.

---
**MEETING NOTES:**
{notes}
"""

REFINEMENT_INSTRUCTIONS = {
    "Expert Meeting": "Pay special attention to industry jargon, technical terms, company names, and domain-specific terminology. Preserve all proper nouns exactly.",
    "Earnings Call": "Pay special attention to financial terminology (EPS, EBITDA, basis points, margin, guidance, revenue, etc.), company names, ticker symbols, analyst names, and numerical data. Preserve all figures exactly as spoken.",
    "Management Meeting": "Pay special attention to names of attendees, action item owners, project names, deadlines, and organizational terminology.",
    "Internal Discussion": "Pay special attention to participant names, project/product names, technical terms, and any referenced documents or systems.",
}

# --- OTG NOTES PROMPTS ---

OTG_EXTRACT_PROMPT = """Analyze the following meeting notes and extract structured metadata. Return ONLY valid JSON with no other text.

{{
  "entities": ["list of company names, product names, and proper nouns mentioned"],
  "people": ["list of people mentioned by name or role"],
  "sector": "the industry sector these notes relate to (e.g., Quick Commerce, Fintech, SaaS, Healthcare, etc.)",
  "topics": ["list of 5-12 distinct topics/themes discussed in the notes, each as a short phrase"]
}}

---
**NOTES:**
{notes}
"""

OTG_CONVERT_PROMPT = """You are writing informal channel check notes — the kind an equity research analyst sends to their team after speaking with industry contacts.

### TASK:
Convert the detailed meeting notes below into a short, plain-text research note.

### STYLE (follow exactly):

1. TITLE: A short, natural title on the first line. Examples: "Channel checks on Quick commerce", "Checks on Hero Motocorp", "Hero demand checks". Keep it simple — no formatting, no colons.

2. INTRO: One sentence starting with "We spoke with..." describing who you spoke with (role/expertise, NOT their name) and what you wanted to understand. Then on the same line or next: "Following were the KTAs:"

3. BODY: Write 4-7 short paragraphs of plain flowing text. Each paragraph makes one clear point.
   - ABSOLUTELY NO markdown formatting. No bold (**), no bullets (-), no numbered lists, no headers (#). Just plain text paragraphs.
   - Use simple, direct language. Write like you're sending a quick note to your team, not writing a formal report.
   - Attribute findings to the source naturally: "The expert estimates...", "She didn't share...", "Dealers felt...", "Managers mentioned...", "Our checks highlight...", "He pointed out..."
   - Weave in your own analyst commentary where relevant: "We will need to monitor...", "This makes it tricky because...", "We have observed earlier that..."

4. TONE: {tone}
   - Very Positive: Frame findings constructively. Strengths, growth, advantages. Challenges are temporary.
   - Positive: Generally constructive. Risks acknowledged but opportunities emphasized.
   - Neutral: Balanced. Facts presented objectively.
   - Negative: Risks and structural problems emphasized. Positive developments are insufficient.
   - Very Negative: Fundamental weaknesses, unsustainable practices. Deeply problematic framing.

5. DATA: {number_focus_instruction}

6. LENGTH: {length_instruction}

7. FOCUS ENTITIES: Center the note around: {entities}. Other entities can appear for context.

8. FOCUS TOPICS: Focus on: {topics}

{custom_instructions_block}

### OUTPUT:
Return ONLY the plain-text note. No preamble, no commentary, no markdown formatting whatsoever.

---
SOURCE NOTES:
{notes}
"""


# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Paste Text"
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    earnings_call_mode: str = "Generate New Notes"
    selected_sector: str = "IT Services"
    notes_model: str = "Gemini 2.5 Pro"
    refinement_model: str =  "Gemini 2.5 Flash Lite"
    transcription_model: str =  "Gemini 3.0 Flash"
    chat_model: str = "Gemini 2.5 Pro"
    refinement_enabled: bool = True
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

def generate_with_retry(model, prompt_or_contents, max_retries=3, stream=False):
    """Wrapper around generate_content with exponential backoff for transient API failures."""
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt_or_contents, stream=stream)
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(kw in error_str for kw in [
                '429', '503', '500', 'deadline', 'timeout', 'unavailable', 'resource_exhausted'
            ])
            if is_transient and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

def stream_and_collect(response, placeholder=None):
    """Consume a streaming response, optionally displaying progress. Returns (full_text, token_count)."""
    full_text = ""
    for chunk in response:
        if chunk.parts:
            full_text += chunk.text
            if placeholder:
                word_count = len(full_text.split())
                lines = full_text.strip().split('\n')
                preview = '\n'.join(lines[-4:])
                placeholder.caption(f"Streaming... {word_count:,} words generated\n{preview}")
    if placeholder:
        placeholder.empty()
    token_count = safe_get_token_count(response)
    return full_text, token_count

def copy_to_clipboard_button(text: str, button_label: str = "Copy Notes"):
    """Render a button that copies text to the clipboard using the browser Clipboard API."""
    # Adapt button colors to the current theme (light vs dark mode)
    theme = st.context.theme
    bg_color = theme.get("primaryColor", "#FF4B4B")
    text_color = theme.get("backgroundColor", "#FFFFFF")

    escaped = html_module.escape(text).replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""
        <button onclick="copyText()" style="
            background-color:{bg_color}; color:{text_color}; border:none; padding:0.4rem 1rem;
            border-radius:0.3rem; cursor:pointer; font-size:0.875rem; width:100%;
        ">{button_label}</button>
        <script>
        function copyText() {{
            const text = `{escaped}`;
            const decoded = new DOMParser().parseFromString(text, 'text/html').body.textContent;
            navigator.clipboard.writeText(decoded).then(() => {{
                const btn = document.querySelector('button');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = '{button_label}', 2000);
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

def create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Creates overlapping chunks of text, ensuring the final fragment is always included."""
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if (i + chunk_size) >= len(words):
            break

    return chunks


def _create_enhanced_context_from_notes(notes_text, chunk_number=0):
    """Create richer context from previous notes"""
    if not notes_text or not notes_text.strip():
        return ""

    headings = re.findall(r"(\*\*.*?\*\*)", notes_text)

    if not headings:
        return ""

    context_headings = headings[-3:] if len(headings) >= 3 else headings

    context_parts = [
        f"**Chunk #{chunk_number} Context Summary:**",
        f"- Total sections processed so far: {len(headings)}",
        f"- Recent topics: {', '.join(q.strip('*') for q in context_headings[-2:])}",
        f"- Last section processed: {headings[-1]}"
    ]

    last_heading = headings[-1]
    answer_match = re.search(
        re.escape(last_heading) + r"(.*?)(?=\*\*|$)",
        notes_text,
        re.DOTALL
    )
    if answer_match:
        last_content = answer_match.group(1).strip()
        context_parts.append(f"- Last section content:\n{last_content[:300]}...")

    return "\n".join(context_parts)

def _get_base_prompt_for_type(state):
    """Returns the base prompt instructions for the selected meeting type."""
    mt = state.selected_meeting_type
    if mt == "Expert Meeting":
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

class ProgressTracker:
    """Manages progress bar and status updates during processing."""

    # Define the processing steps and their approximate weights (totaling 100)
    STEPS = {
        "prepare": {"weight": 5, "label": "Preparing Source Content"},
        "transcribe": {"weight": 15, "label": "Transcribing Audio"},
        "refine": {"weight": 25, "label": "Refining Transcript"},
        "generate": {"weight": 40, "label": "Generating Notes"},
        "proofread": {"weight": 10, "label": "Proofreading Notes"},
        "save": {"weight": 5, "label": "Saving to Database"},
    }

    def __init__(self, status_container):
        self.status = status_container
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.current_progress = 0
        self.completed_steps = set()

    def update(self, step: str, sub_progress: float = 0, detail: str = ""):
        """
        Update progress bar and status.
        step: one of the STEPS keys
        sub_progress: 0-1 progress within the current step
        detail: additional detail text
        """
        # Calculate base progress from completed steps
        base = sum(self.STEPS[s]["weight"] for s in self.completed_steps)

        # Add current step's partial progress
        if step in self.STEPS:
            step_weight = self.STEPS[step]["weight"]
            current = base + (step_weight * sub_progress)
        else:
            current = base

        self.current_progress = min(current / 100, 1.0)
        self.progress_bar.progress(self.current_progress)

        # Update status text
        label = self.STEPS.get(step, {}).get("label", step)
        pct = int(self.current_progress * 100)
        status_msg = f"**{pct}%** - {label}"
        if detail:
            status_msg += f" ({detail})"
        self.status_text.markdown(status_msg)

    def complete_step(self, step: str):
        """Mark a step as completed."""
        self.completed_steps.add(step)
        self.update(step, 1.0)

    def finish(self):
        """Mark all progress complete."""
        self.progress_bar.progress(1.0)
        self.status_text.markdown("**100%** - Complete!")

def send_browser_notification(title: str, body: str):
    """Send a browser notification using the Notifications API."""
    # Escape quotes for JavaScript
    title_escaped = title.replace("'", "\\'").replace('"', '\\"')
    body_escaped = body.replace("'", "\\'").replace('"', '\\"')

    components.html(
        f"""
        <script>
        (function() {{
            // Check if notifications are supported
            if (!("Notification" in window)) {{
                console.log("Browser doesn't support notifications");
                return;
            }}

            // Request permission if needed
            if (Notification.permission === "granted") {{
                new Notification("{title_escaped}", {{
                    body: "{body_escaped}",
                    icon: "https://placehold.co/64x64?text=SN",
                    tag: "synthnotes-complete"
                }});
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(permission) {{
                    if (permission === "granted") {{
                        new Notification("{title_escaped}", {{
                            body: "{body_escaped}",
                            icon: "https://placehold.co/64x64?text=SN",
                            tag: "synthnotes-complete"
                        }});
                    }}
                }});
            }}
        }})();
        </script>
        """,
        height=0,
    )

def process_and_save_task(state: AppState, status_ui, progress: ProgressTracker):
    start_time = time.time()
    notes_model = _get_cached_model(state.notes_model)
    refinement_model = _get_cached_model(state.refinement_model)
    transcription_model = _get_cached_model(state.transcription_model)

    progress.update("prepare", 0, "Loading input...")
    raw_transcript, file_name = "", "Pasted Text"
    pdf_bytes_data = None

    # Handle input (File, Recording, or Text)
    if state.input_method == "Upload / Record":
        file_type, name, pdf_bytes = get_file_content(state.uploaded_file, state.audio_recording)
        file_name = name
        pdf_bytes_data = pdf_bytes

        if file_type == "audio_file":
            progress.update("transcribe", 0, "Processing audio file...")

            if state.audio_recording:
                audio_bytes = state.audio_recording.getvalue()
            else:
                audio_bytes = state.uploaded_file.getvalue()

            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as audio_err:
                raise ValueError(f"Failed to process audio file. It may be corrupted or in an unsupported format. Details: {audio_err}")

            chunk_length_ms = 5 * 60 * 1000
            audio_chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

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
                            while cloud_ref.state.name == "PROCESSING": time.sleep(2); cloud_ref = genai.get_file(cloud_ref.name)
                            if cloud_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {i+1} cloud processing failed.")
                            response = generate_with_retry(transcription_model, ["Transcribe this audio.", cloud_ref])
                            all_transcripts.append(response.text)
                    except Exception as e:
                        raise Exception(f"Transcription failed on chunk {i+1}/{len(audio_chunks)}. Reason: {e}")

                raw_transcript = "\n\n".join(all_transcripts).strip()
                progress.complete_step("transcribe")
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

    if not raw_transcript or not raw_transcript.strip():
        raise ValueError("Source content is empty or contains only whitespace.")

    # Normalize whitespace and remove excessive blank lines
    raw_transcript = re.sub(r'\n{3,}', '\n\n', raw_transcript.strip())

    # Checkpoint: save raw transcript so it survives connection drops
    st.session_state["_checkpoint_raw_transcript"] = raw_transcript
    st.session_state["_checkpoint_file_name"] = file_name

    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0

    speakers = sanitize_input(state.speakers)
    speaker_info = f"Participants: {speakers}." if speakers else ""
    refinement_extra = REFINEMENT_INSTRUCTIONS.get(state.selected_meeting_type, "")

    # --- Step 2: Refinement ---
    if state.refinement_enabled:
        progress.complete_step("prepare")
        progress.update("refine", 0, "Starting refinement...")
        words = raw_transcript.split()

        if len(words) <= CHUNK_WORD_SIZE:
            refine_prompt = f"Refine the following transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n\nTRANSCRIPT:\n{raw_transcript}"
            response = generate_with_retry(refinement_model, refine_prompt)
            refined_transcript = response.text
            total_tokens += safe_get_token_count(response)
        else:
            all_words = raw_transcript.split()
            chunks = [" ".join(all_words[i:i + CHUNK_WORD_SIZE]) for i in range(0, len(all_words), CHUNK_WORD_SIZE)]

            # Pre-build all prompts using raw chunk tails as context (known upfront, enables parallelism)
            prompts = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    prompts.append(f"You are refining a transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n\nTRANSCRIPT CHUNK TO REFINE:\n{chunk}")
                else:
                    prev_chunk_words = chunks[i - 1].split()
                    context = " ".join(prev_chunk_words[-CHUNK_WORD_OVERLAP:])
                    prompts.append(f"""You are continuing to refine a long transcript. Below is the tail end of the previous section for context. Your task is to refine the new chunk provided, ensuring a seamless and natural transition.
{speaker_info} {refinement_extra}
---
CONTEXT FROM PREVIOUS CHUNK (FOR CONTINUITY ONLY):
...{context}
---
NEW TRANSCRIPT CHUNK TO REFINE:
{chunk}""")

            progress.update("refine", 0.1, f"{len(chunks)} chunks in parallel")

            # Process chunks in parallel (max 3 concurrent to respect API rate limits)
            all_refined_chunks = [None] * len(chunks)
            chunk_tokens = [0] * len(chunks)

            def refine_chunk(idx, prompt):
                resp = generate_with_retry(refinement_model, prompt)
                return idx, resp.text, safe_get_token_count(resp)

            with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as executor:
                futures = {executor.submit(refine_chunk, i, p): i for i, p in enumerate(prompts)}
                for future in as_completed(futures):
                    idx, text, tokens = future.result()
                    all_refined_chunks[idx] = text
                    chunk_tokens[idx] = tokens
                    done_count = sum(1 for c in all_refined_chunks if c is not None)
                    progress.update("refine", 0.1 + (0.9 * done_count / len(chunks)), f"{done_count}/{len(chunks)} chunks")

            total_tokens += sum(chunk_tokens)
            refined_transcript = "\n\n".join(c for c in all_refined_chunks if c) if any(all_refined_chunks) else ""

        final_transcript = refined_transcript
    else:
        # Refinement disabled - mark prepare and refine as complete
        progress.complete_step("prepare")
        progress.complete_step("refine")

    # Checkpoint: save refined transcript
    st.session_state["_checkpoint_refined_transcript"] = refined_transcript

    # --- Step 3: Generate Notes ---
    words = final_transcript.split()
    # Earnings calls should not be chunked: their topic-based structure causes
    # repeated sections when the same headings appear across multiple chunks.
    skip_chunking = state.selected_meeting_type == "Earnings Call"
    num_chunks = 1 if skip_chunking else max(1, (len(words) + CHUNK_WORD_SIZE - 1) // CHUNK_WORD_SIZE)
    progress.update("generate", 0, f"{len(words):,} words, {num_chunks} chunk{'s' if num_chunks > 1 else ''}")
    final_notes_content = ""

    if not skip_chunking and len(words) > CHUNK_WORD_SIZE:
        chunks = create_chunks_with_overlap(final_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)

        all_notes_chunks = []
        context_package = ""
        prompt_base = _get_base_prompt_for_type(state)

        for i, chunk in enumerate(chunks):
            progress.update("generate", i / len(chunks), f"Chunk {i+1}/{len(chunks)}")
            prompt_template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION
            prompt = prompt_template.format(base_instructions=prompt_base, chunk_text=chunk, context_package=context_package)

            stream_placeholder = st.empty()
            response = generate_with_retry(notes_model, prompt, stream=True)
            current_notes_text, tokens = stream_and_collect(response, stream_placeholder)
            total_tokens += tokens

            all_notes_chunks.append(current_notes_text)

            cumulative_notes_for_context = "\n\n".join(all_notes_chunks)
            context_package = _create_enhanced_context_from_notes(cumulative_notes_for_context, chunk_number=i + 1)

        if not all_notes_chunks or not any(c.strip() for c in all_notes_chunks):
            raise ValueError("Failed to generate notes from any chunk. Please try again or use a different model.")
        else:
            final_notes_content = all_notes_chunks[0]
            for i in range(1, len(all_notes_chunks)):
                prev_notes = all_notes_chunks[i-1]
                current_notes = all_notes_chunks[i]

                last_q_match = list(re.finditer(r"(\*\*.*?\*\*)", prev_notes))
                if not last_q_match:
                    final_notes_content += "\n\n" + current_notes
                    continue

                last_heading = last_q_match[-1].group(1)

                stitch_point = current_notes.find(last_heading)

                if stitch_point != -1:
                    next_q_match = re.search(r"(\*\*.*?\*\*)", current_notes[stitch_point + len(last_heading):])
                    if next_q_match:
                        final_notes_content += "\n\n" + current_notes[stitch_point + len(last_heading) + next_q_match.start():]
                    else:
                        final_notes_content += "\n\n" + current_notes[stitch_point + len(last_heading):]
                else:
                    st.warning(f"Could not find stitch point for chunk {i+1}. Appending full chunk; check for duplicates.")
                    final_notes_content += "\n\n" + current_notes

    else:
        prompt = get_dynamic_prompt(state, final_transcript)
        stream_placeholder = st.empty()
        response = generate_with_retry(notes_model, prompt, stream=True)
        final_notes_content, tokens = stream_and_collect(response, stream_placeholder)
        total_tokens += tokens

    # Defensive: ensure we have content
    if not final_notes_content or not final_notes_content.strip():
        raise ValueError("The model returned empty notes. Please try again or use a different model.")

    # --- Step 4: Proofread (only when chunking was used — single-chunk notes have no stitching artifacts) ---
    progress.complete_step("generate")
    was_chunked = not skip_chunking and len(final_transcript.split()) > CHUNK_WORD_SIZE
    if was_chunked:
        progress.update("proofread", 0, "Cleaning stitching artifacts...")
        proofread_prompt = PROOFREAD_PROMPT.format(transcript=final_transcript, notes=final_notes_content)
        response = generate_with_retry(refinement_model, proofread_prompt)
        final_notes_content = response.text
        total_tokens += safe_get_token_count(response)
    else:
        progress.update("proofread", 1.0, "Skipped (single-chunk)")
    progress.complete_step("proofread")

    # --- Step 5: Executive Summary (Expert Meeting Option 3 only) ---
    if state.selected_note_style == "Option 3: Less Verbose + Summary" and state.selected_meeting_type == "Expert Meeting":
        progress.update("save", 0, "Generating executive summary...")
        summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(notes=final_notes_content)
        response = generate_with_retry(notes_model, summary_prompt)
        final_notes_content += f"\n\n---\n\n{response.text}"
        total_tokens += safe_get_token_count(response)

    # --- Step 6: Save ---
    progress.update("save", 0.5, "Writing to database...")

    note_data = {
        'id': str(uuid.uuid4()), 'created_at': datetime.now().isoformat(), 'meeting_type': state.selected_meeting_type,
        'file_name': file_name, 'content': final_notes_content, 'raw_transcript': raw_transcript,
        'refined_transcript': refined_transcript, 'token_usage': total_tokens,
        'processing_time': time.time() - start_time,
        'pdf_blob': pdf_bytes_data
    }
    try:
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
            state.uploaded_file = st.file_uploader("Upload a File", type=['pdf', 'txt', 'mp3', 'm4a', 'wav', 'ogg', 'flac'], help="PDF, TXT, MP3, M4A, WAV, OGG, FLAC")
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
        num_chunks = max(1, (wc + CHUNK_WORD_SIZE - 1) // CHUNK_WORD_SIZE)
        info = f"**{wc:,}** words"
        if num_chunks > 1:
            info += f" | **{num_chunks}** chunks"
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
            st.markdown("<br>", unsafe_allow_html=True)
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
            if state.selected_meeting_type != "Custom":
                state.add_context_enabled = st.toggle("Add General Context", value=state.add_context_enabled)
                if state.add_context_enabled: state.context_input = st.text_area("Context Details:", value=state.context_input, placeholder="e.g., Company Name, Date...")

            st.divider()
            state.notes_model = st.selectbox("Notes Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.notes_model))
            state.refinement_model = st.selectbox("Refinement Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.refinement_model))
            state.transcription_model = st.selectbox("Transcription Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.transcription_model), help="Used for audio files.")
            state.chat_model = st.selectbox("Chat Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.chat_model), help="Used for chatting with the final output.")

            st.divider()
            st.caption("Browser notifications for processing completion.")
            if st.button("Enable Notifications", key="enable_notif_btn", use_container_width=True):
                components.html(
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
                    height=0,
                )
    with col_participants:
        state.speakers = st.text_input("Participants (Optional)", value=state.speakers, placeholder="e.g., John Smith (Analyst), Jane Doe (CEO)")

    # --- Generate ---
    st.divider()
    validation_error = validate_inputs(state)

    if st.button("Generate Notes", type="primary", use_container_width=True, disabled=bool(validation_error)):
        state.processing = True; state.error_message = None; state.fallback_content = None; st.rerun()

    if validation_error: st.warning(f"Please fix the following: {validation_error}")

    # --- Prompt Preview (collapsed) ---
    with st.expander("Prompt Preview", expanded=False):
        prompt_preview = get_dynamic_prompt(state, "[...transcript content...]")
        st.code(prompt_preview, language="markdown", height=200)

    # --- Processing ---
    if state.processing:
        with st.status("Processing your request...", expanded=True) as status:
            progress = ProgressTracker(status)
            try:
                final_note = process_and_save_task(state, status, progress)
                state.active_note_id = final_note['id']
                progress.finish()
                status.update(label="Done! Switch to the Output & History tab to view your note.", state="complete")
                st.toast("Notes generated successfully!", icon="\u2705")
                # Send browser notification
                processing_time = final_note.get('processing_time', 0)
                send_browser_notification(
                    "SynthNotes AI - Complete",
                    f"Your notes are ready! Processing took {processing_time:.1f}s"
                )
            except Exception as e:
                state.error_message = f"An error occurred during processing:\n{e}"
                status.update(label=f"Error: {e}", state="error")
                # Send error notification
                send_browser_notification(
                    "SynthNotes AI - Error",
                    "Processing failed. Check the app for details."
                )
        state.processing = False

    if state.error_message:
        st.error("Last run failed. See details below:")
        st.code(state.error_message)
        if state.fallback_content:
            st.download_button("Download Unsaved Note (.txt)", state.fallback_content, "synthnotes_fallback.txt")
        if st.button("Clear Error"):
            state.error_message = None
            state.fallback_content = None
            st.rerun()

@st.dialog("Delete Note")
def _confirm_delete_dialog(note_id: str, note_name: str):
    st.markdown(f"Are you sure you want to delete **{note_name}**?")
    st.caption("This action cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Delete", type="primary", use_container_width=True):
        database.delete_note(note_id)
        if st.session_state.app_state.active_note_id == note_id:
            st.session_state.app_state.active_note_id = None
        st.toast(f"Note '{note_name}' deleted.")
        st.rerun()

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

    active_note = database.get_note_by_id(state.active_note_id)
    if not active_note:
        active_note = database.get_note_by_id(notes[0]['id'])

    # --- Note header with inline metadata ---
    hdr_left, hdr_right = st.columns([3, 2])
    with hdr_left:
        st.markdown(f"### {active_note['file_name']}")
        st.badge(active_note['meeting_type'])
    with hdr_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Time", f"{active_note.get('processing_time', 0):.1f}s")
        m2.metric("Tokens", f"{active_note.get('token_usage', 0):,}")
        m3.metric("Date", datetime.fromisoformat(active_note['created_at']).strftime('%b %d'))

    # --- Side-by-side Notes & Transcript ---
    final_transcript = active_note.get('refined_transcript') or active_note.get('raw_transcript')
    transcript_source = "Refined" if active_note.get('refined_transcript') else "Raw"

    col_notes, col_transcript = st.columns([3, 2])
    with col_notes:
        view_mode = st.pills("View", ["Editor", "Preview"], default="Editor", key=f"view_mode_{active_note['id']}")
        if view_mode == "Editor":
            edited_content = st.text_area("Notes", value=active_note['content'], height=600, key=f"output_editor_{active_note['id']}")
        else:
            edited_content = active_note['content']
            with st.container(height=600, border=True):
                st.markdown(edited_content)
    with col_transcript:
        st.markdown(f"**{transcript_source} Transcript**")
        if final_transcript:
            st.text_area("", value=final_transcript, height=600, disabled=True, label_visibility="collapsed", key=f"side_tx_{active_note['id']}")
        else:
            st.info("No transcript available.")

    # --- Actions bar ---
    note_id = active_note['id']
    fname = active_note.get('file_name', 'note')
    raw_tx = active_note.get('raw_transcript')

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        copy_to_clipboard_button(edited_content)
    dl2.download_button(
        label="Notes (.txt)",
        data=lambda: edited_content,
        file_name=f"SynthNote_{fname}.txt",
        mime="text/plain",
        use_container_width=True
    )
    dl3.download_button(
        label="Notes (.md)",
        data=lambda: edited_content,
        file_name=f"SynthNote_{fname}.md",
        mime="text/markdown",
        use_container_width=True
    )
    if final_transcript:
        dl4.download_button(
            label=f"{transcript_source} Transcript",
            data=lambda: final_transcript,
            file_name=f"{transcript_source}_Transcript_{fname}.txt",
            mime="text/plain",
            use_container_width=True
        )
    elif raw_tx:
        dl4.download_button(
            label="Raw Transcript",
            data=lambda: raw_tx,
            file_name=f"Raw_Transcript_{fname}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        dl4.empty()

    st.feedback("thumbs", key=f"fb_{active_note['id']}")

    # --- CHAT ---
    st.divider()
    st.subheader("Chat with this Note")
    st.caption("Ask questions about the content. The model has access to both the notes and the source transcript for verbatim lookups.")

    st.session_state.chat_histories.setdefault(active_note['id'], [])

    for message in st.session_state.chat_histories[active_note['id']]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the note content..."):
        st.session_state.chat_histories[active_note['id']].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/progress_activity:"):
            full_response = ""
            try:
                transcript_context = final_transcript[:30000] if final_transcript else "Not available."
                system_prompt = f"""You are an expert analyst. Your task is to answer questions based on the provided meeting notes and source transcript.
If the user asks for verbatim quotes or exact wording, refer to the TRANSCRIPT section. For analysis and summary questions, use the NOTES section.

MEETING NOTES:
---
{edited_content}
---
SOURCE TRANSCRIPT:
---
{transcript_context}
---
"""
                chat_model_name = AVAILABLE_MODELS.get(state.chat_model, "gemini-1.5-flash")
                chat_model = genai.GenerativeModel(chat_model_name, system_instruction=system_prompt)
                messages_for_api = [{'role': "model" if m["role"] == "assistant" else "user", 'parts': [m['content']]} for m in st.session_state.chat_histories[active_note['id']]]

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
                st.error(full_response)
                if st.session_state.chat_histories[active_note['id']]:
                    st.session_state.chat_histories[active_note['id']].pop()

        if 'full_response' in locals() and not full_response.startswith("Sorry"):
            st.session_state.chat_histories[active_note['id']].append({"role": "assistant", "content": full_response})

    # --- Analytics ---
    st.divider()
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Notes", summary_dict.get('total_notes', 0))
    c2.metric("Avg. Time / Note", f"{summary_dict.get('avg_time', 0.0):.1f}s")
    c3.metric("Total Tokens", f"{summary_dict.get('total_tokens', 0):,}")

    # --- Search & Filter ---
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        search_query = st.text_input("Search notes", placeholder="Filter by file name...", label_visibility="collapsed")
    with filter_col2:
        type_filter = st.selectbox("Type", ["All"] + MEETING_TYPES, label_visibility="collapsed")

    filtered_notes = notes
    if search_query:
        filtered_notes = [n for n in filtered_notes if search_query.lower() in n.get('file_name', '').lower()]
    if type_filter != "All":
        filtered_notes = [n for n in filtered_notes if n.get('meeting_type') == type_filter]

    if not filtered_notes:
        st.caption("No notes match your search.")

    for note in filtered_notes:
        is_active = note['id'] == state.active_note_id
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"**{note['file_name']}**"
                if is_active:
                    label += " &nbsp; `viewing`"
                st.markdown(label)
                content_text = note.get('content', '')
                if content_text:
                    snippet = content_text[:120].replace('\n', ' ').strip()
                    if len(content_text) > 120:
                        snippet += "..."
                    st.caption(snippet)
                # Badge + date on one line
                badge_col, date_col = st.columns([1, 2])
                badge_col.badge(note['meeting_type'])
                date_col.caption(datetime.fromisoformat(note['created_at']).strftime('%b %d, %Y %H:%M'))
            with col2:
                if st.button("View", key=f"view_{note['id']}", use_container_width=True, disabled=is_active):
                    state.active_note_id = note['id']
                    st.rerun()
                if st.button("Delete", key=f"del_{note['id']}", use_container_width=True):
                    _confirm_delete_dialog(note['id'], note['file_name'])

def render_otg_notes_tab(state: AppState):
    st.subheader("Convert Notes to Research Style")
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
        note_options = {n['file_name']: n['id'] for n in notes}
        selected_name = st.selectbox("Select a saved note", list(note_options.keys()), key="otg_note_selector")
        if selected_name:
            selected_note = database.get_note_by_id(note_options[selected_name])
            if selected_note:
                st.session_state.otg_input = selected_note.get('content', '')
                with st.expander("Preview loaded notes", expanded=False):
                    st.markdown(st.session_state.otg_input[:2000] + ("..." if len(st.session_state.otg_input) > 2000 else ""))

    if not st.session_state.otg_input.strip():
        return

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

    if st.button("Generate Research Note", type="primary", use_container_width=True, key="otg_generate_btn"):
        with st.spinner("Generating research note..."):
            try:
                otg_model = _get_cached_model(state.notes_model)
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
                    notes=st.session_state.otg_input
                )
                response = generate_with_retry(otg_model, prompt)
                st.session_state.otg_output = response.text
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate research note: {e}")

    # --- Display output ---
    if st.session_state.otg_output:
        st.divider()
        st.markdown("### Generated Research Note")
        with st.container(border=True):
            st.markdown(st.session_state.otg_output)

        otg_sector_slug = sector.replace(' ', '_')
        out1, out2, out3 = st.columns(3)
        with out1:
            copy_to_clipboard_button(st.session_state.otg_output, "Copy Research Note")
        out2.download_button(
            label="Download (.txt)",
            data=lambda: st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.txt",
            mime="text/plain",
            use_container_width=True
        )
        out3.download_button(
            label="Download (.md)",
            data=lambda: st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- 6. MAIN APPLICATION RUNNER ---
def run_app():
    st.set_page_config(page_title="SynthNotes AI", layout="wide", page_icon="🤖")

    # Inject app-wide CSS (navigation highlights, spacing, responsive)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.logo("https://placehold.co/64x64?text=SN", link="https://streamlit.io")

    # --- Header with dark mode toggle ---
    title_col, theme_col = st.columns([6, 1])
    with title_col:
        st.title("SynthNotes AI")
    with theme_col:
        # Detect current theme and show toggle
        current_theme = st.context.theme
        is_dark = current_theme.get("backgroundColor", "#ffffff").lower() in (
            "#0e1117", "#111111", "#000000", "#0e1118", "#262730",
        )
        dark_mode = st.toggle(
            "Dark" if is_dark else "Light",
            value=is_dark,
            key="dark_mode_toggle",
            help="Switch between light and dark mode",
        )
        if dark_mode != is_dark:
            # Inject JS to toggle Streamlit's theme via settings menu
            target_theme = "Dark" if dark_mode else "Light"
            components.html(
                f"""
                <script>
                // Toggle theme by updating localStorage and reloading
                const stTheme = '{target_theme.lower()}';
                try {{
                    const key = Object.keys(localStorage).find(k => k.includes('stActiveTheme')) || 'stActiveTheme-/-v1';
                    localStorage.setItem(key, JSON.stringify({{name: stTheme, themeInput: {{}}}}));
                    window.parent.location.reload();
                }} catch(e) {{
                    // Fallback: use URL params
                    const url = new URL(window.parent.location);
                    url.searchParams.set('embed_options', 'dark_theme' === stTheme ? 'dark_theme' : 'light_theme');
                    window.parent.location = url;
                }}
                </script>
                """,
                height=0,
            )

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
                st.error(f"Error in Input tab: {tab_err}")

        def _page_output():
            try:
                render_output_and_history_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in Output tab: {tab_err}")

        def _page_otg():
            try:
                render_otg_notes_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in OTG Notes tab: {tab_err}")

        nav = st.navigation(
            [
                st.Page(_page_input, title="Input & Generate", icon=":material/edit_note:"),
                st.Page(_page_output, title="Output & History", icon=":material/history:"),
                st.Page(_page_otg, title="OTG Notes", icon=":material/quick_phrases:"),
            ],
            position="top",
        )
        nav.run()

    except Exception as e:
        st.error("A critical application error occurred."); st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()

# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/

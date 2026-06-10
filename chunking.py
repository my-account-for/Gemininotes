# /-----------------------------\
# |   START OF chunking.py FILE  |
# \-----------------------------/

"""Transcript chunking and notes-cleanup utilities.

Pure text functions — no Streamlit or API dependencies — so the chunking
behaviour can be unit tested in isolation.

Design notes
------------
Chunks produced by ``create_chunks_with_context`` are **non-overlapping** and
aligned to paragraph boundaries (speaker turns in refined transcripts), so:

1. No content is ever processed twice → no duplicate Q&As in the output and
   no stitching/de-duplication step is needed; chunk outputs are simply
   concatenated.
2. A speaker turn is never split mid-answer (unless a single turn exceeds the
   chunk size, in which case it is split at sentence boundaries).
3. Continuity across the boundary is provided by ``context``: the tail of the
   previous chunk's *transcript*, passed to the model as read-only context.
   Because the context comes from the raw transcript (known upfront) rather
   than previously generated notes, all chunk prompts can be built before any
   generation starts — which lets chunks be processed in parallel.
"""

import math
import re
from typing import Dict, List


def _split_oversized_block(block: str, chunk_size: int) -> List[str]:
    """Split a single block larger than ``chunk_size`` words at sentence
    boundaries, falling back to raw word boundaries for pathological input."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if n > chunk_size:
            # A single "sentence" longer than the chunk size (e.g. text with
            # no punctuation) — split it at raw word boundaries.
            if current:
                pieces.append(" ".join(current))
                current, current_words = [], 0
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                pieces.append(" ".join(words[i : i + chunk_size]))
            continue
        if current and current_words + n > chunk_size:
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def create_chunks_with_context(
    text: str, chunk_size: int, context_words: int
) -> List[Dict[str, str]]:
    """Split ``text`` into non-overlapping chunks aligned to paragraph
    (speaker-turn) boundaries.

    Returns a list of dicts with two keys:
      - ``"text"``: the region to process. Chunks never overlap, and joining
        them with blank lines reproduces the full input content.
      - ``"context"``: the last ``context_words`` words of the previous
        chunk, for continuity only (empty string for the first chunk).
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if len(text.split()) <= chunk_size:
        return [{"text": text, "context": ""}]

    # Blocks = paragraphs, which are speaker turns in refined transcripts.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    expanded: List[str] = []
    for block in blocks:
        if len(block.split()) > chunk_size:
            expanded.extend(_split_oversized_block(block, chunk_size))
        else:
            expanded.append(block)

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for block in expanded:
        n = len(block.split())
        if current and current_words + n > chunk_size:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(block)
        current_words += n
    if current:
        chunks.append("\n\n".join(current))

    # Avoid a tiny trailing chunk: a 100-word leftover doesn't justify its own
    # model call and tends to produce low-quality, contextless notes.
    if len(chunks) >= 2 and len(chunks[-1].split()) < max(1, int(chunk_size * 0.15)):
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    result: List[Dict[str, str]] = []
    for i, chunk in enumerate(chunks):
        if i == 0 or context_words <= 0:
            context = ""
        else:
            prev_words = chunks[i - 1].split()
            context = " ".join(prev_words[-context_words:])
        result.append({"text": chunk, "context": context})
    return result


def estimate_chunk_count(word_count: int, chunk_size: int) -> int:
    """Estimate how many chunks a transcript of ``word_count`` words will
    produce. Exact for word-aligned input; paragraph rounding can shift the
    real count by one, so display this as an approximation."""
    return max(1, math.ceil(word_count / chunk_size))


def cleanup_stitched_notes(notes_text: str) -> str:
    """Deterministic cleanup of concatenated chunk notes — no LLM call, zero
    risk of content loss.

    Handles:
    1. Remove meta-commentary / processing artifacts from chunked generation
    2. Collapse duplicate consecutive headings across chunk boundaries
    3. Fix formatting: excessive blank lines, trailing whitespace
    """
    if not notes_text or not notes_text.strip():
        return notes_text

    # --- 1. Remove known meta-commentary artifacts ---
    artifact_patterns = [
        r'^[\-\*]*\s*(?:Note:|Disclaimer:)?\s*(?:The|This)\s+(?:transcript|section|chunk|portion|segment)\s+(?:does not|doesn\'t|appears to)\s+.*$',
        r'^[\-\*]*\s*(?:No relevant|No additional|No further|No substantive)\s+(?:information|content|data|details).*$',
        r'^[\-\*]*\s*(?:This section (?:appears|seems|is) (?:incomplete|empty|blank)).*$',
        r'^[\-\*]*\s*\[(?:No content|Empty|Continues|Continuation)\].*$',
    ]
    lines = notes_text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        is_artifact = any(re.match(p, stripped, re.IGNORECASE) for p in artifact_patterns)
        if not is_artifact:
            cleaned_lines.append(line)

    # --- 2. Collapse duplicate consecutive bold headings ---
    # Pattern: **Heading** appears twice in a row (possibly separated by blank lines)
    result_lines = []
    last_heading = None
    for line in cleaned_lines:
        stripped = line.strip()
        heading_match = re.match(r'^(\*\*.+?\*\*)\s*$', stripped)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            if current_heading == last_heading:
                # Skip duplicate heading — keep first occurrence
                continue
            last_heading = current_heading
        elif stripped:  # Non-empty, non-heading line resets heading tracker
            last_heading = None
        result_lines.append(line)

    text = '\n'.join(result_lines)

    # --- 3. Collapse 3+ consecutive blank lines to 2 ---
    text = re.sub(r'\n{3,}', '\n\n', text)

    # --- 4. Strip trailing whitespace from each line ---
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    return text.strip()

# /---------------------------\
# |   END OF chunking.py FILE  |
# \---------------------------/

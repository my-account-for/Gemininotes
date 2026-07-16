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

1. No content is ever processed twice, so chunk outputs are (almost) simply
   concatenated. One seam artifact remains: an *answer* can straddle a
   boundary (multi-turn answers, or a single oversized turn split at
   sentence boundaries), in which case the continuation chunk re-states the
   question as an inferred heading. The continuation prompt tags that
   heading with a ``(contd.)`` marker and ``merge_continuation_seams``
   folds its bullets back into the previous section's last block.
2. A speaker turn is never split mid-answer (unless a single turn exceeds the
   chunk size, in which case it is split at sentence boundaries).
3. Continuity across the boundary is provided by ``context``: the tail of the
   previous chunk's *transcript*, passed to the model as read-only context.
   Because the context comes from the raw transcript (known upfront) rather
   than previously generated notes, all chunk prompts can be built before any
   generation starts — which lets chunks be processed in parallel.

``strip_overlap`` is the inverse concern for *audio* chunking: audio chunks
are transcribed with a deliberate time overlap (so no words are lost to a
blind cut), and the duplicated seam words are removed here at join time.
"""

import difflib
import math
import re
from typing import Dict, List, Tuple


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


def _normalized_tokens(text: str) -> List[Tuple[str, int]]:
    """Tokenize ``text`` into (normalized_word, end_char_offset) pairs.

    Normalization (lowercase, strip non-alphanumerics) lets two ASR passes
    over the same audio match even when punctuation/casing differ."""
    tokens: List[Tuple[str, int]] = []
    for m in re.finditer(r"\S+", text):
        norm = re.sub(r"[\W_]+", "", m.group(0)).lower()
        if norm:
            tokens.append((norm, m.end()))
    return tokens


def strip_overlap(
    prev_text: str,
    next_text: str,
    *,
    window_words: int = 120,
    min_match_words: int = 8,
    tail_slack: int = 15,
) -> str:
    """Remove from ``next_text`` the leading region that duplicates the tail
    of ``prev_text``.

    Used when audio chunks are transcribed with a deliberate overlap: the
    seam sentences appear at the end of one chunk's transcript and again at
    the start of the next. Two ASR passes over the same audio rarely agree
    word-for-word (numbers rendered as digits vs. words, fillers, garbles),
    so requiring one long contiguous exact match misses most real seams.
    Instead, the optimal alignment's matching blocks (>= 3 words each) are
    collected; the seam is accepted when they total ``min_match_words`` and
    the last block ends within ``tail_slack`` words of ``prev_text``'s end —
    the true overlap always ends where ``prev_text`` ends, which stops a
    common phrase deeper in the text from being mistaken for the seam.
    ``next_text`` is then cut after the last matched block.

    Conservative by design: when no confident alignment is found,
    ``next_text`` is returned unchanged. Duplicating a few seconds of
    conversation is preferable to silently losing it."""
    if not prev_text or not next_text:
        return next_text
    prev_tokens = _normalized_tokens(prev_text)[-window_words:]
    next_tokens = _normalized_tokens(next_text)[:window_words]
    if not prev_tokens or not next_tokens:
        return next_text

    a = [t[0] for t in prev_tokens]
    b = [t[0] for t in next_tokens]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [blk for blk in matcher.get_matching_blocks() if blk.size >= 3]
    if not blocks:
        return next_text
    if sum(blk.size for blk in blocks) < min_match_words:
        return next_text
    last = blocks[-1]
    if last.a + last.size < len(a) - tail_slack:
        return next_text

    cut_offset = next_tokens[last.b + last.size - 1][1]
    remainder = next_text[cut_offset:]
    return re.sub(r"^[\s.,;:!?\-—–]+", "", remainder)


# Standalone lines that ASR models add around a chunk's transcript. Mid-file
# they read as false endings ("[END OF RECORDING]" halfway through a call).
_ASR_META_LINE = re.compile(
    r"^\s*[\[\(]?\s*(?:the\s+)?(?:beginning|begin|start|end)\s+of\s+(?:the\s+)?"
    r"(?:recording|audio|transcript|transcription|call|meeting)\s*[\]\)]?\s*\.?\s*$"
    r"|^\s*[\[\(]?\s*transcri(?:pt|ption)\s+(?:begins?|ends?|starts?)\s*(?:here)?\s*[\]\)]?\s*\.?\s*$",
    re.IGNORECASE,
)


def strip_asr_meta_markers(text: str) -> str:
    """Remove standalone ASR meta-marker lines the model adds around a chunk's
    transcript ("[END OF RECORDING]", "(Beginning of audio)", ...).

    Only whole lines matching known marker shapes are removed — content
    sentences that merely mention a recording, and pipeline markers like
    "[TRANSCRIPTION GAP: ...]", are untouched."""
    if not text:
        return text
    lines = [line for line in text.split("\n") if not _ASR_META_LINE.match(line)]
    return "\n".join(lines).strip()


# --- Learnings-document merging (Internal Discussion, chunked runs) ---
# Chunked runs emit the document's end sections once per processed section.
# These maps recognise their headings in any of the shapes the model uses
# ("### **10. Unanswered Questions**", "**Consolidated Mental Models**", ...).
_END_SECTION_KEYS = {
    "consolidatedmentalmodels": "mental_models",
    "unansweredquestions": "unanswered",
    "followupsactionitems": "followups",
    "followupactionitems": "followups",
    "followups": "followups",
    "actionitems": "followups",
}
_END_SECTION_TITLES = {
    "mental_models": "Consolidated Mental Models",
    "unanswered": "Unanswered Questions",
    "followups": "Follow-Ups / Action Items",
}


def _heading_text(line: str) -> "str | None":
    """Return the inner text of a markdown heading line, else None.

    Recognises '### Heading', '### **Heading**' and full-line '**Heading**'
    forms; strips bold markers, list numbering, and surrounding quotes."""
    s = line.strip()
    if not s:
        return None
    m = re.match(r"^#{1,6}\s+(.*)$", s)
    if m:
        s = m.group(1).strip()
    elif not re.match(r"^\*\*.+\*\*:?\s*$", s):
        return None
    s = s.strip("*").strip().rstrip(":").strip()
    s = re.sub(r"^\(?\d+[.)]?\)?\s*[-—.:]?\s*", "", s)
    return s.strip("\"'“”‘’").strip()


def _renumber_ordered_items(lines: List[str]) -> List[str]:
    """Renumber '1. ...' items sequentially, preserving original spacing."""
    out: List[str] = []
    counter = 0
    for line in lines:
        m = re.match(r"^(\s*)(\d+)([.)])(\s+)(.*)$", line)
        if m:
            counter += 1
            out.append(f"{m.group(1)}{counter}{m.group(3)}{m.group(4)}{m.group(5)}")
        else:
            out.append(line)
    return out


def _dedup_key(line: str) -> str:
    stripped = re.sub(r"^\s*(?:[*\-+]|\d+[.)])\s*", "", line)
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def merge_learning_doc_sections(notes_text: str) -> str:
    """Merge the per-section end blocks of a chunked learnings document.

    When an Internal Discussion transcript is processed in sections, each
    section emits its own "Consolidated Mental Models", "Unanswered
    Questions" and "Follow-Ups / Action Items" blocks, and topic numbering
    restarts at 1. This deterministically (no LLM, no content loss):

    1. keeps all topic sections in order and renumbers them sequentially,
    2. collects the three end sections from everywhere in the text and
       emits each exactly once, at the end, in the canonical order,
    3. renumbers the items inside Unanswered Questions / Follow-Ups and
       drops exact-duplicate lines within each merged end section.

    Text without any recognised end-section heading is returned unchanged.
    """
    if not notes_text or not notes_text.strip():
        return notes_text

    body: List[str] = []
    buckets: Dict[str, List[str]] = {"mental_models": [], "unanswered": [], "followups": []}
    current: "str | None" = None  # None -> body, else bucket key

    for line in notes_text.split("\n"):
        heading = _heading_text(line)
        if heading is not None:
            key = _END_SECTION_KEYS.get(re.sub(r"[^a-z0-9]+", "", heading.lower()))
            if key:
                current = key
                continue
            current = None
        if current is None:
            body.append(line)
        elif line.strip():
            buckets[current].append(line)

    if not any(buckets.values()):
        return notes_text

    # Renumber topic headings sequentially across the whole body.
    topic_count = 0
    for i, line in enumerate(body):
        if _heading_text(line) is None:
            continue
        m = re.match(r"^(\s*(?:#{1,6}\s+)?\**\s*)(\d+)([.)])(\s+)(.*)$", line)
        if m:
            topic_count += 1
            body[i] = f"{m.group(1)}{topic_count}{m.group(3)}{m.group(4)}{m.group(5)}"

    out_lines = [line for line in body]
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    for key in ("mental_models", "unanswered", "followups"):
        items = buckets[key]
        if not items:
            continue
        seen = set()
        unique: List[str] = []
        for item in items:
            k = _dedup_key(item)
            if k and k in seen:
                continue
            if k:
                seen.add(k)
            unique.append(item)
        if key in ("unanswered", "followups"):
            unique = _renumber_ordered_items(unique)
        title = _END_SECTION_TITLES[key]
        if key == "mental_models" and topic_count:
            topic_count += 1
            title = f"{topic_count}. {title}"
        out_lines.append("")
        out_lines.append(f"### **{title}**")
        out_lines.extend(unique)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


# --- Q&A-block post-processing helpers (Output & History tab) ---
# The LLM's role in post-processing is deliberately narrow: topic grouping
# only ever returns a plan over block indices, and these helpers do the
# actual splitting and reassembly deterministically — so regrouping cannot
# change a single word of the note.


def split_qa_blocks(notes_text: str) -> Tuple[str, List[str]]:
    """Split finished notes into ``(preamble, blocks)`` at full-line bold
    headings.

    A block is a bold question/topic heading — a line that both starts and
    ends with ``**`` — plus everything under it up to the next heading. The
    preamble is anything before the first heading (e.g. an expert-background
    section). Bold speaker attributions inside answers (``- **Name:** ...``)
    start with ``-`` so they never match. Rejoining preamble + blocks with
    blank lines reproduces the note's content."""
    if not notes_text or not notes_text.strip():
        return "", []
    lines = notes_text.split("\n")
    heading_idx = [
        i for i, line in enumerate(lines)
        if line.strip().startswith("**") and line.strip().endswith("**") and len(line.strip()) > 4
    ]
    if not heading_idx:
        return notes_text.strip(), []
    preamble = "\n".join(lines[: heading_idx[0]]).strip()
    blocks: List[str] = []
    for start, end in zip(heading_idx, heading_idx[1:] + [len(lines)]):
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    return preamble, blocks


def flatten_grouping_plan(topics, block_count: int) -> Tuple[List[int], List[Dict]]:
    """Turn an LLM topic-grouping plan into a full ordering of block indices.

    ``topics`` is the parsed plan — a list of ``{"name": str, "blocks":
    [int]}``. The plan is treated as untrusted: duplicate, out-of-range, and
    non-integer indices are dropped, and any index the model forgot is
    re-inserted right after its closest preceding original neighbour, so
    every block appears exactly once and local flow is preserved.

    Returns ``(order, cleaned_topics)`` where ``order`` is a permutation of
    ``range(block_count)`` and ``cleaned_topics`` mirrors the plan with only
    the indices that survived cleaning (empty topics dropped)."""
    order: List[int] = []
    seen = set()
    cleaned: List[Dict] = []
    for topic in topics or []:
        if not isinstance(topic, dict):
            continue
        name = str(topic.get("name", "")).strip() or "Untitled topic"
        indices: List[int] = []
        for idx in topic.get("blocks", []) or []:
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            if 0 <= idx < block_count and idx not in seen:
                seen.add(idx)
                indices.append(idx)
        if indices:
            order.extend(indices)
            cleaned.append({"name": name, "blocks": indices})
    for idx in range(block_count):
        if idx in seen:
            continue
        pos = 0
        for prev in range(idx - 1, -1, -1):
            if prev in seen:
                pos = order.index(prev) + 1
                break
        order.insert(pos, idx)
        seen.add(idx)
    return order, cleaned


def reorder_qa_blocks(preamble: str, blocks: List[str], order: List[int]) -> str:
    """Reassemble a note from whole ``blocks`` in ``order``, preamble first."""
    parts = ([preamble] if preamble and preamble.strip() else []) + [blocks[i] for i in order]
    return "\n\n".join(parts).strip()


def estimate_chunk_count(word_count: int, chunk_size: int) -> int:
    """Estimate how many chunks a transcript of ``word_count`` words will
    produce. Exact for word-aligned input; paragraph rounding can shift the
    real count by one, so display this as an approximation."""
    return max(1, math.ceil(word_count / chunk_size))


_CONTD_MARKER_RE = re.compile(r"\s*\(cont(?:[’']?d|inued)?\.?\)", re.IGNORECASE)


def _contd_heading_cleaned(line: str) -> "str | None":
    """If ``line`` is a bold heading (optionally ``#``-prefixed) carrying a
    ``(contd.)``-style continuation marker, return the line with the marker
    removed; otherwise return None."""
    stripped = line.strip()
    if not stripped or not _CONTD_MARKER_RE.search(stripped):
        return None
    core = re.sub(r"^#{1,6}\s+", "", stripped)
    without = _CONTD_MARKER_RE.sub("", core).strip()
    if without.startswith("**") and without.endswith("**") and len(without) > 4:
        return _CONTD_MARKER_RE.sub("", line.rstrip())
    return None


def normalize_bullet_markers(notes_text: str) -> str:
    """Normalize list markers so independently generated sections agree.

    Parallel chunk generation lets each section pick its own bullet style —
    one section emits ``- point``, the next ``*   point``, and some emit a
    doubled marker (``*   - **Name:** point``). Rewrites every ``*`` list
    marker to ``-``, collapses doubled markers, and normalizes the gap after
    the marker to one space, leaving indentation (and therefore nesting)
    untouched. Bold text (``**``), and marker-only lines such as ``***`` /
    ``* * *`` horizontal rules, are never modified."""
    if not notes_text:
        return notes_text
    out_lines = []
    for line in notes_text.split("\n"):
        if not re.match(r"^\s*[*\-\s]+$", line):
            line = re.sub(r"^(\s*)\*(\s+)", r"\1-\2", line)
            while re.match(r"^\s*-\s+[-*]\s+", line):
                line = re.sub(r"^(\s*)-\s+[-*]\s+", r"\1- ", line, count=1)
            line = re.sub(r"^(\s*)-\s+", r"\1- ", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def merge_continuation_seams(sections: List[str]) -> str:
    """Concatenate per-section chunk notes, attaching mid-answer
    continuations to the block they continue.

    A section that starts in the middle of an answer re-states the question
    as an inferred heading, which the continuation prompt tags with a
    ``(contd.)`` marker. When that marker heading is the first line of a
    section, the same question already closes the previous section — so the
    duplicate heading is dropped and its bullets are appended directly to
    the previous section's last block. A marker anywhere else (or following
    a missing-section placeholder, where there is nothing to attach to)
    leaves the block in place; ``cleanup_stitched_notes`` strips the stray
    marker text afterwards."""
    parts: List[str] = []
    for section in sections:
        text = (section or "").strip()
        if not text:
            continue
        lines = text.split("\n")
        prev_attachable = bool(parts) and "[MISSING SECTION" not in parts[-1].split("\n")[-1]
        if prev_attachable and _contd_heading_cleaned(lines[0]) is not None:
            body = "\n".join(lines[1:]).strip("\n")
            if body.strip():
                parts[-1] = parts[-1].rstrip() + "\n" + body
            continue
        parts.append(text)
    return "\n\n".join(parts).strip()


def cleanup_stitched_notes(notes_text: str) -> str:
    """Deterministic cleanup of concatenated chunk notes — no LLM call, zero
    risk of content loss.

    Handles:
    0. Normalize bullet markers that drift between parallel sections
    1. Remove meta-commentary / processing artifacts from chunked generation,
       and strip stray ``(contd.)`` continuation markers left on headings
    2. Collapse duplicate consecutive headings across chunk boundaries
    3. Fix formatting: excessive blank lines, trailing whitespace
    """
    if not notes_text or not notes_text.strip():
        return notes_text

    # --- 0. Normalize bullet markers ---
    notes_text = normalize_bullet_markers(notes_text)

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
            # A (contd.) marker that survived seam merging (e.g. the model put
            # it on a mid-section heading) is an artifact — drop just the marker.
            without_marker = _contd_heading_cleaned(line)
            cleaned_lines.append(without_marker if without_marker is not None else line)

    # --- 2. Collapse duplicate adjacent headings ---
    # A chunk seam can re-state the previous block's heading verbatim (or
    # verbatim once its (contd.) marker was stripped above). Comparing each
    # heading against the heading of the block it would interrupt — not just
    # the previous line — drops the duplicate so its bullets flow into the
    # block they continue. Repeats further apart (with another heading in
    # between) are legitimate structure and stay.
    result_lines = []
    block_heading = None
    for line in cleaned_lines:
        stripped = line.strip()
        heading_match = re.match(r'^(?:#{1,6}\s+)?(\*\*.+?\*\*)\s*$', stripped)
        if heading_match:
            if stripped == block_heading:
                # Skip duplicate heading — keep first occurrence
                continue
            block_heading = stripped
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

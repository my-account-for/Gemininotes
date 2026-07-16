"""Unit tests for chunking.py — pure text functions, no Streamlit needed."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import (
    create_chunks_with_context,
    estimate_chunk_count,
    cleanup_stitched_notes,
    merge_continuation_seams,
    normalize_bullet_markers,
    strip_overlap,
    strip_asr_meta_markers,
    merge_learning_doc_sections,
    split_qa_blocks,
    flatten_grouping_plan,
    reorder_qa_blocks,
)


def _make_turns(n_turns, words_per_turn, prefix="Speaker"):
    """Build a fake transcript of n_turns paragraphs."""
    turns = []
    for i in range(n_turns):
        words = " ".join(f"w{i}_{j}" for j in range(words_per_turn))
        turns.append(f"**{prefix} {i % 2 + 1}:** {words}")
    return "\n\n".join(turns)


def test_empty_input():
    assert create_chunks_with_context("", 100, 10) == []
    assert create_chunks_with_context("   \n ", 100, 10) == []


def test_short_text_single_chunk():
    text = "Hello world, this is a short transcript."
    chunks = create_chunks_with_context(text, 100, 10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["context"] == ""


def test_chunks_are_non_overlapping_and_lossless():
    text = _make_turns(40, 50)  # 2000 words across 40 turns
    chunks = create_chunks_with_context(text, 300, 50)
    assert len(chunks) > 1
    # Rejoining the chunk texts must reproduce every word exactly once, in order.
    rejoined = "\n\n".join(c["text"] for c in chunks)
    assert rejoined.split() == text.split()


def test_chunks_respect_turn_boundaries():
    text = _make_turns(40, 50)
    chunks = create_chunks_with_context(text, 300, 50)
    for c in chunks:
        # Every chunk must start at a turn boundary, not mid-answer.
        assert c["text"].startswith("**Speaker")


def test_context_is_tail_of_previous_chunk():
    text = _make_turns(40, 50)
    chunks = create_chunks_with_context(text, 300, 50)
    assert chunks[0]["context"] == ""
    for prev, cur in zip(chunks, chunks[1:]):
        expected = " ".join(prev["text"].split()[-50:])
        assert cur["context"] == expected


def test_zero_context_words():
    text = _make_turns(40, 50)
    chunks = create_chunks_with_context(text, 300, 0)
    assert all(c["context"] == "" for c in chunks)


def test_oversized_single_turn_is_split():
    # One giant 1000-word turn with sentences, chunk size 300.
    sentences = " ".join(
        "This is sentence number {} with some filler words to pad.".format(i)
        for i in range(100)
    )
    chunks = create_chunks_with_context(sentences, 300, 30)
    assert len(chunks) > 1
    rejoined_words = []
    for c in chunks:
        rejoined_words.extend(c["text"].split())
    assert rejoined_words == sentences.split()
    for c in chunks:
        assert len(c["text"].split()) <= 300


def test_no_punctuation_pathological_input():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = create_chunks_with_context(text, 300, 30)
    rejoined_words = []
    for c in chunks:
        rejoined_words.extend(c["text"].split())
    assert rejoined_words == text.split()


def test_tiny_trailing_chunk_is_merged():
    # 3 turns of 290 words + 1 turn of 20 words, chunk size 300:
    # without merging the last 20-word turn would be its own chunk.
    text = _make_turns(3, 290) + "\n\n**Speaker 1:** " + " ".join(
        f"tail{j}" for j in range(20)
    )
    chunks = create_chunks_with_context(text, 300, 30)
    assert len(chunks[-1]["text"].split()) >= 30  # not a 20-word orphan


def test_estimate_chunk_count():
    assert estimate_chunk_count(50, 100) == 1
    assert estimate_chunk_count(100, 100) == 1
    assert estimate_chunk_count(101, 100) == 2
    assert estimate_chunk_count(1000, 300) == 4


def test_cleanup_removes_meta_commentary():
    notes = (
        "**Question one?**\n- A point.\n\n"
        "The transcript does not contain an answer to this question.\n\n"
        "**Question two?**\n- Another point."
    )
    cleaned = cleanup_stitched_notes(notes)
    assert "does not contain" not in cleaned
    assert "**Question one?**" in cleaned
    assert "- Another point." in cleaned


def test_cleanup_collapses_duplicate_consecutive_headings():
    notes = "**Same heading**\n\n**Same heading**\n- bullet under it"
    cleaned = cleanup_stitched_notes(notes)
    assert cleaned.count("**Same heading**") == 1
    assert "- bullet under it" in cleaned


def test_cleanup_preserves_distinct_headings():
    notes = "**Heading A**\n- a\n\n**Heading B**\n- b"
    cleaned = cleanup_stitched_notes(notes)
    assert "**Heading A**" in cleaned
    assert "**Heading B**" in cleaned


# --- strip_overlap: dedup of overlapping audio-chunk transcripts ---

def test_strip_overlap_removes_duplicated_seam():
    prev = "So the quick brown fox jumps over the lazy dog near the river bank today"
    nxt = "jumps over the lazy dog near the river bank today and then we discussed pricing"
    out = strip_overlap(prev, nxt)
    assert out == "and then we discussed pricing"


def test_strip_overlap_tolerates_punctuation_and_case_differences():
    # Two ASR passes over the same audio render punctuation/casing differently.
    prev = "He said the package gives you fifty five thousand dollars of credit, right? Yes exactly."
    nxt = "The package gives you fifty five thousand Dollars of credit right... yes, EXACTLY! Then the next topic began."
    out = strip_overlap(prev, nxt)
    assert out == "Then the next topic began."


def test_strip_overlap_no_match_returns_unchanged():
    prev = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    nxt = "one two three four five six seven eight nine ten"
    assert strip_overlap(prev, nxt) == nxt


def test_strip_overlap_short_spurious_match_is_not_cut():
    # A common short phrase must not be mistaken for the seam.
    prev = "we talked about revenue you know and margins were up a lot this quarter overall"
    nxt = "you know the other thing is churn stayed flat and retention improved further still"
    assert strip_overlap(prev, nxt) == nxt


def test_strip_overlap_requires_match_at_prev_tail():
    # The duplicated words exist but prev continues well past them, so this
    # is not the seam — next must be returned unchanged.
    shared = "one two three four five six seven eight nine ten"
    prev = shared + " " + " ".join(f"filler{i}" for i in range(30))
    nxt = shared + " completely new content follows here"
    assert strip_overlap(prev, nxt) == nxt


def test_strip_overlap_empty_inputs():
    assert strip_overlap("", "some text") == "some text"
    assert strip_overlap("some text", "") == ""


def test_strip_overlap_whole_next_duplicated():
    # A tiny tail chunk can be fully contained in the overlap window.
    prev = " ".join(f"word{i}" for i in range(40))
    nxt = " ".join(f"word{i}" for i in range(28, 40))
    assert strip_overlap(prev, nxt) == ""


def test_strip_overlap_tolerates_asr_variance_across_passes():
    # Two ASR passes over the same audio rarely agree word-for-word: numbers
    # come back as digits vs. words and odd words get garbled. The aligned
    # blocks still identify the seam.
    prev = ("earlier discussion continues here and we saw ranges of one five ten "
            "and twenty percent across markets so the average is around four point "
            "five percent depending on the mix")
    nxt = ("we saw ranges of 1, 5, 10 and 20 percent across markets so the average "
           "is about 4.5 percent depending on the mix and then the Philippines "
           "discussion started")
    out = strip_overlap(prev, nxt)
    assert out == "and then the Philippines discussion started"


# --- strip_asr_meta_markers: model-added framing around chunk transcripts ---

def test_strip_asr_meta_markers_removes_framing_lines():
    text = "[Beginning of transcript]\nSpeaker 1: hello there.\n\n[END OF RECORDING]"
    cleaned = strip_asr_meta_markers(text)
    assert "END OF RECORDING" not in cleaned
    assert "Beginning of transcript" not in cleaned
    assert "Speaker 1: hello there." in cleaned


def test_strip_asr_meta_markers_keeps_content_and_gap_markers():
    gap = "[TRANSCRIPTION GAP: audio from 30:00-35:00 could not be transcribed after 3 attempts.]"
    sentence = "He said the end of the recording session was near, then continued."
    text = f"{sentence}\n{gap}"
    assert strip_asr_meta_markers(text) == text


# --- merge_learning_doc_sections: chunked Internal Discussion output ---

STITCHED_DOC = """### **Title: Seek — learnings, insights and mental models**
Subtitle: read-across to Info Edge.

### **1. Topic A**
*   Point one.

### **2. Consolidated Mental Models**
*   **Model one.**

### **3. Unanswered Questions**
1.  Question one?

### **4. Follow-Ups / Action Items**
1.  Task one.

### **1. Topic B**
*   Point two.

### **Consolidated Mental Models**
*   **Model two.**
*   **Model one.**

### **Unanswered Questions**
1.  Question two?

### **Follow-Ups / Action Items**
1.  Task two.
"""


def test_merge_learning_doc_end_sections_appear_once_at_end():
    merged = merge_learning_doc_sections(STITCHED_DOC)
    assert merged.count("Consolidated Mental Models") == 1
    assert merged.count("Unanswered Questions") == 1
    assert merged.count("Follow-Ups / Action Items") == 1
    assert (merged.index("Topic B")
            < merged.index("Consolidated Mental Models")
            < merged.index("Unanswered Questions")
            < merged.index("Follow-Ups / Action Items"))


def test_merge_learning_doc_renumbers_topics_and_items():
    merged = merge_learning_doc_sections(STITCHED_DOC)
    assert "**1. Topic A**" in merged
    assert "**2. Topic B**" in merged  # restarted numbering fixed
    assert "**3. Consolidated Mental Models**" in merged  # final numbered topic
    assert "1.  Question one?" in merged
    assert "2.  Question two?" in merged
    assert "1.  Task one." in merged
    assert "2.  Task two." in merged


def test_merge_learning_doc_dedups_exact_duplicates():
    merged = merge_learning_doc_sections(STITCHED_DOC)
    assert merged.count("Model one.") == 1
    assert merged.count("Model two.") == 1


def test_merge_learning_doc_preserves_all_content():
    merged = merge_learning_doc_sections(STITCHED_DOC)
    for needle in ("Title: Seek", "read-across to Info Edge", "Point one.", "Point two.",
                   "Model one.", "Model two.", "Question one?", "Question two?",
                   "Task one.", "Task two."):
        assert needle in merged


def test_merge_learning_doc_without_end_sections_unchanged():
    doc = "### **1. Topic**\n* a point\n\n### **2. Another**\n* b point"
    assert merge_learning_doc_sections(doc) == doc


# --- Q&A post-processing helpers: split / grouping plan / reorder ---

QA_NOTES = """Expert background:
- 20 years in hotel distribution.

**What is the market size?**
- Roughly $2B, growing 15% a year.

**How do hotels view OTAs?**
- Mixed: they value the demand but resent take rates.
- **Deep Kalra:** attribution bullets must not be treated as headings.

**What are the take rates?**
- 15-25% depending on chain scale.

**Any regulatory risks?**
- Rate-parity clauses under scrutiny in the EU.
"""


def test_split_qa_blocks_basic():
    preamble, blocks = split_qa_blocks(QA_NOTES)
    assert preamble.startswith("Expert background")
    assert len(blocks) == 4
    assert blocks[0].startswith("**What is the market size?**")
    assert "$2B" in blocks[0]


def test_split_qa_blocks_ignores_inline_bold_attribution():
    _, blocks = split_qa_blocks(QA_NOTES)
    # The bold "- **Deep Kalra:** ..." line stays inside its answer block.
    assert "Deep Kalra" in blocks[1]


def test_split_qa_blocks_no_headings():
    text = "Just some plain notes with no bold headings."
    preamble, blocks = split_qa_blocks(text)
    assert preamble == text
    assert blocks == []


def test_split_qa_blocks_empty_input():
    assert split_qa_blocks("") == ("", [])
    assert split_qa_blocks("  \n ") == ("", [])


def test_split_and_reorder_identity_is_lossless():
    preamble, blocks = split_qa_blocks(QA_NOTES)
    rejoined = reorder_qa_blocks(preamble, blocks, list(range(len(blocks))))
    assert rejoined.split() == QA_NOTES.split()


def test_reorder_qa_blocks_moves_whole_blocks():
    preamble, blocks = split_qa_blocks(QA_NOTES)
    text = reorder_qa_blocks(preamble, blocks, [0, 2, 1, 3])
    # Take-rates block now directly follows market size; preamble stays first.
    assert (text.index("Expert background")
            < text.index("**What is the market size?**")
            < text.index("**What are the take rates?**")
            < text.index("**How do hotels view OTAs?**")
            < text.index("**Any regulatory risks?**"))
    # Reordering must not lose or duplicate a single word.
    assert sorted(text.split()) == sorted(QA_NOTES.split())


def test_flatten_grouping_plan_valid_plan():
    topics = [
        {"name": "Market", "blocks": [0, 2]},
        {"name": "Risk", "blocks": [1, 3]},
    ]
    order, cleaned = flatten_grouping_plan(topics, 4)
    assert order == [0, 2, 1, 3]
    assert [t["name"] for t in cleaned] == ["Market", "Risk"]


def test_flatten_grouping_plan_repairs_missing_duplicate_and_out_of_range():
    # The model repeated block 1, forgot block 2, and invented block 9.
    topics = [
        {"name": "A", "blocks": [0, 1]},
        {"name": "B", "blocks": [1, 3, 9]},
    ]
    order, cleaned = flatten_grouping_plan(topics, 4)
    assert sorted(order) == [0, 1, 2, 3]  # a full permutation — nothing lost
    # The forgotten block is re-inserted right after its original neighbour.
    assert order.index(2) == order.index(1) + 1
    assert cleaned == [{"name": "A", "blocks": [0, 1]}, {"name": "B", "blocks": [3]}]


def test_flatten_grouping_plan_garbage_falls_back_to_original_order():
    order, cleaned = flatten_grouping_plan([{"name": "X", "blocks": ["a", None, True]}], 3)
    assert order == [0, 1, 2]
    assert cleaned == []


def test_flatten_grouping_plan_empty_plan():
    order, cleaned = flatten_grouping_plan([], 3)
    assert order == [0, 1, 2]
    assert cleaned == []


# --- normalize_bullet_markers ---


def test_normalize_star_bullets_become_dashes():
    text = "**Q?**\n*   First point.\n*   Second point.\n    * Nested point."
    out = normalize_bullet_markers(text)
    assert out == "**Q?**\n- First point.\n- Second point.\n    - Nested point."


def test_normalize_collapses_doubled_markers():
    # Sections sometimes emit a doubled marker: `*   - **Name:** point`.
    text = "*   - **Mohit Joshi:** Revenue grew 6.1% YoY."
    assert normalize_bullet_markers(text) == "- **Mohit Joshi:** Revenue grew 6.1% YoY."


def test_normalize_leaves_bold_and_rules_alone():
    text = "**Bold heading**\n***\n* * *\n---\n- Already fine."
    assert normalize_bullet_markers(text) == text


def test_normalize_does_not_touch_content_dashes():
    # A bullet whose content starts with a negative number keeps its dash.
    text = "- -5% decline in the segment."
    assert normalize_bullet_markers(text) == text


# --- merge_continuation_seams ---


def test_seam_merge_folds_contd_block_into_previous_section():
    sections = [
        "**Were you surprised by Q1?**\n- **CEO:** Results beat plan.",
        "**Were you surprised by the performance? (contd.)**\n"
        "- Demand challenges include manual testing.\n\n"
        "**What drives growth next?**\n- **CFO:** Large deal ramp-ups.",
    ]
    out = merge_continuation_seams(sections)
    # The duplicate inferred heading is gone; its bullets attach to the
    # previous section's last block, and later blocks are untouched.
    assert "(contd.)" not in out
    assert out.count("**Were you surprised") == 1
    _, blocks = split_qa_blocks(out)
    assert len(blocks) == 2
    assert "Demand challenges include manual testing." in blocks[0]


def test_seam_merge_plain_sections_join_unchanged():
    sections = ["**Q1?**\n- A.", "**Q2?**\n- B."]
    assert merge_continuation_seams(sections) == "**Q1?**\n- A.\n\n**Q2?**\n- B."


def test_seam_merge_handles_marker_variants_and_hash_headings():
    for marker in ["(contd.)", "(contd)", "(cont'd)", "(Continued)"]:
        sections = ["**Q?**\n- A.", f"### **Q? {marker}**\n- B."]
        out = merge_continuation_seams(sections)
        assert marker not in out
        assert out == "**Q?**\n- A.\n- B."


def test_seam_merge_skips_missing_section_placeholder():
    placeholder = "**[MISSING SECTION 1 of 2]** — no notes could be generated."
    sections = [placeholder, "**Q? (contd.)**\n- Tail of an answer."]
    out = merge_continuation_seams(sections)
    # Nothing to attach to — the block stays standalone with its heading.
    assert placeholder in out
    assert "**Q? (contd.)**" in out


def test_seam_merge_ignores_mid_section_markers():
    # A marker deeper inside a section is not a seam; the block stays put
    # (cleanup_stitched_notes strips the stray marker text later).
    sections = [
        "**Q1?**\n- A.",
        "**Q2?**\n- B.\n\n**Q3? (contd.)**\n- C.",
    ]
    out = merge_continuation_seams(sections)
    _, blocks = split_qa_blocks(out)
    assert len(blocks) == 3


def test_cleanup_strips_stray_contd_markers_from_headings():
    text = "**Q1? (contd.)**\n- A point.\n\n- Not a heading (contd.) example."
    out = cleanup_stitched_notes(text)
    assert "**Q1?**" in out
    # Non-heading lines keep their text untouched.
    assert "- Not a heading (contd.) example." in out


def test_cleanup_collapses_heading_duplicated_by_stripped_marker():
    text = "**Q1?**\n- A.\n\n**Q1? (contd.)**\n- B."
    out = cleanup_stitched_notes(text)
    assert out.count("**Q1?**") == 1
    assert "- A." in out and "- B." in out


def test_cleanup_normalizes_bullets_across_sections():
    text = "**Q1?**\n*   - **CEO:** Point one.\n*   Point two."
    out = cleanup_stitched_notes(text)
    assert "- **CEO:** Point one." in out
    assert "- Point two." in out

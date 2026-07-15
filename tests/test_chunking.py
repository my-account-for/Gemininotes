"""Unit tests for chunking.py — pure text functions, no Streamlit needed."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import (
    create_chunks_with_context,
    estimate_chunk_count,
    cleanup_stitched_notes,
    strip_overlap,
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

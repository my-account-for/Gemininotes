# /----------------------------\
# |   START OF prompts.py FILE  |
# \----------------------------/

"""All LLM prompt templates and prompt-adjacent instruction tables.

Pure constants — no Streamlit or API dependencies. Kept separate from
app.py so prompt iterations don't touch application logic.
"""

# Appended to both Expert Meeting prompts. Written for panels/multi-speaker
# calls but the ordering rules apply to 1:1 calls too. Counters the model's
# tendency — strongest on large sections — to reorganize by theme, merge
# similar questions into one heading, and drop per-speaker attribution.
PANEL_HANDLING_SECTION = """

**(3.) ORDER & MULTI-SPEAKER HANDLING:**
-   **PRESERVE INTERVIEW ORDER:** Output Q&A pairs strictly in the order they occur in the transcript. Do NOT reorganize, regroup, or cluster questions by theme — the notes must follow the conversation as it unfolded.
-   **ONE QUESTION = ONE HEADING:** Never merge separate questions into a single heading, even when they are similar or related. If the moderator asks each panellist the same or a similar question in turn (e.g., each founder's origin story), create a SEPARATE bold question for each respondent, with the respondent's name in the heading (e.g., **How did the company get started? — Rajesh Magow**).
-   **ATTRIBUTE EVERY ANSWER:** When more than two people speak and names are known, attribute the content. If multiple panellists contribute to the same question, start each panellist's contribution with their name in bold (e.g., `- **Deep Kalra:** ...`) and keep each panellist's points grouped together under that question.
-   **NEVER BLUR ATTRIBUTION:** Do not collapse different speakers' views into unattributed bullets, and do not fall back to generic labels like "Speaker 1" when the person's name is known anywhere in the transcript or in the speaker list provided above."""

# Injected into refinement and speaker-ID prompts. Audio transcripts garble
# proper nouns phonetically ("Alog Baji" for Aloke Bajpai, "My BusinessME"
# for myBiz) — downstream stages then propagate the garble consistently.
# Correction has to happen here, at the refinement stage, with an explicit
# flag for anything the model can't confidently identify.
ASR_CORRECTION_INSTRUCTION = """PROPER-NOUN CORRECTION: This transcript comes from automatic speech recognition, so names of people, companies, products, and websites are often phonetically garbled. Where a garbled proper noun clearly refers to a real, identifiable entity given the context — a company's founder, a known brand or product, a well-known website — correct it to its canonical real-world spelling (e.g., a garbled rendering of a travel company founder's name should become the founder's actual name; a garbled product name like "My BusinessME" in a MakeMyTrip discussion should become "myBiz"). Apply the same corrected spelling consistently throughout. If you CANNOT identify the real entity with reasonable confidence, keep the transcript's spelling and append [sp?] immediately after it to flag it for human review — never invent a plausible-looking name."""

# Appended to every refinement prompt. Refinement exists to clean up ASR
# output, not to shorten it — without an explicit completeness contract the
# model occasionally skips or condenses passages, and downstream stages have
# no way to detect the loss.
REFINEMENT_COMPLETENESS_INSTRUCTION = """COMPLETENESS: Refine EVERY sentence of the transcript from start to finish. Do NOT omit, summarize, or condense anything — filler words ("um", "uh") may be dropped, but every question, answer, aside, digression, and repeated point must appear in your output. The refined text should be roughly the same length as the input. Preserve any [TRANSCRIPTION GAP: ...] or [POSSIBLE MISSING CONTENT: ...] markers exactly as they appear — they flag audio that could not be transcribed."""

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
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label. The bold question text stands alone.
-   If the questioner provides context, framing, or a multi-part question, capture the full scope — do not reduce a multi-part question to a single line.
-   **LONG QUESTIONS / PREAMBLE:** Sometimes a question is long because the interviewer provides substantial framing, background, or context before asking — this preamble is part of the question and must be preserved as part of the bold question text. Do NOT treat the preamble as part of the answer.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question, so each Q&A pair is visually separated.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question (no blank line between the bold question and its first bullet).
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   Use **multiple bullet points** per answer — do NOT collapse a detailed response into a single bullet.
-   **ZERO SKIPPING RULE:** If the expert said it with substance, it must appear in your notes. Do NOT skip examples, anecdotes, specific sentences, or supporting details even if they seem minor or repetitive. Every distinct point gets its own bullet. If an answer contains 8 substantive points, you must produce at least 8 bullets — never condense them into 3-4.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, specific entities mentioned, time periods, market sizes, growth rates, company names, product names, and geographies.
-   **NAMED LISTS & COINED LABELS:** If the speaker lists named examples (brands, hotels, companies, apps), capture EVERY name in the list — never compress a list of names into "several players" or similar. If a speaker coins or uses a memorable label, metaphor, or framework (e.g., "Two Indias"), preserve that label verbatim in quotes — these frames are often the most quoted part of a call. Preserve any [sp?] flags that appear in the transcript: they mark unverified name spellings and must survive into the notes.
-   **PRIORITY #2: CAPTURE ALL NUANCE & REASONING.** Do not over-summarize or reduce complex answers to surface-level statements. You must retain the following:
    -   **Sentiment & Tone:** Note if the expert is confident, uncertain, speculative, cautious, or enthusiastic (e.g., "The expert was highly confident that...," "He cautioned that...").
    -   **Qualifiers & Conditions:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "except in," "only when," "roughly," "approximately," "a potential risk is...").
    -   **Key Examples & Analogies:** If the expert uses a specific example, anecdote, case study, or analogy to illustrate a point, capture it in full, even if it spans multiple sentences — these are often the most valuable parts of an expert call.
    -   **Cause & Effect:** Retain any reasoning chains provided (e.g., "...because of regulatory changes," "...which led to a 15% decline in...").
    -   **Comparisons & Contrasts:** If the expert compares companies, products, approaches, or time periods, capture both sides of the comparison with the specific details for each.
    -   **Tangential but relevant points:** If the expert volunteers additional context, background, or related information beyond the direct question, include it — do NOT discard it as off-topic.
-   **PRIORITY #3: PRESERVE MULTI-STEP EXPLANATIONS.** If an answer involves a sequence of steps, a timeline, or a logical chain, preserve the full sequence rather than summarizing the conclusion only.""" + PANEL_HANDLING_SECTION

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
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label. The bold question text stands alone.
-   **LONG QUESTIONS / PREAMBLE:** Sometimes a question is long because the interviewer provides substantial framing, background, or context before asking — this preamble is part of the question and must be preserved as part of the bold question text. Do NOT treat the preamble as part of the answer.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question, so each Q&A pair is visually separated.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question (no blank line between the bold question and its first bullet).
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, and specific entities mentioned.
-   **NAMED LISTS & COINED LABELS:** If the speaker lists named examples (brands, hotels, companies, apps), capture EVERY name — never compress a list of names into "several players" or similar. If a speaker coins a memorable label or framework (e.g., "Two Indias"), preserve it verbatim in quotes. Preserve any [sp?] flags from the transcript — they mark unverified name spellings.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Do not over-summarize. You must retain the following:
    -   **Sentiment & Tone:** Note if the speaker is optimistic, hesitant, confident, or speculative (e.g., "The expert was cautiously optimistic about...", "He speculated that...").
    -   **Qualifiers:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "rarely," "a potential risk is...").
    -   **Key Examples & Analogies:** If the speaker uses a specific example to illustrate a point, capture it, even if it's a few sentences long.
    -   **Cause & Effect:** Retain any reasoning provided (e.g., "...because of the new regulations," "...which led to a decrease in...").""" + PANEL_HANDLING_SECTION

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

**COVERAGE RULE:** Your output must contain one Q&A block for EVERY question asked in the transcript below. Count them: if the transcript contains 12 questions, your notes must contain 12 bold question headings. Do NOT merge, combine, or skip questions to save space — long transcripts deserve proportionally long notes.
---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are a High-Fidelity Factual Extraction Engine continuing a note-taking task from a long transcript that is being processed in sequential sections.

### **CONTEXT — ALREADY PROCESSED (do NOT take notes on this)**
The text below is the tail end of the transcript section that was already processed. Notes for it already exist. It is provided ONLY so you can follow the flow of the conversation across the section boundary. Do NOT produce notes for anything that appears only in this context block.

...{context_tail}

### **CONTINUATION INSTRUCTIONS**
1.  **PROCESS THE ENTIRE NEW SECTION:** Take notes on the **entire** new transcript section provided below. Every substantive point, example, data point, and nuanced opinion in the new section MUST appear in your output.
2.  **MID-ANSWER STARTS:** If the new section begins in the middle of a response, capture that content as bullet points under a clear question heading inferred from the context block. Do not skip or discard partial content.
3.  **MAINTAIN FORMAT:** Continue to use the exact same formatting as established in the base instructions.
4.  **NO META-COMMENTARY:** NEVER produce statements about the transcript itself, such as "the transcript does not contain an answer," "no relevant information in this section," "the section starts mid-conversation," or similar. Always extract and document whatever substantive content exists.
5.  **MAINTAIN OUTPUT VOLUME:** This section contains as much content as any other. Your output MUST be equally detailed and equally long. Do NOT produce a shorter or more condensed output just because this is a continuation. Do NOT taper off, summarize, or become briefer.
6.  **COVERAGE RULE:** Your output must contain one Q&A block for EVERY question asked in the new section. Count them: if it contains 12 questions, your notes must contain 12 bold question headings. Do NOT merge, combine, or skip questions to save space.

---
{base_instructions}
---

**MEETING TRANSCRIPT (NEW SECTION — PROCESS ALL OF IT):**
{chunk_text}
"""

VALIDATION_DETAILED_PROMPT = """You are a rigorous Transcript Completeness Auditor performing a fact-by-fact audit of processed meeting notes against the source transcript.

## INPUTS

### FULL PROCESSED NOTES (complete — for reference when checking missing content):
{full_notes}

### PORTION TO ANNOTATE ({chunk_info}):
{chunk_to_annotate}

### SOURCE TRANSCRIPT (Ground Truth):
{transcript}

## CRITICAL UNDERSTANDING

Notes are always paraphrased and restructured versions of the transcript — this is intentional and CORRECT. You must NEVER flag paraphrasing, rephrasing, reorganisation, or compression as errors. The note-taking AI's job is to restructure, not transcribe verbatim.

**Cross-chunk context:** The FULL PROCESSED NOTES above contain all Q&As from this call. When checking for missing content, check the FULL NOTES — if a piece of transcript content is captured *anywhere* in the full notes (even outside the PORTION TO ANNOTATE), do NOT flag it as missing.

## WHAT TO FIND — BE RIGOROUS

**1. MISSING CONTENT** (most important — go fact by fact)

Walk through the TRANSCRIPT systematically, exchange by exchange. For each expert response, check the FULL NOTES for every one of the following:

- **Every specific number, percentage, monetary value, metric, or growth rate** — even a single missing figure is a gap
- **Every named entity** — companies, people, product names, geographies, regulatory bodies, specific time periods
- **Every distinct example, anecdote, or case study** the expert used to illustrate a point — these are high-value and frequently dropped
- **Every qualifier or hedge that changes meaning** — "roughly," "typically," "only in certain cases," "except when," "approximately," "we think," "possibly" — omitting these alters the meaning materially
- **Every distinct reasoning chain or cause-effect link** — e.g., "because X, Y happened, which led to Z"
- **Every comparison or contrast** — if the expert compared two companies, sectors, or time periods, check both sides are captured
- **Every explicitly stated uncertainty or caveat** — if the expert said they were unsure, speculating, or hedging, that tone must be preserved

Only flag as MISSING if the fact, name, number, or nuance genuinely does not appear anywhere in the FULL NOTES.

**2. MISREPRESENTATION** (apply sparingly but precisely)

Content in the PORTION TO ANNOTATE that factually contradicts or distorts the transcript:
- Wrong number (transcript: 30%, notes: 20%)
- Wrong direction (transcript: declining, notes: growing)
- Wrong entity name or wrong speaker attribution
- Expert expressed uncertainty but notes state it as established fact, or vice versa
- A "could" or "might" in the transcript rendered as a definitive claim in the notes

**3. REPEATED Q&A** (check the FULL NOTES)

Scan the FULL PROCESSED NOTES for Q&A pairs that cover substantially the same question or repeat the same answer content. This happens when chunked note generation produces near-duplicate sections due to transcript overlap. Flag as repeated if:
- Two bold questions ask essentially the same thing (even if worded differently)
- An answer block appears twice with the same or very similar bullet points
- A topic or data point is covered in near-identical language in two separate Q&A pairs

## WHAT NOT TO FLAG

- Paraphrasing → CORRECT
- Restructuring or reordering → CORRECT
- Compression where key facts are still present → CORRECT
- Filler, false starts, rambling clean-up → CORRECT
- Minor synonym substitutions that preserve meaning → CORRECT
- A topic mentioned briefly in one Q&A and fully covered in another → NOT a repeat (only flag true duplicates)

## ANNOTATIONS — THREE TYPES ONLY

Do NOT use any markup other than these three exact formats.

**MISSING CONTENT** — insert immediately after the Q&A pair in the PORTION TO ANNOTATE where the gap is most relevant:
`<div style="background:#fef9c3;border-left:3px solid #ca8a04;padding:5px 10px;margin:6px 0;font-size:0.88em;color:#78350f">⚠️ <strong>Missing:</strong> [quote or precisely describe the specific fact, number, name, qualifier, or example from the transcript that is absent from the full notes]</div>`

**MISREPRESENTATION** — wrap only the specific wrong text, immediately followed by an inline correction:
`<del style="color:#dc2626">the wrong text as it appears in the notes</del><span style="color:#16a34a;font-size:0.9em"> → [what the transcript actually says]</span>`

**REPEATED Q&A** — insert immediately before the duplicate bold question in the PORTION TO ANNOTATE:
`<div style="background:#ede9fe;border-left:3px solid #7c3aed;padding:5px 10px;margin:6px 0;font-size:0.88em;color:#5b21b6">🔁 <strong>Duplicate:</strong> This Q&A substantially repeats [describe which earlier Q&A it duplicates and what the overlapping content is]</div>`

**Correct content** — leave exactly as-is. No annotation whatsoever.

## OUTPUT

Output ONLY the annotated PORTION TO ANNOTATE, preserving its exact structure (bold questions, bullet points, spacing). Do NOT output the full notes section, and do NOT add any summary, preamble, footer, or meta-commentary of any kind."""

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

# --- SPEAKER IDENTIFICATION PROMPTS (Option 4 of Expert Meeting) ---

SPEAKER_ID_PROMPT_INITIAL = """You are refining a transcript AND identifying distinct speakers.

## TASK
1. Clean up the transcript: fix spelling, grammar, punctuation, and conversational filler. Translate any non-English content into clear, natural English while preserving meaning and tone.
2. Identify distinct speakers. **ASSUME 2 SPEAKERS by default.** Only introduce a 3rd speaker if you are highly confident a clearly distinct third voice is present (e.g., a different role explicitly introduced, a third name addressed in the conversation, or unambiguously different perspective sustained across multiple turns).
   - **ROLE ANCHORING (critical):** In interview-style calls (expert calls, channel checks, analyst interviews), assign **Speaker 1 to the interviewer/analyst** — the person who asks questions, sets the agenda, and speaks in short turns — and **Speaker 2 to the expert/respondent** — the person giving long, substantive answers. Decide this mapping from the first few turns and apply it consistently for the ENTIRE transcript. Question turns and answer turns must never share a label.
3. Tag any **off-topic logistical chatter** with `**Skip:**` instead of `**Speaker N:**`. Logistics includes:
   - Tech checks: "can you hear me?", "let me share my screen", "your mic is muted", "is the recording on?"
   - Personal/comfort: "can I get a charger?", "do you want water?", "should we order food?", "let me grab my notes"
   - Scheduling/breaks: "let's take a 5 minute break", "are we done early?", "we have a hard stop at 4"
   - Greetings/sign-offs that carry no substantive content
   - Side chatter unrelated to the meeting's subject
   When in doubt about whether something is logistical or substantive, prefer `Speaker N`.
4. Output the cleaned transcript with EVERY turn prefixed by either `**Speaker N:**` or `**Skip:**` on its own line.

## OUTPUT FORMAT (strict)
**Speaker 1:**
<first turn text>

**Skip:**
<logistical side-chatter>

**Speaker 2:**
<next substantive turn>

...

### Rules
- Every turn MUST start with `**Speaker N:**` (N is 1, 2, or 3) or `**Skip:**` on its own line.
- Use ONLY generic labels: `Speaker 1`, `Speaker 2`, optionally `Speaker 3`, or `Skip`. Do NOT use real names even if mentioned.
- Leave exactly ONE blank line between turns.
- Do NOT include any meta-commentary, headings, framing text, or summaries — output ONLY the tagged transcript.
- If two consecutive lines are from the same speaker (or both Skip), merge them under one block. A brief interjection ("right", "okay", "mm-hmm") that merely acknowledges and does not interrupt the flow should be dropped rather than breaking the other speaker's turn into pieces.
- **SELF-CHECK before output:** scan your tagged transcript once. If any segment labelled as the expert is clearly a question directed AT the expert (or vice versa), fix its label. Verify no two consecutive blocks share the same label.

{speaker_info}
{refinement_extra}

## TRANSCRIPT
{transcript}
"""

SPEAKER_ID_PROMPT_CONTINUATION = """You are continuing to refine a long transcript with speaker identification.

You have already established the following speaker labels in earlier chunks: **{speakers_so_far}**.
Continue using EXACTLY these same labels. Do NOT introduce a new speaker unless you are highly confident a clearly new voice appears that was not present earlier.

## CONTEXT FROM PREVIOUS CHUNK (already tagged — for speaker continuity only, do NOT include in output)
{context}

## TASK
Refine this new chunk (fix spelling, grammar, punctuation, translate non-English to English) and tag each turn with `**Speaker N:**` matching the labels above, or `**Skip:**` for off-topic logistical chatter (tech checks, breaks, food/charger requests, greetings, side chatter unrelated to the meeting topic).

## LABEL STABILITY (critical)
- The context above shows the established voice-to-label mapping. Match each new turn to it by ROLE: the interviewer/analyst (short turns, asks questions) keeps the same Speaker label it has in the context; the expert/respondent (long, substantive answers) keeps theirs.
- NEVER swap or re-derive labels from scratch in this chunk.
- If this chunk starts mid-answer, the first segment almost always continues the SAME speaker as the last segment of the context — only assign a different label if the voice clearly changes.

## OUTPUT FORMAT
Same as before: each turn starts with `**Speaker N:**` or `**Skip:**` on its own line, one blank line between turns. Output ONLY the tagged transcript — no headings, no commentary.

{speaker_info}
{refinement_extra}

## NEW TRANSCRIPT CHUNK
{chunk}
"""

SPEAKER_LEGEND_EXTRACT_PROMPT = """Identify the people speaking in this meeting transcript excerpt.

Return ONLY valid JSON with no other text:
{{"speakers": ["Full Name (role/company)", "..."]}}

Rules:
- List each distinct speaker once, in order of first appearance.
- Include the role/company in parentheses when stated or clearly implied.
- **ASR CORRECTION:** This excerpt comes from automatic speech recognition, so names are often phonetically garbled. When the speaker is clearly identifiable from context (their company, role, or product), output the canonical real-world spelling of their name — NOT the garbled transcript spelling. Example: a garbled rendering of a known company founder's name should be replaced with the founder's actual name.
- If you cannot confidently identify the real person behind a garbled name, keep the transcript spelling and append " [sp?]" to it. Never invent a plausible-looking name.
- If a speaker is never named, describe them functionally (e.g., "Moderator", "Analyst").
- List at most 8 speakers.

TRANSCRIPT EXCERPT:
{transcript_sample}
"""

SPEAKER_NAME_MAP_PROMPT = """Below is the beginning of a speaker-tagged meeting transcript and a list of known participants. Map each generic speaker label to the most likely participant.

Return ONLY valid JSON with no other text, in exactly this shape (one entry per speaker label that appears in the transcript):
{{"Speaker 1": "participant name (role)", "Speaker 2": ""}}

Rules:
- Use the participant names/roles EXACTLY as written in the participants list below.
- Map a label only when the evidence is reasonably clear: who asks questions vs who answers, names used when speakers address each other, roles or companies mentioned in introductions.
- If you cannot confidently map a label, use an empty string "" for it. Do NOT guess.
- Do NOT invent names that are not in the participants list.

PARTICIPANTS: {participants}

TAGGED TRANSCRIPT (beginning):
{transcript_sample}
"""

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
   - As Is: Present findings exactly as stated in the notes. Do not add any positive or negative framing — reproduce the sentiment already present in the source material.
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

OTG_REFINE_CHUNK_PROMPT = """You are a research analyst extracting structured Q&A notes from a segment of meeting notes.

Your task: Identify all questions asked and their corresponding responses. Structure them clearly so key information is easy to find.

Rules:
- Restate each question clearly in **bold** on its own line — no "Q:" prefix, no label.
- Use bullet points (-) immediately below for each distinct answer point.
- ONLY capture content from responses/answers. Do NOT transcribe question text as note content.
- Preserve every specific detail: numbers (%, ₹, $, volumes, timelines), names, company mentions, data points.
- If a passage has no clear Q&A structure, organise it by **bold topic header** with bullet points.
- Be comprehensive — every substantive point in the answer gets its own bullet.
- Raw and unpolished is fine. Abbreviate freely (Rev, Vol, GM, EBITDA, QoQ, YoY, etc.).

---
NOTES SEGMENT {chunk_num} of {total_chunks}:
{chunk}"""


# --- INVESTMENT ANALYST PROCESSING PROMPTS ---

IA_MANAGEMENT_KTA_PROMPT = """You are a senior equity research analyst processing a Company Management Meeting transcript.

Generate exactly two sections in this order. You MUST use the exact section headers shown below — do not rename, reformat, or omit them:

KEY TAKEAWAYS

- Map findings to the framework below. Only include sections the meeting covered meaningfully.
- 5–6 bullets in total across all sections. Each bullet is one short, punchy sentence — no padding.
- Do NOT start a bullet with a label or category prefix (e.g., do NOT write "Revenue: ..." or "Execution: ...").
- Include numbers stated (%, bps, ₹, $, multiples, timelines). State direction where clear.
- If management was vague, say so in one brief phrase. No interpretation beyond what was stated.

Framework (in order): Strategy → Industry → Thematic → Org/Structure → Execution → Revenue → Margins → Capital Alloc. → Mgmt Culture

Format — No bold section header, then bullet(s):
- Volume-led growth expected in H2; no price increase guidance.
- EBITDA margin expansion of 20–30 bps expected over the next 2–3 quarters.
- Net debt declining; net-cash target by FY26 — capex quantum not shared.
- Supply chain on track; vague on exact timeline.

---

ROUGH NOTES

IMPORTANT: The text "ROUGH NOTES" above is your required section header. Output it exactly as shown — plain text, on its own line, preceded by "---". Do not use markdown formatting (no ##, no bold) for this header.

- Capture ALL substantive points — comprehensive, not selective.
- Neutral meeting notes. State what was said. No spin.
- Organise by topic with bold headers. Fewer, denser bullets — aim for ~25% of the bullet count you would otherwise use by consolidating related points into a single longer sentence.
- Each bullet should be a complete sentence that bundles multiple related details together (numbers, direction, caveats, qualitative colour) rather than splitting them across separate lines.
- Abbreviations: Mgmt, Rev, Vol, ASP, GM, EBITDA, QoQ, YoY, H1, H2, FY, bps, capex, opex, D/E, WC, etc.
- Include qualitative context alongside numbers — what was stressed, what was avoided.
- No positive/negative spin.
- If unclear or unquantified → note it inline within the sentence.
- In Q&A-style transcripts: capture ONLY management's responses. Use the question only to identify the topic heading.

Format: Bold topic headers, dashes (-) under each.
Use sentence case for all headings—capitalize only the first word and proper nouns; do not use title case.

---
TRANSCRIPT:
{transcript}
"""

IA_EXPERT_KTA_PROMPT = """You are a senior equity research analyst processing an Expert / Industry Expert / Channel Check Meeting transcript.

Generate exactly two sections in this order. You MUST use the exact section headers shown below — do not rename, reformat, or omit them:

KEY TAKEAWAYS

- Map findings to the framework below. Only include sections the meeting covered meaningfully.
- 5–6 bullets in total across all sections. Each bullet is one short, punchy sentence — no padding.
- Do NOT start a bullet with a label or category prefix (e.g., do NOT write "Inventory: ..." or "Demand: ...").
- Include numbers stated (%, bps, ₹, $, multiples, timelines, volumes). State direction where clear.
- Tag the source type naturally within the sentence — weave in [Expert view], [Channel check], or [Industry data] where relevant.
- If the expert was vague, say so briefly. No interpretation beyond what was stated.

Framework (in order): Industry → Demand → Channel/Trade → Inventory → Pricing → Margins → Competition → Regulatory/Macro → Outlook

Format:
- [Channel check] Dealer inventory at 45–60 days vs. norm of 30 — destocking ongoing.
- [Expert view] Demand weakening in Tier-2 cities; discretionary most hit.
- [Industry data] Organised players gaining ~200 bps share annually from unorganised.
- Expert unclear on recovery timeline; cautious on H1.

---

ROUGH NOTES

IMPORTANT: The text "ROUGH NOTES" above is your required section header. Output it exactly as shown — plain text, on its own line, preceded by "---". Do not use markdown formatting (no ##, no bold) for this header.

- Capture ALL substantive points — comprehensive, not selective.
- Neutral meeting notes. State what was said. No spin.
- Organise by topic with bold headers. Fewer, denser bullets — aim for ~25% of the bullet count you would otherwise use by consolidating related points into a single longer sentence.
- Each bullet should be a complete sentence that bundles multiple related details together (numbers, direction, caveats, qualitative colour) rather than splitting them across separate lines.
- Abbreviations: Expert, Ch-check, Rev, GM, EBITDA, QoQ, YoY, H1, H2, FY, bps, T2, T3, ASP, inv, dist, etc.
- Include qualitative context alongside numbers — what was stressed, what was avoided, any caveats.
- No positive/negative spin.
- If unclear or unquantified → note it inline within the sentence.
- In Q&A-style transcripts: capture ONLY the expert's responses. Use the question only to identify the topic heading.

Format: Bold topic headers, dashes (-) under each.
Format headings in sentence case. Only capitalize the first word and proper nouns. Do not capitalize every word.
---
TRANSCRIPT:
{transcript}
"""

IA_REFINE_CHUNK_PROMPT = """You are a research analyst cleaning up a segment of a meeting transcript.

Your task: Identify all questions asked and the corresponding management/expert responses. Restructure them clearly.

Rules:
- Restate each question clearly in **bold** on its own line — no "Q:" prefix, no label.
- Use bullet points (-) immediately below for each distinct point made in the response.
- ONLY capture content from responses/answers. Do NOT include question text as note content.
- Preserve every specific detail: numbers (%, bps, ₹, $, timelines), names, entities, data points, qualifiers.
- Preserve the speaker's tone and caveats (confident, cautious, vague, speculative).
- If a passage is not Q&A (e.g., opening remarks), organise it under a **bold topic header** with bullets.
- Be comprehensive — every substantive point in the answer gets its own bullet.
- Raw and unpolished is fine. Abbreviate freely.

---
TRANSCRIPT SEGMENT {chunk_num} of {total_chunks}:
{chunk}"""

IA_TONE_INSTRUCTIONS = {
    "Very Positive": "Frame Output 1 findings in the most constructive investment light. Lead with strengths, growth, and opportunity. Challenges are acknowledged only as temporary or manageable context.",
    "Positive": "Frame Output 1 findings constructively. Opportunities lead. Risks acknowledged but not alarming. Overall tone is favourable.",
    "Neutral": "Frame Output 1 findings objectively. Present facts as stated. Balanced where evidence is mixed. No tilting positive or negative.",
    "Negative": "Frame Output 1 findings with risks and headwinds leading. Positives are noted but insufficient to offset structural concerns.",
    "Very Negative": "Frame Output 1 findings around structural problems, execution gaps, and risks. Even positives are presented as temporary or inadequate.",
}

# --- EARNINGS CALL MULTI-FILE ANALYSIS PROMPTS ---

EC_TOPIC_DISCOVERY_PROMPT = """You are an expert equity research analyst. Analyze the following earnings call transcripts and identify the key topics discussed.

### TASK:
From the transcripts below, extract a structured topic hierarchy. The topics should reflect the actual business structure and discussion themes of this company/group.

### OUTPUT FORMAT:
Return ONLY valid JSON with no other text, using this exact structure:
{{
  "company_name": "The company or group name",
  "primary_topics": [
    {{
      "name": "Primary Topic Name (e.g., brand name, business segment, division)",
      "description": "Brief description of what this covers",
      "sub_topics": [
        "Sub-topic 1 (e.g., menu innovation, unit economics, store expansion)",
        "Sub-topic 2",
        "Sub-topic 3"
      ]
    }}
  ],
  "cross_cutting_topics": [
    {{
      "name": "Cross-cutting Topic Name (e.g., Capital Allocation, Management Changes, Macro Environment)",
      "description": "Brief description"
    }}
  ]
}}

### GUIDELINES:
1. **Primary topics** are business segments, brands, divisions, or major product lines (e.g., for Jubilant FoodWorks: "Dominos India", "Popeyes", "Dunkin Donuts", "Hong's Kitchen")
2. **Sub-topics** under each primary topic are recurring themes like: strategy, menu innovation, store expansion, unit economics, competitive positioning, pricing, customer acquisition, operational efficiency, supply chain, marketing, org structure changes, incentive changes, etc.
3. **Cross-cutting topics** span the entire company: capital allocation, management commentary, guidance, macro environment, regulatory, ESG, etc.
4. Be SPECIFIC to this business — don't use generic templates. The topics should reflect what is ACTUALLY discussed in these transcripts.
5. Include 3-8 primary topics and 3-10 sub-topics per primary topic, based on what the transcripts actually cover.

---
**TRANSCRIPTS:**

{transcripts}
"""

EC_MULTI_FILE_NOTES_PROMPT = """### **EARNINGS CALL NOTES — STRUCTURED BY TOPICS**

You are generating detailed earnings call notes from the transcript below. Structure your notes STRICTLY under the provided topic hierarchy.

### TOPIC STRUCTURE TO FOLLOW:
{topic_structure}

### RULES:
1. For each topic and sub-topic, extract ALL relevant information from the transcript.
2. If a topic/sub-topic has no relevant information in this transcript, write "No specific commentary in this quarter." under it — do NOT skip the heading.
3. Use **bold headings** for primary topics and sub-topics. Use bullet points for details.
4. **PRIORITY #1: CAPTURE ALL FINANCIAL DATA.** Revenue, margins, EPS, guidance ranges, growth rates, basis points, dollar amounts — every number matters.
5. **PRIORITY #2: CAPTURE FORWARD GUIDANCE.** Any forward-looking statements, guidance ranges, management expectations, or outlook commentary.
6. **PRIORITY #3: PRESERVE MANAGEMENT TONE.** Note confidence, caution, hedging language, or changes from prior quarter tone.
7. **PRIORITY #4: CAPTURE SEGMENT/VERTICAL DETAIL.** Business segment breakdowns, geographic splits, and vertical-specific commentary.
8. Include the quarter/period identifier at the top of your notes if mentioned in the transcript.

### FORMAT EXAMPLE:
**[Primary Topic: Brand/Segment Name]**

**[Sub-topic: Strategy]**
- Bullet point with detail...
- Another bullet point...

**[Sub-topic: Unit Economics]**
- Bullet point with detail...

---
**TRANSCRIPT ({file_label}):**
{transcript}
"""

EC_MULTI_FILE_STITCH_HEADER = """# Earnings Call Topic Analysis — {company_name}
*Generated on {date}*
*Files analyzed: {file_count}*

---

"""

# --- REPORT COMPARISON PROMPT CONSTANTS ---

RC_DIMENSION_DISCOVERY_PROMPT = """You are an expert equity research analyst specializing in annual report analysis. Analyze the following annual reports and identify the key QUALITATIVE dimensions that can be meaningfully compared across years.

### TASK:
From the reports below, extract a structured set of comparison dimensions. Focus ONLY on qualitative and strategic aspects — NOT financial numbers (those will differ year to year and are not the focus).

### OUTPUT FORMAT:
Return ONLY valid JSON with no other text, using this exact structure:
{{
  "company_name": "The company or group name",
  "report_years": ["Year 1", "Year 2", ...],
  "comparison_dimensions": [
    {{
      "name": "Dimension Name",
      "description": "Brief description of what this covers",
      "sub_dimensions": [
        "Sub-dimension 1",
        "Sub-dimension 2"
      ]
    }}
  ]
}}

### FOCUS AREAS (use these as guidance, but be specific to what is actually in the reports):
1. **Management Commentary & Tone** — How does the CEO/Chairman letter read? What is the tone — optimistic, cautious, defensive? What themes are emphasized?
2. **Strategic Direction & Priorities** — What strategic pillars are highlighted? How have priorities shifted? New initiatives vs. continued focus areas?
3. **Business Structure & Organization** — How is the business organized (segments, divisions, subsidiaries)? Any restructuring, new segments, or organizational changes?
4. **Leadership & Governance** — Board composition changes, key management changes, succession planning, governance structure evolution?
5. **Incentive Structures & Compensation** — How are executives compensated? What metrics drive bonuses/ESOPs? Any changes in incentive design?
6. **Risk Factors & Mitigation** — What risks are highlighted? How has the risk landscape changed? New risks vs. dropped risks?
7. **Capital Allocation Philosophy** — How does management talk about deploying capital? Dividends vs. buybacks vs. reinvestment priorities?
8. **ESG / Sustainability** — Environmental, social, governance initiatives. How prominent is ESG in the narrative? Any new commitments?
9. **Market & Competitive Positioning** — How does the company describe its competitive position? Market share commentary, moats, differentiation?
10. **Growth Levers & Outlook** — What growth avenues are highlighted? Organic vs. inorganic? Geographic vs. product expansion?
11. **Culture & People** — Employee-related commentary, talent strategy, culture statements, DEI initiatives?
12. **Technology & Digital** — Digital transformation initiatives, technology investments, IT strategy evolution?

### GUIDELINES:
- Be SPECIFIC to this company — identify dimensions that are actually discussed in these reports.
- Include 5-12 dimensions with 2-6 sub-dimensions each, based on what the reports actually cover.
- The dimensions should enable meaningful year-over-year comparison of QUALITATIVE changes.
- Do NOT include dimensions focused on specific numbers or financial metrics.

---
**ANNUAL REPORTS:**

{reports}
"""

RC_PER_REPORT_EXTRACTION_PROMPT = """### **ANNUAL REPORT ANALYSIS — QUALITATIVE EXTRACTION**

You are extracting qualitative information from an annual report for a specific set of comparison dimensions. Focus on WHAT management says, HOW they say it, and WHAT has changed — NOT on specific numbers.

### DIMENSIONS TO EXTRACT:
{dimension_structure}

### RULES:
1. For each dimension and sub-dimension, extract ALL relevant qualitative information from this report.
2. If a dimension/sub-dimension has no relevant information in this report, write "Not addressed in this report." — do NOT skip the heading.
3. Use **bold headings** for dimensions and sub-dimensions. Use bullet points for details.
4. **FOCUS ON:**
   - Management's language, tone, and emphasis
   - Strategic statements and directional commentary
   - Organizational descriptions and structural details
   - Policy descriptions (compensation, governance, risk)
   - Qualitative characterizations ("strong growth", "challenging environment", "transformational year")
   - Changes in emphasis or new themes compared to what might be typical
5. **AVOID:**
   - Specific revenue/profit/margin numbers (unless they illustrate a qualitative point about strategy)
   - Detailed financial tables or ratios
   - Restating numbers that will obviously differ between years
6. Capture DIRECT QUOTES from management where they are particularly revealing of tone or strategic intent.
7. Note the year/period this report covers at the top.

### FORMAT:
**[Dimension: Name]**

**[Sub-dimension: Name]**
- Bullet point with qualitative detail...
- Another bullet point...

---
**ANNUAL REPORT ({file_label}):**
{report_text}
"""

RC_COMPARISON_PROMPT = """### **ANNUAL REPORT COMPARISON — YEAR-OVER-YEAR QUALITATIVE ANALYSIS**

You are an expert analyst comparing annual reports from different years for the same company. Below are the qualitative extractions from each year's report. Your task is to produce a structured comparison highlighting what has CHANGED, what has STAYED THE SAME, and what is NEW or DROPPED.

### COMPANY: {company_name}
### REPORTS COMPARED: {report_labels}

### COMPARISON DIMENSIONS:
{dimension_structure}

### EXTRACTED DATA FROM EACH REPORT:
{per_report_extractions}

### YOUR TASK:
For each dimension and sub-dimension, produce a comparison that answers:
1. **What changed?** — Shifts in tone, emphasis, strategy, structure, or policy between years.
2. **What remained consistent?** — Themes or approaches that persisted across years.
3. **What is new?** — Themes, initiatives, or structural elements that appear in later reports but not earlier ones.
4. **What was dropped?** — Items emphasized in earlier reports but absent or de-emphasized in later ones.

### FORMAT:
For each dimension, structure your output as:

## [Dimension Name]

### [Sub-dimension Name]

**Evolution across years:**
- [Year-over-year comparison points as bullets]

**Key shifts:**
- [Most significant changes highlighted]

**Consistency:**
- [What stayed the same]

### RULES:
1. Be SPECIFIC — cite which year said what. Use phrases like "In FY2022, management emphasized X, while in FY2024, the focus shifted to Y."
2. Include direct management quotes where they illustrate a meaningful shift.
3. Do NOT simply list what each year said — actually COMPARE and CONTRAST.
4. Highlight the most SIGNIFICANT shifts prominently. Minor changes can be noted briefly.
5. If a dimension shows no meaningful change across years, say so explicitly — consistency is also a finding.
6. Order the comparison chronologically (earliest to latest year).
7. At the end, include a section called "## Key Takeaways" with 5-10 bullet points summarizing the most important qualitative shifts across all dimensions.

---
"""

RC_STITCH_HEADER = """# Annual Report Comparison — {company_name}
*Generated on {date}*
*Reports compared: {report_labels}*

---

"""

# /--------------------------\
# |   END OF prompts.py FILE  |
# \--------------------------/

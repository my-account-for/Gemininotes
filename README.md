# Gemininotes (SynthNotes AI)

A Streamlit app for turning raw transcripts/PDFs/audio into structured investment and research notes with Gemini models.

## What was improved

- **Reliable theme handling:** replaced fragile JS-based Streamlit theme toggling with a stable in-app appearance selector (`System`, `Light`, `Dark`, `Midnight Blue`).
- **Graceful startup when API key is missing:** app no longer hard-stops; users can still browse history and use non-AI UI utilities.
- **UI polish:** improved header, KPI quick-stats, compact/comfortable density mode, and session utility actions.
- **Better resilience for deployment:** clear guidance for Streamlit Cloud stability and runtime settings.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

## Streamlit Cloud stability checklist (important)

If your app "keeps getting down", use this checklist:

1. **Set secrets in Streamlit Cloud** (do not depend on repo `.env`):
   - `GEMINI_API_KEY` in **App settings → Secrets**.
2. **Pin dependency versions** in `requirements.txt` (already partly pinned).
3. **Keep startup lightweight**:
   - avoid expensive operations at import time,
   - defer heavy model creation until needed.
4. **Use `packages.txt` only for required system deps** (`ffmpeg` is already included for audio).
5. **Review logs** for OOM/timeouts and reduce payload sizes where needed.
6. **Handle missing API key gracefully** so non-AI parts still render.
7. **Use Streamlit config** (`.streamlit/config.toml`) for consistent server behavior.

## Streamlit config

This repo includes a `.streamlit/config.toml` for stable defaults:

- headless server mode
- fixed dark base theme and primary color
- broader max upload size for transcript/audio workflows

You can tune these values based on your cloud memory and traffic profile.

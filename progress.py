# /-----------------------------\
# |   START OF progress.py FILE  |
# \-----------------------------/

"""Work-plan based progress tracking for long-running pipelines.

The old tracker had hardcoded step weights (prepare 5 / transcribe 15 /
refine 25 / generate 47 / cleanup 3 / save 5) regardless of which steps
actually ran: pasted-text runs never completed "transcribe", so the bar
topped out near ~82-85% and then snapped to 100%. It also only updated
*between* model calls, freezing for minutes during streamed generation.

This tracker is built from a per-run plan containing ONLY the steps that
will execute, weighted by expected relative duration ("units"). Units can be
re-scaled once the real workload is known (e.g. the chunk count after the
file is loaded), and within-step progress is fed continuously from streaming
callbacks — so the bar genuinely spans 0-100% and moves during generation.

It renders three elements (create it inside a `st.status` container):
  - a progress bar,
  - a step checklist (✅ done / 🔄 active / ⏳ pending) with per-step detail,
  - a percentage + ETA line (ETA from overall elapsed/progress rate).
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st


def _supports_shimmer() -> bool:
    """The :shimmer[] markdown directive shipped in Streamlit 1.57."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 57)
    except Exception:
        return False


_SHIMMER = _supports_shimmer()


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"~{max(1, int(seconds))}s remaining"
    m, s = divmod(int(seconds), 60)
    return f"~{m}m {s:02d}s remaining"


class ProgressTracker:
    """Plan-driven progress bar + step checklist.

    plan: sequence of (key, label, units) tuples in execution order. Include
    only steps that will actually run; units are expected relative durations.
    """

    def __init__(self, plan: Sequence[Tuple[str, str, float]]):
        self._order: List[str] = []
        self._steps: Dict[str, Dict] = {}
        for key, label, units in plan:
            self._order.append(key)
            self._steps[key] = {
                "label": label,
                "units": max(float(units), 0.01),
                "frac": 0.0,
                "state": "pending",  # pending | active | done
                "detail": "",
            }
        self._bar = st.progress(0.0)
        self._checklist = st.empty()
        self._eta_line = st.empty()
        self._start = time.time()
        self._render()

    # ------------------------------------------------------------------ #
    def set_units(self, step: str, units: float):
        """Re-scale a step's weight once the real workload is known
        (e.g. the number of chunks after the transcript is loaded)."""
        if step in self._steps:
            self._steps[step]["units"] = max(float(units), 0.01)
            self._render()

    def update(self, step: str, sub_progress: float = 0.0, detail: str = ""):
        """Mark `step` active with within-step progress in [0, 1]."""
        info = self._steps.get(step)
        if info is None:  # unknown step — ignore rather than crash mid-run
            return
        if info["state"] != "done":
            info["state"] = "active"
            # Monotonic: a 0-progress call just marks activity / sets detail.
            if sub_progress > 0:
                info["frac"] = min(max(info["frac"], sub_progress), 1.0)
        info["detail"] = detail
        self._render()

    def complete_step(self, step: str):
        info = self._steps.get(step)
        if info is None:
            return
        info["state"] = "done"
        info["frac"] = 1.0
        self._render()

    def finish(self):
        for info in self._steps.values():
            info["state"] = "done"
            info["frac"] = 1.0
            info["detail"] = ""
        self._render(final=True)

    # ------------------------------------------------------------------ #
    @property
    def overall_fraction(self) -> float:
        total = sum(s["units"] for s in self._steps.values())
        if total <= 0:
            return 0.0
        done = sum(s["units"] * s["frac"] for s in self._steps.values())
        return min(done / total, 1.0)

    def _render(self, final: bool = False):
        frac = 1.0 if final else self.overall_fraction
        self._bar.progress(frac)

        lines = []
        for key in self._order:
            info = self._steps[key]
            label = info["label"]
            detail = f" — {info['detail']}" if info["detail"] else ""
            if info["state"] == "done":
                lines.append(f"✅ ~~{label}~~")
            elif info["state"] == "active":
                text = f"{label}{detail}"
                if _SHIMMER:
                    text = f":shimmer[{text}]"
                lines.append(f"🔄 **{text}**")
            else:
                lines.append(f"⏳ {label}")
        self._checklist.markdown("  \n".join(lines))

        pct = int(frac * 100)
        if final:
            self._eta_line.caption("**100%** — Complete")
            return
        elapsed = time.time() - self._start
        # Wait for a meaningful sample before extrapolating an ETA.
        if frac >= 0.08 and elapsed > 5:
            remaining = elapsed * (1 - frac) / frac
            self._eta_line.caption(f"**{pct}%** · {_fmt_eta(remaining)}")
        else:
            self._eta_line.caption(f"**{pct}%**")


# ---------------------------------------------------------------------- #
# Plan builders for the app's pipelines. Units are expected relative
# durations; they get re-scaled via set_units once chunk counts are known.

def parallel_batches(n_chunks: int, max_workers: int = 3) -> int:
    """Wall-time proxy for n_chunks processed max_workers at a time."""
    return max(1, (n_chunks + max_workers - 1) // max_workers)


def build_processing_plan(
    *,
    is_audio: bool,
    refinement_enabled: bool,
    with_summary: bool,
    with_audit: bool = False,
) -> List[Tuple[str, str, float]]:
    """Plan for the standard generate-notes pipeline."""
    plan: List[Tuple[str, str, float]] = [("prepare", "Preparing source content", 0.5)]
    if is_audio:
        plan.append(("transcribe", "Transcribing audio", 6.0))
    if refinement_enabled:
        plan.append(("refine", "Refining transcript", 3.0))
    plan.append(("generate", "Generating notes", 8.0))
    if with_summary:
        plan.append(("summary", "Generating executive summary", 1.0))
    plan.append(("save", "Finalizing & saving", 0.3))
    if with_audit:
        plan.append(("audit", "Auditing notes against transcript", 3.0))
    return plan


def build_speaker_id_plan(*, is_audio: bool) -> List[Tuple[str, str, float]]:
    """Plan for the speaker-identification step (Expert Meeting Option 4)."""
    plan: List[Tuple[str, str, float]] = [("prepare", "Preparing source content", 0.5)]
    if is_audio:
        plan.append(("transcribe", "Transcribing audio", 6.0))
    plan.append(("refine", "Refining & tagging speakers", 6.0))
    return plan


def build_notes_only_plan(*, with_summary: bool, with_audit: bool = False) -> List[Tuple[str, str, float]]:
    """Plan for generating notes from an already-tagged transcript."""
    plan: List[Tuple[str, str, float]] = [("generate", "Generating notes", 8.0)]
    if with_summary:
        plan.append(("summary", "Generating executive summary", 1.0))
    plan.append(("save", "Finalizing & saving", 0.3))
    if with_audit:
        plan.append(("audit", "Auditing notes against transcript", 3.0))
    return plan

# /---------------------------\
# |   END OF progress.py FILE  |
# \---------------------------/

"""Component 2d: Evaluation section.

Renders Phase 5's output comprehensively: per-step hop_metrics +
step_faithfulness, then the overall classification verdict. This is the
"comprehensively explain the step-level eval phase" requirement -- a
viewer should understand PASS vs failure and *why* without narration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.styles.style import phase_badge

CLASSIFICATION_COLOR = {
    "PASS": ("#4ADE80", "rgba(34,197,94,0.12)"),
    "PREMATURE_COLLAPSE": ("#FBBF24", "rgba(245,158,11,0.12)"),
    "OVER_EXTENSION": ("#FBBF24", "rgba(245,158,11,0.12)"),
    "RETRIEVAL_FAILURE": ("#F87171", "rgba(239,68,68,0.12)"),
    "REASONING_FAILURE": ("#F87171", "rgba(239,68,68,0.12)"),
    "UNANSWERABLE_CORRECTLY_ABSTAINED": ("#60A5FA", "rgba(59,130,246,0.12)"),
}


def classification_pill(label: str) -> str:
    text_color, bg_color = CLASSIFICATION_COLOR.get(label, ("#94A3B8", "rgba(148,163,184,0.12)"))
    return (
        f'<span style="background:{bg_color}; color:{text_color}; padding:4px 14px; '
        f'border-radius:8px; font-size:14px; font-weight:600;">{label}</span>'
    )


def _hop_metric_for_step(hop_metrics: list[dict], step_number: int) -> dict | None:
    return next((h for h in hop_metrics if h["step_number"] == step_number), None)


def _faithfulness_for_step(step_faithfulness: list[dict], step_number: int) -> dict | None:
    return next((f for f in step_faithfulness if f["step_number"] == step_number), None)


def render_evaluation_section(eval_result: dict, step_numbers: list[int]):
    with st.container(border=True):
        st.markdown(phase_badge("evaluation"), unsafe_allow_html=True)
        st.markdown('<p class="rl-muted" style="margin-top:6px;">step-level scoring</p>', unsafe_allow_html=True)

        hop_metrics = eval_result["hop_metrics"]
        step_faithfulness = eval_result["step_faithfulness"]

        for step_number in step_numbers:
            hm = _hop_metric_for_step(hop_metrics, step_number)
            f = _faithfulness_for_step(step_faithfulness, step_number)

            cols = st.columns([1, 1, 1, 1, 3])
            with cols[0]:
                st.markdown(f'<span class="rl-mono rl-muted">step {step_number}</span>', unsafe_allow_html=True)

            if hm is None and f is None:
                # Exists in the trace (a retrieval/reasoning step happened) but the
                # harness never scored it -- e.g. eval window cut off before this
                # step. This is often itself the over-extension signal, so it's
                # surfaced explicitly rather than hidden.
                with cols[1]:
                    st.markdown('<span class="rl-muted"><em>not evaluated by harness</em></span>', unsafe_allow_html=True)
                continue

            with cols[1]:
                precision = hm["context_precision_at_3"] if hm else None
                st.markdown(
                    f'<span class="rl-mono">P@3: {precision:.2f}</span>' if precision is not None else "—",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                recall = hm["context_recall"] if hm else None
                st.markdown(
                    f'<span class="rl-mono">recall: {recall:.1f}</span>' if recall is not None else "—",
                    unsafe_allow_html=True,
                )
            with cols[3]:
                rank = hm["supporting_chunk_rank"] if hm else None
                rank_str = f"rank {rank}" if rank is not None else "no support"
                st.markdown(f'<span class="rl-mono">{rank_str}</span>', unsafe_allow_html=True)
            with cols[4]:
                if f is not None:
                    faithful = f["faithful"]
                    if faithful is True:
                        icon = "✅"
                    elif faithful is False:
                        icon = "⚠️"
                    else:
                        icon = "➖"  # None -- not applicable, e.g. abstention with no claim to check
                    st.markdown(f'{icon} <span class="rl-muted">{f["reason"]}</span>', unsafe_allow_html=True)

        st.divider()

        classification = eval_result["classification"]
        verdict_cols = st.columns([1, 4])
        with verdict_cols[0]:
            st.markdown(classification_pill(classification["label"]), unsafe_allow_html=True)
        with verdict_cols[1]:
            over_ext = " · <strong>over-extension flagged</strong>" if classification["over_extension_flag"] else ""
            st.markdown(
                f'<span class="rl-muted">{classification["reason"]}{over_ext}</span>',
                unsafe_allow_html=True,
            )


# --- Standalone stress test ---
if __name__ == "__main__":
    st.set_page_config(page_title="Evaluation Section — standalone test", layout="wide")
    from styles import inject_css
    inject_css()
    st.markdown("#### Component 2d: Evaluation Section (standalone)")

    st.markdown("**Case: PASS, all faithful**")
    render_evaluation_section(
        {
            "question_id": "x", "config_id": "hybrid",
            "hop_metrics": [
                {"step_number": 1, "context_precision_at_3": 0.33, "context_recall": 1.0, "supporting_chunk_rank": 1},
                {"step_number": 2, "context_precision_at_3": 0.2, "context_recall": 1.0, "supporting_chunk_rank": 1},
            ],
            "step_faithfulness": [
                {"step_number": 1, "faithful": True, "reason": "Claim matches retrieved chunk."},
                {"step_number": 2, "faithful": True, "reason": "Final answer grounded in supporting chunk."},
            ],
            "classification": {"label": "PASS", "over_extension_flag": False, "reason": "Both hops resolved correctly."},
        },
        step_numbers=[1, 2],
    )

    st.markdown("**Case: RETRIEVAL_FAILURE, no supporting chunk found, over-extension flagged**")
    render_evaluation_section(
        {
            "question_id": "y", "config_id": "naive",
            "hop_metrics": [
                {"step_number": 1, "context_precision_at_3": 0.0, "context_recall": 0.0, "supporting_chunk_rank": None},
            ],
            "step_faithfulness": [
                {"step_number": 1, "faithful": False, "reason": "No retrieved chunk supports the generated claim."},
            ],
            "classification": {"label": "RETRIEVAL_FAILURE", "over_extension_flag": True, "reason": "Retriever never surfaced the gold chunk across 5 attempts."},
        },
        step_numbers=[1],
    )

    st.markdown("**Edge case: a step with no hop_metric or faithfulness entry (pure generate step)**")
    render_evaluation_section(
        {
            "question_id": "z", "config_id": "advanced",
            "hop_metrics": [{"step_number": 1, "context_precision_at_3": 1.0, "context_recall": 1.0, "supporting_chunk_rank": 1}],
            "step_faithfulness": [{"step_number": 1, "faithful": True, "reason": "Grounded."}],
            "classification": {"label": "PASS", "over_extension_flag": False, "reason": "Clean single-hop resolution."},
        },
        step_numbers=[1, 2],  # step 2 has no matching entries -- should surface as "not evaluated", not crash
    )

    st.markdown("**Real case: UNANSWERABLE_CORRECTLY_ABSTAINED, faithful=null, 7 trace steps but only 5 scored "
                 "(steps 6-7 unscored -- itself evidence of the over-extension flag)**")
    render_evaluation_section(
        {
            "question_id": "4hop1__151650_5274_458768_33677", "config_id": "hybrid",
            "hop_metrics": [
                {"step_number": 1, "context_precision_at_3": 0.3333333333333333, "context_recall": 1.0, "supporting_chunk_rank": 2},
                {"step_number": 2, "context_precision_at_3": 0.0, "context_recall": 0.0, "supporting_chunk_rank": None},
                {"step_number": 3, "context_precision_at_3": 0.3333333333333333, "context_recall": 1.0, "supporting_chunk_rank": 1},
                {"step_number": 4, "context_precision_at_3": 0.0, "context_recall": 0.0, "supporting_chunk_rank": None},
                {"step_number": 5, "context_precision_at_3": 0.0, "context_recall": 0.0, "supporting_chunk_rank": None},
            ],
            "step_faithfulness": [
                {"step_number": 1, "faithful": None, "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into."},
                {"step_number": 2, "faithful": None, "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into."},
                {"step_number": 3, "faithful": None, "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into."},
                {"step_number": 4, "faithful": None, "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into."},
                {"step_number": 5, "faithful": None, "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into."},
            ],
            "classification": {
                "label": "UNANSWERABLE_CORRECTLY_ABSTAINED",
                "over_extension_flag": True,
                "reason": "Recall reached 1.0 at steps 1 and 3, but the evidence for Universal Music Group's "
                          "headquarters was ambiguous (the only retrieved sentence could refer to EMI, not UMG), "
                          "and the agent correctly abstained rather than hallucinate an answer.",
            },
        },
        step_numbers=[1, 2, 3, 4, 5, 6, 7],  # steps 6-7 exist in the trace but the harness never scored them
    )
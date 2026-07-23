"""Component 2b: Step card (retrieval + reasoning + routing decision merged).

Your trace schema represents retrieval, reasoning, and the routing decision
as ONE unit per step_number (reasoning_steps[i]["decision"] IS the routing
output) -- so this renders as one card per step, phase-badged for clarity,
rather than three separate cards. Standalone testable with mock data.

Fixed from the earlier version: uses st.container(border=True) as a real
Streamlit primitive instead of splitting a raw <div class="rl-card"> across
multiple st.markdown() calls, which does not wrap content in Streamlit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
import streamlit as st
from ui.styles.style import phase_badge

DECISION_ICON = {"retrieve_more": "🔁", "refine_query": "✏️", "generate": "✅"}
DECISION_LABEL = {"retrieve_more": "retrieve more", "refine_query": "refine query", "generate": "generate"}
DOT = {"green": "🟢", "amber": "🟡", "red": "🔴"}


def chunk_color(rank: int, supporting_chunk_rank, normalised_scores: list[float]) -> str:
    if supporting_chunk_rank is not None and rank == supporting_chunk_rank:
        return "green"
    if not normalised_scores:
        return "red"
    score = normalised_scores[rank - 1]
    cutoff = statistics.median(normalised_scores)
    return "amber" if score >= cutoff else "red"


def render_step_card(reasoning_step: dict, retrieval_step: dict | None, hop_metric: dict | None,
                      faithfulness: dict | None, key_prefix: str):
    step_number = reasoning_step["step_number"]
    decision = reasoning_step["decision"]

    with st.container(border=True):
        badges = phase_badge("retrieval") + " " + phase_badge("reasoning") + " " + phase_badge("routing")
        st.markdown(badges, unsafe_allow_html=True)

        header_cols = st.columns([5, 2])
        with header_cols[0]:
            unfaithful = ""
            if faithfulness is not None and faithfulness["faithful"] is False:
                unfaithful = ' <span style="color:#F87171; font-size:12px;">⚠ unfaithful step</span>'
            st.markdown(
                f'**Step {step_number}** — {DECISION_ICON.get(decision, "•")} '
                f'{DECISION_LABEL.get(decision, decision)}{unfaithful}',
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            if retrieval_step is not None:
                st.markdown(
                    f'<span class="rl-mono rl-muted">{retrieval_step["latency_ms"]:.0f}ms</span>',
                    unsafe_allow_html=True,
                )

        if retrieval_step is not None:
            st.markdown(f'<p class="rl-muted">{retrieval_step["sub_query"]}</p>', unsafe_allow_html=True)

            supporting_rank = hop_metric["supporting_chunk_rank"] if hop_metric else None
            normalised_scores = retrieval_step["normalised_scores"]
            chunks = retrieval_step["retrieved_chunks"]

            dot_cols = st.columns(len(chunks)) if chunks else []
            for i, chunk in enumerate(chunks):
                rank = i + 1
                color = chunk_color(rank, supporting_rank, normalised_scores)
                metadata = chunk.get("metadata", {})
                chunk_id = metadata.get("chunk_id", f"chunk_{rank}")
                score = normalised_scores[i] if i < len(normalised_scores) else None
                with dot_cols[i]:
                    label = f'{DOT[color]} rank {rank}'
                    if st.button(label, key=f"{key_prefix}_s{step_number}_c{rank}",
                                 help=f"score: {score:.2f}" if score is not None else None):
                        st.session_state[f"{key_prefix}_s{step_number}_expanded"] = chunk_id

            expanded_id = st.session_state.get(f"{key_prefix}_s{step_number}_expanded")
            if expanded_id:
                matched = next(
                    (c for c in chunks if c.get("metadata", {}).get("chunk_id") == expanded_id), None
                )
                if matched:
                    st.markdown(
                        f'<div class="rl-mono" style="background:rgba(120,120,120,0.08); '
                        f'padding:10px 12px; border-radius:8px; margin-top:6px;">'
                        f'{matched.get("text", "(no text field on chunk)")}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown('<p class="rl-muted"><em>no retrieval at this step</em></p>', unsafe_allow_html=True)

        if decision == "refine_query" and reasoning_step.get("refine_reason"):
            st.markdown(
                f'<p class="rl-muted" style="border-left:2px solid rgba(120,120,120,0.4); '
                f'padding-left:10px;">{reasoning_step["refine_reason"]}</p>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<p class="rl-muted">{reasoning_step["reasoning_trace"]}</p>', unsafe_allow_html=True)


# --- Standalone stress test ---
if __name__ == "__main__":
    st.set_page_config(page_title="Step Card — standalone test", layout="wide")
    from styles import inject_css
    inject_css()
    st.markdown("#### Component 2b: Step Card (standalone)")

    mock_retrieval_step = {
        "step_number": 1,
        "sub_query": "What creature do annelid larvae live like?",
        "retrieved_chunks": [
            {"text": "Annelid trochophore larvae live as plankton.", "metadata": {"chunk_id": 11633, "is_supporting": True}},
            {"text": "Nannochoristid larvae are elateriform.", "metadata": {"chunk_id": 11635, "is_supporting": False}},
            {"text": "The genus Darwinius resembles a modern lemur.", "metadata": {"chunk_id": 13411, "is_supporting": False}},
        ],
        "raw_scores": [0.76, 0.38, 0.28],
        "normalised_scores": [1.0, 0.5, 0.0],
        "latency_ms": 1662.85,
    }
    mock_reasoning_step_refine = {
        "step_number": 1, "decision": "refine_query",
        "reasoning_trace": "Identified plankton as the creature, but no SpongeBob info retrieved yet.",
        "refine_reason": "Original sub-query returned biology results, not the character voice actor.",
    }
    mock_reasoning_step_generate = {
        "step_number": 2, "decision": "generate",
        "reasoning_trace": "All needed facts present: creature is plankton, character is voiced by Mr. Lawrence.",
        "refine_reason": None,
    }
    mock_hop_metric = {"step_number": 1, "context_precision_at_3": 0.33, "context_recall": 1.0, "supporting_chunk_rank": 1}
    mock_faithfulness_bad = {"step_number": 1, "faithful": False, "reason": "Claim not grounded in retrieved text."}

    st.markdown("**Case: refine_query, with an unfaithful flag**")
    render_step_card(mock_reasoning_step_refine, mock_retrieval_step, mock_hop_metric, mock_faithfulness_bad, key_prefix="test1")

    st.markdown("**Case: generate, no retrieval this step**")
    render_step_card(mock_reasoning_step_generate, None, None, None, key_prefix="test2")

    st.markdown("**Edge case: retrieval_step present but supporting_chunk_rank is None (judge found no support)**")
    mock_hop_metric_none = {"step_number": 1, "context_precision_at_3": 0.0, "context_recall": 0.0, "supporting_chunk_rank": None}
    render_step_card(mock_reasoning_step_refine, mock_retrieval_step, mock_hop_metric_none, None, key_prefix="test3")
"""RAGLens demo dashboard.

Reads already-persisted trace + eval JSONL (no live pipeline/harness calls),
filters by question_id within the config-selected file, and renders the
full vertical timeline in one shot: decomposition -> step cards -> generation
(or exhaustion) -> evaluation section.

Run with: streamlit run app.py
"""

import streamlit as st

from ui.styles.style import inject_css, phase_badge
from ui.loader import get_trace, get_eval, trace_file_path, eval_file_path
from ui.components.input_form import render_input_form
from ui.components.decomposition_card import render_decomposition_card
from ui.components.step_card import render_step_card
from ui.components.generation_card import render_generation_card, render_no_generation_card
from ui.components.eval_section import render_evaluation_section, classification_pill

st.set_page_config(page_title="RAGLens", layout="wide")
inject_css()
 
st.markdown("## RAGLens")
st.markdown('<p class="rl-muted">Step-level agentic RAG evaluation — persisted trace + eval viewer</p>',
            unsafe_allow_html=True)
 
form_result = render_input_form()
 
# st.form_submit_button() only returns True on the exact rerun it was clicked.
# Every other widget interaction below (e.g. a chunk rank button) triggers its
# own rerun, on which form_result would be None again -- so the loaded run is
# stashed in session_state to survive those unrelated reruns.
if form_result is not None:
    st.session_state["rl_question_id"] = form_result["question_id"]
    st.session_state["rl_config_id"] = form_result["config"]
 
if "rl_question_id" in st.session_state:
    question_id = st.session_state["rl_question_id"]
    config_id = st.session_state["rl_config_id"]
 
    with st.spinner(f"Loading persisted trace + eval for {question_id} ({config_id})..."):
        trace = get_trace(question_id, config_id)
        eval_result = get_eval(question_id, config_id)
 
    if trace is None:
        st.error(
            f"No trace found for question_id=\"{question_id}\" in "
            f"{trace_file_path(config_id)}. Check the question_id and config are correct."
        )
        st.stop()
 
    reasoning_steps = trace["reasoning_steps"]
    retrieval_steps = trace["retrieval_steps"]
    all_step_numbers = [s["step_number"] for s in reasoning_steps]
 
    def _retrieval_for_step(step_number):
        return next((r for r in retrieval_steps if r["step_number"] == step_number), None)
 
    def _hop_metric_for_step(step_number):
        if eval_result is None:
            return None
        return next((h for h in eval_result["hop_metrics"] if h["step_number"] == step_number), None)
 
    def _faithfulness_for_step(step_number):
        if eval_result is None:
            return None
        return next((f for f in eval_result["step_faithfulness"] if f["step_number"] == step_number), None)
 
    # --- Page header ---
    st.divider()
    st.markdown("### " + trace["original_query"])
    top_cols = st.columns([2, 2, 2, 2])
    with top_cols[0]:
        if eval_result is not None:
            st.markdown(classification_pill(eval_result["classification"]["label"]), unsafe_allow_html=True)
        else:
            st.markdown(
                '<span style="background:rgba(148,163,184,0.12); color:#94A3B8; padding:4px 14px; '
                'border-radius:8px; font-size:14px; font-weight:600;">NOT YET EVALUATED</span>',
                unsafe_allow_html=True,
            )
    with top_cols[1]:
        st.markdown(f'<span class="rl-mono rl-muted">config: {config_id}</span>', unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown(f'<span class="rl-mono rl-muted">hops: {trace["hop_count"]}</span>', unsafe_allow_html=True)
    with top_cols[3]:
        st.markdown(
            f'<span class="rl-mono rl-muted">{trace["total_retrieval_latency_ms"]:.0f}ms total</span>',
            unsafe_allow_html=True,
        )
    st.divider()
 
    # --- Decomposition ---
    render_decomposition_card(trace["original_query"], trace["sub_queries"])
 
    # --- Step cards (retrieval + reasoning + routing, merged per your trace schema) ---
    key_prefix = f'{trace["question_id"]}_{trace["config_id"]}'
    for reasoning_step in reasoning_steps:
        step_number = reasoning_step["step_number"]
        render_step_card(
            reasoning_step=reasoning_step,
            retrieval_step=_retrieval_for_step(step_number),
            hop_metric=_hop_metric_for_step(step_number),
            faithfulness=_faithfulness_for_step(step_number),
            key_prefix=key_prefix,
        )
 
    # --- Generation (or exhaustion) ---
    reached_generate = any(s["decision"] == "generate" for s in reasoning_steps)
    if reached_generate:
        render_generation_card(trace["final_answer"], trace["total_retrieval_latency_ms"])
    else:
        render_no_generation_card(
            status=trace["status"],
            final_answer=trace["final_answer"],
            total_retrieval_latency_ms=trace["total_retrieval_latency_ms"],
            step_count=len(reasoning_steps),
        )
 
    # --- Evaluation ---
    st.divider()
    if eval_result is not None:
        render_evaluation_section(eval_result, step_numbers=all_step_numbers)
    else:
        st.markdown(phase_badge("evaluation"), unsafe_allow_html=True)
        st.markdown(
            f'<p class="rl-muted" style="margin-top:8px;">Not yet evaluated — no file found at '
            f'{eval_file_path(config_id)}, or this question_id isn\'t in it yet '
            f'(expected mid-sweep).</p>',
            unsafe_allow_html=True,
        )
 
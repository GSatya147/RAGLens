"""Component 2c: Generation card -- the final answer, terminal node of the pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.styles.style import phase_badge


def render_generation_card(final_answer: str, total_retrieval_latency_ms: float):
    with st.container(border=True):
        st.markdown(phase_badge("generation"), unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:18px; margin-top:10px;"><strong>{final_answer}</strong></p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="rl-mono rl-muted">total retrieval latency: {total_retrieval_latency_ms:.0f}ms</span>',
            unsafe_allow_html=True,
        )


def render_no_generation_card(status: str, final_answer: str, total_retrieval_latency_ms: float, step_count: int):
    """For traces that never reach a 'generate' decision -- e.g. status
    'insufficient_context' after exhausting retries. Deliberately styled
    distinct from a normal generation card so it isn't mistaken for success."""
    with st.container(border=True):
        st.markdown(phase_badge("error"), unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:16px; margin-top:10px;">Pipeline exhausted after '
            f'{step_count} steps without reaching generation.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<p class="rl-muted">status: <strong>{status}</strong> · final_answer: "{final_answer}"</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="rl-mono rl-muted">total retrieval latency: {total_retrieval_latency_ms:.0f}ms</span>',
            unsafe_allow_html=True,
        )


# --- Standalone stress test ---
if __name__ == "__main__":
    st.set_page_config(page_title="Generation Card — standalone test", layout="wide")
    from styles import inject_css
    inject_css()
    st.markdown("#### Component 2c: Generation Card (standalone)")
    render_generation_card("Mr. Lawrence", 2717.38)

    st.markdown("**Edge case: pipeline exhausted without reaching generate**")
    render_no_generation_card("insufficient_context", "insufficient context", 6942.97, step_count=7)
"""Component 2a: Decomposition card.

Shows the original query breaking into sub_queries -- the very first
thing a viewer sees after the spinner ends. Standalone testable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.styles.style import phase_badge


def render_decomposition_card(original_query: str, sub_queries: list[str]):
    with st.container(border=True):
        st.markdown(phase_badge("decomposition"), unsafe_allow_html=True)
        st.markdown(f'<p class="rl-muted" style="margin-top:8px;">"{original_query}"</p>', unsafe_allow_html=True)
        st.markdown('<p class="rl-mono rl-muted">broken into:</p>', unsafe_allow_html=True)
        for i, sq in enumerate(sub_queries, start=1):
            st.markdown(f'<p class="rl-mono" style="margin:2px 0;">{i}. {sq}</p>', unsafe_allow_html=True)


# --- Standalone stress test ---
if __name__ == "__main__":
    st.set_page_config(page_title="Decomposition Card — standalone test", layout="wide")
    from styles import inject_css
    inject_css()
    st.markdown("#### Component 2a: Decomposition Card (standalone)")
    render_decomposition_card(
        original_query="Who is the voice of the character in Spongebob Squarepants with the "
                        "same name as the creature that annelid larvae live like?",
        sub_queries=[
            "What creature do annelid larvae live like?",
            "Who voices the character Plankton in SpongeBob SquarePants?",
        ],
    )
    st.divider()
    st.markdown("**Edge case: single sub-query (1-hop)**")
    render_decomposition_card(original_query="What is the capital of France?", sub_queries=["What is the capital of France?"])
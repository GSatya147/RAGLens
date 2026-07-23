"""Component 1: Input form.

Standalone-testable: run this file directly to see the form with no
pipeline dependency at all. Returns None until submitted, then returns
a dict of inputs for the caller to pass into the live pipeline runner.
"""

import streamlit as st


def render_input_form() -> dict | None:
    """Renders the query/question_id/hop_count/config input form.

    Returns a dict {query, question_id, hop_count, config} on submit,
    or None if not yet submitted.
    """
    with st.form("rl_input_form", clear_on_submit=False):
        query = st.text_area(
            "Query",
            placeholder="Who is the voice of the character in Spongebob Squarepants "
                        "with the same name as the creature that annelid larvae live like?",
            height=80,
        )
        cols = st.columns([2, 1, 1])
        with cols[0]:
            question_id = st.text_input("Question ID", placeholder="2hop__28287_89399")
        with cols[1]:
            hop_count = st.selectbox("Hop count", options=[2, 3, 4], index=0)
        with cols[2]:
            config = st.selectbox("Config", options=["naive", "advanced", "hybrid"], index=2)

        submitted = st.form_submit_button("▶ Run Trace", use_container_width=True, type="primary")

    if submitted:
        if not query or not question_id:
            st.error("Query and Question ID are both required.")
            return None
        return {
            "query": query,
            "question_id": question_id,
            "hop_count": hop_count,
            "config": config,
        }
    return None


# --- Standalone stress test ---
if __name__ == "__main__":
    st.set_page_config(page_title="Input Form — standalone test", layout="wide")
    st.markdown("#### Component 1: Input Form (standalone)")
    result = render_input_form()
    if result:
        st.success("Submitted:")
        st.json(result)
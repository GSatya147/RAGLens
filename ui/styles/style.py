"""Styling for the live-streaming RAGLens demo dashboard.

Two SEPARATE color systems, deliberately kept apart so a viewer never
conflates them:
  1. PHASE colors -- which node of the pipeline is currently executing
     (decomposition / retrieval / reasoning / routing / generation / eval)
  2. CHUNK colors -- retrieval quality within a step (green/amber/red),
     reused from the inspector dashboard, sourced from hop_metrics only.
"""

import streamlit as st

CSS = """
<style>
.rl-phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.rl-phase-decomposition { background: rgba(139,92,246,0.15); color: #A78BFA; }
.rl-phase-retrieval     { background: rgba(59,130,246,0.15); color: #60A5FA; }
.rl-phase-reasoning     { background: rgba(148,163,184,0.15); color: #CBD5E1; }
.rl-phase-routing       { background: rgba(245,158,11,0.15); color: #FBBF24; }
.rl-phase-generation    { background: rgba(34,197,94,0.15); color: #4ADE80; }
.rl-phase-evaluation    { background: rgba(20,184,166,0.15); color: #2DD4BF; }
.rl-phase-error         { background: rgba(239,68,68,0.15); color: #F87171; }

.rl-node-card {
    border-left: 3px solid transparent;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.rl-node-decomposition { border-left-color: #8B5CF6; background: rgba(139,92,246,0.05); }
.rl-node-retrieval     { border-left-color: #3B82F6; background: rgba(59,130,246,0.05); }
.rl-node-reasoning     { border-left-color: #94A3B8; background: rgba(148,163,184,0.05); }
.rl-node-routing       { border-left-color: #F59E0B; background: rgba(245,158,11,0.05); }
.rl-node-generation    { border-left-color: #22C55E; background: rgba(34,197,94,0.05); }
.rl-node-evaluation    { border-left-color: #14B8A6; background: rgba(20,184,166,0.05); }
.rl-node-error         { border-left-color: #EF4444; background: rgba(239,68,68,0.06); }

.rl-mono { font-family: "Source Code Pro", monospace; font-size: 12px; }
.rl-muted { color: rgba(150,150,150,0.9); font-size: 13px; }
.rl-elapsed {
    font-family: "Source Code Pro", monospace;
    font-size: 13px;
    color: #94A3B8;
}
.rl-chunk-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px;
}
.rl-dot-green { background: #22C55E; }
.rl-dot-amber { background: #F59E0B; }
.rl-dot-red   { background: #EF4444; }
</style>
"""

PHASE_META = {
    "decomposition": {"emoji": "🟣", "label": "Decomposition", "css": "decomposition"},
    "retrieval":     {"emoji": "🔵", "label": "Retrieval",     "css": "retrieval"},
    "reasoning":     {"emoji": "⚪", "label": "Reasoning",     "css": "reasoning"},
    "routing":       {"emoji": "🟠", "label": "Routing",       "css": "routing"},
    "generation":    {"emoji": "🟢", "label": "Generation",    "css": "generation"},
    "evaluation":    {"emoji": "🔷", "label": "Evaluation",    "css": "evaluation"},
    "error":         {"emoji": "🔴", "label": "Error",         "css": "error"},
}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def phase_badge(phase_key: str) -> str:
    meta = PHASE_META.get(phase_key, PHASE_META["error"])
    return (
        f'<span class="rl-phase-badge rl-phase-{meta["css"]}">'
        f'{meta["emoji"]} {meta["label"]}</span>'
    )
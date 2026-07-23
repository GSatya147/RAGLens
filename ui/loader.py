"""Loads persisted trace/eval JSONL, filtered by question_id.

File selection is by config_id (traces/traces_{config_id}.jsonl and
eval_traces/traces_{config_id}.jsonl -- same filename, different folder),
then each line is scanned for a matching question_id.

Malformed lines are skipped with a warning rather than crashing the whole
load -- a single bad line in a 270-question sweep file shouldn't take
down the dashboard.
"""

import json
from pathlib import Path

import streamlit as st

from configs.config import TRACE_DIR, EVAL_DIR


def _load_filtered(base_dir: str, config_id: str, question_id: str, question_id_key: str = "question_id") -> dict | None:
    file_path = Path(base_dir) / f"traces_{config_id}.jsonl"

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed line rather than crash the whole load.
                continue
            if record.get(question_id_key) == question_id:
                return record

    return None


@st.cache_data(show_spinner=False)
def get_trace(question_id: str, config_id: str) -> dict | None:
    """Returns the trace dict for (question_id, config_id), or None if not found."""
    return _load_filtered(TRACE_DIR, config_id, question_id)


@st.cache_data(show_spinner=False)
def get_eval(question_id: str, config_id: str) -> dict | None:
    """Returns the eval dict for (question_id, config_id), or None if not found
    (e.g. mid-sweep: trace exists but hasn't been evaluated yet)."""
    return _load_filtered(EVAL_DIR, config_id, question_id)


def trace_file_path(config_id: str) -> Path:
    return Path(TRACE_DIR) / f"traces_{config_id}.jsonl"


def eval_file_path(config_id: str) -> Path:
    return Path(EVAL_DIR) / f"traces_{config_id}.jsonl"
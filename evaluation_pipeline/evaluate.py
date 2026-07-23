import json
import os

from evaluation_pipeline.hop_metrics import compute_hop_metrics
from evaluation_pipeline.faithfulness import step_level_faithfulness
from evaluation_pipeline.classification import classify_failure
from configs.config import EVAL_DIR

def write_trace(trace: dict, output_dir: str) -> str:
    filename = f"traces_{trace['config_id']}.jsonl"
    path = os.path.join(output_dir, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    return path

def evaluate_trace(trace: dict) -> dict:
    hop_metrics    = compute_hop_metrics(trace)
    faithfulness   = step_level_faithfulness(trace)
    classification = classify_failure(trace, hop_metrics, faithfulness)

    eval_trace = {
        "question_id": trace["question_id"],
        "config_id": trace["config_id"],
        "hop_metrics": hop_metrics,
        "step_faithfulness": faithfulness,
        "classification": classification,
    }

    path = write_trace(trace=eval_trace, output_dir=EVAL_DIR)
    
    return eval_trace



from evaluation_pipeline.hop_metrics import compute_hop_metrics
from evaluation_pipeline.faithfulness import step_level_faithfulness
from evaluation_pipeline.classification import classify_failure

def evaluate_trace(trace: dict) -> dict:
    hop_metrics    = compute_hop_metrics(trace)
    faithfulness   = step_level_faithfulness(trace)
    classification = classify_failure(trace, hop_metrics, faithfulness)
    return {
        "question_id": trace["question_id"],
        "config_id": trace["config_id"],
        "hop_metrics": hop_metrics,
        "step_faithfulness": faithfulness,
        "classification": classification,
    }


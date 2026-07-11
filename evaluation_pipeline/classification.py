import json 

from client.client import Client

client = Client()

# 5-class taxonomy
SYSTEM_PROMPT_STANDARD = """You are classifying the failure mode (or success) of one run of a multi-hop agentic RAG system, given its full reasoning trace and hop-level retrieval metrics.

You are NOT re-judging retrieval or faithfulness independently — those scores are already computed and given to you below. Your job is to assign the label that best NAMES the pattern the evidence already shows.

IMPORTANT: hop-level recall reaching 1.0 at a given step means THAT STEP's own sub-query succeeded — it does NOT mean the overall question was already answerable at that point. A multi-hop question requires multiple successful steps by design; each one showing recall=1.0 for its own sub-query is expected and correct, not evidence of unnecessary retrieval. Do NOT infer over-extension from per-step recall alone.

Choose exactly one label:
- PASS: the number of retrieval steps matches what the question's hop structure required, and the final answer is faithfully grounded in the retrieved evidence.
- PREMATURE_COLLAPSE: the agent stopped retrieving and answered before it had gathered evidence for all the hops the question required.
- OVER_EXTENSION: the agent ran MORE retrieval steps than the question's hop count required — re-querying, refining, or retrieving again after it already had everything needed across all hops, not just one step's own recall.
- RETRIEVAL_FAILURE: hop-level recall was low across steps — retrieval never surfaced the needed evidence for one or more hops.
- REASONING_FAILURE: retrieval succeeded across all required hops (recall was high) but the final answer misuses, ignores, or misreads the retrieved evidence.

Respond with ONLY a JSON object, no other text:
{"label": "ONE_OF_THE_ABOVE", "reason": "one sentence citing the specific metric or reasoning step that supports this"}

Example:
Question hop_count: 2. Retrieval steps taken: 2.
Hop recall per step: [1.0, 1.0]
Reasoning: step 1 identified the show, step 2 found the composer and instrument, agent then answered.
Final answer faithful: true
{"label": "PASS", "reason": "Two hops were required and exactly two retrieval steps were taken, each succeeding at its own sub-query — no wasted steps, faithful final answer."}

Example:
Question hop_count: 2. Retrieval steps taken: 5.
Hop recall per step: [1.0, 1.0, 1.0, 1.0, 1.0]
Reasoning: step 1 succeeded, but the agent re-queried three additional times after already having what step 1's sub-query needed.
{"label": "OVER_EXTENSION", "reason": "Only 2 hops were required but 5 retrieval steps were taken — recall being 1.0 at every step reflects each step succeeding at its own query, not that extra steps were needed."}
"""

def build_user_turn_standard(trace: dict, hop_metrics: list[dict], step_faithfulness: list[dict]) -> str:
    recall_series = [h["context_recall"] for h in hop_metrics]
    faithful_series = [f["faithful"] for f in step_faithfulness]
    reasoning_summary = "\n".join(
        f"  Step {r['step_number']} ({r['decision']}): {r['reasoning_trace']}"
        for r in trace["reasoning_steps"]
    )
    return f"""Original question: {trace['original_query']}
            Final answer: {trace['final_answer']}

            Question hop_count: {trace['hop_count']}
            Retrieval steps taken: {len(trace['retrieval_steps'])}
            Hop-level context_recall per step: {recall_series}
            Step-level faithfulness per step: {faithful_series}

            Reasoning steps:
            {reasoning_summary}"""


# 3-class taxonomy for cases where the system declined
SYSTEM_PROMPT_ABSTENTION = """You are classifying WHY a multi-hop agentic RAG system abstained (returned "insufficient context") instead of producing an answer, given its reasoning trace and hop-level retrieval metrics.

Choose exactly one label:
- UNANSWERABLE_CORRECTLY_ABSTAINED: retrieval found the correct supporting evidence (recall reached 1.0 at some step), but that evidence genuinely does not contain the fact needed to answer, and the agent correctly recognized this rather than hallucinating an answer.
- RETRIEVAL_FAILURE: recall never reached 1.0 — the agent abstained because it never found the right evidence, not because the evidence was insufficient.
- OVER_EXTENSION: this is not a separate label choice — it is a flag applied alongside whichever label above fits, when the trace continued retrying after recall had already reached 1.0. Do not use this as a label value.

Respond with ONLY a JSON object, no other text:
{"label": "UNANSWERABLE_CORRECTLY_ABSTAINED or RETRIEVAL_FAILURE", "reason": "one sentence citing the specific recall values or reasoning step that supports this"}

Example:
Hop recall per step: [1.0, 1.0]
Reasoning: agent found the release date of a song but the question asked for the writing date, which never appeared in any retrieved chunk; agent stated it lacked the specific fact and abstained.
{"label": "UNANSWERABLE_CORRECTLY_ABSTAINED", "reason": "Recall was 1.0 from step 1, and the reasoning explicitly identifies that the needed fact (writing date, not release date) was never present in any retrieved chunk."}

Example:
Hop recall per step: [0.0, 0.0, 0.0]
Reasoning: agent's queries kept returning off-topic chunks, none matching the entity in the question; agent abstained after exhausting retries.
{"label": "RETRIEVAL_FAILURE", "reason": "Recall never reached 1.0 across any step — the correct evidence was never surfaced, so abstention reflects a retrieval gap, not a genuinely unanswerable question."}
"""

def build_user_turn_declined(trace: dict, hop_metrics: list[dict]) -> str:
    recall_series = [h["context_recall"] for h in hop_metrics]
    reasoning_summary = "\n".join(
        f"  Step {r['step_number']} ({r['decision']}): {r['reasoning_trace']}"
        for r in trace["reasoning_steps"]
    )
    return f"""Original question: {trace['original_query']}
        System status: {trace['status']}

        Hop-level context_recall per step: {recall_series}

        Reasoning steps:
        {reasoning_summary}"""

# Deterministic over_extension class
def compute_over_extension_flag(trace: dict) -> bool:
    return len(trace["retrieval_steps"]) > trace["hop_count"]

# classification

def classify_failure(trace: dict, hop_metrics: list[dict], step_faithfulness: list[dict]) -> dict:
    """
    Returns: {"label": str, "over_extension_flag": bool, "reason": str}
    """
    over_extension_flag = compute_over_extension_flag(trace)

    if trace["status"] == "insufficient_context":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ABSTENTION},
            {"role": "user", "content": build_user_turn_declined(trace, hop_metrics)},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STANDARD},
            {"role": "user", "content": build_user_turn_standard(trace, hop_metrics, step_faithfulness)},
        ]

    response = client.generate_response(messages=messages, temperature=0)  # expects {"label": str, "reason": str}
    content = json.loads(response["content"])

    return {
        "label": content["label"],
        "over_extension_flag": over_extension_flag,  # always the deterministic value, never response's
        "reason": content["reason"],
    }
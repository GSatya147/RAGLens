import json

from client.client import Client

SYSTEM_PROMPT = """You are auditing a multi-step retrieval-augmented QA system. You will be shown chunks retrieved at ONE step of a multi-hop reasoning process, and the FINAL answer the system produced after all steps.

Your job: decide whether this step's retrieved content was actually USED in reaching the final answer — either because it appears directly in the final answer's wording, OR because it supplied a fact that a LATER step depended on to reach that answer (e.g. identifying an entity, a name, or a date that the next sub-query then searched for).

This is NOT about whether this step's exact words appear in the final answer. Many steps in a multi-hop chain are scaffolding — they exist to identify something needed for the next step, not to appear in the final wording. A scaffolding step that correctly supplied what the next step needed is FAITHFUL, even if none of its content is echoed in the final answer text.

IMPORTANT — watch for indirect/derived links, not just literal string overlap: a chunk may state a fact using a different form than the entity a later step searches for — a common noun instead of a proper noun, a lowercase term instead of a capitalized name, a description instead of a label, or a pun/wordplay relationship between the two. Check whether the LATER STEP'S SUB-QUERY targets something this chunk conceptually identifies, even if the exact surface text differs. Do not mark a step unfaithful just because its wording doesn't textually match the final answer or a later query — trace the underlying concept, not the string.

Mark a step UNFAITHFUL only when its content was irrelevant, wrong, or contradicted by what the system actually used to reach its answer — not merely because the final answer or later query doesn't repeat this step's exact words.

Respond with ONLY a JSON object, no other text:
{"faithful": true or false, "reason": "one sentence, referencing specific content and whether it was scaffolding or directly used"}

Examples:

Sub-query at this step: "Which show won the 1975 Emmy for Outstanding Drama Series?"
Retrieved chunks: ["Police Story won the Primetime Emmy Award for Outstanding Drama Series in 1975."]
Final answer: "Violin"
{"faithful": true, "reason": "This step correctly identified Police Story, which the next step's sub-query then used to find the composer — a scaffolding step that supplied what later retrieval depended on, even though 'Police Story' itself doesn't appear in the final answer."}

Sub-query at this step: "Who composed the theme song for Police Story, and what instrument did they play?"
Retrieved chunks: ["The theme music for Police Story was composed by Jerry Goldsmith, who was classically trained on violin."]
Final answer: "Violin"
{"faithful": true, "reason": "The instrument fact from this chunk appears directly in the final answer."}

Sub-query at this step: "Where was the author born?"
Retrieved chunks: ["The author's publisher was based in New York City."]
Final answer: "The author was born in Chicago."
{"faithful": false, "reason": "This chunk is about the publisher's location, not the author's birthplace, and doesn't match or support the final answer's claim of Chicago — it was neither used directly nor as scaffolding for a later step."}

Sub-query at this step: "What year did the museum open?"
Retrieved chunks: ["The museum's founding director was appointed in 1962, two years before the museum opened."]
Final answer: "The museum opened in 1964."
{"faithful": true, "reason": "The chunk states the museum opened two years after 1962, which correctly derives to 1964 — the final answer, so this step's content was used even though '1964' isn't stated verbatim in the chunk."}

Sub-query at this step: "What creature do annelid larvae live like?"
Retrieved chunks: ["Trochophore larvae live as plankton before metamorphosing into adults."]
Next step's sub-query: "Who voices the character Plankton in SpongeBob SquarePants?"
Final answer: "Mr. Lawrence"
{"faithful": true, "reason": "This chunk identifies 'plankton' as the creature, which the next step's sub-query directly targets by name ('the character Plankton') — a scaffolding step correctly used to identify the entity, even though the chunk's lowercase ecological term and the proper-noun character name are different surface forms of the same link."}
"""

client = Client()

def build_user_turn(sub_query: str, chunks: list[dict], final_answer: str) -> str:
    chunk_texts = "\n".join(f"- {c['text']}" for c in chunks[:3])
    return f"""Sub-query at this step: {sub_query}

            Retrieved chunks (top-3):
            {chunk_texts}

            Final answer (after all steps): {final_answer}"""


def step_level_faithfulness(trace: dict) -> list[dict]:
    if trace["status"] == "insufficient_context":
        return [
            {
                "step_number": step["step_number"],
                "faithful": None,
                "reason": "Trace is an abstention (insufficient_context) — no final answer content to trace faithfulness into.",
            }
            for step in trace["retrieval_steps"]
        ]

    results = []
    for step in trace["retrieval_steps"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_turn(
                    sub_query=step["sub_query"],
                    chunks=step["retrieved_chunks"],
                    final_answer=trace["final_answer"],
                ),
            },
        ]
        response = client.generate_response(messages=messages, temperature=0)  # expects {"faithful": bool, "reason": str}
        content = json.loads(response["content"])

        results.append({
            "step_number": step["step_number"],
            "faithful": content["faithful"],
            "reason": content["reason"],
        })

    return results
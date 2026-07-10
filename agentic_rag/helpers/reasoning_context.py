from client.client import Client

REASONING_SYSTEM_PROMPT: str = """<role>
You are the reasoning component of a multi-step retrieval agent. After each
search, you review what has been gathered and decide what happens next.
You do NOT know how many total steps this question will require — decide
based only on whether the gathered context is sufficient, not on a target
number of steps.
</role>

<decision_options>
- retrieve_more: the context so far is on the right track but incomplete —
  another fact is still needed to fully answer the question.
- refine_query: the most recent search did not return useful information
  (wrong entity, off-topic chunks, or a dead end) — the search itself needs
  to change, not just be repeated.
- generate: the accumulated context is sufficient to answer the original
  question fully and accurately.
</decision_options>

<constraints>
- Base your decision only on whether the CONTENT gathered so far answers
  the question — never on how many searches have already happened.
- Do not pad or repeat searches for a "target number of hops." If the
  context is already sufficient after one search, choose generate.
- If choosing refine_query, you must explain specifically what was wrong
  with the last search (irrelevant results, wrong entity, missing link) —
  not just "need more information."
- If choosing retrieve_more, briefly state what specific fact is still
  missing.
</constraints>

<output_format>
Respond in exactly this json structure, nothing else:
{
    "decision"      : "retrieve_more|refine_query|generate",
    "reasoning"     : "one or two sentences justifying the decision",
    "refine_reason" : "only include this tag if decision is refine_query"
}
</output_format>

<examples>
<example>
<scenario>insufficient_context</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context_so_far>Christopher Nolan directed Oppenheimer.</context_so_far>
<output>
{
    "decision"  : "retrieve_more",
    "reasoning" : "We know who directed the film, but not where that person was born, which is still needed."
}
</output>
</example>

<example>
<scenario>bad_retrieval</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context_so_far>Christopher Nolan directed Oppenheimer. Search results: a list of Nolan's box office earnings and awards.</context_so_far>
<output>
{
    "decision"      : "refine_query",
    "reasoning"     : "The last search returned box office and award information, not birthplace details.",
    "refine_reason" : "The search for "Christopher Nolan" was too broad and returned career facts instead of biographical/birthplace information."
}
</output>
</example>

<example>
<scenario>sufficient_context</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context_so_far>Christopher Nolan directed Oppenheimer. Christopher Nolan was born in London, England. The capital of England is London.</context_so_far>
<output>
{
    "decision"  : "generate",
    "reasoning" : "We now know the director, his country of birth, and that country's capital — enough to answer fully."
}
</output>
</example>
</examples>"""

def build_reasoning_messages(
    original_query: str,
    accumulated_context: str,
    step_counter: int,
) -> list[dict]:
    
    user_content: str = (
        f"<question>{original_query}</question>\n"
        f"<searches_performed_so_far>{step_counter}</searches_performed_so_far>\n"
        f"<context_so_far>{accumulated_context}</context_so_far>\n\n"
        f"Decide what happens next."
    )

    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

if __name__=="__main__":
    client = Client()
    response = client.generate_response(messages=build_reasoning_messages(original_query="what is my name", accumulated_context="the user name was satya, and he is 22 at the moment", step_counter=2))
    print(response["content"])
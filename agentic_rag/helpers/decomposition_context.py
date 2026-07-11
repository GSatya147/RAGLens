from typing import Optional

from client.client import Client

DECOMPOSITION_SYSTEM_PROMPT: str = """<role>
You are the query decomposition component of a multi-step retrieval agent.
Your job is to identify the SINGLE next fact that must be looked up in order
to eventually answer a multi-hop question. You do not answer the question,
and you do not plan out all the steps in advance.
</role>

<constraints>
- Produce exactly ONE search query — never a list, never multiple questions.
- You do not know how many total hops this question requires. Do not guess
  or reference a hop count. Focus only on the immediate next fact needed.
- If this is the first step, base your query only on the original question.
- If a previous search attempt was insufficient, you will be told why —
  use that reason to search differently, not to simply reword the same query.
- If you already have partial evidence and simply need the next fact to
  continue (not because the last search failed), identify what new
  information is still missing and search for THAT — never repeat a query
  already listed in previously tried queries, even worded differently.
- Never answer the original question yourself, even partially.
- Output ONLY the search query text. No explanation, no preamble, no labels.
</constraints>

<examples>
<example>
<scenario>first_step</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<output>Who directed the film Oppenheimer?</output>
</example>

<example>
<scenario>first_step</scenario>
<question>The university that Grace Hopper taught at is located in which U.S. state?</question>
<output>Which university did Grace Hopper teach at?</output>
</example>

<example>
<scenario>refine_step</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context_so_far>Christopher Nolan directed Oppenheimer.</context_so_far>
<previous_attempt_failed_because>The search only returned Nolan's filmography, not his place of birth.</previous_attempt_failed_because>
<output>Where was Christopher Nolan born?</output>
</example>

<example>
<scenario>continue_step</scenario>
<question>Who is the voice of the character in SpongeBob SquarePants with the same name as the creature that annelid larvae live like?</question>
<context_so_far>Annelid larvae (trochophores) live as plankton. The SpongeBob character with that name is Plankton.</context_so_far>
<previously_tried_queries>What creature do annelid larvae live like?</previously_tried_queries>
<output>Who voices the character Plankton in SpongeBob SquarePants?</output>
</example>

<example>
<scenario>continue_step</scenario>
<question>The university that Grace Hopper taught at is located in which U.S. state?</question>
<context_so_far>Grace Hopper taught at Vassar College.</context_so_far>
<previously_tried_queries>Which university did Grace Hopper teach at?</previously_tried_queries>
<output>What U.S. state is Vassar College located in?</output>
</example>
</examples>

<reminder>
Respond with the search query only — a single line of plain text, nothing else.
</reminder>"""


def build_decomposition_messages(
    original_query: str,
    step_counter: int,
    accumulated_context: str = "",
    refine_reason: Optional[str] = None,
    previous_sub_queries: Optional[list[str]] = None,
) -> list[dict]:

    previous_sub_queries = previous_sub_queries or []

    if step_counter == 0:
        user_content = (
            f"<question>{original_query}</question>\n"
            f"<scenario>first_step</scenario>\n\n"
            f"What is the first fact you need to look up?"
        )
        
    elif refine_reason:
        tried_queries_block = "\n".join(f"- {q}" for q in previous_sub_queries) or "(none)"
        user_content = (
            f"<question>{original_query}</question>\n"
            f"<scenario>refine_step</scenario>\n"
            f"<context_so_far>{accumulated_context}</context_so_far>\n"
            f"<previous_attempt_failed_because>{refine_reason}</previous_attempt_failed_because>\n"
            f"<previously_tried_queries>\n{tried_queries_block}\n</previously_tried_queries>\n\n"
            f"What should be searched next? Do not repeat any of the "
            f"previously tried queries above — try a different angle or phrasing "
            f"strategy entirely."
        )

    else:
        tried_queries_block = "\n".join(f"- {q}" for q in previous_sub_queries) or "(none)"
        user_content = (
            f"<question>{original_query}</question>\n"
            f"<scenario>continue_step</scenario>\n"
            f"<context_so_far>{accumulated_context}</context_so_far>\n"
            f"<previously_tried_queries>\n{tried_queries_block}\n</previously_tried_queries>\n\n"
            f"You have evidence so far but not enough to answer. Based on what is "
            f"still missing, what should be searched next? Do not repeat any of "
            f"the previously tried queries above."
        )

    return [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

if __name__=="__main__":
    client = Client()
    response = client.generate_response(messages=build_decomposition_messages(original_query="what is my name", accumulated_context="the user name was satya, and he is 22 at the moment", refine_reason="user name is satya, but we need full name", step_counter=3))
    print(response["content"])
from client.client import Client

GENERATION_SYSTEM_PROMPT = """<role>
You are the answer generation component of a multi-step retrieval agent.
You produce the final answer to a question using ONLY the context provided
to you. You do not use outside knowledge, even if you are confident about
the correct answer from your own training.
</role>

<constraints>
- Answer using ONLY the information present in the provided context.
- If the context does not contain enough information to answer the
  question fully and confidently, respond with exactly:
  "insufficient context"
  Do not guess, infer beyond what is stated, or fill gaps with your own
  knowledge — treat "insufficient context" as a valid, correct answer
  when the evidence genuinely does not support one.
- Do not explain your reasoning, cite sources, or add caveats. Give the
  direct answer only, or the exact phrase "insufficient context".
- Keep the answer as short as possible while still being complete — a
  name, a date, a place, a short phrase. Do not write full sentences
  unless the question requires one.
</constraints>

<examples>
<example>
<scenario>sufficient</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context>Christopher Nolan directed Oppenheimer. Christopher Nolan was born in London, England. The capital of England is London.</context>
<output>London</output>
</example>

<example>
<scenario>insufficient</scenario>
<question>What is the capital of the country where the director of "Oppenheimer" was born?</question>
<context>Christopher Nolan directed Oppenheimer. Nolan has won several awards for his filmmaking.</context>
<output>insufficient context</output>
</example>
</examples>"""


def build_generation_messages(
    original_query: str,
    accumulated_context: str,
) -> list[dict]:

    user_content: str = (
        f"<question>{original_query}</question>\n"
        f"<context>{accumulated_context}</context>\n\n"
        f"Answer the question using only the context above."
    )

    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

if __name__=="__main__":
    client = Client()
    response = client.generate_response(messages=build_generation_messages(original_query="what is my name", accumulated_context="the user is 22 at the moment"))
    print(response["content"])
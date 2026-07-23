import time

from agentic_rag.pipeline import run_agent_pipeline
from evaluation_pipeline.evaluate import evaluate_trace 
from db.db_client import execute_query

batch_size = 5
offset = 103

i=0
while i<4:
   
    sql = """
    SELECT * FROM questions
    ORDER BY question_id
    LIMIT %s
    OFFSET %s;
    """

    rows = execute_query(sql=sql, params=(batch_size, offset))

    if not rows:
        break

    for row in rows:
        print(f"id: {row["question_id"]}, question: {row["question_text"]}, hop_count: {row["hop_count"]}")

        # HYBRID
        hybrid_trace = run_agent_pipeline(user_query=row["question_text"], hop_count=row["hop_count"], question_id=row["question_id"], config="hybrid")

        time.sleep(3)

        hybrid_evaluation_results = evaluate_trace(trace=hybrid_trace)

        # ADVANCED
        advanced_trace = run_agent_pipeline(user_query=row["question_text"], hop_count=row["hop_count"], question_id=row["question_id"], config="advanced")
        
        time.sleep(3)
        
        advanced_evaluation_results = evaluate_trace(trace=advanced_trace)

        # NAIVE
        naive_trace = run_agent_pipeline(user_query=row["question_text"], hop_count=row["hop_count"], question_id=row["question_id"], config="naive")
        
        time.sleep(3)
        
        naive_evaluation_results = evaluate_trace(trace=naive_trace)

    print(f"Batch {i+1} complete")
    offset+=batch_size
    i+=1


# trace = run_agent_pipeline(user_query="Who owns the record label where Neil Young's co-singer on Four Strong Winds records?", hop_count=3, question_id="3hop1__71549_486583_84283", config="hybrid")
# print(trace["final_answer"])

# evaluation_results = evaluate_trace(trace=trace)
# print(evaluation_results)
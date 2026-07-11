import time

from agentic_rag.pipeline import run_agent_pipeline
from evaluation_pipeline.evaluate import evaluate_trace 
from db.db_client import execute_query

# sql = """
# SELECT setseed(0.42);
# SELECT * FROM questions ORDER BY random() LIMIT 5;
# """
# rows = execute_query(sql=sql)
# for row in rows:
#    print(f"id: {row["question_id"]}, question: {row["question_text"]}, hop_count: {row["hop_count"]}")

#    trace = run_agent_pipeline(user_query=row["question_text"], hop_count=row["hop_count"], question_id=row["question_id"], config="hybrid")
#    print(trace["final_answer"])

#    time.sleep(3)

#    evaluation_results = evaluate_trace(trace=trace)
#    print(evaluation_results)


trace = run_agent_pipeline(user_query="Who plays the wife of Big Stan's director in Grown Ups?", hop_count=2, question_id="2hop__803903_78606", config="hybrid")
print(trace["final_answer"])

evaluation_results = evaluate_trace(trace=trace)
print(evaluation_results)
from db.db_client import execute_query, execute_batch, execute_write

def insert_questions(df) -> None: 
        sql = """
            INSERT INTO questions (question_id, hop_count, question_text, ground_truth_answer)
            VALUES (%s, %s, %s, %s);
        """

        params_list = [(
            entry["id"],
            entry["hop_count"],         
            entry["question"],
            entry["answer"],
        ) for id, entry in df.iterrows()]

        execute_batch(sql=sql, params_list=params_list)

def insert_chunks(chunks_list: list[dict]) -> None: 
        sql = """
            INSERT INTO chunks (question_id, hop_count, is_supporting, content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s);
        """

        params_list = [(
            entry["question_id"],
            entry["hop_count"],         
            entry["is_supporting"],
            entry["content"],
            entry["token_count"],
            entry["embedding"]
        ) for entry in chunks_list]

        execute_batch(sql=sql, params_list=params_list)

def get_questions_db() -> None:
    sql    = "SELECT * FROM questions;"
    rows   = execute_query(sql=sql)

    return rows

def get_chunks_db() -> None:
    sql    = "SELECT * FROM chunks;"
    rows   = execute_query(sql=sql)

    return rows

def clear_db() -> None:
    sql    = "TRUNCATE TABLE questions CASCADE; TRUNCATE TABLE chunks CASCADE;"
    execute_write(sql=sql)

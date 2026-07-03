-- Building it against an empty table gives IVFFLAT no real data to compute cluster centroids from.

CREATE INDEX idx_chunks_embedding ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
 
from retrieval.naive_rag.naive_retrieval import NaiveRetriever
from retrieval.advanced_rag.advanced_retrieval import AdvancedRetriever
from retrieval.hybrid_rag.hybrid_retrieval import HybridRetriever

def run_pipeline(user_query: str, retriever: str):
    if retriever.lower() == "naive":
        retrieval_obj = NaiveRetriever()
        nodes = retrieval_obj.retrieve(user_query)

        return nodes
    
    elif retriever.lower() == "advanced":
        obj = AdvancedRetriever()
        nodes = obj.retrieve(user_query)

        return nodes
    
    elif retriever.lower() == "hybrid":
        obj = HybridRetriever()
        nodes = obj.retrieve(user_query)

        return nodes
    
    else: 
        return None

if __name__=="__main__":
    query = "What date saw the writing of the song where the devil went down to the state where WDXQ is located?"
    retriever = input("(naive/advanced/hybrid): ")

    nodes = run_pipeline(user_query=query, retriever=retriever)
    if nodes:
        for node in nodes:
            print(f"text: {node.node.text}, score: {node.score}\n")


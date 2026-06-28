from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    return db

def retrieve(query: str, k: int = 3) -> list:
    db = load_retriever()
    results = db.similarity_search_with_score(query, k=k)
    print(f"\nTop {k} chunks for: '{query}'\n" + "-"*50)
    for i, (doc, score) in enumerate(results):
        print(f"\nChunk {i+1} | similarity score: {round(score, 3)}")
        print(doc.page_content[:300])
    return results

if __name__ == "__main__":
    retrieve("payment due invoices billing net days")
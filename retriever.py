from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import pickle, os

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_bm25_index(db: Chroma):
    all_docs = db.get()
    texts = all_docs["documents"]
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]

    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    return bm25, texts, metadatas

def retrieve(query: str, k_final: int = 5, k_fetch: int = 10) -> list:
    embedding_model = get_embedding_model()

    bge_query = f"Represent this sentence for searching relevant passages: {query}"

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    vector_results = db.similarity_search(bge_query, k=k_fetch)

    bm25, all_texts, all_metas = build_bm25_index(db)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    import numpy as np
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_fetch]
    bm25_results = [
        Document(page_content=all_texts[i], metadata=all_metas[i])
        for i in top_bm25_indices
    ]

    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    print(f"\nCombined pool: {len(combined)} unique chunks")

    pairs = [[query, doc.page_content] for doc in combined]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, combined), key=lambda x: x[0], reverse=True)
    top_chunks = [doc for score, doc in ranked[:k_final]]

    print(f"\nTop {k_final} chunks after reranking:")
    print("-" * 50)
    for i, (score, doc) in enumerate(ranked[:k_final]):
        print(f"\nChunk {i+1} | reranker score: {round(float(score), 3)}")
        print(doc.page_content[:300])

    return [(doc, score) for score, doc in ranked[:k_final]]

if __name__ == "__main__":
    retrieve("What are the payment terms?")
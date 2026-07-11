from sentence_transformers import SentenceTransformer, util
import re

sim_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def split_into_claims(answer: str) -> list[str]:
    clean = re.sub(r'\[SOURCE \d+\]', '', answer).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    return [
        s.strip() for s in sentences
        if len(s.strip()) > 40
        and not s.strip().startswith("According to Section")
        and not s.strip().startswith("NOT FOUND")
    ]

def check_claim_semantic(claim: str, context: str) -> dict:
    context_sentences = re.split(r'(?<=[.!?])\s+', context)
    context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 20]

    if not context_sentences:
        return {"claim": claim, "verdict": "neutral", "confidence": 0.0, "faithful": False}
    claim_embedding = sim_model.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    context_embeddings = sim_model.encode(context_sentences, convert_to_tensor=True, normalize_embeddings=True)


    similarities = util.cos_sim(claim_embedding, context_embeddings)[0]
    best_score = float(similarities.max())
    best_match_idx = int(similarities.argmax())
    best_match = context_sentences[best_match_idx]

    is_faithful = best_score >= 0.71

    return {
        "claim": claim,
        "verdict": "entailment" if is_faithful else "neutral",
        "confidence": round(best_score, 3),
        "faithful": is_faithful,
        "best_matching_context": best_match[:150]
    }

def evaluate_faithfulness(answer: str, context: str) -> dict:
    if "NOT FOUND IN DOCUMENT" in answer:
        print(f"\n{'='*50}")
        print("FAITHFULNESS EVALUATION")
        print(f"{'='*50}")
        print("Model correctly reported information not found.")
        print(f"FAITHFULNESS SCORE: 1.0 (model did not hallucinate)")
        print(f"{'='*50}\n")
        return {"faithfulness_score": 1.0, "claims": []}

    claims = split_into_claims(answer)

    if not claims:
        print("No evaluable claims found.")
        return {"faithfulness_score": 0.0, "claims": []}

    print(f"\n{'='*50}")
    print("FAITHFULNESS EVALUATION")
    print(f"{'='*50}")
    print(f"Evaluating {len(claims)} claim(s) via semantic similarity...\n")

    results = []
    for i, claim in enumerate(claims):
        result = check_claim_semantic(claim, context)
        results.append(result)

        status = "SUPPORTED" if result["faithful"] else "⚠️  NOT SUPPORTED"
        print(f"Claim {i+1}: {claim[:100]}...")
        print(f"  → {status} | similarity: {result['confidence']}")
        if result["faithful"]:
            print(f"  ↳ Matched: \"{result['best_matching_context']}\"")
        print()

    faithful_count = sum(1 for r in results if r["faithful"])
    faithfulness_score = round(faithful_count / len(claims), 2) if claims else 0.0

    print(f"{'='*50}")
    print(f"FAITHFULNESS SCORE: {faithfulness_score} ({faithful_count}/{len(claims)} claims supported)")
    if faithfulness_score == 1.0:
        print("Fully grounded in the source document.")
    elif faithfulness_score >= 0.6:
        print("Partially grounded — review flagged claims.")
    else:
        print("Low grounding — likely contains hallucinations.")
    print(f"{'='*50}\n")

    return {"faithfulness_score": faithfulness_score, "claims": results}

if __name__ == "__main__":
    from retriever import retrieve
    from generator import generate_answer

    query = "What are the payment terms?"
    chunks = retrieve(query, k_final=5, k_fetch=10)
    result = generate_answer(query, chunks)
    evaluate_faithfulness(result["answer"], result["context"])
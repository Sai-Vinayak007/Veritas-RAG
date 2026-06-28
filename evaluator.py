import torch
from transformers import pipeline

nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small"
)

def split_into_claims(answer: str) -> list[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s for s in sentences if len(s) > 10]

def check_claim(claim: str, context: str) -> dict:
    result = nli_model(
        f"{context} [SEP] {claim}",
        truncation=True,
        max_length=512
    )[0]

    label = result["label"].lower()
    score = round(result["score"], 3)
    is_faithful = label == "entailment"

    return {
        "claim": claim,
        "verdict": label,
        "confidence": score,
        "faithful": is_faithful
    }

def evaluate_faithfulness(answer: str, context: str) -> dict:
    claims = split_into_claims(answer)

    print(f"\n{'='*50}")
    print(f"FAITHFULNESS EVALUATION")
    print(f"{'='*50}")
    print(f"Checking {len(claims)} claim(s) against retrieved context...\n")

    results = []
    for i, claim in enumerate(claims):
        result = check_claim(claim, context)
        results.append(result)

        status = "SUPPORTED" if result["faithful"] else "NOT SUPPORTED"
        print(f"Claim {i+1}: {claim}")
        print(f"  → {status} | verdict: {result['verdict']} | confidence: {result['confidence']}")
        print()

    faithful_count = sum(1 for r in results if r["faithful"])
    faithfulness_score = round(faithful_count / len(claims), 2) if claims else 0.0

    print(f"{'='*50}")
    print(f"FAITHFULNESS SCORE: {faithfulness_score} ({faithful_count}/{len(claims)} claims supported)")
    if faithfulness_score == 1.0:
        print("Answer is fully grounded in the source document.")
    elif faithfulness_score >= 0.5:
        print("Answer is partially grounded — review flagged claims.")
    else:
        print("Answer is mostly unsupported — likely hallucination.")
    print(f"{'='*50}\n")

    return {
        "faithfulness_score": faithfulness_score,
        "claims": results
    }

if __name__ == "__main__":
    from retriever import retrieve
    from generator import generate_answer

    query = "payment due invoices billing net days"
    chunks = retrieve(query)
    result = generate_answer(query, chunks)
    evaluate_faithfulness(result["answer"], result["context"])
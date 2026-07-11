import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def format_context_with_citations(chunks: list) -> tuple[str, list]:
    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(chunks):
        label = f"[SOURCE {i+1}]"
        page = doc.metadata.get("page", "?")
        context_parts.append(f"{label} (page {page}):\n{doc.page_content}")
        sources.append({
            "label": label,
            "page": page,
            "preview": doc.page_content[:120] + "...",
            "reranker_score": round(float(score), 3)
        })

    return "\n\n".join(context_parts), sources


def build_prompt(query: str, context: str) -> str:
    return f"""You are a precise legal document assistant. Your job is to answer 
questions strictly based on the provided source excerpts.

STRICT RULES YOU MUST FOLLOW:
1. Only use information explicitly stated in the sources below.
2. After every factual claim, cite the source using [SOURCE X] notation.
3. If the answer is not found in any source, respond with exactly:
   "NOT FOUND IN DOCUMENT: This information is not available in the provided context."
4. Never infer, assume, or use outside knowledge.
5. If sources partially answer the question, say what is found and what is missing.
6. Quote exact phrases from the sources where possible.
7. Be concise — state the answer directly. Do not explain your reasoning process or add commentary about the sources.

SOURCES:
{context}

QUESTION:
{query}

ANSWER (with citations):"""


def check_failsafe(answer: str) -> dict:
    red_flags = [
        "i believe", "i think", "probably", "likely", "might be",
        "in general", "typically", "usually", "often",
        "as per industry standard", "in most contracts", "generally speaking",
        "it can be assumed", "it is reasonable to",
    ]

    answer_lower = answer.lower()
    flagged = [phrase for phrase in red_flags if phrase in answer_lower]

    has_citation = "[source" in answer_lower
    not_found = "not found in document" in answer_lower

    return {
        "has_citation": has_citation,
        "not_found_response": not_found,
        "speculation_flags": flagged,
        "is_safe": len(flagged) == 0 and (has_citation or not_found)
    }


def generate_answer(query: str, chunks: list) -> dict:
    if not chunks:
        return {
            "query": query,
            "answer": "NOT FOUND IN DOCUMENT: No relevant chunks were retrieved.",
            "context": "",
            "sources": [],
            "failsafe": {"is_safe": True, "not_found_response": True,
                        "has_citation": False, "speculation_flags": []}
        }

    context, sources = format_context_with_citations(chunks)
    prompt = build_prompt(query, context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal document analysis assistant. "
                    "You ONLY answer based on provided source text. "
                    "You ALWAYS cite sources using [SOURCE X] notation. "
                    "You NEVER use outside knowledge or make assumptions. "
                    "If information is missing from sources, say so explicitly."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    failsafe = check_failsafe(answer)

    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}")
    print(f"\nANSWER:\n{answer}")

    print(f"\n--- SOURCES USED ---")
    for s in sources:
        print(f"{s['label']} | page {s['page']} | reranker score: {s['reranker_score']}")
        print(f"  {s['preview']}")

    print(f"\n--- FAILSAFE CHECK ---")
    if failsafe["is_safe"]:
        print("Answer appears grounded")
    else:
        print("Potential issues detected:")
        if not failsafe["has_citation"] and not failsafe["not_found_response"]:
            print("  → No citations found in answer")
        if failsafe["speculation_flags"]:
            print(f"  → Speculation language detected: {failsafe['speculation_flags']}")

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "sources": sources,
        "failsafe": failsafe
    }


if __name__ == "__main__":
    from retriever import retrieve
    from evaluator import evaluate_faithfulness

    query = "What are the payment terms?"
    chunks = retrieve(query, k_final=5, k_fetch=10)
    result = generate_answer(query, chunks)
    evaluate_faithfulness(result["answer"], result["context"])
from retriever import retrieve
from generator import generate_answer
from evaluator import evaluate_faithfulness

def run():
    print("="*60)
    print("  VeritasRAG — Legal Document Q&A with Hallucination Detection")
    print("="*60)
    print("  Document: Master SaaS Agreement (Armorblox/Cisco)")
    print("  Type 'quit' to exit")
    print("="*60)

    while True:
        print()
        query = input("Ask a question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("\nExiting VeritasRAG. Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        print("\nSearching document...")

        chunks = retrieve(query, k_final=5, k_fetch=10)

        result = generate_answer(query, chunks)

        eval_result = evaluate_faithfulness(result["answer"], result["context"])

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Query       : {query}")
        print(f"Faithfulness: {eval_result['faithfulness_score']} / 1.0")
        print(f"Failsafe    : {'Passed' if result['failsafe']['is_safe'] else 'Flagged'}")
        print(f"Sources used: {len(result['sources'])} chunks from document")
        print("="*60)

if __name__ == "__main__":
    run()
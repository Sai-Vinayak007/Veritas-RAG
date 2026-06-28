import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query: str, chunks: list) -> dict:
    context = "\n\n---\n\n".join([doc.page_content for doc, score in chunks])
    prompt = f"""You are a helpful assistant. Answer the question below using ONLY 
the context provided. If the answer is not in the context, say 
"I could not find this in the document." Do not use any outside knowledge.
CONTEXT:
{context}
QUESTION:
{query}
ANSWER:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return {
        "query": query,
        "context": context,
        "chunks": chunks,
        "answer": answer
    }

if __name__ == "__main__":
    from retriever import retrieve
    query = "payment due invoices billing net days"
    chunks = retrieve(query)
    result = generate_answer(query, chunks)
    print("\n" + "="*50)
    print("ANSWER:")
    print(result["answer"])
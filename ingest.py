import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_pdf(pdf_path: str) -> list[Document]:
    reader = pypdf.PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(page_content=text, metadata={"page": i}))
    print(f"Loaded {len(docs)} pages")
    return docs

def ingest(pdf_path: str):
    documents = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("Saved to ./chroma_db — ingestion complete!")

if __name__ == "__main__":
    ingest("data/master-saas-agreement.pdf")
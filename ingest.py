import pypdf
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 320 

def load_pdf(pdf_path: str) -> list[Document]:
    reader = pypdf.PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        docs.append(Document(page_content=text, metadata={"page": i}))
    print(f"Loaded {len(docs)} pages")
    return docs

def ingest(pdf_path: str):
    documents = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("Saved to ./chroma_db — ingestion complete!")

if __name__ == "__main__":
    ingest("data/master-saas-agreement.pdf")
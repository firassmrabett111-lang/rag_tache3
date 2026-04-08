import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
RAW_DOCS_FILE = os.path.join(os.path.dirname(__file__), "raw_docs.json")
COLLECTION_NAME = "fallahtech_docs"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def build_embeddings():
    if not os.path.exists(RAW_DOCS_FILE):
        from fallahtech_rag.ingest import ingest_documents
        ingest_documents()

    with open(RAW_DOCS_FILE, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_metadatas = []
    all_ids = []
    idx = 0

    for doc in raw_docs:
        chunks = chunk_text(doc["text"])
        source = doc.get("source", "unknown")
        page = doc.get("page", doc.get("sheet", "N/A"))
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": source, "page": str(page)})
            all_ids.append(f"chunk_{idx}")
            idx += 1

    if all_chunks:
        embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

    return len(all_chunks)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def query_documents(query_text, n_results=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results


if __name__ == "__main__":
    n = build_embeddings()
    print(f"Built {n} chunk embeddings in ChromaDB.")

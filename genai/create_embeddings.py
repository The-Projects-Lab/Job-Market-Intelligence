import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAG_FILE = BASE_DIR / "data" / "gold" / "rag_documents.csv"
DB_DIR = BASE_DIR / "vector_db" / "chroma_jobs"

COLLECTION_NAME = "job_market_rag"

BATCH_SIZE = 500


def main():
    print("Loading RAG documents...")
    df = pd.read_csv(RAG_FILE)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "document" not in df.columns:
        raise ValueError("rag_documents.csv must contain a 'document' column")

    df = df.dropna(subset=["document"]).reset_index(drop=True)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Job market intelligence RAG documents"}
    )

    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        documents = batch["document"].astype(str).tolist()
        ids = [f"job_{i}" for i in range(start, end)]

        embeddings = model.encode(
            documents,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        metadatas = []

        for _, row in batch.iterrows():
            metadatas.append({
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "location": str(row.get("location", "")),
                "skills": str(row.get("skills", "")),
                "source": str(row.get("source", ""))
            })

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Inserted {end}/{total}")

    print("DONE")
    print("Vector DB saved at:", DB_DIR)


if __name__ == "__main__":
    main()

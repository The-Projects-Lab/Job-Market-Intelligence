from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Import the new AIService (Groq + Guardrails)
from genai.ai_service import AIService

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "vector_db" / "chroma_jobs"

COLLECTION_NAME = "job_market_rag"


class JobMarketRAG:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_collection(COLLECTION_NAME)

        print("Initializing Groq AI & Guardrails Service...")
        self.ai_service = AIService()

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    def generate_answer(self, question):
        # 1. Check Guardrail FIRST on the user question
        # If unsafe, stop immediately without wasting ChromaDB query
        guard_check = self.ai_service.guardrail.validate_prompt(question)
        if not guard_check.is_safe:
            return f"⚠️ **Security Notice**: {guard_check.reason}"

        # 2. Retrieve context from ChromaDB
        results = self.retrieve(question, top_k=5)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""

        for i, doc in enumerate(docs):
            meta = metadatas[i]

            context += f"""
Job {i+1}
Title: {meta.get("title")}
Company: {meta.get("company")}
Location: {meta.get("location")}
Skills: {meta.get("skills")}
Description:
{doc}
"""

        # 3. Clean separation of system prompt & user prompt for Groq
        system_prompt = (
            "You are an AI Career Advisor for Indian data jobs. "
            "Answer the user's question using ONLY the job market context below. "
            "Be practical, placement-oriented, and concise. "
            "Mention skills, job roles, and market demand where relevant."
        )

        prompt = f"""
JOB MARKET CONTEXT:
{context}

USER QUESTION:
{question}
"""

        # 4. Generate answer using Groq Provider
        return self.ai_service.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )


if __name__ == "__main__":
    rag = JobMarketRAG()

    while True:
        q = input("\nAsk career question: ")

        if q.lower() in ["exit", "quit", "bye"]:
            break

        ans = rag.generate_answer(q)
        print("\nAI Answer:")
        print(ans)

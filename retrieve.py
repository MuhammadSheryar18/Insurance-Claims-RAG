import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(question, index, chunks, model, top_k=5):
    question_embedding = model.encode([question])
    distances, indices = index.search(
        np.array(question_embedding, dtype=np.float32), top_k
    )
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "distance": distances[0][i]
        })
    return results

if __name__ == "__main__":
    print("Loading index and model...")
    index, chunks = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question = "What is a deductible in health insurance?"
    print(f"Question: {question}")
    print("-" * 50)
    results = retrieve(question, index, chunks, model)
    for r in results:
        print(f"Rank {r['rank']} | Source: {r['source']}")
        print(f"Text: {r['text'][:200]}")
        print("-" * 50)

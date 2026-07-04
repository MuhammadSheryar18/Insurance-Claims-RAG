import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

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
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"]
        })
    return results

def build_prompt(question, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n\n"
    prompt = f"""You are a helpful insurance expert. Answer the question using ONLY the context below.
If the answer is not in the context, say 'I cannot find this in the provided documents.'
Always mention which source document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def generate_answer(prompt, generator):
    response = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        temperature=1.0
    )
    full_text = response[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    return answer

if __name__ == "__main__":
    print("Loading index...")
    index, chunks = load_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loading LLM (this may take a minute)...")
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto"
    )
    question = "What is a deductible in health insurance?"
    print(f"\nQuestion: {question}")
    retrieved = retrieve(question, index, chunks, embed_model)
    prompt = build_prompt(question, retrieved)
    print("Generating answer...")
    answer = generate_answer(prompt, generator)
    print(f"\nAnswer:\n{answer}")
    print("\nSources used:")
    for r in retrieved[:3]:
        print(f"  - {r['source']}")

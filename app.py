import pickle
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

print("Loading index and models...")
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)
print("All models loaded.")

def retrieve(question, top_k=5):
    question_embedding = embed_model.encode([question])
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

def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)
    response = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        temperature=1.0
    )
    full_text = response[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    sources = "\n".join(
        [f"- {r['source']}" for r in retrieved[:3]]
    )
    return answer, sources

with gr.Blocks(title="Insurance Claims RAG") as demo:
    gr.Markdown("# Insurance Claims Document QA System")
    gr.Markdown("Ask questions about healthcare insurance policies. Answers are grounded in source documents.")
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is a deductible?",
                lines=2
            )
            submit_btn = gr.Button("Get Answer", variant="primary")
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", lines=6)
            sources_output = gr.Textbox(label="Sources", lines=3)
    gr.Examples(
        examples=[
            ["What is a deductible?"],
            ["What is covered under preventive care?"],
            ["What is coinsurance?"],
            ["What is the out of pocket maximum?"]
        ],
        inputs=question_input
    )
    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

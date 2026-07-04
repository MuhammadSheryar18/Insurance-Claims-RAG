import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_PATH = "documents"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

def load_pdfs(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            reader = PdfReader(os.path.join(folder_path, filename))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_chunks.append({"source": filename, "text": text})
    return all_chunks

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({"source": doc["source"], "text": split})
    return chunks

def create_embeddings(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

if __name__ == "__main__":
    documents = load_pdfs(DOCS_PATH)
    chunks = chunk_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    embeddings, model = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("Saved FAISS index and chunks to disk.")
    print("Step 3 complete.")

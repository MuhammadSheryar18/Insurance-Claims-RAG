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

def build_index():
    print("Building index from documents...")
    all_chunks = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            reader = PdfReader(os.path.join(DOCS_PATH, filename))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_chunks.append({"source": filename, "text": text})

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in all_chunks:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append({"source": doc["source"], "text": split})

    print(f"Total chunks: {len(chunks)}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("Index built and saved.")

if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    build_index()
else:
    print("Index already exists, skipping build.")

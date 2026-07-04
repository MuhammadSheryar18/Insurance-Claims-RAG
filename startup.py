import os
import pickle
import faiss
import numpy as np
import urllib.request
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_PATH = "documents"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

PDFS = {
    "nsa-health-insurance-basics.pdf": "https://www.cms.gov/files/document/nsa-health-insurance-basics.pdf",
    "sbc-sample.pdf": "https://www.cms.gov/cciio/resources/forms-reports-and-other-resources/downloads/sbc-sample-completed-mm-508-fixed-4-12-16.pdf",
    "medicare-chapter15.pdf": "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/bp102c15.pdf"
}

def download_docs():
    os.makedirs(DOCS_PATH, exist_ok=True)
    for filename, url in PDFS.items():
        filepath = os.path.join(DOCS_PATH, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")

def build_index():
    download_docs()
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

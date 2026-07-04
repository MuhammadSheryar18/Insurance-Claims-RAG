import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "documents"

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
    print(f"Total chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    documents = load_pdfs(DOCS_PATH)
    chunks = chunk_documents(documents)
    print(f"Sample chunk:\n{chunks[0]['text'][:300]}")

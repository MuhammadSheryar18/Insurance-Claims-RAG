
# Insurance Claims Document QA System

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about healthcare insurance policies, grounded in real source documents with explicit citation of which document each answer comes from.

**Live Demo:** [huggingface.co/spaces/Sheryar1998/Insurance-Claims-RAG](https://huggingface.co/spaces/Sheryar1998/Insurance-Claims-RAG)

---

## Problem Statement

Healthcare insurance claims processing requires navigating large volumes of unstructured policy documents. Claims adjusters and care providers need quick, accurate answers to specific questions buried across these documents. Manual lookup is slow and error-prone, and generic LLM responses risk hallucinated answers, an unacceptable failure mode in a domain involving patient care and financial liability.

This system solves that by grounding every answer in retrieved source passages, and refusing to answer when the information is not in the documents.

---

## Architecture

```text
PDF Documents
      |
      v
Text Extraction (pypdf)
      |
      v
Chunking (500 chars, 50 char overlap)
      |
      v
Embeddings: all-MiniLM-L6-v2
      |
      v
FAISS Vector Store (IndexFlatL2)
      |
      v
User Query --> Embed Query --> Retrieve Top-5 Chunks --> Build Prompt --> TinyLlama --> Answer + Source
```

## Document Corpus

Real public CMS (Centers for Medicare and Medicaid Services) documents:

* NSA Health Insurance Basics Guide
* CMS Summary of Benefits and Coverage Sample
* Medicare Benefit Policy Manual Chapter 15

---

## Tech Stack

| Component | Technology |
|---|---|
| Document Parsing | pypdf |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (IndexFlatL2) |
| Language Model | TinyLlama-1.1B-Chat-v1.0 |
| Interface | Gradio |
| Deployment | Hugging Face Spaces |

---

## Example Questions

* What is a deductible in health insurance?
* What is covered under preventive care?
* What is coinsurance?
* What is the out of pocket maximum?
* What is the difference between in-network and out-of-network coverage?

---

## Key Design Decisions

**Source attribution:** Every answer includes which document it was retrieved from. In regulated domains like insurance and healthcare, auditability is not optional.

**Grounded generation:** The LLM prompt explicitly instructs the model to answer only from retrieved context and say so when it cannot find the answer, preventing hallucination.

**Chunk overlap:** 50 character overlap between chunks prevents information loss at chunk boundaries, a common failure mode in naive chunking approaches.

**On-startup index building:** The FAISS index is built at runtime by downloading public CMS documents, keeping the repository clean and avoiding binary file storage in git.

---

## Limitations and Next Steps

* TinyLlama is a small model; answer quality would improve significantly with a larger LLM
* Currently limited to 3 source documents; expanding the corpus would improve coverage
* No reranking of retrieved chunks; adding a cross-encoder reranker would improve retrieval precision
* Evaluation: a formal groundedness evaluation against a labeled QA test set is a planned next step

---

## Author

Muhammad Sheryar | [LinkedIn](https://www.linkedin.com/in/muhammad-sheryar-8342ab1b0/) | [GitHub](https://github.com/MuhammadSheryar18)

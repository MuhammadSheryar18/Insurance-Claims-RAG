---
title: Insurance Claims RAG
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Insurance Claims Document QA System

A Retrieval-Augmented Generation (RAG) system for querying healthcare insurance policy documents.

## How it works
1. Loads real CMS healthcare insurance policy PDFs
2. Chunks and embeds documents using Sentence Transformers
3. Stores vectors in FAISS for fast retrieval
4. Uses TinyLlama to generate grounded answers with source attribution

## Tech Stack
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS vector store
- TinyLlama-1.1B-Chat
- Gradio interface

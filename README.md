# 📚 Retrieval-Augmented Generation (RAG) System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for querying a knowledge base of PDF documents using a Large Language Model (LLM).

The system supports:

* PDF ingestion and indexing
* Hybrid retrieval (semantic + keyword search)
* Intent-aware orchestration (deciding when to use RAG)
* Answer generation with citations
* Hallucination filtering (evidence validation)
* Lightweight chat UI

---

## System Architecture

```
                ┌────────────────────┐
                │   User Query       │
                └─────────┬──────────┘
                          ↓
              ┌──────────────────────┐
              │  Intent Classifier   │
              └─────────┬────────────┘
                        ↓
        ┌───────────────┼────────────────┐
        │               │                │
   Greeting        General LLM       KB Lookup
 (no RAG)          (no search)       (RAG)
                                          ↓
                              ┌──────────────────┐
                              │ Query Rewriting  │
                              └─────────┬────────┘
                                        ↓
                              ┌──────────────────┐
                              │ Hybrid Retrieval │
                              └─────────┬────────┘
                                        ↓
                              ┌──────────────────┐
                              │ Re-ranking       │
                              └─────────┬────────┘
                                        ↓
                              ┌──────────────────┐
                              │ LLM Generation   │
                              └─────────┬────────┘
                                        ↓
                              ┌──────────────────┐
                              │ Evidence Check   │
                              └─────────┬────────┘
                                        ↓
                              ┌──────────────────┐
                              │ Final Response   │
                              └──────────────────┘
```

---

## Data Ingestion

### Pipeline

1. Upload PDF(s) via API
2. Extract text per page
3. Chunk text into overlapping segments
4. Generate embeddings using Mistral API
5. Store:

   * Chunk text
   * Source file name
   * Page number
   * Chunk ID
   * Embeddings (NumPy)

---

### Chunking Strategy

* Chunk size: 300–500 tokens
* Overlap: 50–100 tokens
* Rationale:

  * Preserves semantic coherence
  * Improves recall during retrieval
  * Prevents context fragmentation

---

## Query Processing

### Intent Detection

The system classifies queries into:

| Type      | Behavior          |
| --------- | ----------------- |
| GREETING  | No retrieval      |
| GENERAL   | Direct LLM        |
| KB_LOOKUP | Full RAG pipeline |
| UNSAFE    | Refusal           |

---

### Query Rewriting

Queries are rewritten using the LLM to:

* Expand abbreviations
* Clarify intent
* Improve retrieval recall

---

## Semantic Search

### Hybrid Retrieval

We combine:

* **Semantic similarity** (cosine similarity on embeddings)
* **Keyword matching** (token overlap)

Final score:

```
score = 0.7 * semantic + 0.3 * keyword
```

---

## 🔄 Post-processing

### Re-ranking

Top-K results are sorted based on hybrid score.

---

### Metadata-Aware Retrieval

Each chunk includes:

```json
{
  "id": 42,
  "text": "...",
  "source": "file.pdf",
  "page": 12
}
```

This enables **traceable citations**.

---

## 🤖 Generation

The system uses Mistral API to generate answers.

### Prompt Design

* Strict grounding instructions
* Context-only answering
* Refusal if insufficient evidence

---

### Answer Shaping

Output format is selected dynamically:

| Query        | Format        |
| ------------ | ------------- |
| "List..."    | Bullet points |
| "Compare..." | Table         |
| "Explain..." | Paragraph     |

---

## Hallucination Filter

A post-hoc validation step ensures factual grounding:

1. Split answer into sentences
2. Embed each sentence
3. Compare against retrieved chunks
4. Reject answer if unsupported

If validation fails:

```
→ "insufficient evidence"
```

---

## 💬 API Endpoints

### Ingestion

```
POST /ingest
```

* Upload one or more PDF files
* Extracts, chunks, embeds, and stores data

---

### 🔎 Query

```
GET /query?q=...
```

Returns:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "file.pdf",
      "page": 3,
      "preview": "..."
    }
  ]
}
```

---

## UI

A lightweight HTML/JS chat interface:

* Supports querying the backend
* Displays responses and citations
* Allows PDF uploads

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Start backend

```
uvicorn app.main:app --reload
```

---

### 3. Open API docs

```
http://127.0.0.1:8000/docs
```

---

### 4. Run UI

```
cd ui
python -m http.server 3000
```

Open:

```
http://localhost:3000
```

---

## Tech Stack

* FastAPI (backend API)
* Mistral AI API (LLM + embeddings)
* NumPy (vector storage)
* PyPDF2 (PDF parsing)
* Vanilla JS (UI)

---

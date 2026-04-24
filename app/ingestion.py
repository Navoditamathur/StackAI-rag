import os
import json
import numpy as np
from PyPDF2 import PdfReader
from mistralai.client import Mistral
from app.config import *
import uuid


client = Mistral(api_key=MISTRAL_API_KEY)

DOC_PATH = "data/documents.json"
EMB_PATH = "data/embeddings.npy"

def extract_text(file, filename):
    reader = PdfReader(file)
    data = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text:
            continue

        data.append({
            "text": text,
            "page": page_num + 1,
            "source": filename
        })

    return data
    
def chunk_pages(pages):
    doc_id = str(uuid.uuid4())
    chunks = []

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "doc_id": doc_id
            })

    return chunks

import time

def call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries exceeded")

def embed_batch(batch):
    return call_with_retry(lambda: client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=batch
    ))
      
def embed_texts(texts, batch_size=16):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embed_batch(batch)
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def save_data(chunks, embeddings):
    if not os.path.exists(DOC_PATH):
        os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    
    if not os.path.exists(EMB_PATH):
        os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)

    if os.path.exists(DOC_PATH) and os.path.getsize(DOC_PATH) > 0:
        with open(DOC_PATH, "r") as f:
            docs = json.load(f)
    else:
        docs = []

    start_id = len(docs)

    for i, chunk in enumerate(chunks):
        chunk["id"] = start_id + i

    docs.extend(chunks)

    with open(DOC_PATH, "w") as f:
        json.dump(docs, f)

    if os.path.exists(EMB_PATH):
        existing = np.load(EMB_PATH)
        embeddings = np.vstack([existing, embeddings])

    np.save(EMB_PATH, embeddings)

def ingest(file):
    filename = os.path.basename(file.filename)
    
    pages = extract_text(file.file, filename)
    chunks = chunk_pages(pages)
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embed_texts(texts)
    
    save_data(chunks, embeddings)
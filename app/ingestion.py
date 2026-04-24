import os
import json
import numpy as np
from PyPDF2 import PdfReader
from mistralai.client import Mistral
from app.config import *

client = Mistral(api_key=MISTRAL_API_KEY)

DOC_PATH = "data/documents.json"
EMB_PATH = "data/embeddings.npy"

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks


def embed_texts(texts, batch_size=16):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=batch
        )

        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def save_data(chunks, embeddings):
    if os.path.exists(DOC_PATH) and os.path.getsize(DOC_PATH) > 0:
        try:
            with open(DOC_PATH, "r") as f:
                docs = json.load(f)
        except json.JSONDecodeError:
            docs = []
    else:
        docs = []

    docs.extend(chunks)

    with open(DOC_PATH, "w") as f:
        json.dump(docs, f)

    if os.path.exists(EMB_PATH):
        existing = np.load(EMB_PATH)
        embeddings = np.vstack([existing, embeddings])

    np.save(EMB_PATH, embeddings)


def ingest(file):
    text = extract_text(file)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    save_data(chunks, embeddings)
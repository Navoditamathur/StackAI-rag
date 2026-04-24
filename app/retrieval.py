import json
import numpy as np
from app.config import *
from mistralai.client import Mistral

client = Mistral(api_key=MISTRAL_API_KEY)

def load_data():
    with open("data/documents.json") as f:
        docs = json.load(f)
    embeddings = np.load("data/embeddings.npy")
    return docs, embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def keyword_score(query, doc):
    return sum(word in doc.lower() for word in query.lower().split())


def embed_query(query):
    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[query]
    )
    return res.data[0].embedding


def search(query):
    docs, embeddings = load_data()
    q_emb = embed_query(query)

    scores = []
    for i, emb in enumerate(embeddings):
        sem = cosine_similarity(q_emb, emb)
        key = keyword_score(query, docs[i])
        score = 0.7 * sem + 0.3 * key
        scores.append((score, docs[i]))

    scores.sort(reverse=True)
    print("Scores: ", scores)
    top = scores[:TOP_K]

    if top[0][0] < SIM_THRESHOLD:
        return []

    return [doc for _, doc in top]
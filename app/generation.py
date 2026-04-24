from mistralai.client import Mistral
from app.config import *
from app.ingestion import embed_texts
from app.retrieval import cosine_similarity
import re   

client = Mistral(api_key=MISTRAL_API_KEY)

def build_prompt(query, context, memory, answer_type="PARAGRAPH"):
    
    if answer_type == "LIST":
        format_instruction = "Return the answer as a bullet-point list."
    
    elif answer_type == "TABLE":
        format_instruction = "Return the answer as a markdown table."

    else:
        format_instruction = "Return a clear paragraph answer."

    return f"""
You are a strict QA system.

Rules:
- Use ONLY the provided context
- DO NOT guess
- If missing info → say "insufficient evidence"

{format_instruction}

Context:
{context}

if {memory != ""}:
    Conversation history:
    {memory}

Question:
{query}
"""

def generate_answer(query, docs, memory="", answer_type="PARAGRAPH"):
    context = "\n\n".join([d["text"] for d in docs])
    prompt = build_prompt(query, context, memory, answer_type)
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "sources": [
        {
            "source": d["source"],
            "page": d["page"],
            "preview": d["text"][:200],
        }
        for d in docs
        ]
}

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

def hallucination_check(answer, docs, threshold=0.65):
    sentences = split_sentences(answer)

    sentence_embeddings = embed_texts(sentences)
    sources = [s['source'] for s in docs]
    doc_embeddings = embed_texts(sources)

    unsupported = []

    for i, sent_emb in enumerate(sentence_embeddings):
        supported = False

        for doc_emb in doc_embeddings:
            sim = cosine_similarity(sent_emb, doc_emb)
            if sim >= threshold:
                supported = True
                break

        if not supported:
            unsupported.append(sentences[i])

    return unsupported
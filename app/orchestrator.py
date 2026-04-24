from mistralai.client import Mistral
from app.config import *

client = Mistral(api_key=MISTRAL_API_KEY)

def detect_answer_type(query: str):
    prompt = f"""
Classify the expected answer format:

- PARAGRAPH → normal explanation
- LIST → multiple bullet points
- TABLE → structured comparison or data

Return ONLY one label.

Query: {query}
"""

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip().upper()

def classify_intent(query: str):
    """
    Decide whether RAG is needed.
    """

    prompt = f"""
Classify the user query into one of the following:

- GREETING
- GENERAL
- KB_LOOKUP
- UNSAFE

Rules:
- Greetings or casual chat → GREETING
- Questions answerable without documents → GENERAL
- Questions about uploaded files / internal knowledge → KB_LOOKUP
- Requests involving PII, legal/medical advice → UNSAFE

Return ONLY the label.

Query: {query}
"""

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    label = response.choices[0].message.content.strip()
    return label

def needs_memory(query: str):
    """
    Decide whether RAG needs memory.
    """

    prompt = f"""
Does this query depend on previous conversation context?

Answer ONLY:
YES or NO

Examples:
- "What about its limitations?" → YES
- "Explain transformers" → NO

Query: {query}
"""

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip() == "YES"
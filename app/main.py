from app.orchestrator import classify_intent, needs_memory
from app.memory import add_turn, get_context
from fastapi import FastAPI, UploadFile, File
from app.ingestion import ingest
from app.retrieval import search
from app.generation import generate_answer, hallucination_check
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    for file in files:
        ingest(file)
    return {"status": "ingested"}


@app.get("/query")
def query(q: str):
    intent = classify_intent(q)
    if intent == "GREETING":
        return {
            "answer": "Hello! How can I help?",
            "sources": []
        }

    if intent == "UNSAFE":
        return {
            "answer": "I cannot help with that.",
            "sources": []
        }

    use_memory = needs_memory(q)

    memory_context = ""
    if use_memory:
        history = get_context()
        memory_context = "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
        )

    if intent == "GENERAL":
        answer = generate_answer(q, [], memory_context)
        return answer

    if intent == "KB_LOOKUP":
        docs = search(q)
        if not docs:
            return {
                "answer": "insufficient evidence",
                "sources": []
            }

        answer = generate_answer(q, docs, memory_context)
        unsupported = hallucination_check(answer["answer"], docs)

        if len(unsupported) > 0:
            return {
                "answer": "insufficient evidence",
                "unsupported_claims": unsupported
            }

        return answer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import get_answer

app = FastAPI(title="RAG Customer Support Chatbot")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"message": "RAG Customer Support Chatbot is running"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = get_answer(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

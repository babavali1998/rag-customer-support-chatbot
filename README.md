# RAG-Powered Customer Support Chatbot

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with LangChain, OpenAI, and Pinecone for intelligent customer support automation. Originally developed during graduate research at the University of Southern Mississippi, achieving 80%+ answer relevance and 30% reduction in repetitive support inquiries.

## Tech Stack
- **LangChain** — orchestration and chain management
- **OpenAI API** — embeddings and GPT-4 completions
- **Pinecone** — vector database for semantic search
- **FastAPI** — REST API microservice
- **Docker** — containerized deployment
- **Python-dotenv** — environment variable management

## Architecture

User Query
    ↓
FastAPI (/ask endpoint)
    ↓
LangChain RetrievalQA Chain
    ↓
OpenAI Embeddings (text-embedding-ada-002)
    ↓
Pinecone Vector Store (top-3 semantic chunks)
    ↓
GPT-4 + Custom Prompt Template
    ↓
Final Answer + Source Documents

## Project Structure

rag-customer-support-chatbot/
├── main.py            # FastAPI app with /ask and /health endpoints
├── rag_chain.py       # LangChain RAG pipeline logic
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # Project documentation

## Results
- 80%+ answer relevance on custom evaluation set
- 30% reduction in repetitive support inquiries
- Deployed as containerized FastAPI microservice
- Top-3 semantic chunk retrieval with source attribution

## How to Run

1. Clone the repository
git clone https://github.com/babavali1998/rag-customer-support-chatbot
cd rag-customer-support-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Set up environment variables
cp .env.example .env

4. Run the application
uvicorn main:app --reload

5. Test the API
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "How do I reset my password?"}'

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Root health check |
| POST | /ask | Submit a question, get RAG answer |
| GET | /health | Service health status |

## Environment Variables

OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=your_pinecone_environment_here
PINECONE_INDEX=your_pinecone_index_name_here

## Dependencies

fastapi==0.104.1
uvicorn==0.24.0
langchain==0.0.350
openai==1.3.7
pinecone-client==2.2.4
python-dotenv==1.0.0
pydantic==2.5.2
tiktoken==0.5.2

## Author
Babavali Kotcherla
MS Computer Science — University of Southern Mississippi
babavali.kotcherla@gmail.com
Ocean Springs, MS

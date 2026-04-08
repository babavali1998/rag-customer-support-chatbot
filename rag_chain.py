from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Prompt template
PROMPT_TEMPLATE = """
You are a helpful customer support assistant. 
Use the following context to answer the question accurately.
If you don't know the answer, say so clearly.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def get_answer(question: str) -> dict:
    # Load embeddings and vector store
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = Pinecone.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"),
        embedding=embeddings
    )
    
    # Build RAG chain
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Get response
    result = qa_chain({"query": question})
    
    # Extract source documents
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    ]
    
    return {
        "answer": result["result"],
        "sources": sources
    }

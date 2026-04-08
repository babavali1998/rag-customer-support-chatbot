from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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

# Initialize once at module load — not per request
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX"),
    embedding=embeddings
)

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

def get_answer(question: str) -> dict:
    # Get response
    result = qa_chain.invoke({"query": question})

    # Extract source documents
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    ]

    return {
        "answer": result["result"],
        "sources": sources
    }

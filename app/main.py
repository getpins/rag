from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
import logging
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Simple RAG API")

# Auth secret from environment
API_SECRET = os.getenv("API_SECRET", "changeme")

logger.info(f"API_SECRET loaded: {'***' if API_SECRET != 'changeme' else 'changeme (default!)'}")

def verify_bearer_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    if token != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

# Connect to Chroma service
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

logger.info("Initializing embedding function with model: all-MiniLM-L6-v2")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)
logger.info("ChromaDB collection ready.")

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Document(BaseModel):
    content: str
    metadata: dict = {}
    id: str | None = None

class Query(BaseModel):
    question: str
    max_results: int = 5

@app.post("/documents")
async def add_or_update_document(doc: Document, _: str = Depends(verify_bearer_token)):
    try:
        doc_id = doc.id if doc.id else str(uuid.uuid4())
        existing = collection.get(ids=[doc_id])
        if existing['ids']:
            collection.update(
                ids=[doc_id],
                documents=[doc.content],
                metadatas=[doc.metadata]
            )
            return {"document_id": doc_id, "status": "Document updated successfully"}
        else:
            collection.add(
                documents=[doc.content],
                metadatas=[doc.metadata],
                ids=[doc_id]
            )
            return {"document_id": doc_id, "status": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/answer")
async def answer_query(query: Query, _: str = Depends(verify_bearer_token)):
    try:
        results = collection.query(
            query_texts=[query.question],
            n_results=query.max_results
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        if not documents:
            return {"question": query.question, "answer": "No relevant documents found", "retrieved_documents": []}

        context = "\n\n".join(documents)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the provided documents."},
                {"role": "user", "content": f"Question: {query.question}\n\nDocuments:\n{context}\n\nPlease provide the answer based on the documents."}
            ]
        )

        answer = response.choices[0].message.content.strip()

        return {
            "question": query.question,
            "answer": answer,
            "retrieved_documents": [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(documents, metadatas)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

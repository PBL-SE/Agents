import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, select, Table, MetaData
from sqlalchemy.orm import sessionmaker
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


# Load environment variables
load_dotenv()

# ----------------------------
# Environment Variables & Configuration
# ----------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NEONDB_URL = os.getenv("NEONDB_URL")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, NEONDB_URL]):
    raise Exception("One or more required environment variables are missing.")

# ----------------------------
# Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------------------
# Initialize SQLAlchemy (NeonDB Postgres)
# ----------------------------
""" engine = create_engine(NEONDB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()
papers_table = Table("papers", metadata, autoload_with=engine) """


# ----------------------------
# Load BERT tokenizer and MODEL for embeddings
# ----------------------------
device = "cpu"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# ----------------------------
# FastAPI Application Initialization
# ----------------------------
app = FastAPI(
    title="Curated Research Paper Recommendation API",
    description="Agentically curate and personalize research paper recommendations for users using Gemini.",
    version="1.0",
)

# ----------------------------
# Pydantic Models
# ----------------------------
class QueryRequest(BaseModel):
    query: str
    difficulty_level: Optional[int] = None
    user_id: Optional[str] = None

# ----------------------------
# LangChain Integration
# ----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
    # other params...
)

# ----------------------------
# Query Refinement & Summary Generation
# ----------------------------
refine_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=(
        """
            You are an intelligent query refiner. Your goal is to make the user's search query slightly more specific if necessary, but avoid overcomplicating it.

            **Rules:**
            1. If the query is already clear (like "Deep learning"), keep it as is.
            2. Only add minimal keywords if the query is vague, but don't introduce unnecessary complexity.
            3. Do NOT create Boolean operators or complex logical conditions unless explicitly requested.

            **Original Query:** "{original_query}"
            **Refined Query:** 
        """
    ),
)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# ----------------------------
# Helper Functions
# ----------------------------
def refine_query(original_query: str) -> str:
    if len(original_query.split()) <= 3:
        return original_query  # Return as-is for short, specific queries
    else:
        return refine_chain.run(original_query=original_query).strip()

def get_bert_embedding(text):
    """Generates BERT embeddings for text."""
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token ([0, :]) as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()  # Convert NumPy array to Python list


def query_pinecone(refined_query: str, top_k: int = 10) -> List[str]:
    query_embedding = get_bert_embedding(refined_query)
    response = index.query(vector=query_embedding, top_k=top_k, namespace="ns1")
    print("Response: ", response)
    return [match.id for match in response.matches]

""" def get_papers_from_db(arxiv_ids: List[str]) -> List[dict]:
    with SessionLocal() as session:
        stmt = select(papers_table).where(papers_table.c.arxiv_id.in_(arxiv_ids))
        return [dict(row._mapping) for row in session.execute(stmt).fetchall()]
 """
# ----------------------------
# API Endpoint: /query
# ----------------------------
@app.post("/query", response_model=List[str])  # Change the response model to List[str] to return candidate_ids
def query_papers(request: QueryRequest):
    refined_query = refine_query(request.query)
    print(refined_query)
    candidate_ids = query_pinecone(refined_query)
    
    if not candidate_ids:
        raise HTTPException(status_code=404, detail="No papers found from vector search.")
    
    # Return only the candidate_ids (arxiv_id)
    return candidate_ids

@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello World"}

""" # ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("query_refiner:app", host="0.0.0.0", port=8000, reload=True)
 """
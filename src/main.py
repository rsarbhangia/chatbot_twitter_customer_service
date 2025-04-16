from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import json

from database import get_db, CustomerInteraction
from rag import RAGSystem
from chatbot import CustomerSupportChatbot

app = FastAPI(title="Customer Support AI API")
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

# Initialize the RAG system and chatbot
rag_system = RAGSystem()
chatbot = CustomerSupportChatbot(rag_system=rag_system)

class Query(BaseModel):
    text: str

class Response(BaseModel):
    response: str
    contexts: List[str]
    confidence_score: int

class InteractionResponse(BaseModel):
    id: int
    query: str
    response: str
    timestamp: str
    context_used: str
    confidence_score: int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def process_query(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    query = data.get("text", "")
    
    if not query:
        return {"error": "No query provided"}
    
    try:
        print(f"Processing query: {query}")
        # Use the chatbot to process the query
        response, contexts, confidence_scores = chatbot.process_query(query)
        print(f"Response: {response}")
        print(f"Contexts: {contexts}")
        print(f"Confidence scores: {confidence_scores}")
        
        # Store the interaction in the database
        db_interaction = CustomerInteraction(
            query=query,
            response=response,
            context_used=json.dumps(contexts),
            confidence_score=confidence_scores[0] if confidence_scores else 0
        )
        db.add(db_interaction)
        db.commit()
        
        return {
            "response": response,
            "contexts": contexts,
            "confidence_scores": confidence_scores
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/interactions", response_model=List[InteractionResponse])
async def get_interactions(limit: int = 10, db: Session = Depends(get_db)):
    interactions = db.query(CustomerInteraction).order_by(
        CustomerInteraction.timestamp.desc()
    ).limit(limit).all()
    
    return [
        InteractionResponse(
            id=interaction.id,
            query=interaction.query,
            response=interaction.response,
            timestamp=interaction.timestamp.isoformat(),
            context_used=interaction.context_used,
            confidence_score=interaction.confidence_score
        )
        for interaction in interactions
    ]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
from fastapi import FastAPI, Depends, Request, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import pathlib
import uuid
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import get_db, CustomerInteraction
from rag import RAGSystem
from chatbot import CustomerSupportChatbot

load_dotenv()

app = FastAPI(title="Customer Support AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot-twitter-customer-service.onrender.com"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize the RAG system and chatbot
rag_system = RAGSystem()
chatbot = CustomerSupportChatbot(rag_system=rag_system)

# Store chat sessions in memory (in production, use Redis or similar)
chat_sessions: Dict[str, list] = {}

class ChatSession:
    def __init__(self):
        self.history = []
        self.last_activity = datetime.utcnow()

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
async def process_query(
    request: Request, 
    db: Session = Depends(get_db),
    session_id: Optional[str] = Cookie(None)
):
    # Get or create session ID
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get or create chat session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession()
    
    data = await request.json()
    query = data.get("text", "")
    
    if not query:
        return {"error": "No query provided"}
    
    try:
        # Get chat history for this session
        session = chat_sessions[session_id]
        
        # Use the chatbot to process the query with session history
        response, contexts, confidence_scores = chatbot.process_query(query)
        
        # Update session history
        session.history.append({"role": "user", "content": query})
        session.history.append({"role": "assistant", "content": response})
        session.last_activity = datetime.utcnow()
        
        # Store in database as before
        db_interaction = CustomerInteraction(
            session_id=session_id,  # Add this field to the model
            query=query,
            response=response,
            context_used=json.dumps(contexts),
            confidence_score=confidence_scores[0] if confidence_scores else 0
        )
        db.add(db_interaction)
        db.commit()
        
        response = {
            "response": response,
            "contexts": contexts,
            "confidence_scores": confidence_scores
        }
        
        # Set cookie in response
        response_obj = JSONResponse(content=response)
        response_obj.set_cookie(key="session_id", value=session_id)
        return response_obj
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/interactions", response_model=List[InteractionResponse])
async def get_interactions(limit: int = 10, db: Session = Depends(get_db)):
    """
    Get the last N interactions from the database
    """
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

# Initialize the scheduler
scheduler = AsyncIOScheduler()

# Session cleanup function
async def cleanup_old_sessions():
    current_time = datetime.utcnow()
    expired_sessions = [
        sid for sid, session in chat_sessions.items()
        if (current_time - session.last_activity) >= timedelta(hours=24)  # Remove sessions inactive for 24 hours
    ]
    for sid in expired_sessions:
        del chat_sessions[sid]
        print(f"Cleaned up session {sid}")

# Start the scheduler on app startup
@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(cleanup_old_sessions, 'interval', hours=1)
    scheduler.start()

# Stop the scheduler on app shutdown
@app.on_event("shutdown")
async def stop_scheduler():
    scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Get port from environment variable or default to 8000
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Changed from 127.0.0.1 to allow external connections
        port=port
    )
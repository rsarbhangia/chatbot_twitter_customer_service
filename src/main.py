from fastapi import FastAPI, Depends, Request, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import pathlib
import uuid
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

from database import get_db, CustomerInteraction
from rag import RAGSystem
from chatbot import CustomerSupportChatbot
from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

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
        logger.info(f"Created new session ID: {session_id}")
    
    # Get or create chat session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession()
        logger.debug(f"Created new chat session for ID: {session_id}")
    
    data = await request.json()
    query = data.get("text", "")
    
    if not query:
        logger.warning("Empty query received")
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
        
        # Store in database
        db_interaction = CustomerInteraction(
            session_id=session_id,
            query=query,
            response=response,
            context_used=json.dumps(contexts),
            confidence_score=confidence_scores[0] if confidence_scores else 0
        )
        db.add(db_interaction)
        db.commit()
        
        logger.info(f"Successfully processed query for session {session_id}")
        logger.debug(f"Query: {query}, Response: {response}, Confidence: {confidence_scores}")
        
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
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
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
        if (current_time - session.last_activity) >= timedelta(hours=24)
    ]
    for sid in expired_sessions:
        del chat_sessions[sid]
        logger.info(f"Cleaned up expired session {sid}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    scheduler.add_job(cleanup_old_sessions, 'interval', hours=1)
    scheduler.start()
    yield
    # Shutdown
    scheduler.shutdown()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
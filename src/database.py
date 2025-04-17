from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

# Create database directory if it doesn't exist
DB_DIR = Path("database")
DB_DIR.mkdir(exist_ok=True)

# Update database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///database/customer_support.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class CustomerInteraction(Base):
    __tablename__ = "customer_interactions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    query = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_used = Column(Text)
    confidence_score = Column(Integer)

# Check if tables exist before creating them
def init_db():
    inspector = inspect(engine)
    if not inspector.has_table("customer_interactions"):
        Base.metadata.create_all(bind=engine)
        print("Created customer_interactions table")
    else:
        print("Table customer_interactions already exists")

# Initialize database
init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 
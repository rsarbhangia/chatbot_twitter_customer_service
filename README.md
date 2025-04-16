# AI-Powered Customer Support Assistant

This is an end-to-end AI-powered customer support assistant that uses Retrieval-Augmented Generation (RAG) to provide intelligent responses to customer queries. The system uses a dataset of customer support tweets and provides a FastAPI-based API for interaction.

## Features

- Retrieval-Augmented Generation (RAG) for context-aware responses
- FastAPI-based REST API
- SQLite database for storing interaction history
- Confidence scoring for responses
- Easy-to-use API endpoints

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-support-ai
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Submit a Query
- **Endpoint**: `/query`
- **Method**: POST
- **Body**:
```json
{
    "text": "Your customer support question here"
}
```
- **Response**:
```json
{
    "response": "AI-generated response",
    "contexts": ["relevant context 1", "relevant context 2"],
    "confidence_score": 85
}
```

### 2. Get Recent Interactions
- **Endpoint**: `/interactions`
- **Method**: GET
- **Query Parameters**:
  - `limit`: Number of interactions to retrieve (default: 10)
- **Response**: List of recent interactions with their details

### 3. Root Endpoint
- **Endpoint**: `/`
- **Method**: GET
- **Response**: API information and available endpoints

## Architecture

The system consists of three main components:

1. **RAG System** (`src/rag.py`):
   - Handles the retrieval and generation of responses
   - Uses sentence transformers for embeddings
   - Implements FAISS for efficient similarity search

2. **Database** (`src/database.py`):
   - SQLite database using SQLAlchemy
   - Stores customer interactions and their metadata

3. **API** (`src/main.py`):
   - FastAPI application
   - Handles HTTP requests
   - Integrates RAG system with database

## Contributing

Feel free to submit issues and enhancement requests! 
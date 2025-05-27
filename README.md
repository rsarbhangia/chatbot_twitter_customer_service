# Customer Support AI Chatbot

A sophisticated customer support chatbot powered by FastAPI and RAG (Retrieval-Augmented Generation) technology. This application provides intelligent customer service responses by leveraging historical interactions and context-aware responses.

## 🚀 Features

- **RAG-Powered Responses**: Utilizes Retrieval-Augmented Generation for contextually relevant answers
- **Session Management**: Maintains chat history and session state
- **Response Evaluation**: Built-in evaluation system for response quality
- **Database Integration**: Stores interactions for analysis and improvement
- **Configurable Search Methods**: Supports multiple search algorithms (KNN by default)
- **Reranking Capabilities**: Optional response reranking for improved accuracy
- **Health Monitoring**: Built-in health check endpoints
- **Session Cleanup**: Automatic cleanup of inactive sessions
- **CORS Support**: Configured for secure cross-origin requests

## 🏗️ Project Structure

```
.
├── src/
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # Database models and connection
│   ├── rag.py              # RAG system implementation
│   ├── chatbot.py          # Chatbot logic
│   ├── evaluator.py        # Response evaluation system
│   └── logger_config.py    # Logging configuration
├── static/                 # Static files (CSS, JS, images)
├── templates/             # HTML templates
├── cache/                 # Cache directory for RAG system
├── database/             # Database files
├── tmp/                  # Temporary files
├── Dockerfile           # Container configuration
└── requirements.txt     # Python dependencies
```

## 🛠️ Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- PostgreSQL (recommended for production)

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```env
PORT=8000
SEARCH_METHOD=knn
RERANK_METHOD=None
STORAGE_TYPE=local
OPENAI_API_KEY=your_api_key_here
```

## 🚀 Getting Started

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
cd src
python main.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t customer-support-chatbot .
```

2. Run the container:
```bash
docker run -p 8000:8000 customer-support-chatbot
```

## 📡 API Endpoints

- `GET /`: Main application interface
- `POST /query`: Process customer queries
- `GET /interactions`: Retrieve interaction history
- `GET /health`: Health check endpoint

## 🔍 RAG System Configuration

The application supports different search and reranking methods:

- **Search Methods**: KNN (default), configurable via `SEARCH_METHOD`
- **Reranking**: Optional, configurable via `RERANK_METHOD`
- **Storage**: Local storage by default, configurable via `STORAGE_TYPE`

## 📊 Database Schema

The application uses SQLAlchemy ORM with the following main model:

- `CustomerInteraction`: Stores query, response, context, and evaluation data

## 🔐 Security

- CORS is configured for specific origins
- Session management with secure cookies
- Environment variable based configuration
- Automatic session cleanup after 24 hours of inactivity

## 📈 Monitoring

- Built-in logging system
- Health check endpoint
- Response evaluation metrics
- Confidence scoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

[Add your license information here]

## 👥 Authors

[Add author information here]

## 🙏 Acknowledgments

- FastAPI framework
- SQLAlchemy ORM
- OpenAI API
- APScheduler for background tasks 

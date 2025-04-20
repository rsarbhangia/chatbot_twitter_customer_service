from typing import List, Tuple
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from rag import RAGSystem
from logger_config import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

class CustomerSupportChatbot:
    def __init__(self, rag_system: RAGSystem = None):
        """
        Initialize the chatbot with an optional RAG system
        
        Args:
            rag_system: An instance of RAGSystem to provide context for responses
        """
        # Initialize the RAG system if not provided
        self.rag_system = rag_system or RAGSystem()
        logger.info("CustomerSupportChatbot initialized")
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer support assistant. You're provided with customer support tweets(Question) and the response from company to those tweets(Answer) as context.
            
            Use the following context to answer the customer's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Previous conversation:
            {chat_history}
            """),
            ("human", "{question}")
        ])
        
        # Create the chain
        self.chain = (
            {"context": self._get_context, "chat_history": self._get_chat_history, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Initialize chat history
        self.chat_history = []
    
    def _get_context(self, query: str) -> str:
        """
        Get relevant context from the RAG system
        
        Args:
            query: The user's question
            
        Returns:
            A string containing relevant context
        """
        # Get relevant contexts from the RAG system
        retrieved = self.rag_system.retrieve(query)

        if not retrieved:
            return "No relevant context found."
        
        # Format the context
        context_parts = []
        for i, result in enumerate(retrieved):
            # Convert distance to a similarity score (lower distance = higher similarity)
            similarity = 1.0 - min(result['distance'] / 5.0, 1.0)  # Normalize to 0-1 range
            similarity_percent = int(similarity * 100)
            
            context_parts.append(f"Context {i+1} (Relevance: {similarity_percent}%):")
            context_parts.append(f"Question: {result['input_text']}")
            context_parts.append(f"Answer: {result['output_text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_chat_history(self, question: str) -> str:
        """
        Get the chat history formatted as a string
        
        Args:
            question: The current question (not used, but required by the chain)
            
        Returns:
            A string containing the formatted chat history
        """
        if not self.chat_history:
            return "No previous conversation."
        
        history_parts = []
        for i, (q, a) in enumerate(self.chat_history[-5:]):  # Only include last 5 exchanges
            history_parts.append(f"Customer: {q}")
            history_parts.append(f"Assistant: {a}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def process_query(self, query: str) -> Tuple[str, List[str], List[float]]:
        """
        Process a user query and return a response with metadata
        
        Args:
            query: The user's question
            
        Returns:
            A tuple containing (response, contexts, confidence_scores)
        """
        try:
            logger.debug(f"Processing query: {query}")
            
            # Get the response from the LLM
            response = self.chain.invoke(query)
            logger.debug(f"LLM Chain Response: {response}")
            
            # Get the contexts and confidence scores from the RAG system
            retrieved = self.rag_system.retrieve(query)
            
            contexts = []
            confidence_scores = []
            
            for result in retrieved:
                contexts.append(result['input_text'])
                # Convert distance to confidence score (lower distance = higher confidence)
                max_distance = 5.0
                normalized_distance = min(result['distance'], max_distance) / max_distance
                confidence = int((1 - normalized_distance) * 100)
                confidence_scores.append(confidence)
            
            # Update chat history
            self.chat_history.append((query, response))
            
            logger.debug(f"Retrieved {len(contexts)} relevant contexts")
            logger.debug(f"Generated response: {response}")
            
            return response, contexts, confidence_scores
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            raise
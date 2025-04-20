from typing import List, Tuple, Optional
import os
import json
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
    def __init__(self, rag_system: RAGSystem = None, search_method: str = "knn", rerank_method: Optional[str] = None):
        """
        Initialize the chatbot with an optional RAG system
        
        Args:
            rag_system: An instance of RAGSystem to provide context for responses
            search_method: The search method to use in RAGSystem. Options are:
                - "knn": Exact K-Nearest Neighbors search (default)
                - "ann": Approximate Nearest Neighbors search
            rerank_method: The reranking method to use. Options are:
                - None: No reranking (default)
                - "cross_encoder": Use Cross-Encoder for reranking
                - "context_aware": Use context-aware reranking
                - "both": Use both Cross-Encoder and context-aware reranking
        """
        # Initialize the RAG system if not provided
        self.rag_system = rag_system or RAGSystem(search_method=search_method, rerank_method=rerank_method)
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
            
            When responding, follow these steps:
            1. Analyze the customer's question and identify the key issues or concerns
            2. Review the provided context and identify relevant information
            3. Consider the conversation history for any relevant context
            4. Formulate a clear, helpful response
            5. Explain your reasoning process in detail
            
            Your response should be in JSON format with the following structure:
            {{
                "response": "Your direct answer to the customer",
                "reasoning": {{
                    "question_analysis": "Your analysis of what the customer is asking",
                    "context_relevance": "How the provided context relates to the question",
                    "history_consideration": "How previous conversation context influenced your response",
                    "response_formation": "How you combined information to form the response",
                    "confidence_explanation": "Why you believe this response is appropriate"
                }}
            }}
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
        for i, (q, a, _) in enumerate(self.chat_history[-5:]):  # Only include last 5 exchanges
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
            llm_response = self.chain.invoke(query)
            logger.debug(f"LLM Chain Response: {llm_response}")
            
            # Parse the JSON response
            try:
                parsed_response = json.loads(llm_response)
                response = parsed_response.get("response", "I couldn't generate a proper response.")
                reasoning = parsed_response.get("reasoning", {})
                
                # Log the reasoning for debugging
                logger.debug(f"Response reasoning: {json.dumps(reasoning, indent=2)}")
                
                # Store the reasoning in the chat history for future reference
                self.chat_history.append((query, response, reasoning))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {llm_response}")
                response = llm_response  # Use the raw response if JSON parsing fails
                reasoning = {}
                self.chat_history.append((query, response, reasoning))
            
            # Get the contexts and confidence scores from the RAG system
            retrieved = self.rag_system.retrieve(query)
            
            contexts = []
            confidence_scores = []
            
            for result in retrieved:
                contexts.append("User Query: " + result['input_text'] + "\nSupport Response: " + result['output_text'])
                # Convert distance to confidence score (lower distance = higher confidence)
                max_distance = 5.0
                normalized_distance = min(result['distance'], max_distance) / max_distance
                confidence = int((1 - normalized_distance) * 100)
                confidence_scores.append(confidence)
            
            # Update chat history
            self.chat_history.append((query, response, reasoning))
            
            logger.debug(f"Retrieved {len(contexts)} relevant contexts")
            logger.debug(f"Generated response: {response}")
            
            return response, reasoning, contexts, confidence_scores
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            raise
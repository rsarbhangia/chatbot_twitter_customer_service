from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import pickle
from pathlib import Path
from logger_config import setup_logger

logger = setup_logger(__name__)

class RAGSystem:
    def __init__(self, search_method: str = "knn", rerank_method: Optional[str] = None):
        """
        Initialize RAG System
        
        Args:
            search_method: The search method to use. Options are:
                - "knn": Exact K-Nearest Neighbors search (default)
                - "ann": Approximate Nearest Neighbors search
            rerank_method: The reranking method to use. Options are:
                - None: No reranking (default)
                - "cross_encoder": Use Cross-Encoder for reranking
                - "context_aware": Use context-aware reranking
                - "both": Use both Cross-Encoder and context-aware reranking
        """
        logger.info("Initializing RAG System")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.input_texts = []
        self.output_texts = []
        self.combined_texts = []  # Store combined input-output pairs
        self.chat_history = []  # Store chat history as (query, response) pairs
        
        # Validate search method
        if search_method not in ["knn", "ann"]:
            raise ValueError("search_method must be either 'knn' or 'ann'")
        self.search_method = search_method
        
        # Validate rerank method
        if rerank_method not in [None, "cross_encoder", "context_aware", "both"]:
            raise ValueError("rerank_method must be None, 'cross_encoder', 'context_aware', or 'both'")
        self.rerank_method = rerank_method
        
        # Initialize cross-encoder if needed
        if rerank_method in ["cross_encoder", "both"]:
            logger.info("Initializing Cross-Encoder model for reranking")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Define cache paths
        self.cache_dir = Path("../cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_cache_path = self.cache_dir / "customer_support_dataset.pkl"
        self.embeddings_cache_path = self.cache_dir / "embeddings.npy"
        self.index_cache_path = self.cache_dir / "faiss_index.bin"
        
        self._prepare_index()
        logger.debug("RAG System initialized successfully")

    def _prepare_index(self):
        """Prepare the FAISS index with the customer support tweets dataset"""
        logger.info("Preparing the index with customer support tweets dataset...")
        
        # Check if we have cached data
        if self._load_cached_data():
            logger.info("Using cached dataset and embeddings.")
            return
            
        try:
            # Load the dataset using the Hugging Face datasets API
            logger.info("Loading dataset from Hugging Face...")

            df = pd.read_json("hf://datasets/MohammadOthman/mo-customer-support-tweets-945k/preprocessed_data.json")
            logger.info(f"Dataset loaded and converted to DataFrame with {len(df)} rows")
            
            # Extract text from the DataFrame
            # Check what columns are available
            logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Try to find input and output columns
            input_column = None
            output_column = None
            
            input_column, output_column = "input", "output"
            
            if input_column and output_column:
                logger.info(f"Using '{input_column}' as input column and '{output_column}' as output column")
                # Store both input and output texts
                self.input_texts = df[input_column].tolist()[:100000]  # Using first 10k examples
                self.output_texts = df[output_column].tolist()[:100000]
                
                # Create combined input-output pairs for better context
                self.combined_texts = [
                    f"Customer: {input_text}\nResponse: {output_text}" 
                    for input_text, output_text in zip(self.input_texts, self.output_texts)
                ]
            else:
                raise ValueError("Both input and output columns must exist in the dataset")
            
            logger.info(f"Loaded {len(self.combined_texts)} conversation pairs from the dataset")
            
            if len(self.combined_texts) == 0:
                raise ValueError("No conversation pairs loaded from dataset")
            
            # Cache the dataset
            self._cache_dataset()
            
            # Create embeddings for combined input-output pairs
            logger.info("Creating embeddings for combined input-output pairs...")
            embeddings = self.model.encode(self.combined_texts, show_progress_bar=True)
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            # Cache the embeddings
            self._cache_embeddings(embeddings)
            
            # Create FAISS index with combined texts
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            
            if self.search_method == "knn":
                logger.info("Using KNN (exact) search with IndexFlatL2")
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(np.array(embeddings).astype('float32'))
            else:  # ann
                logger.info("Using ANN (approximate) search with IndexIVFFlat")
                # Create a quantizer for the index
                quantizer = faiss.IndexFlatL2(dimension)
                
                # Create the IVF index with 100 clusters
                nlist = 100  # number of clusters/centroids
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                
                # Train the index
                logger.info("Training the IVF index...")
                self.index.train(np.array(embeddings).astype('float32'))
                
                # Add vectors to the index
                self.index.add(np.array(embeddings).astype('float32'))
                
                # Set search parameters
                self.index.nprobe = 10  # number of clusters to visit during search
            
            # Cache the FAISS index
            self._cache_index()
            
            logger.info("Customer support tweets index preparation completed!")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.info("Please download the dataset.")
            
            raise ValueError("Failed to prepare index. Please ensure the dataset is accessible and properly formatted.")

    def _cache_dataset(self):
        """Cache the dataset to disk"""
        logger.info(f"Caching dataset to {self.dataset_cache_path}")
        with open(self.dataset_cache_path, 'wb') as f:
            pickle.dump({
                'input_texts': self.input_texts,
                'output_texts': self.output_texts,
                'combined_texts': self.combined_texts
            }, f)
        logger.info("Dataset cached successfully")

    def _cache_embeddings(self, embeddings):
        """Cache the embeddings to disk"""
        logger.info(f"Caching embeddings to {self.embeddings_cache_path}")
        np.save(self.embeddings_cache_path, embeddings)
        logger.info("Embeddings cached successfully")

    def _cache_index(self):
        """Cache the FAISS index to disk"""
        logger.info(f"Caching FAISS index to {self.index_cache_path}")
        faiss.write_index(self.index, str(self.index_cache_path))
        logger.info("FAISS index cached successfully")

    def _load_cached_data(self):
        """Load cached data if available"""
        # Check if all cache files exist
        if not (self.dataset_cache_path.exists() and 
                self.embeddings_cache_path.exists() and 
                self.index_cache_path.exists()):
            logger.info("Cache files not found. Will download and process data.")
            return False
        
        try:
            # Load dataset
            logger.info(f"Loading cached dataset from {self.dataset_cache_path}")
            with open(self.dataset_cache_path, 'rb') as f:
                data = pickle.load(f)
                self.input_texts = data['input_texts']
                self.output_texts = data['output_texts']
                self.combined_texts = data['combined_texts']
            
            # Load embeddings
            logger.info(f"Loading cached embeddings from {self.embeddings_cache_path}")
            embeddings = np.load(self.embeddings_cache_path)
            
            # Load FAISS index
            logger.info(f"Loading cached FAISS index from {self.index_cache_path}")
            self.index = faiss.read_index(str(self.index_cache_path))
            
            logger.info("Successfully loaded all cached data")
            return True
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}")
            logger.info("Will download and process data instead.")
            return False

    def _rerank_with_cross_encoder(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using Cross-Encoder model"""
        if not results:
            return results
            
        # Prepare pairs for cross-encoder
        pairs = [(query, result['input_text']) for result in results]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to results
        for result, score in zip(results, cross_scores):
            result['cross_score'] = float(score)
        
        # Sort by cross-encoder score (higher is better)
        results.sort(key=lambda x: x['cross_score'], reverse=True)
        
        return results

    def _rerank_with_context(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using context awareness"""
        if not results or not self.chat_history:
            return results
            
        # Get recent chat history
        recent_history = self.chat_history[-3:]  # Last 3 exchanges
        
        # Calculate context relevance scores
        for result in results:
            # Initialize context score
            context_score = 0.0
            
            # Check relevance to recent history
            for hist_query, hist_response in recent_history:
                # Calculate similarity with historical query
                query_similarity = self.model.encode([hist_query, result['input_text']])
                query_sim_score = np.dot(query_similarity[0], query_similarity[1]) / (
                    np.linalg.norm(query_similarity[0]) * np.linalg.norm(query_similarity[1])
                )
                
                # Calculate similarity with historical response
                response_similarity = self.model.encode([hist_response, result['output_text']])
                response_sim_score = np.dot(response_similarity[0], response_similarity[1]) / (
                    np.linalg.norm(response_similarity[0]) * np.linalg.norm(response_similarity[1])
                )
                
                # Combine scores (you can adjust weights)
                context_score += 0.6 * query_sim_score + 0.4 * response_sim_score
            
            # Normalize context score
            context_score /= len(recent_history)
            result['context_score'] = float(context_score)
        
        # Sort by context score (higher is better)
        results.sort(key=lambda x: x['context_score'], reverse=True)
        
        return results

    def _combine_reranking_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine scores from different reranking methods"""
        for result in results:
            # Initialize combined score
            combined_score = 0.0
            weights = []
            
            # Add cross-encoder score if available
            if 'cross_score' in result:
                combined_score += result['cross_score']
                weights.append(0.6)  # Weight for cross-encoder
            
            # Add context score if available
            if 'context_score' in result:
                combined_score += result['context_score']
                weights.append(0.4)  # Weight for context
            
            # Normalize by weights
            if weights:
                result['combined_score'] = combined_score / sum(weights)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant contexts using combined input-output pairs"""
        logger.debug(f"Retrieving contexts for query: {query}")
        
        # If we have chat history, enhance the query with recent context
        enhanced_query = self._enhance_query_with_history(query)
        logger.debug(f"Enhanced query: {enhanced_query}")
        
        # Get query vector
        query_vector = self.model.encode([enhanced_query])
        logger.debug(f"Query vector shape: {query_vector.shape}")
        
        if self.index is None:
            logger.warning("Warning: Index is None, returning empty results")
            return []
        
        # Retrieve more candidates for reranking if needed
        initial_k = k * 3 if self.rerank_method else k
        
        # Retrieve using combined input-output pairs
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), initial_k
        )
        logger.debug(f"Distances: {distances}")
        logger.debug(f"Indices: {indices}")
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.combined_texts):  # Ensure index is valid
                # Return combined text, distance, and corresponding input/output texts
                results.append({
                    'combined_text': self.combined_texts[idx],
                    'input_text': self.input_texts[idx],
                    'output_text': self.output_texts[idx],
                    'distance': float(distance)
                })
        
        # Apply reranking if specified
        if self.rerank_method in ["cross_encoder", "both"]:
            logger.debug("Applying Cross-Encoder reranking")
            results = self._rerank_with_cross_encoder(query, results)
            
        if self.rerank_method in ["context_aware", "both"]:
            logger.debug("Applying context-aware reranking")
            results = self._rerank_with_context(query, results)
            
        if self.rerank_method == "both":
            logger.debug("Combining reranking scores")
            results = self._combine_reranking_scores(results)
        
        # Return top k results
        return results[:k]
    
    def _enhance_query_with_history(self, query: str, max_history: int = 3) -> str:
        """Enhance the current query with context from chat history"""
        if not self.chat_history:
            return query
        
        # Get the most recent conversations (up to max_history)
        recent_history = self.chat_history[-max_history:]
        
        # Build a context string from recent history
        context = ""
        for i, (hist_query, hist_response) in enumerate(recent_history):
            context += f"Previous question {i+1}: {hist_query}\n"
            context += f"Previous answer {i+1}: {hist_response}\n\n"
        
        # Combine context with current query
        enhanced_query = f"{context}Current question: {query}"
        
        return enhanced_query

    def generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a response using the retrieved contexts"""
        # If no contexts are found, return a clear message
        if not results:
            return "I don't have enough relevant information to answer your question. Please try rephrasing or contact human support."
        
        # For a more complete response, combine information from multiple contexts
        # In a production environment, you would use a proper LLM here
        
        # Start with a clear introduction
        response = "Here's what I found based on similar customer inquiries:\n\n"
        
        # Process the most relevant context first
        main_result = results[0]
        main_input = main_result['input_text']
        main_output = main_result['output_text']
        
        # Clean up the main input and output
        main_input = ' '.join(main_input.split())
        main_output = ' '.join(main_output.split())
        
        # Add the main output as the primary response
        response += main_output
        
        # Add additional information from other contexts if available
        if len(results) > 1:
            response += "\n\nAdditional details from similar cases:\n"
            
            # Process each additional context
            for i, result in enumerate(results[1:], 1):
                # Clean up the input and output
                clean_input = ' '.join(result['input_text'].split())
                clean_output = ' '.join(result['output_text'].split())
                
                # Only add if it provides new information
                if clean_output not in response:
                    # Format as a bullet point for readability
                    response += f"\n• {clean_output}"
        
        # Add a note about the Similar Conversations tab for reference
        response += "\n\n(You can view the original conversations in the Similar Conversations tab)"
        
        return response

    def process_query(self, query: str) -> Tuple[str, List[str], List[float]]:
        """Process a query and return response with metadata"""
        # Retrieve relevant contexts
        retrieved = self.retrieve(query)
        
        # Extract contexts, outputs, and calculate individual confidence scores
        contexts = []
        outputs = []
        confidence_scores = []
        
        for result in retrieved:
            contexts.append(result['input_text'])
            outputs.append(result['output_text'])
            # Convert distance to confidence score (higher distance = lower confidence)
            # Using a more realistic formula that better reflects semantic similarity
            # Normalize distance to a 0-1 range and then convert to percentage
            # Typical FAISS L2 distances for similar texts are often in the range of 0.5-2.0
            # We'll use a scaling factor to make the scores more realistic
            max_distance = 5.0  # Consider distances above this as very low confidence
            normalized_distance = min(result['distance'], max_distance) / max_distance
            confidence = int((1 - normalized_distance) * 100)
            # Ensure confidence is between 0 and 100
            confidence = max(0, min(100, confidence))
            confidence_scores.append(confidence)
        
        # If no contexts or all confidence scores are too low, return a clear message
        if not contexts or all(score < 30 for score in confidence_scores):
            response = "I don't have enough relevant information to answer your question. Please try rephrasing or contact human support."
        else:
            # Generate response using the retrieved contexts
            response = self.generate_response(query, retrieved)
        
        # Update chat history with the current query and response
        self.chat_history.append((query, response))
        
        return response, contexts, confidence_scores 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple
import pickle
from pathlib import Path

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
        self.input_texts = []
        self.output_texts = []
        self.chat_history = []  # Store chat history as (query, response) pairs
        
        # Define cache paths
        self.cache_dir = Path("../cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_cache_path = self.cache_dir / "customer_support_dataset.pkl"
        self.embeddings_cache_path = self.cache_dir / "embeddings.npy"
        self.index_cache_path = self.cache_dir / "faiss_index.bin"
        
        self._prepare_index()

    def _prepare_index(self):
        """Prepare the FAISS index with the customer support tweets dataset"""
        print("Preparing the index with customer support tweets dataset...")
        
        # Check if we have cached data
        if self._load_cached_data():
            print("Using cached dataset and embeddings.")
            return
            
        try:
            # Load the dataset using the Hugging Face datasets API
            print("Loading dataset from Hugging Face...")

            df = pd.read_json("hf://datasets/MohammadOthman/mo-customer-support-tweets-945k/preprocessed_data.json")
            print(f"Dataset loaded and converted to DataFrame with {len(df)} rows")
            
            # Extract text from the DataFrame
            # Check what columns are available
            print(f"Available columns: {df.columns.tolist()}")
            
            # Try to find input and output columns
            input_column = None
            output_column = None
            
            # Check for common input column names
            for col in ['input', 'query', 'question', 'text', 'tweet', 'message']:
                if col in df.columns:
                    input_column = col
                    break
            
            # Check for common output column names
            for col in ['output', 'response', 'answer', 'reply', 'label']:
                if col in df.columns:
                    output_column = col
                    break
            
            if input_column and output_column:
                print(f"Using '{input_column}' as input column and '{output_column}' as output column")
                # Store both input and output texts
                self.input_texts = df[input_column].tolist()[:10000]  # Using first 10k examples
                self.output_texts = df[output_column].tolist()[:10000]
                # Use input texts for indexing
                self.texts = self.input_texts
            elif input_column:
                print(f"Only found input column '{input_column}', using it for both input and output")
                self.input_texts = df[input_column].tolist()[:10000]
                self.output_texts = self.input_texts
                self.texts = self.input_texts
            else:
                # If no input column found, use the first column
                print(f"No input column found, using first column: {df.columns[0]}")
                self.input_texts = df[df.columns[0]].tolist()[:10000]
                self.output_texts = self.input_texts
                self.texts = self.input_texts
            
            print(f"Loaded {len(self.texts)} documents from the dataset")
            
            if len(self.texts) == 0:
                raise ValueError("No texts loaded from dataset")
            
            # Cache the dataset
            self._cache_dataset()
            
            # Create embeddings for the dataset
            print("Creating embeddings...")
            embeddings = self.model.encode(self.texts, show_progress_bar=True)
            print(f"Embeddings shape: {embeddings.shape}")
            
            # Cache the embeddings
            self._cache_embeddings(embeddings)
            
            # Create FAISS index with the dataset
            dimension = embeddings.shape[1]
            print(f"Creating FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Cache the FAISS index
            self._cache_index()
            
            print("Customer support tweets index preparation completed!")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Please download the dataset.")
            
            raise ValueError("Failed to prepare index. Please ensure the dataset is accessible and properly formatted.")

    def _cache_dataset(self):
        """Cache the dataset to disk"""
        print(f"Caching dataset to {self.dataset_cache_path}")
        with open(self.dataset_cache_path, 'wb') as f:
            pickle.dump({
                'input_texts': self.input_texts,
                'output_texts': self.output_texts,
                'texts': self.texts
            }, f)
        print("Dataset cached successfully")

    def _cache_embeddings(self, embeddings):
        """Cache the embeddings to disk"""
        print(f"Caching embeddings to {self.embeddings_cache_path}")
        np.save(self.embeddings_cache_path, embeddings)
        print("Embeddings cached successfully")

    def _cache_index(self):
        """Cache the FAISS index to disk"""
        print(f"Caching FAISS index to {self.index_cache_path}")
        faiss.write_index(self.index, str(self.index_cache_path))
        print("FAISS index cached successfully")

    def _load_cached_data(self):
        """Load cached data if available"""
        # Check if all cache files exist
        if not (self.dataset_cache_path.exists() and 
                self.embeddings_cache_path.exists() and 
                self.index_cache_path.exists()):
            print("Cache files not found. Will download and process data.")
            return False
        
        try:
            # Load dataset
            print(f"Loading cached dataset from {self.dataset_cache_path}")
            with open(self.dataset_cache_path, 'rb') as f:
                data = pickle.load(f)
                self.input_texts = data['input_texts']
                self.output_texts = data['output_texts']
                self.texts = data['texts']
            
            # Load embeddings
            print(f"Loading cached embeddings from {self.embeddings_cache_path}")
            embeddings = np.load(self.embeddings_cache_path)
            
            # Load FAISS index
            print(f"Loading cached FAISS index from {self.index_cache_path}")
            self.index = faiss.read_index(str(self.index_cache_path))
            
            print("Successfully loaded all cached data")
            return True
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
            print("Will download and process data instead.")
            return False

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
        """Retrieve relevant contexts for the query"""
        print(f"Retrieving contexts for query: {query}")
        
        # If we have chat history, enhance the query with recent context
        enhanced_query = self._enhance_query_with_history(query)
        print(f"Enhanced query: {enhanced_query}")
        
        query_vector = self.model.encode([enhanced_query])
        print(f"Query vector shape: {query_vector.shape}")
        
        if self.index is None:
            print("Warning: Index is None, returning empty results")
            return []
            
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.texts):  # Ensure index is valid
                # Return input text, distance, and corresponding output text
                results.append((self.input_texts[idx], float(distance), self.output_texts[idx]))
        
        print(f"Retrieved {len(results)} results")
        return results
    
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

    def generate_response(self, query: str, contexts: List[str], outputs: List[str]) -> str:
        """Generate a response using the retrieved contexts and outputs"""
        # If no contexts are found or confidence is too low, return a clear message
        if not contexts:
            return "I don't have enough relevant information to answer your question. Please try rephrasing or contact human support."
        
        # For a more complete response, combine information from multiple contexts and outputs
        # In a production environment, you would use a proper LLM here
        
        # Start with a clear introduction
        response = "Here's what I found based on the available information:\n\n"
        
        # Process the most relevant context and output first
        main_context = contexts[0]
        main_output = outputs[0]
        
        # Clean up the main context and output
        main_context = ' '.join(main_context.split())
        main_output = ' '.join(main_output.split())
        
        # Add the main output as the primary response
        response += main_output
        
        # Add additional information from other contexts and outputs if available
        if len(contexts) > 1:
            response += "\n\nAdditional details:\n"
            
            # Process each additional context and output
            for i, (context, output) in enumerate(zip(contexts[1:], outputs[1:]), 1):
                # Clean up the context and output
                clean_context = ' '.join(context.split())
                clean_output = ' '.join(output.split())
                
                # Only add if it provides new information
                if clean_output not in response:
                    # Format as a bullet point for readability
                    response += f"\n• {clean_output}"
        
        # No longer truncating the response to ensure all information is included
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
        
        for ctx, distance, output in retrieved:
            contexts.append(ctx)
            outputs.append(output)
            # Convert distance to confidence score (higher distance = lower confidence)
            # The previous formula was causing very high confidence scores
            # Using a more realistic formula that better reflects semantic similarity
            # Normalize distance to a 0-1 range and then convert to percentage
            # Typical FAISS L2 distances for similar texts are often in the range of 0.5-2.0
            # We'll use a scaling factor to make the scores more realistic
            max_distance = 5.0  # Consider distances above this as very low confidence
            normalized_distance = min(distance, max_distance) / max_distance
            confidence = int((1 - normalized_distance) * 100)
            # Ensure confidence is between 0 and 100
            confidence = max(0, min(100, confidence))
            confidence_scores.append(confidence)
        
        # If no contexts or all confidence scores are too low, return a clear message
        if not contexts or all(score < 30 for score in confidence_scores):
            response = "I don't have enough relevant information to answer your question. Please try rephrasing or contact human support."
        else:
            # Generate response using both contexts and outputs
            response = self.generate_response(query, contexts, outputs)
        
        # Update chat history with the current query and response
        self.chat_history.append((query, response))
        
        return response, contexts, confidence_scores 
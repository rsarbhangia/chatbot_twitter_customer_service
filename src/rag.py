from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
import torch
import os
from pathlib import Path
import pandas as pd

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
        self.input_texts = []
        self.output_texts = []
        self.chat_history = []  # Store chat history as (query, response) pairs
        self._prepare_index()

    def _prepare_index(self):
        """Prepare the FAISS index with the customer support tweets dataset"""
        print("Preparing the index with customer support tweets dataset...")
        
        try:
            # Load the dataset using pandas
            print("Loading dataset with pandas...")
            
            # Try to load the dataset from the Hugging Face datasets
            try:
                df = pd.read_json("hf://datasets/MohammadOthman/mo-customer-support-tweets-945k/preprocessed_data.json")
                print(f"Dataset loaded successfully with {len(df)} rows")
            except Exception as e:
                print(f"Error loading from Hugging Face: {str(e)}")
                print("Trying to download the dataset first...")
                
                # Try to download the dataset first
                dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
                
                # Convert to pandas DataFrame
                if 'train' in dataset:
                    df = dataset['train'].to_pandas()
                else:
                    # If no 'train' split, use the first available split
                    first_split = list(dataset.keys())[0]
                    df = dataset[first_split].to_pandas()
                
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
            
            # Create embeddings for the dataset
            print("Creating embeddings...")
            embeddings = self.model.encode(self.texts, show_progress_bar=True)
            print(f"Embeddings shape: {embeddings.shape}")
            
            # Create FAISS index with the dataset
            dimension = embeddings.shape[1]
            print(f"Creating FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            print("Customer support tweets index preparation completed!")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Using sample data instead...")
            
            # Use a small sample dataset as fallback
            self.input_texts = [
                "How do I reset my password?",
                "I can't access my account",
                "The website is not loading properly",
                "How do I contact customer support?",
                "My order hasn't arrived yet",
                "I need to update my shipping address",
                "Can I get a refund for my recent purchase?",
                "The product I received is damaged",
                "How do I track my order?",
                "I forgot my username",
                "The app keeps crashing",
                "How do I change my email preferences?",
                "I want to cancel my subscription",
                "The payment was declined",
                "How do I download my invoice?",
                "I need to change my delivery date",
                "The product is out of stock",
                "How do I apply a discount code?",
                "I'm having trouble with the checkout process",
                "Can you help me find a specific item?"
            ]
            
            # Create sample outputs
            self.output_texts = [
                "To reset your password, click on the 'Forgot Password' link on the login page and follow the instructions sent to your email.",
                "If you can't access your account, try clearing your browser cache or using a different browser. If the issue persists, contact our support team.",
                "Please try refreshing the page or clearing your browser cache. If the problem continues, it might be a temporary server issue.",
                "You can contact our customer support team by email at support@example.com or by phone at 1-800-123-4567.",
                "We apologize for the delay. Please check your order tracking number in your account. If it's been more than 5 business days, contact our support team.",
                "You can update your shipping address in your account settings under 'Shipping Information' or contact our support team for assistance.",
                "Refund requests can be submitted through your account under 'Order History'. Select the order and click 'Request Refund'.",
                "We're sorry to hear that. Please take photos of the damaged product and contact our support team for a replacement or refund.",
                "You can track your order by logging into your account and going to 'Order History', or by using the tracking number sent to your email.",
                "If you've forgotten your username, you can retrieve it by entering your email address on the login page and clicking 'Forgot Username'.",
                "Please try updating the app to the latest version. If the issue persists, try uninstalling and reinstalling the app.",
                "You can change your email preferences in your account settings under 'Communication Preferences'.",
                "To cancel your subscription, go to your account settings, select 'Subscriptions', and click 'Cancel Subscription'.",
                "The payment might have been declined due to insufficient funds or incorrect card information. Please verify your payment details and try again.",
                "You can download your invoice by logging into your account, going to 'Order History', selecting the order, and clicking 'Download Invoice'.",
                "You can change your delivery date by logging into your account, going to 'Order History', selecting the order, and clicking 'Change Delivery Date'.",
                "We apologize for the inconvenience. You can sign up for notifications when the product is back in stock by clicking the 'Notify Me' button.",
                "To apply a discount code, enter it in the 'Promo Code' field at checkout and click 'Apply'.",
                "If you're having trouble with checkout, try using a different browser or clearing your cache. If the issue persists, contact our support team.",
                "You can search for specific items using the search bar at the top of our website. You can also browse categories or use filters to narrow down results."
            ]
            
            self.texts = self.input_texts
            
            # Create embeddings for sample data
            print("Creating embeddings for sample data...")
            embeddings = self.model.encode(self.texts, show_progress_bar=True)
            print(f"Sample embeddings shape: {embeddings.shape}")
            
            # Create FAISS index with sample data
            dimension = embeddings.shape[1]
            print(f"Creating FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            print("Sample data index preparation completed!")

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
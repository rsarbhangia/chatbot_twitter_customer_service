from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import os
import numpy as np

class DataProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        # Create data directory if it doesn't exist
        os.makedirs("data/chroma_db", exist_ok=True)
        
        # Updated ChromaDB configuration
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="huggingface_doc",
            metadata={"description": "Hugging Face documentation embeddings"}
        )

    def load_and_process_dataset(self, batch_size=1000):
        """Load and process the Hugging Face documentation dataset"""
        print("Processing Hugging Face documentation dataset...")
        
        try:
            # Load the Hugging Face documentation dataset
            dataset = load_dataset("m-ric/huggingface_doc")
            
            # Extract text from the dataset
            # The structure might be different, so we'll check what's available
            if 'text' in dataset['train'].features:
                texts = dataset['train']['text'][:10000]  # Using first 10k examples
            elif 'content' in dataset['train'].features:
                texts = dataset['train']['content'][:10000]
            elif 'document' in dataset['train'].features:
                texts = dataset['train']['document'][:10000]
            else:
                # If we can't find a text field, use the first column
                first_column = list(dataset['train'].features.keys())[0]
                texts = dataset['train'][first_column][:10000]
            
            print(f"Loaded {len(texts)} documents from the dataset")
            
            # Generate embeddings for the dataset
            embeddings = self.embedding_model.encode(texts)
            
            # Add to ChromaDB with updated API
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=[f"doc_{i}" for i in range(len(texts))]
            )
            
            print(f"Processed {len(texts)} documents from Hugging Face documentation")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Using sample data instead...")
            
            # Use a small sample dataset as fallback
            sample_data = [
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
            
            # Generate embeddings for sample data
            embeddings = self.embedding_model.encode(sample_data)
            
            # Add to ChromaDB with updated API
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=sample_data,
                ids=[f"sample_{i}" for i in range(len(sample_data))]
            )
            
            print(f"Processed {len(sample_data)} sample documents")

    def query_similar(self, query_text, n_results=5):
        """Query similar support messages"""
        query_embedding = self.embedding_model.encode(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_and_process_dataset() 
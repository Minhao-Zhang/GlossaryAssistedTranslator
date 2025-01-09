"""
Glossary Retrieval-Augmented Generation (RAG) system for document retrieval.

This module provides functionality for managing and searching domain-specific terminology
in a glossary using a Retrieval-Augmented Generation (RAG) system.
"""

import logging
import chromadb
import pandas as pd
import os
import glob
import requests
from typing import List
import numpy as np

class EmbeddingFunction:
    """Base class for custom embedding functions"""
    def __init__(self, document_prefix: str = "", query_prefix: str = ""):
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        
    def __call__(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for texts with optional query prefix"""
        prefixed_texts = [
            (self.query_prefix if is_query else self.document_prefix) + text
            for text in texts
        ]
        return self._embed(prefixed_texts)
        
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Internal method to implement actual embedding logic"""
        raise NotImplementedError

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Embedding function using Ollama API"""
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
                 document_prefix: str = "", query_prefix: str = ""):
        super().__init__(document_prefix, query_prefix)
        self.model_name = model_name
        self.base_url = base_url
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            if response.status_code != 200:
                raise ValueError(f"Ollama API error: {response.text}")
            embeddings.append(response.json()["embedding"])
        return embeddings

class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Embedding function using HuggingFace models"""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu",
                 document_prefix: str = "", query_prefix: str = ""):
        super().__init__(document_prefix, query_prefix)
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()


class GlossaryRAG:
    """Glossary Retrieval-Augmented Generation (RAG) system for document retrieval."""

    def __init__(self, collection_name: str, embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_prefix: str = "Definition of ",
                 query_prefix: str = "Identify and retrieve definitions for the key terms related to the following sentence: "):
        """
        Initialize the GlossaryRAG system.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model_name: Name of the SentenceTransformer model
            embedding_prefix: Prefix to add to definitions when storing them
            query_prefix: Prefix to add to queries when searching
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_prefix = embedding_prefix
        self.query_prefix = query_prefix

        try:
            # Initialize embedding function based on model name
            if embedding_model_name.startswith("ollama:"):
                # Example: "ollama:nomic-embed-text"
                model = embedding_model_name.split(":")[1]
                self.embedding_fn = OllamaEmbeddingFunction(model_name=model)
            elif embedding_model_name.startswith("hf:"):
                # Example: "hf:nomic-ai/nomic-embed-text-v1.5"
                model = embedding_model_name.split(":")[1]
                self.embedding_fn = HuggingFaceEmbeddingFunction(model_name=model)
            else:
                raise ValueError(f"Unsupported embedding model format: {embedding_model_name}")
            
            # Initialize Chroma client and collection
            self.client = chromadb.PersistentClient()
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )
            
            # Test embedding function
            test_embedding = self.embedding_fn(["test"])[0]
            if not isinstance(test_embedding, list) or not all(isinstance(x, float) for x in test_embedding):
                raise ValueError("Embedding function returned invalid format")
            
            logging.info(f"Initialized GlossaryRAG system with collection '{collection_name}'")
        except Exception as e:
            logging.error(f"Failed to initialize GlossaryRAG system: {str(e)}")
            raise

    def insert(self, term: str, translation: str, definition: str, example: str) -> None:
        """
        Insert a single document into the vector store.

        Args:
            term: The glossary term
            translation: The term's translation
            definition: The definition text
            example: Example usage text
        """
        try:
            # Generate a unique ID for the document
            doc_id = f"{term}-{hash(definition)}"
            
            # Add document to collection with proper prefix
            self.collection.add(
                ids=[doc_id],
                documents=[f"{self.embedding_prefix}{definition}"],
                metadatas=[{
                    "term": term,
                    "translation": translation,
                    "definition": definition,
                    "example": example
                }],
                embeddings=self.embedding_fn([definition], is_query=False)
            )
            logging.debug(f"Inserted document for term: {term}")
        except Exception as e:
            logging.error(f"Failed to insert document: {str(e)}")
            raise

    def load_from_dir(self, dir_path: str = "data") -> None:
        """
        Load glossary data from CSV files in a directory.

        Args:
            dir_path: Path to directory containing glossary CSV files
        """
        all_files = glob.glob(os.path.join(dir_path, "*.csv"))

        for file in all_files:
            data = pd.read_csv(file)
            for _, row in data.iterrows():
                self.insert(row["Term"], row["Translation"], row["Definition"], row["Example"])

    def query(self, question: str, n_results: int = 3) -> pd.DataFrame:
        """
        Query the vector store for relevant documents.

        Args:
            question: The query text
            n_results: Number of results to return

        Returns:
            DataFrame containing matching terms, definitions, and examples
        """
        try:
            # Query the collection with proper prefix
            results = self.collection.query(
                query_embeddings=self.embedding_fn([question], is_query=True),
                n_results=n_results,
                include=["metadatas"]
            )
            
            # Convert results to DataFrame
            matches = []
            for metadata in results["metadatas"][0]:
                matches.append({
                    "Term": metadata["term"],
                    "Definition": metadata["definition"],
                    "Example": metadata["example"]
                })
                
            return pd.DataFrame(matches)
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize GlossaryRAG instance
        glossary_rag = GlossaryRAG(
            "default_collection", "nomic-ai/nomic-embed-text-v1.5")

        # Load data from directory
        glossary_rag.load_from_dir()

        # Interactive question-answering loop
        while True:
            print("=" * 80)
            question = input("Enter a question (or 'exit' to quit): ")
            if question.lower() == "exit":
                break

            results = glossary_rag.query(question)
            if not results.empty:
                print("\nMatching glossary terms:")
                print(results.to_string(index=False))
            else:
                print("No matching glossary terms found.")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")

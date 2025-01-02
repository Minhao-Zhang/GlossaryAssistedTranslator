"""
Glossary Retrieval-Augmented Generation (RAG) system for document retrieval.

This module provides functionality for managing and searching domain-specific terminology
in a glossary using a Retrieval-Augmented Generation (RAG) system.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd
import os
import glob


class GlossaryRAG:
    """Glossary Retrieval-Augmented Generation (RAG) system for document retrieval."""

    def __init__(self, collection_name: str, embedding_model_name: str,
                 embedding_prefix: str = "Definition of ",
                 query_prefix: str = "Find definitions that relates to the sentence: "):
        """
        Initialize the GlossaryRAG system.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model_name: Name of the HuggingFace embedding model
            embedding_prefix: Prefix to add to definitions when storing them
            query_prefix: Prefix to add to queries when searching
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_prefix = embedding_prefix
        self.query_prefix = query_prefix

        try:
            model_kwargs = {"trust_remote_code": True}
            encode_kwargs = {}

            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            self.vectorstore = Chroma(
                collection_name,
                embedding_function=self.embedding_model
            )
            logging.info(f"Initialized GlossaryRAG system with collection '{
                         collection_name}'")
        except Exception as e:
            logging.error(f"Failed to initialize GlossaryRAG system: {str(e)}")
            raise

    def insert(self, term: str, definition: str, example: str) -> None:
        """
        Insert a single document into the vector store.

        Args:
            term: The glossary term
            definition: The definition text
            example: Example usage text
        """
        try:
            doc = Document(
                page_content=f"{self.embedding_prefix}{definition}",
                metadata={
                    "term": term,
                    "definition": definition,
                    "example": example
                }
            )
            self.vectorstore.add_documents([doc])
            logging.debug(f"Inserted document for term: {term}")
        except Exception as e:
            logging.error(f"Failed to insert document: {str(e)}")
            raise

    def load_from_dir(self, dir_path: str = "rag_db") -> None:
        """
        Load glossary data from CSV files in a directory.

        Args:
            dir_path: Path to directory containing glossary CSV files
        """
        all_files = glob.glob(os.path.join(dir_path, "*.csv"))

        for file in all_files:
            data = pd.read_csv(file)
            for _, row in data.iterrows():
                self.insert(row["Term"], row["Definition"], row["Example"])

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
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": n_results}
            )
            docs = retriever.invoke(self.query_prefix + question)

            # Convert results to DataFrame
            results = []
            for doc in docs:
                results.append({
                    "Term": doc.metadata["term"],
                    "Definition": doc.metadata["definition"],
                    "Example": doc.metadata["example"]
                })

            return pd.DataFrame(results)
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

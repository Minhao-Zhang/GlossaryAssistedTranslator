import logging
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd


class RAG:
    """Retrieval-Augmented Generation (RAG) system for document retrieval."""

    def __init__(self, collection_name: str, embedding_model_name: str):
        """
        Initialize the RAG system.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model_name: Name of the HuggingFace embedding model
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.doc_count = 0

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
            logging.info(f"Initialized RAG system with collection '{
                         collection_name}'")
        except Exception as e:
            logging.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def insert(self, definition: str, example_translation: str) -> None:
        """
        Insert a single document into the vector store.

        Args:
            definition: The definition text
            example_translation: Example translation text
        """
        try:
            if self.embedding_model_name[:5] == "nomic":
                definition = f"search_document: {definition}"

            self.doc_count += 1
            doc = Document(
                page_content=definition,
                metadata={
                    "id": f"id_{self.doc_count}",
                    "example_translation": example_translation
                }
            )
            self.vectorstore.add_documents([doc])
            logging.debug(f"Inserted document {self.doc_count}")
        except Exception as e:
            logging.error(f"Failed to insert document: {str(e)}")
            raise

    def insert_all(self, definitions: List[str], examples: List[str]) -> None:
        """
        Insert multiple documents into the vector store.

        Args:
            definitions: List of definition texts
            examples: List of example translation texts
        """
        try:
            documents = []
            for definition, example in zip(definitions, examples):
                content = definition
                if self.embedding_model_name[:5] == "nomic":
                    content = f"search_document: {definition}"

                self.doc_count += 1
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "id": f"id_{self.doc_count}",
                        "example_translation": example
                    }
                ))

            self.vectorstore.add_documents(documents)
            logging.info(f"Inserted {len(documents)} documents")
        except Exception as e:
            logging.error(f"Failed to insert documents: {str(e)}")
            raise

    def query(self, question: str, n_results: int = 3) -> List[Document]:
        """
        Query the vector store for relevant documents.

        Args:
            question: The query text
            n_results: Number of results to return

        Returns:
            List of relevant documents
        """
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": n_results}
            )
            return retriever.invoke("Find possible definitions that relates to the sentence: " + question)
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            raise


def load_and_insert_data(rag_instances: List[RAG], csv_path: str) -> None:
    """
    Load data from CSV and insert into RAG instances.

    Args:
        rag_instances: List of RAG instances
        csv_path: Path to CSV file
    """
    try:
        data = pd.read_csv(csv_path)
        definitions = list(data["Definition"])
        examples = list(data["Example"])

        for rag in rag_instances:
            rag.insert_all(definitions, examples)

        logging.info(f"Loaded data from {csv_path}")
    except Exception as e:
        logging.error(f"Failed to load data from {csv_path}: {str(e)}")
        raise


def display_results(results: List[Document], rag_name: str) -> None:
    """
    Display query results in a formatted way.

    Args:
        results: List of documents to display
        rag_name: Name of the RAG instance
    """
    print(f"\n{rag_name} Results:")
    for doc in results:
        print(doc.page_content)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize RAG instances
        rag_instances = [
            RAG("rag1", "nomic-ai/nomic-embed-text-v1.5"),
            RAG("rag2", "dunzhang/stella_en_400M_v5"),
            RAG("rag3", "Snowflake/snowflake-arctic-embed-l-v2.0")
        ]

        # Load data from CSVs
        csv_files = [
            "rag_db/general.csv",
            "rag_db/weapons.csv",
            "rag_db/players.csv",
            "rag_db/teams.csv",
            "rag_db/agents.csv"
        ]

        for csv_file in csv_files:
            load_and_insert_data(rag_instances, csv_file)

        # Interactive question-answering loop
        while True:
            print("=" * 80)
            question = input("Enter a question (or 'exit' to quit): ")
            if question.lower() == "exit":
                break

            for rag in rag_instances:
                results = rag.query(question)
                display_results(results, rag.collection_name)

    except Exception as e:
        logging.error(f"Application error: {str(e)}")

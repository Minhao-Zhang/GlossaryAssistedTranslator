from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd


class RAG:
    def __init__(self, collection_name: str, embedding_model_name: str):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        model_kwargs = {
            "trust_remote_code": True,
            'device': 'cpu'
        }
        encode_kwargs = {
        }

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,

        )

        # Initialize Chroma vector store with the embedding function
        self.vectorstore = Chroma(
            collection_name, embedding_function=self.embedding_model)

        self.doc_count = 0

    def insert(self, definition: str, example_translation: str):
        # nomic uses `search_document: ` as a prefix to the definition
        if self.embedding_model_name[:5] == "nomic":
            definition = f"search_document: {definition}"
        self.doc_count += 1
        doc = Document(page_content=definition,
                       metadata={
                           "id": f"id_{self.doc_count}",
                           "example_translation": example_translation
                       })
        self.vectorstore.add_documents([doc])

    def insert_all(self, definition: list[str], example_translation: list[str]):
        doc = []
        for d, s in zip(definition, example_translation):
            if self.embedding_model_name[:5] == "nomic":
                definition = f"search_document: {d}"
            self.doc_count += 1
            doc.append(Document(page_content=d,
                                metadata={
                                    "id": f"id_{self.doc_count}",
                                    "example_translation": s
                                }))
        self.vectorstore.add_documents(doc)

    def query(self, question: str, n_results: int = 3):
        """
        Retrieve documents relevant to the question.
        :param question: Query text.
        :param n_results: Number of results to return.
        :return: List of relevant documents.
        """
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": n_results})
        docs = retriever.invoke(question)
        return docs


if __name__ == "__main__":
    rag1 = RAG("rag1", "nomic-ai/nomic-embed-text-v1.5")
    rag2 = RAG("rag2", "dunzhang/stella_en_400M_v5")
    rag3 = RAG("rag3", "Snowflake/snowflake-arctic-embed-l-v2.0")

    # Load data from CSVs and insert into vector store
    general_terms = pd.read_csv("rag_db/general_terms.csv")
    rag1.insert_all(list(general_terms["Definition"]), list(
        general_terms["Example"]))
    rag2.insert_all(list(general_terms["Definition"]), list(
        general_terms["Example"]))
    rag3.insert_all(list(general_terms["Definition"]), list(
        general_terms["Example"]))

    weapons = pd.read_csv("rag_db/weapons.csv")
    rag1.insert_all(list(weapons["Definition"]), list(
        weapons["Example"]))
    rag2.insert_all(list(weapons["Definition"]), list(
        weapons["Example"]))
    rag3.insert_all(list(weapons["Definition"]), list(
        weapons["Example"]))

    players = pd.read_csv("rag_db/players.csv")
    rag1.insert_all(list(players["Definition"]), list(
        players["Example"]))
    rag2.insert_all(list(players["Definition"]), list(
        players["Example"]))
    rag3.insert_all(list(players["Definition"]), list(
        players["Example"]))

    teams = pd.read_csv("rag_db/teams.csv")
    rag1.insert_all(list(teams["Definition"]), list(
        teams["Example"]))
    rag2.insert_all(list(teams["Definition"]), list(
        teams["Example"]))
    rag3.insert_all(list(teams["Definition"]), list(
        teams["Example"]))

    agents = pd.read_csv("rag_db/agents.csv")
    rag1.insert_all(list(agents["Definition"]), list(
        agents["Example"]))
    rag2.insert_all(list(agents["Definition"]), list(
        agents["Example"]))
    rag3.insert_all(list(agents["Definition"]), list(
        agents["Example"]))

    # Interactive question-answering loop
    while True:
        print("=====================================================================")
        question = input("Enter a question: ")
        if question.lower() == "exit":
            break
        print("---------------------------------------------------------------------")

        print("RAG1 Results:")
        for doc in rag1.query(question):
            print(doc.page_content)
            print(doc.metadata["example_translation"])

        print("\nRAG2 Results:")
        for doc in rag2.query(question):
            print(doc.page_content)
            print(doc.metadata["example_translation"])

        print("\nRAG3 Results:")
        for doc in rag3.query(question):
            print(doc.page_content)
            print(doc.metadata["example_translation"])

import chromadb
import pandas as pd
import ollama


class RAG():
    def __init__(self, chroma_client: chromadb.Client, collection_name: str, ollama_embed_model: str):
        self.chroma_client = chroma_client
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.ollama_embed_model = ollama_embed_model
        self.doc_count = 0

    def embed_ollama(self, text, task_type="docoment"):
        if self.ollama_embed_model == 'nomic-embed-text':
            if task_type == "docoment":
                prefix = "search_document: "
            elif task_type == "question":
                prefix = "search_query: "
            else:
                prefix = ""
            response = ollama.embeddings(
                model=self.ollama_embed_model,
                prompt=prefix + text
            )
        elif self.ollama_embed_model == 'snowflake-arctic-embed2':
            response = ollama.embeddings(
                model=self.ollama_embed_model,
                prompt="Represent this sentence for searching relevant passages: " + text
            )
        else:
            response = ollama.embeddings(
                model=self.ollama_embed_model,
                prompt=text
            )
        return response['embedding']

    def query(self, text, n_results=3):
        embed_text = self.embed_ollama(text, "question")
        result = self.collection.query(embed_text, n_results=n_results)
        return result['documents'][0]

    def insert(self, text_to_encode, text_to_store):
        embed = self.embed_ollama(text_to_encode, 'nomic-embed-text')
        self.doc_count += 1
        self.collection.upsert(documents=text_to_store,
                               ids="id_" + str(self.doc_count),
                               embeddings=embed)


if __name__ == "__main__":
    chroma_client = chromadb.Client()
    rag1 = RAG(chroma_client, "rag1", "nomic-embed-text")
    rag2 = RAG(chroma_client, "rag2", "snowflake-arctic-embed2")

    general_terms = pd.read_csv('rag_db/general_terms.csv')
    for i, row in general_terms.iterrows():
        rag1.insert(row['Definition'], row['Definition'] +
                    '\n' + row['Sample'])
        rag2.insert(row['Definition'], row['Definition'] +
                    '\n' + row['Sample'])
    weapons = pd.read_csv('rag_db/weapons.csv')
    for i, row in weapons.iterrows():
        rag1.insert(row['Definition'], row['Definition'] +
                    '\n' + row['Sample'])
        rag2.insert(row['Definition'], row['Definition'] +
                    '\n' + row['Sample'])

    while True:
        print("=====================================================================")
        question = input("Enter a question: ")
        if question == "exit":
            break
        print("---------------------------------------------------------------------")

        for doc in rag1.query(question):
            print(doc)
        print()
        for doc in rag2.query(question):
            print(doc)

import chromadb
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()
os.environ["ORT_DISABLE_COREML"] = "1"

class RAGSystem:
    def __init__(self, embedding_model="nomic", llm_model="ollama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.llm, self.llm_model_name = self.setup_llm()
        self.embedding_fn = self.setup_embedding_function()

        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name, self.collection = self.setup_collection()

    def setup_llm(self):
        if self.llm_model == "openai":
            llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            llm_model_name = "gpt-4o-mini"
        else:
            llm = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            llm_model_name = "llama3.2:latest"
        return llm, llm_model_name

    def setup_embedding_function(self):
        embedding_fn = None
        try:
            if self.embedding_model == "openai":
                embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
                                                                           model_name=os.getenv("OPENAI_MODEL"))
            elif self.embedding_model == "nomic":
                embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key="ollama",
                                                                           model_name="nomic-embed-text",
                                                                           api_base="http://localhost:11434/v1")
            else:
                embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as error:
            print(str(error))
        return embedding_fn

    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        try: # Firstly, Try to get existing collection
            collection = self.db.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)
        except ValueError as e:
            print(str(e))
        return collection_name, collection

    def add_documents(self, chunks):
        try:
           self.collection.add(
               ids=[chunk["id"] for chunk in chunks],
               documents=[chunk["text"] for chunk in chunks],
               metadatas=[chunk["metadata"] for chunk in chunks]
           )

           return True
        except Exception as e:
            print(str(e))
            return False

    def query_documents(self, query, n_results=2):
        results = self.collection.query(query_texts=[query],
                                        n_results=n_results)
        return results

    def generate_response(self, query, context):
        try:
            prompt = (f"Based on the following context, please answer the below question.\n"
                      f"If you cannot find the answer in the context, say don't know politely."
                      f"Don't need to explain your resources.\n"
                      f"Context: {context}\n"
                      f"Question: {query}")

            response = self.llm.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": "you are are helpful assistant."},
                    {"role": "user", "content": prompt}
                ])
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

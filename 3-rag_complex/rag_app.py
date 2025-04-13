import chromadb
import pandas as pd
import dotenv
dotenv.load_dotenv()
from typing import Dict, List, Tuple
from embedding_model import EmbeddingModel
from llm_model import LLMModel


CHROMA_COLLECTION = "nutrition_facts"
CHROMA_DB_PATH = "./db/chromadb_rag.db"

class RAGPipeline:
    """RAG = Retrieval Augmented Generation pipeline
    This class manages the complete RAG pipeline including:
    - Data loading and preprocessing
    - Vector database setup
    - Query processing
    - LLM response generation

    Attributes:
        embedding_model: Instance of EmbeddingModel for text embedding
        llm_model: Instance of LLMModel for response generation
        collection_name: Name of the ChromaDB collection
        db_path: Path to the ChromaDB database
        collection: ChromaDB collection instance
        reset: True/False - reset the ChromaDB or keep the existing db.
        """
    def __init__(self, embedding_model: EmbeddingModel,
                 llm_model:LLMModel,
                 collection_name:str=CHROMA_COLLECTION,
                 db_path:str=CHROMA_DB_PATH,
                 reset=True
                 ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.db_path = db_path
        self.collection = None
        self.reset = reset

    @staticmethod
    def _sanitize_input_text(text:str) -> str:
        """Sanitize the user input text to prevent injection"""
        sanitized = "".join(char for char in text if ord(char) >=32)
        return sanitized[:1000] #hard stop user input query at 1000 characters.

    @staticmethod
    def load_csv():
        csv_file = "nutrient_facts.csv"
        df = pd.read_csv(csv_file, delimiter=";")
        documents = df["details"].to_list()
        ids = df["id"].to_list()
        print(f"Loaded {len(documents)} documents from {csv_file}")
        return documents, ids

    def setup_chromadb(self, documents: List[str]):
        chromadb_client = chromadb.PersistentClient(path=self.db_path)
        try:
            chromadb_client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = chromadb_client.create_collection(name=self.collection_name,
                                                            embedding_function=self.embedding_model.embedding_fn)
        self.collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])


    def find_relevant_chunks(self, query_text:str, top_k:int = 2):
        sanitized_text = self._sanitize_input_text(query_text)
        results = self.collection.query(query_texts=[sanitized_text], n_results=top_k)

        print("====Retrieved relevant chunks:")
        # for doc in results["documents"][0]:
        #     print(f"- {doc}")
        return list(
            zip(
                results["documents"][0],
                results["metadatas"][0] if results["metadatas"][0] else [{}] * len(results["documents"][0])
            )
        )

    def augment_prompt(self, query_text:str, related_chunks:List[Tuple[str, Dict]]) -> str:
        context = "\n".join([chunk[0] for chunk in related_chunks])
        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}"
        print(f"\nAugmented Prompt: {augmented_prompt}")
        return augmented_prompt

    def process_query(self, query_text:str, top_k=2):
        print(f"\n===Processing query: {query_text}")
        related_chunks = self.find_relevant_chunks(query_text,  top_k)
        augmented_prompt = self.augment_prompt(query_text, related_chunks)
        try:
            response = self.llm_model.generate_completion_response(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who provides accurate answers based only on the given context."
                                                  "If you cannot find the answer in the given context, just say you don't know."},
                    {"role": "user", "content": augmented_prompt}
                ]
            )
            references = [chunk[0] for chunk in related_chunks] # Extract references for citation

            return response, references
        except Exception as error:
            print(f"Error processing query: {str(error)}")

def select_a_model():
    list_embedding_models = [
        "====Select an Embedding model:",
        "1-Ollama-Nomic Embedded",
        "2-OpenAI Embedding"]
    print(*list_embedding_models, sep="\n")
    while (embedding_model_choice := input("Please enter 1, 2: ")) not in {"1", "2"}: print("Try again!")
    print(embedding_model_choice)
    print(f"==> User choose {list_embedding_models[int(embedding_model_choice)]}\n")

    list_chat_models = ["====Select a LLM model:","1-Ollama Llama2", "2-OpenAI GPT-4"]
    print(*list_chat_models, sep="\n")
    while (llm_model_choice := input("Please enter 1 or 2: ")) not in {"1", "2"}: print("Try again!")
    print(f"==> User choose {list_chat_models[int(llm_model_choice)]}")

    return embedding_model_choice, llm_model_choice



def main():
    print("======RAG Pipeline======")
    # embedding_model_choice, llm_model_choice = select_a_model()
    embedding_model_choice ="1"
    llm_model_choice="1"
    embedding_model = EmbeddingModel(embedding_model_choice)
    llm_model = LLMModel(llm_model_choice)

    pipeline = RAGPipeline(embedding_model=embedding_model,
                           llm_model=llm_model,
                           collection_name=CHROMA_COLLECTION,
                           db_path=CHROMA_DB_PATH,
                           reset=True
                           )
    # Load and setup database
    documents, ids = pipeline.load_csv()
    pipeline.setup_chromadb(documents)

    # Process the query

    query_text = input("Please input your question related to fruits' nutrient: \n")
    response, references = pipeline.process_query(query_text=query_text,top_k=2)
    print(f"\n\n===> Final Response:\n{response}\n\n")



if __name__ == '__main__':
    main()














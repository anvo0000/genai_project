from chromadb.utils import embedding_functions
import os
import dotenv
from typing import Literal
dotenv.load_dotenv()

class EmbeddingModel:
    """Embedding model selection class supporting multiple providers."""

    def __init__(self, model_type: Literal["1", "2", "3"]):
        """Initialize the embedding model based on the selected type:
            1. Ollama with nomic-embed-text model (local)
            2. OpenAI embeddings API
            3. ChromaDB's default embedding function
            """
        self.model_type = model_type
        self.embedding_fn = None
        self.model_name = ""
        if self.model_type == "1": # Using Ollama nomic-embed-text model locally
            try:
                model_name = "nomic-embed-text"
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key="ollama",
                                                                                api_base="http://localhost:11434/v1",
                                                                                model_name="nomic-embed-text")
            except Exception as error:
                raise Exception(str(error))
        elif self.model_type == "2":
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                model_name = os.getenv("OPENAI_MODEL")
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key,model_name=model_name)
            except ValueError as error:
                print(str(error))
        else:
            model_name = "ChromaDB DefaultEmbeddingFunction"
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        print(f"----> {model_name}")

# em = EmbeddingModel("1")
# print(em.embedding_fn)

class ModelSelector:
    """Class to handle the models selection"""
    def __init__(self):
        self.llm_models = {
            "ollama": "Llama2",
            "openai": "GPT-4"
        }

        self.embedding_models = {
            "nomic": {
                "name": "Nomic Embed Text",
                "model_name": "nomic-embed-text",
                "dimensions": 768
            },
            "openai":{
                "name": "OpenAI Embeddings",
                "model_name": "text-embedding-3-small",
                "dimensions": 1536
            },
            "chroma": {
                "name": "ChromaDB Default",
                "model_name": None,
                "dimensions": 384
            },
        }



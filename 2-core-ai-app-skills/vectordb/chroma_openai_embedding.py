import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()

chromadb_client = chromadb.PersistentClient(path="./db/chromadb_openai.db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
                                                        model_name=os.getenv("OPENAI_MODEL"))
collection =chromadb_client.get_or_create_collection(name="openai_story",
                                                     embedding_function=openai_ef
                                                     )

documents = [
    {"id": "doc1", "text": "hello Amazon!"},
    {"id": "doc3", "text": "how are you?"},
    {"id": "doc4", "text": "goodbye!"},
    {"id": "doc2", "text": "The AWS Certified Machine Learning Engineer - Associate (MLA-C01) exam validates a candidate’s ability to build, operational, deploy, and maintain machine learning(ML) solutions and pipelines by using the AWS Cloud."},
    {"id": "doc5", "text": "Chroma is the open-source AI application database. Batteries included. Embeddings, vector search, document storage, full-text search, metadata filtering, and multi-modal. All in one place. Retrieval that just works. As it should be."},
    {"id": "doc6", "text": "Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms."}
]

for doc in documents:
    collection.upsert(ids=doc["id"],
                      documents=[doc["text"]]
                      )

user_query = "Find technical certificate documents_url"

results = collection.query(query_texts=user_query,
                           n_results=2)

print(f"====User Query: {user_query}\nFound:")
for idx, document in enumerate(results["documents_url"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"====ID: {doc_id}\nDistance: {distance}\nText: {document}")
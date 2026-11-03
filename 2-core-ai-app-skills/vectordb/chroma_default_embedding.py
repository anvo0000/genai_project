import chromadb
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db/chromadb_persist.db")
collection = chroma_client.get_or_create_collection(name="my_story",
                                                    embedding_function=default_ef)

documents = [
    {"id": "doc1", "text": "hello Amazon!"},
    {"id": "doc3", "text": "how are you?"},
    {"id": "doc4", "text": "goodbye!"},
    {"id": "doc2", "text": "The AWS Certified Machine Learning Engineer - Associate (MLA-C01) exam validates a candidate’s ability to build, operationalize, deploy, and maintain machine learning(ML) solutions and pipelines by using the AWS Cloud."}
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

query = "AWS"
results = collection.query(
    query_texts=[query],
    n_results=2
)
# print(results)
print(f"User Query: {query}\nFound: ")
for idx, document in enumerate(results["documents_url"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"ID: {doc_id}\nDistance: {distance}\nText: {document}")
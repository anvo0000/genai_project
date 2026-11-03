import chromadb
chroma_client = chromadb.Client()
collection_name = "av_collection"

collection = chroma_client.get_or_create_collection(collection_name)
documents = [
    {"id": "doc1", "text": "hello world!"},
    {"id": "doc3", "text": "how are you?"},
    {"id": "doc4", "text": "goodbye!"},
]

for doc in documents:
    collection.upsert(
        ids=doc["id"],
        documents=doc["text"]
    )


# define user_query
user_query = "hello"

results = collection.query(
    query_texts=[user_query],
    n_results=2
)
# print(results)
for idx, result in enumerate(results["documents_url"][0]):
    # print(idx, result)
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]

    print(f"Doc ID: {doc_id}")
    print(f"Document Text: {result}")
    print(f"Distance: {distance}")


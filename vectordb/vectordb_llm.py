import chromadb
from chromadb.utils import embedding_functions
import os
import dotenv
from openai import OpenAI
dotenv.load_dotenv()

openai_client = OpenAI()
chromadb_client = chromadb.PersistentClient(path="./db/chromadb_qa.db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
                                                        model_name=os.getenv("OPENAI_MODEL"))
collection = chromadb_client.get_or_create_collection(name="openai_new_story",
                                                      embedding_function=openai_ef
                                                     )

def load_documents_from_directory(dir_path:str) -> list:
    print(f"====Loading documents from directory '{dir_path}' ====")
    documents = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path,filename)) as file:
                documents.append({"id": filename,
                                  "text": file.read()
                                  })
    return documents

def split_text(text, chunk_size=500, chunk_overlap=20) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start: end])
        start = end - chunk_overlap
    return chunks

def get_openai_embedding(text):
    response = openai_client.embeddings.create(input=text, model=os.getenv("OPENAI_MODEL"))
    embedding = response.data[0].embedding
    total_tokens = response.usage.total_tokens
    print(f"Embedding dimension: {len(embedding)}")
    return embedding, total_tokens

def main_save_vectordb():
    # 1.Load documents from the directory
    directory_path = "./text_files"
    documents = load_documents_from_directory(directory_path)
    print(f"Loaded {len(documents)} documents.")
    # print(documents[0])

    # 2.Split the documents into chunks of text
    chunked_documents = []
    for doc in documents:
        print(f"\n== Splitting {doc['id']} ==")
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            print(f"==== chunk-{i+1}- {chunk[:10]}... ====")
            chunked_documents.append({"id": f"{doc['id']}-chunk-{i+1}",
                                      "text": chunk})
    # print(len(chunked_documents))

    # 3.Generate embeddings from each item in the chunked_documents list.
    for doc in chunked_documents:
        print(f"\n== Generating Embedding {doc['id']} ==")
        embedding, total_tokens = get_openai_embedding(doc["text"])
        doc["embedding"] = embedding
        doc["total_tokens"] = total_tokens

    # 4. Upsert documents with embeddings into ChromaDB
    print(f"\n== Upsert documents(id, text, embedding, total_tokens) into ChromaDB ==")
    for doc in chunked_documents:
        print(f'\n== id: {doc["id"]}, text: {doc["text"][:10]}..., total_tokens: {doc["total_tokens"]} ==')
        collection.upsert(ids=doc["id"],
                          documents=[doc["text"]],
                          embeddings=[doc["embedding"]],
                          metadatas=[{"total_tokens": doc["total_tokens"]}]
                          )


def query_from_vectordb(question, n_result=2) -> list:
    global collection
    relevant_chunks = []
    result = collection.query(query_texts=question, n_results=n_result)

    for sublist in result["documents"]:
        for doc in sublist:
            relevant_chunks.append(doc)
    return relevant_chunks

def llm_generate_response(question, relevant_chunks:list):
    context = "\n\n".join(relevant_chunks)
    system_prompt = ("You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question."
            "If you don't know the answer, say I don't know " 
            "Use three sentences maximum and keep the answer concise."
            f"Context:\n{context}".strip()
    )
    response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    print("=====response\n",response, "\n\n\n")
    return response.choices[0].message.content.strip()


def main_query_documents():
    question = "what is token?"
    relevant_chunks = query_from_vectordb(question)
    print("=====relevant_chunks\n",relevant_chunks, "\n\n\n")
    answer = llm_generate_response(question, relevant_chunks)
    print("=====Final Answer\n",answer)



if __name__ == '__main__':
    # main_save_vectordb()
    main_query_documents()












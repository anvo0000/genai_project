from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# import os
from dotenv import load_dotenv
load_dotenv()

import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

documents_url = ["https://beebom.com/what-is-meta-ai-app-features-overview/",
                 "https://beebom.com/13-websites-that-will-make-you-smarter/",
                 "https://beebom.com/openai-drops-plan-to-become-for-profit-company/"]

def scrape_docs(urls: List[str]) -> List[Document]:
    try:
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        print(f"Load {len(raw_docs)} documents.")

        # Documents metadata
        for doc in raw_docs:
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Content length: {len(doc.page_content)}")
            print(f"Sample Content: {doc.page_content[:100]}")
        return raw_docs

    except Exception as e:
        print(e)
        return []

def create_vector_store(texts: List[str], metadata: List[Dict]):
    # embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL"))
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_texts(texts=texts,
                           metadatas=metadata,
                           embedding=embeddings,
                           persist_directory="db/chromadb")
    return db


def setup_qa_chain(db):
    llm = ChatOllama(model="llama3.2:latest", temperature=0)
    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
    System Information:
    - Date (UTC): 2025-05-06 10:45:34
    - User: anvo0000

    ### Context: {context}

    ### Question: {question}

    ### Response Guidelines:
    Please provide a response that follows these criteria:
    1. Format the answer in clear, short paragraphs
    2. Use simple English suitable for ESL readers
    3. Break down complex information into bullet points
    4. Include relevant examples or references if available

    ### Structure your response as:

    Main Answer:
    - [2-3 short sentences addressing the main question]

    Key Points:
    - [Bullet points for important details]
    - [Supporting information if applicable]

    Additional Context:
    - [Any relevant background information]
    - [Examples or references if available]

    Follow-up:
    Would you like to know more about any specific aspect? I'm happy to clarify further.

    Remember to:
    - Keep sentences concise and clear
    - Use simple, everyday language
    - Maintain a friendly and supportive tone
    - Provide practical examples when possible
    """)

    # Create the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def process_query(chain_and_retriever, query: str):
    """Process a query and return response."""
    try:
        chain, retriever = chain_and_retriever  # unpack the tuple from input param
        response = chain.invoke(query)
        docs = retriever.invoke(query)

        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])
        return {"answer": response, "source": sources_str}
    except Exception as e:
        print(str(e))
        return {"answer": "Internal error", "source": ""}

def split_documents(pages_content: List[Document]) -> tuple:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts, all_metadata = [], []
    for document in pages_content:
        text = document.page_content
        source = document.metadata.get("source", "")
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadata.append({"source": source})
    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadata

def main():
    print("====Step 1: Scraping the web page")
    pages_content = scrape_docs(urls=documents_url)

    print("====Step 2: Splitting the pages_content into small chunks")
    all_texts, all_metadata = split_documents(pages_content=pages_content)

    print("====Step 3: Creating a vector store")
    db = create_vector_store(texts=all_texts, metadata=all_metadata)

    print("====Step 4: Setting up a QA chain")
    qa_chain = setup_qa_chain(db)

    print("====> Finally, ready for your questions! (Type 'exit' if you want to stop it)")
    while True:
        user_query = input("\n Enter your question: ".strip())
        if user_query.lower() == 'exit':
            break

        result = process_query(chain_and_retriever=qa_chain, query=user_query)
        print(f"Response:\n{result['answer']}")

        if result["source"]:
            print("References:")
            for source in result["source"].split(","):
                print("- ", source)

if __name__ == '__main__':
    main()
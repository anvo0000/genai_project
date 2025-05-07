from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pprint
import re
load_dotenv()

# Cleaning function
def clean_text(text):
    # removes all non-alphabetic and non-whitespace characters from a text string.
    text = re.sub(pattern=r"[^a-zA-Z\s]", repl="", string=text)

    # Remove irregular spaces by 1 space, tabs \t, newlines \n,
    # Remove space from the beginning and ending of text.
    text = re.sub(pattern="\s+", repl=" ", string=text).strip()
    return text.lower()

# 1-Document Loader
documents = TextLoader("../files/twin_ai_agent.txt").load()

# 2-Text Splitters: will do the splitting text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents=documents)

# 3-Clean Text
texts = [clean_text(text.page_content) for text in texts]
# documents_url = [Document(page_content=text) for text in texts]
# texts = [Document(page_content=clean_text(text.page_content)) for text in texts]


# 4-Embeddings
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL"))

#5-VectorStore and 6-Retriever
retriever = FAISS.from_texts(texts=texts, embedding=embeddings).as_retriever(search_kwargs={"k":2})

# query the retriever
query = "what do the twin labs do?"
docs = retriever.invoke(query, k=1)
pprint.pprint(f"=> DOCS: {docs}")


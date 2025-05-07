from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import pprint
import re
load_dotenv()

import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

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
query = "give me a summary of the speech in bullet points?"
docs = retriever.invoke(query, k=1)
# pprint.pprint(f"=> DOCS: {docs}")

#define a prompt from template
model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"))
prompt = ChatPromptTemplate.from_template("Please use following docs {docs}, and answer my question: {query}")

# Langchain Chains are including: Prompt + Model + Output
chain = prompt | model | StrOutputParser()

# Invoke the chain
response = chain.invoke({"docs": documents,
                         "query": query})
print(f"Model response: {response}")

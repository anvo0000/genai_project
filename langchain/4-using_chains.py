import os
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()

import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

#define a prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"))

# Langchain Chains are including: Prompt + Model + Output
chain = prompt | model | StrOutputParser()

# Invoke the chain
response = chain.invoke({"topic": "langchain"})
print(response)
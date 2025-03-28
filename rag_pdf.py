from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

BEDROCK_REGION = "us-west-2"
BEDROCK_TEXT_MODEL = "amazon.titan-text-express-v1"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v1"
BEDROCK_SERVICE = "bedrock-runtime"

bedrock = boto3.client(service_name=BEDROCK_SERVICE,region_name=BEDROCK_REGION)
bedrock_text = BedrockLLM(model_id=BEDROCK_TEXT_MODEL, client=bedrock)
bedrock_embeddings = BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, client=bedrock)

#Loading Data
loader = PyPDFLoader(file_path="files/MLA-C01.pdf")
spliter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=200)
docs = loader.load()
spliter_docs = spliter.split_documents(docs)


#create a vector store
vector_store = FAISS.from_documents(spliter_docs, bedrock_embeddings)
question = "What are the requirement tasks?"
#create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
results = retriever.invoke(question)
print(results)

results_string = []
for result in results:
    results_string.append(result.page_content)

#build a template
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}")
    ]
)
chain = template.pipe(bedrock_text)
#answer the question from the results_string.

response = chain.invoke({"input": question,
                        "context": results_string })
print(response)
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
import boto3

BEDROCK_REGION = "us-west-2"
BEDROCK_TEXT_MODEL = "amazon.titan-text-express-v1"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v1"
BEDROCK_SERVICE = "bedrock-runtime"

my_data = [
    "The weather is hot today.",
    "Bob likes to eat apple.",
    "Bob likes to eat bread."
    "Red is Bob favorite color."
]
question = "What is Bob favorite food?"

bedrock = boto3.client(service_name=BEDROCK_SERVICE,region_name=BEDROCK_REGION)
bedrock_text = BedrockLLM(model_id=BEDROCK_TEXT_MODEL, client=bedrock)
bedrock_embeddings = BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, client=bedrock)

#create a vector store
vector_store = FAISS.from_texts(texts=my_data, embedding = bedrock_embeddings)
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
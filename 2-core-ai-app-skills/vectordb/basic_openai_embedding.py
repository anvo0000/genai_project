from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL") #"text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.embeddings.create(input="text",
                                    model=OPENAI_MODEL)

result = response.data[0].embedding
print(result)
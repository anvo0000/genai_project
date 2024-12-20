import os
from dotenv import load_dotenv
import dataset
import matplotlib.pyplot as plt
import transformers

# Load the .env file
load_dotenv()

# Access the Hugging Face token
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#Text Classification

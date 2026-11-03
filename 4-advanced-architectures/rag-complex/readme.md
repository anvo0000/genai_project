# RAG Knowledge Base Explorer
A Streamlit application that demonstrates a Retrieval Augmented Generation (RAG) pipeline. 
This application allows you to upload your data or use sample data, then ask questions to get AI-generated answers based on the relevant information in your knowledge base.
![Preview RAG KB Explorer.png](Preview%20RAG%20KB%20Explorer.png)
# Overview
This application integrates several components:
- Streamlit for the web interface
- ChromaDB for vector storage
- Multiple embedding model options (Ollama, OpenAI, ChromaDB)
- Multiple LLM options (Ollama Llama3, OpenAI GPT)
- RAG pipeline for knowledge-based AI responses

# Requirements
- Python 3.11

## Required Packages
- streamlit>=1.11.0
- pandas
- chromadb
- openai>=1.0.0
- python-dotenv

You can install all required packages with:
`pip install -r requirements.txt`

## External Requirements
Depending on your configuration choices:

### 1. For Ollama embedding and LLM models:
- Ollama must be installed and running locally.
- The required models (nomic-embed-text, llama3.2) should be available in Ollama.

### 2. For OpenAI embedding and LLM models:
Create a .env file in the root directory with your API keys (if using OpenAI):
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```
## Project Structure
``` 
.
├── app.py                  # Streamlit application
├── rag_app.py              # RAG pipeline implementation
├── embedding_model.py      # Embedding model selection and initialization
├── llm_model.py            # LLM model selection and initialization
├── nutrient_facts.csv      # Sample dataset (or your own data)
├── requirements.txt        # Project dependencies
├── db/                     # Directory for ChromaDB database files
└── README.md               # This file
```

## Usage
1. Start the Streamlit application: 
`streamlit run app.py`
2. Access the application in your web browser (usually at http://localhost:8501)
3. Configure the application:
- Select embedding and LLM models
- Choose database settings
- Select a data source (upload CSV, use demo data, or sample text)
4. Initialize the pipeline by clicking "Initialize Pipeline".
5. Ask questions in the query section and view AI-generated answers based on your knowledge base.

## Data Format
If uploading your own CSV file, ensure it has the following structure:
- Must include a column named `details` containing the text data
- Should use semicolon `(;)` as the delimiter
- Each row will be treated as a separate document

Example:
```csv
id;details
1;Bananas are rich in potassium, vitamin C, and vitamin B6. They're a great source of energy.
2;Apples contain fiber, vitamin C, and various antioxidants.
```

## Features
- Multiple Model Options: Choose between local Ollama models or OpenAI cloud models
- Flexible Data Sources: Upload your data or use built-in samples
- Interactive Interface: User-friendly UI with configuration options
- Query History: Track previous queries and responses
- References: See which documents were used to generate responses

## Troubleshooting
- Model Connection Issues: Ensure Ollama is running locally if using Ollama models
- API Key Errors: Check your .env file for correct OpenAI API keys
- Database Errors: Try using the "Reset Database" option to recreate the vector database
- CSV Format Issues: Ensure your CSV uses semicolons as delimiters and has a "details" column


_Developed by: anvo0000 - April 2025_
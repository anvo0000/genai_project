from openai import OpenAI
import logging
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv
load_dotenv()

def setup_logging():
    """Save logging info to JSON files format"""
    logger = logging.getLogger("Chatbot")
    logger.setLevel(logging.INFO)

    # Create a file handler for JSON file
    file_handler = logging.FileHandler("chatbot_logs.json")
    formatter = logging.Formatter("%(message)s") # Log raw JSON
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler for human-readable logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    return logger

def initialize_client(use_ollama:bool=True) -> OpenAI:
    """Initialize OpenAI client for either OpenAI or Ollama"""
    if use_ollama:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        return OpenAI() # if false, then use OpenAI



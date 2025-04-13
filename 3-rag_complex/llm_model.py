from openai import OpenAI
import os
from typing import Literal
import dotenv
dotenv.load_dotenv()

class LLMModel:
    """LLM model selection class supporting multiple providers."""
    def __init__(self, model_type: Literal["1", "2"]):
        """Initialize the LLM model based on the selected type:
        1-Ollama Llama2
        2-OpenAI GPT-4
        """
        self.model_type = model_type
        self.client = None
        self.model_name = None

        if self.model_type == "1":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2:latest"
        else:
            try:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model_name = os.getenv("OPENAI_CHAT_MODEL")
            except ValueError as error:
                raise ValueError("Missing required environment variables: OPENAI_API_KEY or OPENAI_CHAT_MODEL")
        print(f"----> {self.model_name}")

    def generate_completion_response(self, messages):
        try:
            response = self.client.chat.completions.create(model=self.model_name,
                                                           messages=messages,
                                                           temperature=0.1)
            return response.choices[0].message.content
        except Exception as error:
            return f"Error generate_response_completion: {str(error)}."


# ll = LLMModel("1")
# print(ll)

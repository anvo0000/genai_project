import json
import uuid
from datetime import datetime
from implement_logging import setup_logging, initialize_client
EXIT_CODE = "exit"
EXIT_MESSAGE = "Thanks for contacting, goodbye!"

class ChatBot:
    def __init__(self, use_ollama:bool = True):
        self.logger = setup_logging()
        self.session_id = str(uuid.uuid4())
        self.use_ollama = use_ollama
        self.client = initialize_client(self.use_ollama)
        self.model_name = "llama3.2:1b" if self.use_ollama else "gpt-4o-mini"

        # Define system prompt (messages)
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful customer support assistant."
            }
        ]

    def chat_func(self, user_input:str) -> str:
        try:
            # Log user_input with metadata
            user_input_log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "user_input",
                "user_input": user_input,
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name
                }
            }
            self.logger.info(json.dumps(user_input_log_entry))

            # Append the user_input to the conversation
            self.messages.append({
                "role": "user",
                "content": user_input
            })

            # Generate a response using the API
            start_time = datetime.now()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            end_time = datetime.now()

            # Calculate the duration of a response
            response_time = (end_time - start_time).total_seconds()

            # Extract the LLM response
            assistant_response = response.choices[0].message.content

            # Log the LLM response
            model_response_log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "model_response",
                "model_response": assistant_response,
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name,
                    "response_time_in_seconds": response_time,
                    "token_used": (response.usage.total_tokens if hasattr(response, "usage") else None)
                }
            }
            self.logger.info(json.dumps(model_response_log_entry))

            # Append the assistant's response to the conversation
            self.messages.append({
                "role": "assistant",
                "content": assistant_response
            })
            return assistant_response
        except Exception as err: # log any error occur
            exception_log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "error",
                "error_message": str(err),
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name
                }
            }
            self.logger.info(json.dumps(exception_log_entry))
            return f"Something went wrong: {str(err)}"


def main():
    # Model selection
    print("Select a model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama3.2 (Local)")


    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            # same return with ->use_ollama = choice == "2"
            use_ollama = True if choice == "2" else False
            break
        print("Try again, only accept 1 or 2.")

    # Initialize ChatBot
    chatbot = ChatBot(use_ollama)
    print(f"\n ----Chat Session Started using {'Ollama' if use_ollama else 'OpenAI' }----")
    print(f"Session ID: {chatbot.session_id}\n")
    print(f"Type '{EXIT_CODE}' to end the converation")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == EXIT_CODE:
            print(EXIT_MESSAGE)
            break

        if not user_input:
            continue

        response = chatbot.chat_func(user_input)
        print(f"Assistant: {response}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n ----Chat Ended.\n {EXIT_MESSAGE}")




from chatbot import ChatBot

SAMPLE_QUESTIONS = [
    "Hi, I need help resetting my password.",
    "I didn't receive the reset email. What should I do?",
    "Thanks for your help!"
]


def main():
    # Run a scripted conversation against the local Ollama model
    chatbot = ChatBot(use_ollama=True)
    print(f"Session ID: {chatbot.session_id}\n")

    for question in SAMPLE_QUESTIONS:
        print(f"User: {question}")
        response = chatbot.chat_func(question)
        print(f"Assistant: {response}\n")


if __name__ == '__main__':
    main()

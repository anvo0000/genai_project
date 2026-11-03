from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()
model_id="gpt-4o-mini"
messages = [
    {
        "role": "system",
        "content": "You are a GenAI expert."
    },
    {
        "role": "user",
        "content": "Write a 10-words summary of the benefits of instructional prompt to OpenAI LLM."
    }
]

completion = client.chat.completions.create(
    model=model_id,
    messages=messages
)
print(completion.choices[0].message.content)

# Result: exactly 10 words.
#   -> Instructional prompts enhance clarity, engagement, relevance, creativity, and user satisfaction.
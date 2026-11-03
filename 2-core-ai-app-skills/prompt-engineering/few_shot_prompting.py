from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
model_id="gpt-4o-mini"

#few-shot prompting
completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system",
         "content": "You are a translator"
         },
        {
            "role": "user",
            "content": "translate these sentences: 'hello' -> 'hola', 'goodbye' -> 'adiós'."
                       "Now translate 'thank you'."
        }
    ]
)
print(completion.choices[0].message.content)
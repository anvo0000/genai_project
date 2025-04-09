from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
model_id="gpt-4o-mini"

#chain of thoughts prompting
completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system",
         "content": "You are a Math Tutor"
         },
        {
            "role": "user",
            "content": "Solve this Math problem Step by Step:"
                       "If I have 5 apples, I give you 2, how many do I have left?"
        }
    ]
)
print(completion.choices[0].message.content)
# Result:
# Let's solve the problem step by step:
#
# 1. **Start with the initial quantity of apples**: You have 5 apples.
#
# 2. **Determine how many apples you give away**: You give away 2 apples.
#
# 3. **Subtract the number of apples you gave away from the initial quantity**:
#    \[
#    5 - 2 = 3
#    \]
#
# 4. **Conclusion**: After giving away 2 apples, you have 3 apples left.
#
# So, you have **3 apples left**.
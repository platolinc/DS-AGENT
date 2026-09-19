import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

while True:
    user_input = input("[User Input]: ").strip()
    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_input},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    print(f"[Agent Response]:{response.choices[0].message.content}\n")
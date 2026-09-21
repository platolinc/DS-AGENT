import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 历史记忆
messages = [
    {"role": "system", "content": "你是一个聊天机器人，每次回复完都要滴一声"},
]

while True:
    user_input = input("[User Input]: ").strip()
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    print(f"[Agent Response]: {assistant_message.content}\n")

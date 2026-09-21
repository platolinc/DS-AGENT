import os
import subprocess
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 工具列表
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在终端执行一条 shell 命令并返回输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

def run_command(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )

    return result.stdout or result.stderr or "(无输出)"

TOOL_HANDLERS = {
    "run_command": lambda args: run_command(args["command"]),
}

# 历史记忆
messages = [
    {"role": "system", "content": "你是一个聊天机器人，每次回复完都要滴一声"},
]

while True:
    user_input = input("[User Input]: ").strip()
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        assistant_message = response.choices[0].message

        print(f"[Assistant消息]: {assistant_message.content}")
        print(f"[Assistant工具调用]: {assistant_message.tool_calls}")

        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(f"[Agent回答]: {assistant_message.content or ''}\n")
            break

        for tc in assistant_message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "run_command":
                print(f"[执行命令]: {args['command']}")
            else:
                print(f"[执行工具]: {name}({args})")

            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                output = f"未知工具: {name}"
            else:
                try:
                    output = handler(args)
                except Exception as e:
                    output = f"工具执行失败: {e}"
            print(f"[命令输出]: {output}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

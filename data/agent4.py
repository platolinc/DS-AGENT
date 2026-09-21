import os
import subprocess
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# ---------- RAG: 预置检索文档----------
# 真实 RAG 会走向量库检索；此处写死字符串，命中关键词时注入提示词
RAG_GREETING_DOC = """【RAG检索文档】
文件路径: D:/hello.py
说明：运行后可打印问候语与当前时间，支持命令行参数 --name 指定称呼。
"""

# 关键词
RAG_KEYWORDS = {
    "问候语": RAG_GREETING_DOC,
}


def rag_retrieve(user_text: str) -> str:
    """
    RAG 检索：若用户输入包含特定关键词，返回对应预置文档。
    多个关键词命中时，文档按出现顺序拼接。
    """
    chunks = []
    for keyword, doc in RAG_KEYWORDS.items():
        if keyword in user_text:
            chunks.append(doc.strip())
    if not chunks:
        return ""
    return "\n\n".join(chunks)


def build_user_message_with_rag(user_text: str) -> str:
    """把 RAG 命中的文档追加到用户问题后面。"""
    retrieved = rag_retrieve(user_text)
    if not retrieved:
        return user_text
    return (
        f"{user_text}\n\n"
        f"--- 以下为RAG检索注入的知识 ---\n"
        f"{retrieved}"
    )

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


    # 增加 RAG 检索
    user_content = build_user_message_with_rag(user_input)
    if user_content != user_input:
        print("[RAG:]命中关键词，已将检索文档注入用户信息")

    messages.append({"role": "user", "content": user_content})

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

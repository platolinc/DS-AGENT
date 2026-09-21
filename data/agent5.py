import os
import subprocess
import json
import math

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 对话模型（DeepSeek）
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 向量模型（硅基流动，免费 embedding）
embed_client = OpenAI(
    api_key=os.environ.get('SILICONFLOW_API_KEY'),
    base_url="https://api.siliconflow.cn/v1")
EMBED_MODEL = "BAAI/bge-m3"

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

# ---------- RAG: 知识库文档 ----------
# 真实场景中这些文档来自文件/数据库，此处直接写在代码里
RAG_DOCS = [
    """文件路径: D:/hello.py
说明：问候语脚本，运行后可打印问候语与当前时间，支持命令行参数 --name 指定称呼。
示例：python D:/hello.py --name 小明""",

    """项目部署说明：本项目使用 Python 3.12，依赖见 requirements.txt。
启动命令：python data/agent5.py。环境变量配置在 data/.env 文件中。""",

    """常见问题：DeepSeek 接口返回 402 表示余额不足，需要在 platform.deepseek.com 充值。
返回 401 表示 API key 错误或缺失，检查 .env 中的 DEEPSEEK_API_KEY。""",
]


def split_chunks(docs: list[str]) -> list[str]:
    """文档切分：按段落切分并去除空白。真实场景会按 token 长度重叠切分。"""
    chunks = []
    for doc in docs:
        for para in doc.split("\n\n"):
            para = para.strip()
            if para:
                chunks.append(para)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 embedding API，把文本批量转成向量。"""
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：越接近 1 语义越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# 启动时：切分文档并一次性向量化，缓存在内存中
print("[RAG] 正在初始化知识库向量...")
RAG_CHUNKS = split_chunks(RAG_DOCS)
RAG_VECTORS = embed_texts(RAG_CHUNKS)
print(f"[RAG] 知识库就绪，共 {len(RAG_CHUNKS)} 个分块\n")


def rag_retrieve(user_text: str, top_k: int = 2, threshold: float = 0.4) -> str:
    """
    RAG 检索：把用户问题向量化，与所有分块算余弦相似度，
    返回最相似的 top_k 个分块（低于阈值则丢弃）。
    """
    query_vec = embed_texts([user_text])[0]
    scored = [
        (cosine_similarity(query_vec, vec), chunk)
        for vec, chunk in zip(RAG_VECTORS, RAG_CHUNKS)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    hits = [chunk for score, chunk in scored[:top_k] if score >= threshold]
    print(f"[RAG] 命中 {len(hits)} 个分块，最高分: {scored[0][0]:.3f}")
    return "\n\n".join(hits)


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


# 历史记忆
messages = [
    {"role": "system", "content": "你是一个聊天机器人，每次回复完都要滴一声"},
]

while True:
    user_input = input("[User Input]: ").strip()
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": build_user_message_with_rag(user_input)})

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

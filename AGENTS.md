# AGENTS.md

本文件面向所有 AI 编程助手（Trae / Cursor / Codex / Claude Code 等），描述本项目的约定。

## 项目简介

DeepSeek Agent 学习项目：从零手写 agent 核心机制，逐步演进。

- `data/agent1.py` — 最简单单轮对话
- `data/agent2.py` — 多轮对话（messages 历史记忆）
- `data/agent3.py` — tool calling（ReAct 循环 + run_command 工具）
- `data/agent4.py` — 关键词版 RAG（演示用）
- `data/agent5.py` — 真实 RAG（embedding + 余弦相似度，向量在内存中）

## 环境

- Python 3.12（64 位），安装在 `D:\Python312`
- 虚拟环境在项目根 `.venv`
- 密钥放在 `data/.env`（`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`），**绝不提交**
- 对话模型：DeepSeek（`api.deepseek.com`）；embedding：硅基流动 `BAAI/bge-m3`

## 运行

```powershell
.venv\Scripts\python.exe data\agent5.py
```

## 代码约定

- 每个 agentN.py 是一个自包含的演进阶段，不抽公共模块，保持教学可读性
- 注释用中文，风格简洁
- 对话历史用标准 OpenAI messages 格式；tool 结果必须以 `role: "tool"` + 对应 `tool_call_id` 回传，且 append 在工具循环内部
- RAG 检索逻辑与 agent 主循环解耦，便于将来替换为 Chroma 等向量库

## Git

- 提交信息格式：`Add agentN: 简述`
- `.venv/`、`.env`、`__pycache__/` 不提交

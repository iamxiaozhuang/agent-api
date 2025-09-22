# fast_agent.py
from datetime import datetime
import os
import pytz
from fastapi import FastAPI, Request
from cachetools import TTLCache
from pydantic import BaseModel, SecretStr
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient


app = FastAPI()

# -----------------------------
# MCP Adapter 连接 MCP Server
# -----------------------------
mcp_client = MultiServerMCPClient({
    "IOC_MCP_Server": {
        "transport": "streamable_http",
        "url": "http://localhost:8001/mcp",
    }
})

# -----------------------------
# 通义千问 LLM（OpenAI 接口兼容方式）
# -----------------------------
QWEN_API_KEY: str | None = os.environ.get("QWEN_API_KEY")
if QWEN_API_KEY is None:
    raise ValueError("千问大模型的API_KEY环境变量未配置！")

llm = ChatOpenAI(
    model="qwen3-coder-flash",
    temperature=0,
    api_key=SecretStr(QWEN_API_KEY),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# -----------------------------
# TTLCache 管理每用户 Agent
# -----------------------------
user_agents = TTLCache(maxsize=1000, ttl=3600)

# 获取当前 Asia/Shanghai 日期
today = datetime.now(pytz.timezone("Asia/Shanghai")).date()
today_str = today.isoformat()

SYSTEM_PROMPT = """
你是一个智能接单助手。
你的功能包括两部分：
1. 发单（创建服务供给单或需求单，列表展示、选择服务单、修改、删除）。
2. 接单（将自己选择的服务单与其他人发布的服务单进行匹配，并展示匹配列表，用户可向对方发起沟通）。
业务规则：
你提供服务匹配功能是帮助用户接单，用“接单”而不用“匹配”让用户理解。
如果用户输入违法乱纪的内容，请直接拒绝并提示用户。
如果用户未选择服务单，请直接提示用户先从服务单列表种选择一个服务单，不能直接接单。
如果用户服务单列表为空，请直接提示用户先创建服务单（发单），不能直接接单。
"""


# -----------------------------
# 获取或创建用户 Agent
# -----------------------------
async def get_or_create_agent(user_id: str):
    if user_id not in user_agents:
        # 获取 MCP tools
        mcp_tools = await mcp_client.get_tools()
        # 直接返回执行工具的结果
        for t in mcp_tools:
            t.return_direct = True

        # 用 LangGraph 的预置 React Agent
        agent = create_react_agent(
            model=llm,                   
            tools=mcp_tools,
            prompt=f"{SYSTEM_PROMPT}\n当前用户ID: {user_id}\n当前日期: {today_str}",
            name=f"agent_{user_id}"
        )

        # 存储在 TTLCache
        user_agents[user_id] = agent
    else:
        # 刷新 TTL
        user_agents[user_id] = user_agents[user_id]

    return user_agents[user_id]

class UserRequest(BaseModel):
    user_message: str
    user_ID: str

# -----------------------------
# FastAPI 路由
# -----------------------------
@app.post("/test")
async def test_api(request: UserRequest):

    agent = await get_or_create_agent(request.user_ID)
    response_text = ""

    # 推荐直接用 ainvoke 拿完整响应；如果要流式再换成 astream
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request.user_message}]})
    response_text = result["messages"][-1].content  # 最后一条回复

    # # 这里 agent 是一个 Graph，可以直接调用 astream
    # inputs = {"messages": [("user", user_text)]}
    # async for event in agent.astream(inputs, stream_mode="values"):
    #     response_text = event["messages"][-1].content  # 最后一条回复

    return {"AI_message": response_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_agent:app", host="0.0.0.0", port=8000)

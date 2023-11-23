import os
from typing import Awaitable, Callable
from openai import AsyncOpenAI

MODEL_GPT_4_VISION = "gpt-4-vision-preview"

# 设置 OPENAI_API_BASE 环境变量
os.environ["OPENAI_API_BASE"] = "https://ai.fakeopen.com"  # 将 "https://your-proxy-url.com" 替换为实际的代理 URL

async def stream_openai_response(
    messages, api_key: str, callback: Callable[[str], Awaitable[None]]
):
    client = AsyncOpenAI(api_key=api_key)

    model = MODEL_GPT_4_VISION

    # Base parameters
    params = {"model": model, "messages": messages, "stream": True, "timeout": 600}

    # Add 'max_tokens' only if the model is a GPT4 vision model
    if model == MODEL_GPT_4_VISION:
        params["max_tokens"] = 4096
        params["temperature"] = 0

    completion = await client.chat.completions.create(**params)
    full_response = ""
    async for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        full_response += content
        await callback(content)

    return full_response

"""DeepSeek API 客户端 —— 使用 openai 库（OpenAI 兼容协议）。"""
from __future__ import annotations
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


def chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """单轮对话，返回文本响应。失败时区分错误类型并返回空字符串。"""
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except AuthenticationError as e:
        print(f"[llm] 认证失败，请检查 DEEPSEEK_API_KEY: {e}")
        return ""
    except RateLimitError as e:
        print(f"[llm] 速率限制，建议稍后重试: {e}")
        return ""
    except APIConnectionError as e:
        print(f"[llm] 网络连接失败: {e}")
        return ""
    except APIError as e:
        print(f"[llm] API 错误 (status={getattr(e, 'status_code', '?')}): {e}")
        return ""
    except Exception as e:
        print(f"[llm] 未预期错误: {type(e).__name__}: {e}")
        return ""


def _extract_json(text: str) -> str | None:
    """从 LLM 响应中提取第一个完整 JSON 对象（平衡括号匹配）。"""
    import re
    # 优先匹配 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return m.group(1)
    # 平衡括号提取：从第一个 { 开始，计数匹配 }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def chat_json(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict:
    """调用 LLM 并尝试解析 JSON 响应。失败返回空 dict。"""
    text = chat(system_prompt, user_message, temperature, max_tokens)
    if not text:
        return {}

    extracted = _extract_json(text)
    if extracted is None:
        print(f"[llm] No JSON found in response: {text[:200]}...")
        return {}

    try:
        import json
        return json.loads(extracted)
    except json.JSONDecodeError:
        print(f"[llm] Failed to parse JSON: {extracted[:200]}...")
        return {}

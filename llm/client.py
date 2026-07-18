"""DeepSeek API 客户端 —— 使用 openai 库（OpenAI 兼容协议）。"""
from __future__ import annotations
from openai import OpenAI
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
    """单轮对话，返回文本响应。"""
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
    except Exception as e:
        print(f"[llm] API call failed: {e}")
        return ""


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

    # 尝试提取 JSON 块
    import re
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    try:
        import json
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[llm] Failed to parse JSON from response: {text[:200]}...")
        return {}

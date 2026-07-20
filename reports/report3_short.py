"""报告三：做空审查 —— 隔离输入 + 全量技术指标 + LLM 五维尽调。

现在传给 LLM 的是 compute_summary() 的完整结构化摘要（趋势/动量/成交量/波动率/统计），
让 LLM 从技术面有充分数据支撑做空判断，而非只能看价格和市值。
"""
from __future__ import annotations
from datetime import datetime
from llm.client import chat_json
from llm.prompts import REPORT3_SYSTEM, REPORT3_USER_TEMPLATE
from data.indicators import format_summary_for_llm


def generate_report3(
    picks: list[dict],
    llm_enabled: bool = True,
) -> dict:
    """生成报告三。

    关键约束：只传股票代码+价格+全量技术指标给 LLM，不传报告一/二的看多分析。

    Args:
        picks: 报告二的选股列表（每只含 code, name, close, summary 等字段）

    Returns:
        结构化报告数据
    """
    if not picks:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stocks": [],
            "systemic_risk": "无候选股，无需审查",
        }

    # --- 构建隔离输入（传全量技术指标，不传看多信息） ---
    stock_list_lines = []
    for p in picks:
        parts = [
            f"- {p.get('name', p['code'])}（{p['code']}）",
            f"现价: ¥{p['close']:.2f}",
            f"涨停价: ¥{p['limit_up_price']:.2f}",
            f"市值: {p['market_cap']}亿",
            f"{'20cm' if p.get('is_20cm') else '10cm'}",
            format_summary_for_llm(p.get("summary", {})),
            f"日涨跌: {p['daily_return']:+.2f}% 连涨: {p['consecutive_up']}日",
        ]
        stock_list_lines.append(" | ".join(parts))

    result = {"stocks": [], "systemic_risk": ""}

    if llm_enabled:
        prompt = REPORT3_USER_TEMPLATE.format(
            stock_list="\n".join(stock_list_lines),
        )
        llm_result = chat_json(REPORT3_SYSTEM, prompt)
        result["stocks"] = llm_result.get("stocks", [])
        result["systemic_risk"] = llm_result.get("systemic_risk", "")
    else:
        for p in picks:
            result["stocks"].append({
                "code": p["code"],
                "name": p.get("name", p["code"]),
                "signals": [],
                "signal_count": 0,
                "level": "—",
                "verdict": "LLM 未启用，跳过做空审查",
                "remove_recommendation": False,
            })
        result["systemic_risk"] = ""

    result["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result

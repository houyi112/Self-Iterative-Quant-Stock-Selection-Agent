"""报告三：做空审查 —— 隔离输入 + 全量技术指标 + LLM 五维尽调。

现在传给 LLM 的是 compute_summary() 的完整结构化摘要（趋势/动量/成交量/波动率/统计），
让 LLM 从技术面有充分数据支撑做空判断，而非只能看价格和市值。
"""
from __future__ import annotations
from datetime import datetime
from llm.client import chat_json
from llm.prompts import REPORT3_SYSTEM, REPORT3_USER_TEMPLATE


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
        summary = p.get("summary", {})
        trend = summary.get("trend", {})
        mom = summary.get("momentum", {})
        vol = summary.get("volume", {})
        vola = summary.get("volatility", {})
        sig = summary.get("signal", {})

        # 基础信息
        parts = [
            f"- {p.get('name', p['code'])}（{p['code']}）",
            f"现价: ¥{p['close']:.2f}",
            f"涨停价: ¥{p['limit_up_price']:.2f}",
            f"市值: {p['market_cap']}亿",
            f"{'20cm' if p.get('is_20cm') else '10cm'}",
        ]
        # 趋势
        parts.append(f"MA: {p['ma5']:.2f}/{p['ma10']:.2f}/{p['ma20']:.2f}({trend.get('ma_alignment', '?')})")
        parts.append(f"MACD: {trend.get('macd_signal', '?')} DIF={trend.get('macd_dif', 0):.3f} DEA={trend.get('macd_dea', 0):.3f}")
        parts.append(f"ADX: {trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) PSAR: {trend.get('psar_signal', '?')}")
        parts.append(f"Aroon: ↑{trend.get('aroon_up', 0):.0f} ↓{trend.get('aroon_down', 0):.0f}")
        # 动量
        parts.append(f"RSI: 6={mom.get('rsi_6', 0):.0f}/14={mom.get('rsi_14', 0):.0f}({mom.get('rsi_regime', '?')})")
        parts.append(f"KDJ: K{mom.get('stoch_k', 0):.0f}/D{mom.get('stoch_d', 0):.0f}/J{mom.get('stoch_j', 0):.0f}({mom.get('stoch_signal', '?')})")
        parts.append(f"CCI: {mom.get('cci_14', 0):.0f} MFI: {mom.get('mfi_14', 0):.0f} WillR: {mom.get('willr_14', 0):.0f} UO: {mom.get('uo', 0):.0f}")
        # 成交量
        parts.append(f"量比: {vol.get('vol_ratio', 1):.2f} OBV趋势: {vol.get('obv_trend', '?')} ForceIdx: {vol.get('force_idx', 0):.0f}")
        # 波动率
        parts.append(f"BOLL: 上{vola.get('bb_upper', 0):.2f}/下{vola.get('bb_lower', 0):.2f} pos={vola.get('bb_position', 0.5):.2f}{' 收窄' if vola.get('bb_squeeze') else ''}")
        parts.append(f"ATR: {vola.get('atr_14', 0):.2f}({vola.get('atr_pct', 0):.1f}%) 历史波动率: {vola.get('hist_vol_20', 0):.2f}")
        # 信号
        parts.append(f"趋势偏向: {sig.get('trend_bias', '?')} 动量偏向: {sig.get('momentum_bias', '?')}")
        parts.append(f"量质: {sig.get('volume_quality', '?')} 风险: {sig.get('risk_warning', '?')}")
        # 日涨跌
        parts.append(f"日涨跌: {p['daily_return']:+.2f}% 连涨: {p['consecutive_up']}日")

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

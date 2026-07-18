"""报告二：涨停潜力股 —— 硬过滤 + LLM 全量判断。

screener 只做硬过滤（价格/市值/数据量），不做评分。
所有候选股全量交给 LLM，LLM 自行判断哪些能涨停、概率多少。
"""
from __future__ import annotations
from datetime import datetime
from config import SECTOR_STOCKS, ALL_STOCK_CODES, INDEX_TO_SECTOR, STOCK_NAMES
from engine.screener import screen_candidates
from data.fetcher import batch_update
from llm.client import chat_json
from llm.prompts import REPORT2_SYSTEM, REPORT2_USER_TEMPLATE


def generate_report2(
    leading_sectors: list[str],
    ohlcv_cache: dict[str, list[dict]],
    llm_enabled: bool = True,
) -> dict:
    """生成报告二。

    硬过滤后全量交给 LLM，LLM 自行判断哪些值得推荐、概率多少。

    Args:
        leading_sectors: 报告一确认的领涨板块名称列表
        ohlcv_cache: 所有股票的 OHLCV 缓存

    Returns:
        结构化报告数据
    """
    top_sectors = leading_sectors[:6]

    # --- 硬过滤：价格<15、市值50-500亿、数据≥60天 ---
    all_candidates = []
    for sector in top_sectors:
        pool_name = INDEX_TO_SECTOR.get(sector, sector)
        codes = SECTOR_STOCKS.get(pool_name, [])
        candidates = screen_candidates(sector, codes, ohlcv_cache)
        for c in candidates:
            c["name"] = STOCK_NAMES.get(c["code"], c["code"])
        all_candidates.extend(candidates)

    # 去重（同一只股票可能出现在多个板块）
    seen = set()
    unique = []
    for c in all_candidates:
        if c["code"] not in seen:
            seen.add(c["code"])
            unique.append(c)

    # --- LLM：全量候选股分析，自行决定哪些推荐 ---
    if llm_enabled and unique:
        picks = _get_llm_picks(top_sectors, unique)
    else:
        picks = []
        for c in unique:
            c["thesis"] = "（LLM 未启用）"
            c["limit_up_probability"] = 0
            picks.append(c)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors_covered": "、".join(top_sectors),
        "total_candidates": len(unique),
        "picks": picks,
    }


def _get_llm_picks(sectors: list[str], candidates: list[dict]) -> list[dict]:
    """把所有候选股的全量指标交给 LLM，LLM 自行挑出涨停潜力股。"""
    stock_lines = []
    for c in candidates:
        summary = c.get("summary", {})
        trend = summary.get("trend", {})
        mom = summary.get("momentum", {})
        vol = summary.get("volume", {})
        vola = summary.get("volatility", {})
        sig = summary.get("signal", {})

        parts = [
            f"- {c['code']}（{c.get('name', '')}）",
            f"板块: {c['sector']}",
            f"现价: ¥{c['close']:.2f}",
            f"市值: {c['market_cap']}亿",
            f"{'20cm' if c['is_20cm'] else '10cm'}",
        ]
        # 趋势
        parts.append(f"MA: {c['ma5']:.2f}/{c['ma10']:.2f}/{c['ma20']:.2f}({trend.get('ma_alignment', '?')})")
        parts.append(f"MACD: {trend.get('macd_signal', '?')}(DIF{trend.get('macd_dif', 0):.3f})")
        parts.append(f"ADX: {trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) PSAR: {trend.get('psar_signal', '?')}")
        # 动量
        parts.append(f"RSI: 6={mom.get('rsi_6', 0):.0f}/14={mom.get('rsi_14', 0):.0f}({mom.get('rsi_regime', '?')})")
        parts.append(f"KDJ: K{mom.get('stoch_k', 0):.0f}/D{mom.get('stoch_d', 0):.0f}/J{mom.get('stoch_j', 0):.0f}")
        parts.append(f"CCI: {mom.get('cci_14', 0):.0f} MFI: {mom.get('mfi_14', 0):.0f}")
        # 成交量
        parts.append(f"量比: {vol.get('vol_ratio', 1):.2f} OBV: {vol.get('obv_trend', '?')} Force: {vol.get('force_idx', 0):.0f}")
        # 波动率
        parts.append(f"BOLL: pos={vola.get('bb_position', 0.5):.2f}{' 收窄' if vola.get('bb_squeeze') else ''}")
        parts.append(f"ATR%: {vola.get('atr_pct', 0):.1f}%")
        # 信号
        parts.append(f"趋势: {sig.get('trend_bias', '?')} 动量: {sig.get('momentum_bias', '?')} 量质: {sig.get('volume_quality', '?')}")
        # 日涨跌
        parts.append(f"日涨跌: {c['daily_return']:+.2f}% 连涨: {c['consecutive_up']}日")

        stock_lines.append(" | ".join(parts))

    prompt = REPORT2_USER_TEMPLATE.format(
        leading_sectors="、".join(sectors),
        stock_data="\n".join(stock_lines),
    )

    result = chat_json(REPORT2_SYSTEM, prompt)
    llm_picks = result.get("picks", [])

    # 把 LLM 返回的 thesis/probability 合并回原始候选股数据
    final_picks = []
    for lp in llm_picks:
        code = lp.get("code", "")
        # 找到原始候选股数据
        match = next((c for c in candidates if c["code"] == code), None)
        if match:
            match["thesis"] = lp.get("thesis", "—")
            match["limit_up_probability"] = lp.get("limit_up_probability", 0)
            final_picks.append(match)

    return final_picks

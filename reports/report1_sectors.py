"""报告一：明日领涨板块预判 —— 全指标直接交给 LLM 分析排序。

不硬编码打分。把所有板块的 compute_summary() 完整摘要 + 风险评分交给 LLM，
让 LLM 自行判断哪些板块领涨、为什么领涨、概率多少。
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
from config import SECTOR_INDICES, TRACKED_INDICES
from data.indicators import compute_all, compute_summary, last_val, format_summary_for_llm
from engine.risk_scorer import score_all_sectors
from llm.client import chat, chat_json
from llm.prompts import REPORT1_SYSTEM, REPORT1_USER_TEMPLATE

# 板块名称 → 用于排名
SECTOR_NAMES = list(SECTOR_INDICES.values())


def generate_report1(
    sector_data: dict[str, list[dict]],
    market_context: dict | None = None,
    llm_enabled: bool = True,
    precomputed_indicators: dict | None = None,
    analyzer=None,
) -> dict:
    """生成报告一。

    核心思路：把所有板块的全量指标 + 风险评分直接交给分析器。

    Args:
        sector_data: {sector_name: ohlcv_list} 所有指数数据（含市场和板块）
        market_context: 市场上下文
        llm_enabled: 是否启用 LLM（兼容旧参数，实际由 analyzer 决定）
        precomputed_indicators: {name: DataFrame} 预计算的指标
        analyzer: BaseAnalyzer 实例，默认自动获取

    Returns:
        结构化报告数据
    """
    market_context = market_context or {}
    precomputed = precomputed_indicators or {}

    # --- 1. 计算所有板块的全部指标 ---
    all_indicators = {}
    all_summaries = {}
    for name, ohlcv in sector_data.items():
        if ohlcv and len(ohlcv) > 20:
            ind = precomputed[name] if name in precomputed else compute_all(ohlcv)
            all_indicators[name] = ind
            all_summaries[name] = compute_summary(ohlcv, indicators=ind)
        else:
            all_indicators[name] = {}
            all_summaries[name] = {}

    # --- 2. 风险评分 ---
    risk_scores = score_all_sectors(sector_data, market_context)

    # --- 3. 构建板块数据列表（只做数据整理，不打分） ---
    sectors = []
    for name in SECTOR_NAMES:
        ohlcv = sector_data.get(name, [])
        ind = all_indicators.get(name, {})
        summary = all_summaries.get(name, {})
        risk = risk_scores.get(name, {})

        change = last_val(ind, "daily_return", 0.0)
        cons_up = int(last_val(ind, "consecutive_up_days", 0))
        vr = last_val(ind, "volume_ratio", 1.0)

        sectors.append({
            "name": name,
            "daily_return": round(change, 2),
            "consecutive_up": cons_up,
            "volume_ratio": round(vr, 2),
            "risk_level": risk.get("level", "未知"),
            "risk_total": risk.get("total", 0),
            "risk_scores": risk,
            "summary": summary,
            "indicators": ind,
        })

    # --- 4. LLM：分析全部板块，自行排序 + 叙事 ---
    if analyzer is None:
        from engine.analyzer import get_analyzer
        analyzer = get_analyzer()

    if llm_enabled:
        llm_result = analyzer.rank_sectors(sectors, sector_data)
        rankings = llm_result.get("rankings", [])
        market_narrative = llm_result.get("market_narrative", "")

        if not rankings:
            # LLM 失败，降级为涨幅排序
            print("  ⚠️ LLM 返回空 rankings，降级为当日涨幅排序")
            market_narrative = ""
            sectors.sort(key=lambda x: x["daily_return"], reverse=True)
            for i, s in enumerate(sectors):
                s["probability"] = max(5, 50 - i * 3)
                s["narrative"] = {
                    "driver_logic": "（LLM 不可用，按涨幅排序）",
                    "sustainability": "—",
                    "key_risk": "—",
                }
        else:
            # 将 LLM 返回的 narrative 合并回 sector
            for s in sectors:
                match = next((r for r in rankings if r.get("name") == s["name"]), {})
                s["probability"] = match.get("probability", 0)
                s["narrative"] = {
                    "driver_logic": match.get("driver_logic", ""),
                    "sustainability": match.get("sustainability", ""),
                    "key_risk": match.get("key_risk", ""),
                }
            # 按 LLM 给的 probability 排序
            sectors.sort(key=lambda x: x.get("probability", 0), reverse=True)
    else:
        market_narrative = ""
        # 无 LLM 时按当日涨幅简单排序
        sectors.sort(key=lambda x: x["daily_return"], reverse=True)
        for i, s in enumerate(sectors):
            s["probability"] = max(5, 50 - i * 3)
            s["narrative"] = {
                "driver_logic": "（LLM 未启用）",
                "sustainability": "—",
                "key_risk": "—",
            }

    # --- 5. 市场概况 ---
    market_overview = {}
    for code, name in TRACKED_INDICES.items():
        ohlcv = sector_data.get(name)
        if ohlcv and len(ohlcv) >= 2:
            close = ohlcv[-1]["close"]
            prev_close = ohlcv[-2]["close"]
            change_pct = (close - prev_close) / prev_close * 100
            market_overview[name] = {"close": close, "change": round(change_pct, 2)}

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_overview": market_overview,
        "market_narrative": market_narrative,
        "rankings": sectors,
        "iching_comparison": "",  # 由 main.py 填充
    }


def _get_llm_analysis(sectors: list[dict], sector_data: dict) -> dict:
    """把所有板块的全量指标交给 LLM，让它自行排序分析。"""
    # 构建每个板块的完整指标文本
    sector_lines = []
    for s in sectors:
        risk_items = []
        for fid, score in s.get("risk_scores", {}).items():
            if fid not in ("total", "level") and score > 0:
                risk_items.append(f"{fid}=+{score}")
        risk_str = ", ".join(risk_items) if risk_items else "无"

        parts = [
            f"**{s['name']}**",
            f"涨跌{s['daily_return']:+.2f}%",
            f"连涨{s['consecutive_up']}日",
            f"风险: {s['risk_level']}({s['risk_total']}分) [{risk_str}]",
            format_summary_for_llm(s.get("summary", {})),
        ]
        sector_lines.append(" | ".join(parts))

    # 大盘
    def _chg(name):
        ohlcv = sector_data.get(name)
        if ohlcv and len(ohlcv) >= 2:
            return (ohlcv[-1]["close"] - ohlcv[-2]["close"]) / ohlcv[-2]["close"] * 100
        return 0.0

    sh = sector_data.get("上证指数", [{}])[-1]
    sz = sector_data.get("深证成指", [{}])[-1]
    cy = sector_data.get("创业板指", [{}])[-1]

    prompt = REPORT1_USER_TEMPLATE.format(
        sh_close=sh.get("close", 0),
        sh_change=_chg("上证指数"),
        sz_close=sz.get("close", 0),
        sz_change=_chg("深证成指"),
        cy_close=cy.get("close", 0),
        cy_change=_chg("创业板指"),
        all_sectors="\n\n".join(sector_lines),
    )

    result = chat_json(REPORT1_SYSTEM, prompt)
    return result

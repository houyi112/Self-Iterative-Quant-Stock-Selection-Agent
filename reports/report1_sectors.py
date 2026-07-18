"""报告一：明日领涨板块预判 —— 全指标直接交给 LLM 分析排序。

不硬编码打分。把所有板块的 compute_summary() 完整摘要 + 风险评分交给 LLM，
让 LLM 自行判断哪些板块领涨、为什么领涨、概率多少。
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
from config import SECTOR_INDICES, TRACKED_INDICES
from data.indicators import compute_all, compute_summary
from engine.risk_scorer import score_all_sectors
from llm.client import chat, chat_json
from llm.prompts import REPORT1_SYSTEM, REPORT1_USER_TEMPLATE

# 板块名称 → 用于排名
SECTOR_NAMES = list(SECTOR_INDICES.values())


def _last_val(ind, col: str, default=0.0):
    """从 DataFrame 或 dict 中安全提取最后一个标量值。"""
    if isinstance(ind, pd.DataFrame) and col in ind.columns:
        val = ind[col].iloc[-1]
        return default if pd.isna(val) else float(val)
    if isinstance(ind, dict):
        val = ind.get(col, [None])
        if isinstance(val, list):
            v = val[-1] if val else None
            return default if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
        return default if val is None else val
    return default


def generate_report1(
    sector_data: dict[str, list[dict]],
    market_context: dict | None = None,
    llm_enabled: bool = True,
) -> dict:
    """生成报告一。

    核心思路：把所有板块的全量指标 + 风险评分直接交给 LLM，
    LLM 自行排序并给出领涨概率和驱动逻辑，不做硬编码打分。

    Args:
        sector_data: {sector_name: ohlcv_list} 所有指数数据（含市场和板块）
        market_context: 市场上下文
        llm_enabled: 是否启用 LLM

    Returns:
        结构化报告数据
    """
    market_context = market_context or {}

    # --- 1. 计算所有板块的全部指标 ---
    all_indicators = {}
    all_summaries = {}
    for name, ohlcv in sector_data.items():
        if ohlcv and len(ohlcv) > 20:
            all_indicators[name] = compute_all(ohlcv)
            all_summaries[name] = compute_summary(ohlcv)
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

        change = _last_val(ind, "daily_return", 0.0)
        cons_up = int(_last_val(ind, "consecutive_up_days", 0))
        vr = _last_val(ind, "volume_ratio", 1.0)

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
    if llm_enabled:
        llm_result = _get_llm_analysis(sectors, sector_data)
        rankings = llm_result.get("rankings", [])
        market_narrative = llm_result.get("market_narrative", "")
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
        summary = s.get("summary", {})
        trend = summary.get("trend", {})
        mom = summary.get("momentum", {})
        vol = summary.get("volume", {})
        vola = summary.get("volatility", {})
        sig = summary.get("signal", {})

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
            # 趋势
            f"MA={trend.get('ma_alignment', '?')} MA5/10/20/60={trend.get('ma5', 0):.1f}/{trend.get('ma10', 0):.1f}/{trend.get('ma20', 0):.1f}/{trend.get('ma60', 0):.1f}",
            f"MACD={trend.get('macd_signal', '?')} DIF={trend.get('macd_dif', 0):.2f} DEA={trend.get('macd_dea', 0):.2f}",
            f"ADX={trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) PSAR={trend.get('psar_signal', '?')}",
            f"Aroon ↑{trend.get('aroon_up', 0):.0f} ↓{trend.get('aroon_down', 0):.0f}",
            # 动量
            f"RSI6/14={mom.get('rsi_6', 0):.0f}/{mom.get('rsi_14', 0):.0f}({mom.get('rsi_regime', '?')})",
            f"KDJ=K{mom.get('stoch_k', 0):.0f} D{mom.get('stoch_d', 0):.0f} J{mom.get('stoch_j', 0):.0f}({mom.get('stoch_signal', '?')})",
            f"CCI={mom.get('cci_14', 0):.0f} MFI={mom.get('mfi_14', 0):.0f} WillR={mom.get('willr_14', 0):.0f} UO={mom.get('uo', 0):.0f}",
            # 成交量
            f"量比={vol.get('vol_ratio', 1):.2f} OBV={vol.get('obv_trend', '?')} ForceIdx={vol.get('force_idx', 0):.0f}",
            # 波动率
            f"BOLL pos={vola.get('bb_position', 0.5):.2f} {'收窄' if vola.get('bb_squeeze') else '正常'} 上{vola.get('bb_upper', 0):.1f}/下{vola.get('bb_lower', 0):.1f}",
            f"ATR%={vola.get('atr_pct', 0):.1f}% HVOL={vola.get('hist_vol_20', 0):.2f}",
            # 综合信号
            f"趋势偏向={sig.get('trend_bias', '?')} 动量偏向={sig.get('momentum_bias', '?')} 量质={sig.get('volume_quality', '?')} 风险={sig.get('risk_warning', '?')}",
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

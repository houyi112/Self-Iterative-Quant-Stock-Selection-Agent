"""Markdown 报告格式化输出。"""
from __future__ import annotations
from datetime import date
from pathlib import Path
from config import OUTPUT_DIR


def _ensure_dir(date_str: str) -> Path:
    d = OUTPUT_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# 报告一：领涨板块预判
# ============================================================

def write_report1(date_str: str, data: dict) -> str:
    """输出报告一：明日领涨板块预判。"""
    d = _ensure_dir(date_str)
    path = d / "report1_领涨预判.md"

    lines = [
        f"# 📊 报告一：明日领涨板块预判",
        f"",
        f"**日期：** {date_str}",
        f"**生成时间：** {data.get('generated_at', '')}",
        f"",
        f"---",
        f"",
        f"## 今日市场概况",
        f"",
    ]

    market = data.get("market_overview", {})
    for idx_name, vals in market.items():
        lines.append(f"- **{idx_name}**：{vals.get('close', 'N/A'):.2f}（{vals.get('change', 0):+.2f}%）")

    # 市场叙事
    market_narrative = data.get("market_narrative", "")
    if market_narrative:
        lines.extend([
            f"",
            f"> {market_narrative}",
        ])

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 板块领涨概率排序",
        f"",
    ])

    for i, sector in enumerate(data.get("rankings", []), 1):
        name = sector.get("name", "未知")
        prob = sector.get("probability", 0)
        level = sector.get("risk_level", "未知")
        scores = sector.get("risk_scores", {})

        lines.append(f"### {i}. {name} — 领涨概率 {prob}%")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 风险等级 | **{level}**（{scores.get('total', 0)}分）|")

        # 风险因子
        for fid, score in scores.items():
            if fid == "total" or fid == "level":
                continue
            if score > 0:
                lines.append(f"| ⚠️ {fid} | +{score} |")

        lines.append(f"| 当日涨跌 | {sector.get('daily_return', 0):+.2f}% |")
        lines.append(f"| 连涨天数 | {sector.get('consecutive_up', 0)} 日 |")
        lines.append(f"| 量比 | {sector.get('volume_ratio', 1):.2f} |")

        # 全量指标摘要
        summary = sector.get("summary", {})
        if summary:
            trend = summary.get("trend", {})
            mom = summary.get("momentum", {})
            vol = summary.get("volume", {})
            vola = summary.get("volatility", {})
            sig = summary.get("signal", {})
            lines.append(f"| MA排列 | {trend.get('ma_alignment', '?')} MA5={trend.get('ma5', 0):.1f} MA20={trend.get('ma20', 0):.1f} |")
            lines.append(f"| MACD | {trend.get('macd_signal', '?')} DIF={trend.get('macd_dif', 0):.2f} ADX={trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) |")
            lines.append(f"| RSI | 6={mom.get('rsi_6', 0):.0f} / 14={mom.get('rsi_14', 0):.0f} ({mom.get('rsi_regime', '?')}) |")
            lines.append(f"| KDJ | K{mom.get('stoch_k', 0):.0f} D{mom.get('stoch_d', 0):.0f} J{mom.get('stoch_j', 0):.0f} |")
            lines.append(f"| CCI / MFI | {mom.get('cci_14', 0):.0f} / {mom.get('mfi_14', 0):.0f} |")
            lines.append(f"| BOLL | pos={vola.get('bb_position', 0.5):.2f}{' 收窄' if vola.get('bb_squeeze') else ''} ATR%={vola.get('atr_pct', 0):.1f}% |")
            lines.append(f"| 量质 | {sig.get('volume_quality', '?')} OBV={vol.get('obv_trend', '?')} |")

        narrative = sector.get("narrative", {})
        if narrative:
            lines.extend([
                f"",
                f"**驱动逻辑：** {narrative.get('driver_logic', '')}",
                f"",
                f"**可持续性：** {narrative.get('sustainability', '未知')}",
                f"",
                f"**关键风险：** {narrative.get('key_risk', '')}",
            ])
        lines.append(f"")

    # 梅花易数对比
    iching = data.get("iching_comparison", "")
    if iching:
        lines.extend([
            f"---",
            f"",
            f"## ☯️ 量化+易学共振",
            f"",
            f"{iching}",
            f"",
        ])

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return str(path)


# ============================================================
# 报告二：涨停潜力股
# ============================================================

def write_report2(date_str: str, data: dict) -> str:
    """输出报告二：涨停潜力股。LLM 自行判断，不设固定得分。"""
    d = _ensure_dir(date_str)
    path = d / "report2_涨停潜力.md"

    picks = data.get("picks", [])
    total = data.get("total_candidates", len(picks))

    lines = [
        f"# 🎯 报告二：涨停潜力股（价格<15元）",
        f"",
        f"**日期：** {date_str}",
        f"**筛选范围：** {data.get('sectors_covered', '')}",
        f"**通过硬过滤：** {total} 只 → **LLM 推荐：** {len(picks)} 只",
        f"",
        f"---",
        f"",
    ]

    for i, pick in enumerate(picks, 1):
        code = pick.get("code", "")
        name = pick.get("name", code)
        close = pick.get("close", 0)
        limit_up = pick.get("limit_up_price", 0)
        cap = pick.get("market_cap", "N/A")
        prob = pick.get("limit_up_probability", 0)
        is_20cm = "20cm" if pick.get("is_20cm") else "10cm"

        # 概率标记
        if prob >= 20:
            prob_tag = f"🔥 {prob}%"
        elif prob >= 10:
            prob_tag = f"⚡ {prob}%"
        else:
            prob_tag = f"{prob}%"

        lines.extend([
            f"### {i}. {name}（{code}）{is_20cm}  |  涨停概率：{prob_tag}",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 现价 | ¥{close:.2f} |",
            f"| 涨停价 | ¥{limit_up:.2f} |",
            f"| 流通市值 | {cap}亿 |",
            f"| MA5 / MA10 / MA20 | {pick.get('ma5', 0):.2f} / {pick.get('ma10', 0):.2f} / {pick.get('ma20', 0):.2f} |",
            f"| 量比 | {pick.get('volume_ratio', 0):.2f} |",
            f"| RSI(14) | {pick.get('rsi', 0):.1f} |",
            f"| 当日涨幅 | {pick.get('daily_return', 0):+.2f}% |",
            f"| 连涨天数 | {pick.get('consecutive_up', 0)} 日 |",
        ])

        # 全量指标摘要
        summary = pick.get("summary", {})
        if summary:
            trend = summary.get("trend", {})
            mom = summary.get("momentum", {})
            vola = summary.get("volatility", {})
            sig = summary.get("signal", {})
            lines.extend([
                f"| MACD | {trend.get('macd_signal', '?')} DIF={trend.get('macd_dif', 0):.3f} |",
                f"| ADX / PSAR | {trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) / {trend.get('psar_signal', '?')} |",
                f"| KDJ | K{mom.get('stoch_k', 0):.0f} D{mom.get('stoch_d', 0):.0f} J{mom.get('stoch_j', 0):.0f} |",
                f"| CCI / MFI | {mom.get('cci_14', 0):.0f} / {mom.get('mfi_14', 0):.0f} |",
                f"| BOLL 位置 | {vola.get('bb_position', 0.5):.2f}{' 📍收窄' if vola.get('bb_squeeze') else ''} |",
                f"| ATR% / HVOL | {vola.get('atr_pct', 0):.1f}% / {vola.get('hist_vol_20', 0):.2f} |",
                f"| 综合信号 | 趋势{sig.get('trend_bias', '?')} 动量{sig.get('momentum_bias', '?')} {sig.get('volume_quality', '?')} |",
            ])

        lines.extend([
            f"",
            f"**LLM 判断：** {pick.get('thesis', '—')}",
            f"",
        ])

        # 买入区间和止损
        if close > 0:
            buy_low = round(close * 0.98, 2)
            stop_loss = round(close * 0.95, 2)
            lines.extend([
                f"| 操作 | 价格 |",
                f"|------|------|",
                f"| 买入区间 | ¥{buy_low:.2f} — ¥{close:.2f} |",
                f"| 止损位 | ¥{stop_loss:.2f} |",
                f"",
            ])

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return str(path)


# ============================================================
# 报告三：做空审查
# ============================================================

def write_report3(date_str: str, data: dict) -> str:
    """输出报告三：做空审查。"""
    d = _ensure_dir(date_str)
    path = d / "report3_做空审查.md"

    lines = [
        f"# 🔴 报告三：做空审查",
        f"",
        f"**日期：** {date_str}",
        f"**审查对象：** 报告二的 {len(data.get('stocks', []))} 只股票",
        f"",
        f"---",
        f"",
    ]

    # 系统性风险
    systemic = data.get("systemic_risk", "")
    if systemic:
        lines.extend([
            f"## 市场系统性风险",
            f"",
            f"{systemic}",
            f"",
            f"---",
            f"",
        ])

    for stock in data.get("stocks", []):
        code = stock.get("code", "")
        name = stock.get("name", code)
        level = stock.get("level", "🔴")
        verdict = stock.get("verdict", "")
        signals = stock.get("signals", [])
        remove = stock.get("remove_recommendation", False)

        remove_flag = " ⚠️ **建议剔除**" if remove else ""

        lines.extend([
            f"### {name}（{code}）做空等级：{level}{remove_flag}",
            f"",
            f"**判定：** {verdict}",
            f"",
            f"**做空信号：**",
        ])
        for s in signals:
            lines.append(f"- {s}")

        lines.append(f"")

    # 汇总
    removes = [s for s in data.get("stocks", []) if s.get("remove_recommendation")]
    if removes:
        lines.extend([
            f"---",
            f"",
            f"## ⚠️ 剔除建议汇总",
            f"",
        ])
        for s in removes:
            lines.append(f"- **{s.get('name', s.get('code', ''))}**（{s.get('code', '')}）：{s.get('verdict', '')}")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return str(path)


# ============================================================
# 报告四：复盘
# ============================================================

def write_report4(date_str: str, data: dict) -> str:
    """输出报告四：复盘摘要（硬编码统计 + LLM 洞察）。"""
    d = _ensure_dir(date_str)
    path = d / "report4_复盘摘要.md"
    summary = data.get("summary", "")
    if not summary:
        summary = f"# 复盘摘要\n\n生成时间：{data.get('generated_at', '')}\n\n（暂无复盘数据）"
    path.write_text(summary, encoding="utf-8")
    return str(path)


# ============================================================
# 梅花易数
# ============================================================

def write_iching(date_str: str, data: dict) -> str:
    """输出梅花易数报告。"""
    d = _ensure_dir(date_str)
    path = d / "iching_梅花易数.md"

    lines = [
        f"# ☯️ 梅花易数",
        f"",
        f"**日期：** {date_str}",
        f"",
        f"---",
        f"",
        f"## 卦象",
        f"",
        f"- **本卦：** {data.get('ben_gua_name', '')} {data.get('ben_gua_symbol', '')}",
        f"- **变卦：** {data.get('bian_gua_name', '')} {data.get('bian_gua_symbol', '')}",
        f"- **互卦：** {data.get('hu_gua_name', '')} {data.get('hu_gua_symbol', '')}",
        f"- **体卦：** {data.get('ti_gua_name', '')}（{data.get('ti_element', '')}）",
        f"- **用卦：** {data.get('yong_gua_name', '')}（{data.get('yong_element', '')}）",
        f"- **动爻：** 第{data.get('dong_yao', '')}爻",
        f"- **体用关系：** {data.get('ti_yong_relation', '')}",
        f"",
        f"---",
        f"",
        f"## 卦象解读",
        f"",
        f"{data.get('hexagram_interpretation', '')}",
        f"",
        f"**动爻启示：** {data.get('dong_yao_revelation', '')}",
        f"",
        f"**大盘判断：** {data.get('market_judgment', '')}",
        f"",
        f"---",
        f"",
        f"## 五行板块吉凶",
        f"",
        f"| 五行 | 吉凶 | 板块 | 理由 |",
        f"|------|------|------|------|",
    ]

    for row in data.get("five_element_table", []):
        lines.append(f"| {row.get('element', '')} | {row.get('rating', '')} | {row.get('sectors', '')} | {row.get('reason', '')} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 一言",
        f"",
        f"> {data.get('one_sentence', '')}",
        f"",
    ])

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return str(path)


def write_ganzhi(date_str: str, data: dict) -> str:
    """输出干支分析报告。"""
    d = _ensure_dir(date_str)
    path = d / "ganzhi_干支分析.md"

    lines = [
        f"# 🌿 干支分析（预测下一个交易日）",
        f"",
        f"**生成日期：** {date_str}",
        f"",
        f"---",
        f"",
        f"## 预测日干支",
        f"",
        f"**{data.get('day_ganzhi', '')}**",
        f"",
        f"- **体（日干）：** {data.get('ti', '')}",
        f"- **用（月支）：** {data.get('yong', '')}",
        f"- **体用关系：** {data.get('ti_yong_relation', '')}",
        f"",
    ]

    key_rels = data.get("key_relations", [])
    if key_rels:
        lines.append("## 关键关系")
        lines.append("")
        for r in key_rels:
            lines.append(f"- {r}")
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 大盘判断",
        f"",
        f"{data.get('market_judgment', '')}",
        f"",
        f"---",
        f"",
        f"## 五行板块吉凶",
        f"",
        f"| 五行 | 吉凶 | 板块 | 理由 |",
        f"|------|------|------|------|",
    ])

    for row in data.get("five_element_table", []):
        lines.append(f"| {row.get('element', '')} | {row.get('rating', '')} | {row.get('sectors', '')} | {row.get('reason', '')} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 一言",
        f"",
        f"> {data.get('one_sentence', '')}",
        f"",
    ])

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return str(path)

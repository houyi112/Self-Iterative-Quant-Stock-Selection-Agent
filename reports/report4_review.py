"""报告四：每日复盘 — 硬编码偏差 + 累积统计 + LLM 增量洞察。

架构：
  硬编码层（不调 LLM）：计算今日偏差 → 追加到 backtest.json → 重算累积统计
  LLM 层（有条件调用）：基于聚合统计 + 今日偏差，产出 1-3 条增量洞察
"""
from __future__ import annotations
import json
from datetime import datetime
from config import STATE_DIR, SECTOR_ELEMENT
from data.backtest import load_log, load_insights, append_insights
from llm.client import chat_json
from llm.prompts import REPORT4_SYSTEM, REPORT4_USER_TEMPLATE


# ---------- 偏差计算（硬编码） ----------

def _compute_deviation(prediction: dict | None, actual: dict) -> dict:
    """硬编码：计算预测与实际之间的偏差。"""
    if prediction is None:
        return {"status": "no_prediction", "message": "首次运行，无历史预测"}

    rankings = prediction.get("rankings", [])
    predicted_order = [s.get("name", "") for s in rankings]
    predicted_top = predicted_order[:3] if len(predicted_order) >= 3 else predicted_order

    actual_sorted = sorted(actual.items(), key=lambda x: x[1].get("change", -999), reverse=True)
    actual_top = [name for name, _ in actual_sorted[:3]]

    hits = list(set(predicted_top) & set(actual_top))
    misses = list(set(predicted_top) - set(actual_top))
    surprises = list(set(actual_top) - set(predicted_top))

    details = []
    total_err = 0.0
    for sector in predicted_order[:5]:
        act_data = actual.get(sector, {})
        act_change = act_data.get("change", 0) if isinstance(act_data, dict) else 0
        pred_prob = next((s.get("probability", 0) for s in rankings if s.get("name") == sector), 0)
        err = abs(pred_prob - act_change * 5)  # 标准化: 概率(0-55) vs 涨跌(-10~+10) → 乘以5
        total_err += err
        details.append({
            "sector": sector,
            "predicted_probability": pred_prob,
            "actual_change": round(act_change, 2),
            "error": round(err, 1),
        })

    return {
        "predicted_top3": predicted_top,
        "actual_top3": actual_top,
        "hits": hits,
        "misses": misses,
        "surprises": surprises,
        "hit_rate": f"{len(hits)}/3",
        "mae": round(total_err / len(details), 1) if details else 0,
        "details": details,
    }


def _compute_ganzhi_deviation(ganzhi: dict | None, actual: dict) -> dict | None:
    """对比干支五行吉凶预测与实际板块表现。"""
    if ganzhi is None:
        return None

    element_table = ganzhi.get("five_element_table", [])
    if not element_table:
        return None

    actual_sorted = sorted(actual.items(), key=lambda x: x[1].get("change", -999), reverse=True)
    actual_top = actual_sorted[:3] if len(actual_sorted) >= 3 else actual_sorted

    top_elements = {}
    for name, change in actual_top:
        el = SECTOR_ELEMENT.get(name, "?")
        top_elements[el] = top_elements.get(el, 0) + 1

    details = []
    hits = 0
    total = 0
    for row in element_table:
        el = row.get("element", "")
        predicted = row.get("rating", "—")
        in_top = top_elements.get(el, 0)
        match = False
        if predicted in ("吉", "大吉") and in_top > 0:
            match = True; hits += 1
        elif predicted in ("凶", "大凶") and in_top == 0:
            match = True; hits += 1
        elif predicted == "平":
            match = True
        total += 1
        details.append({
            "element": el,
            "predicted_rating": predicted,
            "predicted_reason": row.get("reason", ""),
            "actual_top3_presence": "领涨" if in_top > 0 else "未进TOP3",
            "match": match,
        })

    accuracy = f"{hits}/{total}" if total > 0 else "N/A"
    rate = hits / total if total > 0 else 0
    if rate >= 0.8: assessment = "准确"
    elif rate >= 0.5: assessment = "部分准确"
    else: assessment = "不准确"

    return {
        "accuracy": accuracy,
        "assessment": assessment,
        "details": details,
        "actual_top3_elements": {k: v for k, v in top_elements.items()},
    }


# ---------- 格式化（供 LLM prompt + 人类阅读） ----------

def _format_stats_for_llm(stats: dict) -> str:
    """将累积统计格式化为 LLM 可读文本。"""
    if not stats:
        return "（尚无累积统计）"

    lines = [
        f"- 总运行天数: {stats.get('total_days', 0)}",
        f"- 累计命中率: {stats.get('cumulative_hit_rate', 'N/A')} ({stats.get('cumulative_hit_pct', 0)}%)",
        f"- 最近7天命中: {stats.get('recent_7d_hits', [])}",
    ]

    sector_bias = stats.get("sector_bias", {})
    if sector_bias:
        biases = []
        for sector, bias in sector_bias.items():
            direction = "高估" if bias > 0 else "低估"
            biases.append(f"{sector} 持续{direction}(偏差{bias:+.1f})")
        lines.append(f"- 板块系统性偏差: {'; '.join(biases[:5])}")

    if stats.get("ganzhi_accuracy", "N/A") != "N/A":
        lines.append(f"- 干支准确率: {stats['ganzhi_accuracy']} ({stats.get('ganzhi_accuracy_pct', 0)}%)")

    return "\n".join(lines)


def _format_summary(deviation: dict, ganzhi_deviation: dict | None, stats: dict, insights: list[dict]) -> str:
    """生成人类可读的复盘摘要 Markdown。"""
    lines = [
        f"# 📋 复盘摘要",
        f"",
        f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## 今日偏差",
        f"",
    ]

    if deviation.get("status") == "no_prediction":
        lines.append("首次运行，无历史预测。")
    else:
        lines.append(f"- 预测 TOP3：{', '.join(deviation['predicted_top3'])}")
        lines.append(f"- 实际 TOP3：{', '.join(deviation['actual_top3'])}")
        lines.append(f"- 命中：{deviation['hit_rate']}（{'✅' if '/3' in str(deviation.get('hit_rate','')) else '❌'}）")
        lines.append(f"- 漏判：{', '.join(deviation['misses']) if deviation.get('misses') else '无'}")
        lines.append(f"- 黑马：{', '.join(deviation['surprises']) if deviation.get('surprises') else '无'}")
        lines.append(f"- 平均误差：{deviation.get('mae', 'N/A')}")
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 累积统计",
        f"",
        _format_stats_for_llm(stats),
        f"",
    ])

    if ganzhi_deviation:
        lines.extend([
            f"---",
            f"",
            f"## 干支偏差",
            f"",
            f"- 准确率：{ganzhi_deviation['accuracy']}",
            f"- 综合评估：{ganzhi_deviation['assessment']}",
        ])
        for d in ganzhi_deviation.get("details", []):
            mark = "✅" if d.get("match") else "❌"
            lines.append(f"  {mark} {d['element']}: 预测{d['predicted_rating']} → 实际{d['actual_top3_presence']}")
        lines.append("")

    if insights:
        lines.extend([
            f"---",
            f"",
            f"## 新洞察",
            f"",
        ])
        for ins in insights:
            tag = {"pattern": "📊 规律", "anomaly": "⚠️ 异常", "rule_suggestion": "💡 规则建议"}.get(ins.get("type", ""), "📌")
            lines.append(f"- {tag} {ins.get('content', '')}（置信度: {ins.get('confidence', 0):.0%}）")
        lines.append("")

    return "\n".join(lines)


# ---------- 主入口 ----------

def generate_report4(
    yesterdays_prediction: dict | None,
    todays_actual: dict,
    llm_enabled: bool = True,
    yesterdays_ganzhi: dict | None = None,
    todays_prediction: dict | None = None,
) -> dict:
    """生成报告四：硬编码偏差 + 累计统计 + LLM 增量洞察。

    Args:
        yesterdays_prediction: 昨日对今日的预测
        todays_actual: 今日实际市场数据
        yesterdays_ganzhi: 昨日干支预测
        todays_prediction: 今日对明日的预测（存入 backtest.json）
    """
    # --- 1. 硬编码偏差 ---
    deviation = _compute_deviation(yesterdays_prediction, todays_actual)
    ganzhi_deviation = _compute_ganzhi_deviation(yesterdays_ganzhi, todays_actual)

    # --- 2. 加载累积统计 ---
    log_data = load_log()
    stats = log_data.get("stats", {})

    # --- 3. LLM 增量洞察 ---
    new_insights: list[dict] = []
    if llm_enabled and yesterdays_prediction is not None:
        recent_insights = load_insights(limit=10)
        prompt = REPORT4_USER_TEMPLATE.format(
            cumulative_stats=_format_stats_for_llm(stats),
            todays_deviation=json.dumps(deviation, ensure_ascii=False, indent=2),
            ganzhi_deviation=json.dumps(ganzhi_deviation, ensure_ascii=False, indent=2) if ganzhi_deviation else "（无）",
            recent_insights=json.dumps(recent_insights, ensure_ascii=False, indent=2) if recent_insights else "（尚无历史洞察）",
        )
        result = chat_json(REPORT4_SYSTEM, prompt, temperature=0.3)
        raw_insights = result.get("insights", [])
        for ins in raw_insights:
            ins.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
            ins.setdefault("confidence", 0.5)
        new_insights = raw_insights
        if new_insights:
            append_insights(new_insights)

    # --- 4. 人类可读摘要 ---
    summary = _format_summary(deviation, ganzhi_deviation, stats, new_insights)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deviation": deviation,
        "ganzhi_deviation": ganzhi_deviation,
        "summary": summary,
        "new_insights": new_insights,
        "stats": stats,
    }

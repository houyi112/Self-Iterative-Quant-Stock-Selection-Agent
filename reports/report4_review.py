"""报告四：每日复盘 + 规则自进化 —— 对比预测vs实际 + LLM 提取规则。"""
from __future__ import annotations
import json
from datetime import datetime, date
from pathlib import Path
from config import STATE_DIR
from engine.risk_scorer import _load_rules, _save_rules, update_factor_stats, apply_lifecycle
from llm.client import chat, chat_json
from llm.prompts import REPORT4_SYSTEM, REPORT4_USER_TEMPLATE


REVIEW_LOG_PATH = STATE_DIR / "review_log.json"


def _load_review_log() -> list[dict]:
    if not REVIEW_LOG_PATH.exists():
        return []
    return json.loads(REVIEW_LOG_PATH.read_text(encoding="utf-8"))


def _save_review_log(log: list[dict]) -> None:
    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_report4(
    yesterdays_prediction: dict | None,
    todays_actual: dict,
    llm_enabled: bool = True,
) -> dict:
    """生成报告四：复盘 + 规则提取。

    Args:
        yesterdays_prediction: 昨日报告一的预测数据，None 表示首次运行
        todays_actual: 今日实际市场数据 {sector_name: {change, close, ...}}

    Returns:
        结构化复盘报告
    """
    today_str = str(date.today())

    # --- 1. 硬编码：偏差度量 ---
    deviation = _compute_deviation(yesterdays_prediction, todays_actual)

    # --- 2. LLM：根因分析 + 规则提取 ---
    existing_rules = _format_rules_for_llm()
    llm_result = {}

    if llm_enabled and yesterdays_prediction is not None:
        prompt = REPORT4_USER_TEMPLATE.format(
            yesterdays_prediction=json.dumps(yesterdays_prediction, ensure_ascii=False, indent=2),
            todays_actual=json.dumps(todays_actual, ensure_ascii=False, indent=2),
            deviation_analysis=json.dumps(deviation, ensure_ascii=False, indent=2),
            existing_rules=existing_rules,
        )
        llm_result = chat_json(REPORT4_SYSTEM, prompt)
    else:
        llm_result = {
            "root_cause": "首次运行，无历史预测可对比",
            "new_rules": [],
            "rule_adjustments": [],
            "narrative": "系统首次运行，开始积累复盘数据。",
        }

    # --- 3. 硬编码：应用规则变更 ---
    rule_changes = []

    # 新增规则
    for rule in llm_result.get("new_rules", []):
        rule_changes.append(f"[新增] IF {rule.get('if', '')} THEN {rule.get('then', '')}")

    # 因子调整
    for adj in llm_result.get("rule_adjustments", []):
        fid = adj.get("factor_id", "")
        action = adj.get("action", "")
        reason = adj.get("reason", "")
        rule_changes.append(f"[{action}] {fid}: {reason}")

    # 应用生命周期
    lifecycle_changes = apply_lifecycle()
    rule_changes.extend(lifecycle_changes)

    # --- 4. 持久化复盘记录 ---
    review_entry = {
        "date": today_str,
        "prediction": yesterdays_prediction,
        "actual": todays_actual,
        "deviation": deviation,
        "root_cause": llm_result.get("root_cause", ""),
        "new_rules": llm_result.get("new_rules", []),
        "narrative": llm_result.get("narrative", ""),
    }

    log = _load_review_log()
    # 去重：同一天不重复记录
    log = [e for e in log if e.get("date") != today_str]
    log.append(review_entry)
    # 只保留最近 60 条
    log = log[-60:]
    _save_review_log(log)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deviation_analysis": _format_deviation(deviation),
        "root_cause": llm_result.get("root_cause", ""),
        "rule_changes": rule_changes,
        "new_rules": llm_result.get("new_rules", []),
        "rule_adjustments": llm_result.get("rule_adjustments", []),
        "narrative": llm_result.get("narrative", ""),
    }


def _compute_deviation(prediction: dict | None, actual: dict) -> dict:
    """硬编码：计算预测与实际之间的偏差。"""
    if prediction is None:
        return {"status": "no_prediction", "message": "首次运行，无历史预测"}

    # 从预测中提取板块排序
    rankings = prediction.get("rankings", [])
    predicted_order = [s.get("name", "") for s in rankings]
    predicted_top = predicted_order[:3] if len(predicted_order) >= 3 else predicted_order

    # 实际板块排序（按当日涨跌幅）
    actual_sorted = sorted(actual.items(), key=lambda x: x[1].get("change", -999), reverse=True)
    actual_top = [name for name, _ in actual_sorted[:3]]

    # 交集
    hits = set(predicted_top) & set(actual_top)
    misses = set(predicted_top) - set(actual_top)
    surprises = set(actual_top) - set(predicted_top)

    deviations = []
    for sector in predicted_order[:5]:
        act = actual.get(sector, {})
        pred_prob = next((s.get("probability", 0) for s in rankings if s.get("name") == sector), 0)
        act_change = act.get("change", 0)
        error = abs(pred_prob / 100 * 10 - act_change)  # 粗略误差
        deviations.append({
            "sector": sector,
            "predicted_probability": pred_prob,
            "actual_change": round(act_change, 2),
            "error": round(error, 2),
        })

    return {
        "predicted_top3": predicted_top,
        "actual_top3": actual_top,
        "hits": list(hits),
        "misses": list(misses),
        "surprises": list(surprises),
        "hit_rate": f"{len(hits)}/3",
        "details": deviations,
    }


def _format_deviation(deviation: dict) -> str:
    """格式化偏差分析为可读文本。"""
    if deviation.get("status") == "no_prediction":
        return deviation.get("message", "无数据")

    lines = [
        f"**预测 TOP3：** {', '.join(deviation['predicted_top3'])}",
        f"**实际 TOP3：** {', '.join(deviation['actual_top3'])}",
        f"**命中率：** {deviation['hit_rate']}",
        f"**命中：** {', '.join(deviation['hits']) if deviation['hits'] else '无'}",
        f"**漏判：** {', '.join(deviation['misses']) if deviation['misses'] else '无'}",
        f"**黑马：** {', '.join(deviation['surprises']) if deviation['surprises'] else '无'}",
        f"",
        f"**逐板块偏差：**",
    ]
    for d in deviation.get("details", []):
        lines.append(
            f"- {d['sector']}：预测 {d['predicted_probability']}% → 实际 {d['actual_change']:+.2f}%，"
            f"误差 {d['error']:.1f}"
        )

    return "\n".join(lines)


def _format_rules_for_llm() -> str:
    """将当前规则库格式化为 LLM 可读的文本。"""
    rules = _load_rules()
    lines = []
    for f in rules.get("factors", []):
        if not f.get("active", True):
            continue
        cond_texts = [c.get("description", "") for c in f.get("conditions", [])]
        lines.append(
            f"- **{f['id']} {f['name']}** (权重: {f.get('weight', 1)}, 封顶: {f.get('max_score', '无')})"
            f"\n  条件: {'; '.join(cond_texts)}"
            f"\n  来源: {f.get('source', '')}"
        )
    return "\n".join(lines) if lines else "（无现有规则）"

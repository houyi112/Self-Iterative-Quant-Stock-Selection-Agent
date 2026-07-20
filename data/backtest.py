"""回测数据层 —— 统一复盘日志 + 累积统计 + 增量洞察。

单一事实来源: state/backtest.json
    {daily: [{date, todays_prediction, actual, deviation, ganzhi}, ...],
     stats: {total_days, cumulative_hit_rate, sector_bias, ...}}

洞察日志: state/insights.jsonl  （每行一个 JSON，LLM 增量追加）
CSV 导出: 按需从 backtest.json 生成
"""
from __future__ import annotations
import json
from pathlib import Path
from config import STATE_DIR, OUTPUT_DIR

BACKTEST_PATH = STATE_DIR / "backtest.json"
INSIGHTS_PATH = STATE_DIR / "insights.jsonl"

# ---------- backtest.json 核心读写 ----------

def load_log() -> dict:
    """加载完整回测日志，不存在返回空结构。"""
    if not BACKTEST_PATH.exists():
        return {"daily": [], "stats": {}}
    try:
        data = json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"daily": [], "stats": {}}
        data.setdefault("daily", [])
        data.setdefault("stats", {})
        return data
    except (json.JSONDecodeError, OSError):
        return {"daily": [], "stats": {}}


def save_log(data: dict) -> None:
    """写入回测日志。"""
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yesterdays_prediction() -> dict | None:
    """从 backtest.json 中加载最近一条 todays_prediction（供 Step 2 复盘使用）。"""
    log = load_log()
    for entry in reversed(log.get("daily", [])):
        pred = entry.get("todays_prediction")
        if pred and pred.get("rankings"):
            return pred
    return None


def append_daily(
    date_str: str,
    yesterdays_pred: dict | None,
    todays_prediction: dict | None,
    actual: dict,
    deviation: dict,
    ganzhi_deviation: dict | None,
) -> None:
    """追加一天的复盘数据，自动重算累积统计。

    Args:
        date_str: 运行日期
        yesterdays_pred: 昨日对今日的预测（用于 deviation 计算，展示用）
        todays_prediction: 今日对明日的预测（存入供下次复盘用）
        actual: 今日所有板块实际涨跌
        deviation: _compute_deviation() 返回的偏差
        ganzhi_deviation: 干支偏差（可能为 None）
    """
    log = load_log()

    # 构建精简 actual（只存 change%，不存 close 价格）
    actual_slim = {
        name: round(info.get("change", 0), 2)
        for name, info in actual.items()
    }

    # 构建精简 prediction（只存 name + probability）
    pred_slim = None
    if todays_prediction and todays_prediction.get("rankings"):
        pred_slim = {
            "rankings": [
                {"name": r["name"], "probability": r.get("probability", 0)}
                for r in todays_prediction["rankings"]
            ]
        }

    entry = {
        "date": date_str,
        "todays_prediction": pred_slim,
        "actual": actual_slim,
        "deviation": deviation,
    }
    if ganzhi_deviation is not None:
        entry["ganzhi"] = ganzhi_deviation

    # 去重写入
    daily = [e for e in log["daily"] if e.get("date") != date_str]
    daily.append(entry)
    daily.sort(key=lambda e: e["date"])
    log["daily"] = daily

    # 重算累积统计
    log["stats"] = _recalc_stats(daily)

    save_log(log)


# ---------- 累积统计（硬编码计算） ----------

def _recalc_stats(daily: list[dict]) -> dict:
    """从 daily 数组全量重算累积统计。"""
    if not daily:
        return {}

    total_days = len(daily)
    total_hits = 0
    total_slots = 0
    sector_errors: dict[str, list[float]] = {}  # {sector: [pred_prob - actual_change, ...]}
    ganzhi_hits = 0
    ganzhi_total = 0

    for entry in daily:
        dev = entry.get("deviation", {})
        if dev.get("status") != "no_prediction":
            total_hits += len(dev.get("hits", []))
            total_slots += 3  # TOP3

        for d in dev.get("details", []):
            sector = d.get("sector", "")
            err = d.get("predicted_probability", 0) - d.get("actual_change", 0)
            sector_errors.setdefault(sector, []).append(err)

        gz = entry.get("ganzhi")
        if gz:
            parts = gz.get("accuracy", "0/0").split("/")
            if len(parts) == 2:
                try:
                    ganzhi_hits += int(parts[0])
                    ganzhi_total += int(parts[1])
                except ValueError:
                    pass

    # 板块偏差：取每板块平均误差，排序
    sector_bias = {}
    for sector, errors in sector_errors.items():
        if len(errors) >= 2:
            avg = sum(errors) / len(errors)
            if abs(avg) > 5:
                sector_bias[sector] = round(avg, 1)

    # 按误差绝对值排序
    sector_bias = dict(sorted(sector_bias.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

    # 胜率趋势（最近 7 天滑动窗口）
    hit_trend = []
    for i in range(max(0, total_days - 7), total_days):
        entry = daily[i]
        dev = entry.get("deviation", {})
        if dev.get("status") != "no_prediction":
            hits = len(dev.get("hits", []))
            hit_trend.append(hits)

    return {
        "total_days": total_days,
        "cumulative_hit_rate": f"{total_hits}/{total_slots}" if total_slots > 0 else "N/A",
        "cumulative_hit_pct": round(total_hits / total_slots * 100, 1) if total_slots > 0 else 0,
        "sector_bias": sector_bias,
        "recent_7d_hits": hit_trend,
        "ganzhi_accuracy": f"{ganzhi_hits}/{ganzhi_total}" if ganzhi_total > 0 else "N/A",
        "ganzhi_accuracy_pct": round(ganzhi_hits / ganzhi_total * 100, 1) if ganzhi_total > 0 else 0,
        "last_updated": daily[-1]["date"] if daily else "",
    }


# ---------- insights.jsonl 增量洞察 ----------

def load_insights(limit: int = 20) -> list[dict]:
    """加载最近的洞察（用于喂给 LLM）。"""
    if not INSIGHTS_PATH.exists():
        return []
    insights = []
    with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    insights.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return insights[-limit:]


def append_insights(new_insights: list[dict]) -> None:
    """追加洞察到 insights.jsonl（每行一个 JSON）。"""
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INSIGHTS_PATH, "a", encoding="utf-8") as f:
        for ins in new_insights:
            f.write(json.dumps(ins, ensure_ascii=False) + "\n")


# ---------- CSV 导出 ----------

def export_csv() -> str:
    """从 backtest.json 导出 CSV，供模型微调使用。"""
    import csv

    log = load_log()
    path = OUTPUT_DIR / "backtest_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["日期", "预测情况", "实际情况", "差异"])
        writer.writeheader()
        for entry in log.get("daily", []):
            dev = entry.get("deviation", {})
            if dev.get("status") == "no_prediction":
                continue
            pred = entry.get("todays_prediction", {})
            writer.writerow({
                "日期": entry["date"],
                "预测情况": json.dumps(_csv_prediction(pred), ensure_ascii=False),
                "实际情况": json.dumps(_csv_actual(entry.get("actual", {})), ensure_ascii=False),
                "差异": json.dumps(dev, ensure_ascii=False),
            })

    return str(path)


def _csv_prediction(prediction: dict) -> dict:
    rankings = prediction.get("rankings", [])
    return {
        "top5": [{"name": r["name"], "probability": r.get("probability", 0)} for r in rankings[:5]],
        "total_sectors": len(rankings),
    }


def _csv_actual(actual: dict) -> dict:
    sorted_a = sorted(actual.items(), key=lambda x: x[1], reverse=True)
    return {
        "top5": [{"name": n, "change": c} for n, c in sorted_a[:5]],
        "bottom5": [{"name": n, "change": c} for n, c in sorted_a[-5:]],
    }

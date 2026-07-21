"""训练数据采集 —— 每日管线运行时自动记录特征 + 标签，积累训练集。

文件结构:
    state/training/
        sector_samples.json   ← 板块排序样本（预测明日领涨）
        stock_samples.json    ← 涨停选股样本（预测涨停概率）

每条样本包含:
    features:  {字段名: 数值}    ← compute_summary() 的 66 个字段展平
    label:     实际结果          ← 板块: 次日涨跌%, 股票: 是否涨停

导出:
    python -m data.training_log export  → 生成 train.csv / val.csv
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from datetime import date
from config import STATE_DIR

TRAINING_DIR = STATE_DIR / "training"
SECTOR_SAMPLES = TRAINING_DIR / "sector_samples.json"
STOCK_SAMPLES = TRAINING_DIR / "stock_samples.json"


# ============================================================
# 特征提取：将 compute_summary() 的嵌套 dict → 展平数值
# ============================================================

def flatten_summary(summary: dict) -> dict[str, float]:
    """将 compute_summary() 输出展平为纯数值特征字典。

    跳过非数值字段（如 'ma_alignment'='多头'），只保留可计算的值。
    """
    flat: dict[str, float] = {}

    # 趋势
    t = summary.get("trend", {})
    for k in ("ma5", "ma10", "ma20", "ma60", "ma120", "ema_12", "ema_26",
              "macd_dif", "macd_dea", "macd_hist", "adx", "adx_di_diff",
              "aroon_up", "aroon_down", "psar_value"):
        _set(flat, f"trend_{k}", t.get(k))

    # 动量
    m = summary.get("momentum", {})
    for k in ("rsi_6", "rsi_14", "stoch_k", "stoch_d", "stoch_j",
              "cci_14", "mfi_14", "willr_14", "uo"):
        _set(flat, f"mom_{k}", m.get(k))

    # 成交量
    v = summary.get("volume", {})
    for k in ("vol_ratio", "force_idx", "eom", "cmf", "vwap_dev"):
        _set(flat, f"vol_{k}", v.get(k))

    # 波动率
    vl = summary.get("volatility", {})
    for k in ("bb_position", "kc_position", "dc_position",
              "atr_pct", "hist_vol_20", "amp_now"):
        _set(flat, f"vola_{k}", vl.get(k))

    # 统计
    s = summary.get("statistical", {})
    for k in ("zscore_20", "skew_20", "kurt_20"):
        _set(flat, f"stat_{k}", s.get(k))
    _set(flat, "stat_is_20d_high", 1.0 if s.get("is_20day_high") else 0.0)
    _set(flat, "stat_is_60d_high", 1.0 if s.get("is_60day_high") else 0.0)

    return flat


def _set(d: dict, key: str, val):
    """安全写入数值，跳过 None / NaN / 非数字。"""
    if val is None:
        return
    try:
        f = float(val)
        if not math.isnan(f) and not math.isinf(f):
            d[key] = f
    except (ValueError, TypeError):
        pass


# ============================================================
# 样本写入
# ============================================================

def record_sector_samples(
    run_date: str,
    sectors: list[dict],    # 来自 report1 rankings，含 summary + daily_return
    next_day_actual: dict,  # 次日实际涨跌 {sector_name: change%}
) -> int:
    """记录板块排序训练样本。每个板块一行。

    Args:
        run_date: 预测日期 YYYY-MM-DD
        sectors: 当日报告一的板块列表（含 summary）
        next_day_actual: 次日实际涨跌幅（复盘时填入，首次运行无此数据则为 None 的值）

    Returns:
        写入的样本数
    """
    samples = _load(SECTOR_SAMPLES)
    count = 0
    for s in sectors:
        name = s.get("name", "")
        summary = s.get("summary", {})
        if not summary:
            continue

        features = flatten_summary(summary)
        label = next_day_actual.get(name) if next_day_actual else None
        # label 可能是 float(change%) 或 None(暂无次日数据)

        sample = {
            "date": run_date,
            "sector": name,
            "features": features,
            "label_change_pct": round(label, 2) if isinstance(label, (int, float)) else None,
        }
        samples.append(sample)
        count += 1

    _save(SECTOR_SAMPLES, samples)
    return count


def record_stock_samples(
    run_date: str,
    candidates: list[dict],  # 来自 report2 candidates/picks，含 summary + code
    next_day_results: dict[str, bool] | None,  # {code: 是否涨停}
) -> int:
    """记录涨停选股训练样本。每只候选股一行。

    Args:
        run_date: 预测日期
        candidates: 当日候选/选股列表（含 summary）
        next_day_results: 次日结果，key 为股票代码，value 为是否涨停

    Returns:
        写入的样本数
    """
    samples = _load(STOCK_SAMPLES)
    count = 0
    for c in candidates:
        code = c.get("code", "")
        summary = c.get("summary", {})
        if not summary:
            continue

        features = flatten_summary(summary)
        # 附加基础特征
        _set(features, "price", c.get("close"))
        _set(features, "market_cap", c.get("market_cap"))
        features["is_20cm"] = 1.0 if c.get("is_20cm") else 0.0
        features["daily_return"] = c.get("daily_return", 0)
        features["consecutive_up"] = c.get("consecutive_up", 0)

        label = next_day_results.get(code) if next_day_results else None

        sample = {
            "date": run_date,
            "code": code,
            "name": c.get("name", ""),
            "sector": c.get("sector", ""),
            "features": features,
            "label_limit_up": label,  # True / False / None
        }
        samples.append(sample)
        count += 1

    _save(STOCK_SAMPLES, samples)
    return count


# ============================================================
# 标签回填（次日复盘时填充昨天的标签）
# ============================================================

def backfill_labels(prediction_date: str, actual: dict) -> int:
    """用今天的实际涨跌，回填昨天记录的板块样本标签。

    Args:
        prediction_date: 昨天做预测的日期 YYYY-MM-DD
        actual: 今天的实际涨跌 {sector_name: change%} 或 {sector_name: {change: x, close: y}}

    Returns:
        回填的样本数
    """
    samples = _load(SECTOR_SAMPLES)
    count = 0
    for s in samples:
        if s.get("label_change_pct") is not None:
            continue
        if s.get("date") != prediction_date:
            continue
        label = actual.get(s.get("sector", ""))
        if label is None:
            continue
        # 兼容 {change: x} dict 和裸 float
        if isinstance(label, dict):
            label = label.get("change", 0)
        try:
            s["label_change_pct"] = round(float(label), 2)
            count += 1
        except (ValueError, TypeError):
            pass
    if count > 0:
        _save(SECTOR_SAMPLES, samples)
    return count


# ============================================================
# 导出
# ============================================================

def export_csv(output_dir: str | None = None) -> dict[str, str]:
    """将训练数据导出为 CSV 文件。

    Returns:
        {"sector": path, "stock": path}
    """
    import csv

    out = Path(output_dir) if output_dir else TRAINING_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 板块样本
    sector_samples = _load(SECTOR_SAMPLES)
    if sector_samples:
        # 推断字段顺序：取第一条的 features keys
        feature_keys = list(sector_samples[0]["features"].keys())
        fieldnames = ["date", "sector", "label_change_pct"] + feature_keys
        path = out / "sector_train.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for s in sector_samples:
                row = {"date": s["date"], "sector": s["sector"],
                       "label_change_pct": s.get("label_change_pct")}
                row.update(s.get("features", {}))
                w.writerow(row)
        paths["sector"] = str(path)

    # 股票样本
    stock_samples = _load(STOCK_SAMPLES)
    if stock_samples:
        feature_keys = list(stock_samples[0]["features"].keys())
        fieldnames = ["date", "code", "name", "sector", "label_limit_up"] + feature_keys
        path = out / "stock_train.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for s in stock_samples:
                row = {"date": s["date"], "code": s["code"], "name": s["name"],
                       "sector": s["sector"], "label_limit_up": s.get("label_limit_up")}
                row.update(s.get("features", {}))
                w.writerow(row)
        paths["stock"] = str(path)

    return paths


def stats() -> dict:
    """查看当前训练数据统计。"""
    sector = _load(SECTOR_SAMPLES)
    stock = _load(STOCK_SAMPLES)
    return {
        "sector_samples": len(sector),
        "sector_labeled": sum(1 for s in sector if s.get("label_change_pct") is not None),
        "sector_dates": len(set(s["date"] for s in sector)),
        "stock_samples": len(stock),
        "stock_labeled": sum(1 for s in stock if s.get("label_limit_up") is not None),
        "stock_dates": len(set(s["date"] for s in stock)),
    }


# ============================================================
# 内部
# ============================================================

def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

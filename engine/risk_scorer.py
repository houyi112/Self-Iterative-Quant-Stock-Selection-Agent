"""风险因子评分引擎 —— F001-F008 逐条检测 + 生命周期管理。

F003/F004/F005/F007 的字段从 OHLCV + 指标数据硬编码计算，
不依赖外部 API。只有 F002(高标情绪)、F006(资金流向)、F008(利好兑现)
需要外部数据（连板高度、北向、新闻），无数据时默认不触发。
"""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from config import STATE_DIR
from data.indicators import compute_all, last_val


def _last_n(ind, col: str, n: int = 3):
    """从 DataFrame 列中安全提取最后 n 个值。"""
    if isinstance(ind, pd.DataFrame) and col in ind.columns:
        return ind[col].tail(n).tolist()
    return []


RULES_PATH = STATE_DIR / "rules.json"


def _load_rules() -> dict:
    if not RULES_PATH.exists():
        return {"factors": [], "scoring_tiers": []}
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _save_rules(rules: dict) -> None:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 条件评估
# ============================================================

def _eval_condition(field_value: float | int | bool, op: str, target: float | int | bool | list) -> bool:
    """评估单个条件。"""
    if op == "gt":
        return float(field_value) > float(target)
    elif op == "gte":
        return float(field_value) >= float(target)
    elif op == "lt":
        return float(field_value) < float(target)
    elif op == "lte":
        return float(field_value) <= float(target)
    elif op == "eq":
        if isinstance(target, bool):
            return bool(field_value) is target
        return field_value == target
    elif op == "between":
        if isinstance(target, list) and len(target) == 2:
            return float(target[0]) <= float(field_value) <= float(target[1])
        return False
    return False


# ============================================================
# 单板块评分
# ============================================================

def score_sector(
    sector_name: str,
    sector_ohlcv: list[dict] | None,
    market_context: dict | None = None,
    indicators: "pd.DataFrame | None" = None,
) -> dict:
    """对单个板块计算风险分。

    Args:
        sector_name: 板块名称
        sector_ohlcv: 板块指数的 OHLCV 数据
        market_context: 市场上下文，包含 board height, northbound 等外部数据。
        indicators: 预计算的指标 DataFrame（可选，避免重复 compute_all）

    Returns:
        {factor_id: score, ..., "total": sum, "level": "低风险"}
    """
    rules = _load_rules()
    market_context = market_context or {}

    # 计算板块指标（优先使用预计算结果）
    if indicators is not None:
        ind = indicators
    elif sector_ohlcv and len(sector_ohlcv) > 20:
        ind = compute_all(sector_ohlcv)
    else:
        ind = pd.DataFrame()

    # ================================================================
    # 硬编码计算 F003/F004/F005/F007 字段（不依赖外部 API）
    # ================================================================

    # --- F003: 逆市补跌 ---
    # 需大盘数据：从 market_context 传入，否则默认未触发
    sh_change = market_context.get("_market_changes", {}).get("上证指数", 0.0)
    sector_change = last_val(ind, "daily_return", 0.0)
    counter_market_rise = bool(sector_change > 0 and sh_change < 0)

    # 连续逆市上涨天数（从 OHLCV 计算）
    cons_counter = 0
    if sector_ohlcv and len(sector_ohlcv) >= 3 and sh_change < 0:
        for i in range(len(sector_ohlcv) - 1, -1, -1):
            if i < 2:
                break
            s_chg = (sector_ohlcv[i]["close"] - sector_ohlcv[i-1]["close"]) / sector_ohlcv[i-1]["close"] * 100
            if s_chg > 0:
                cons_counter += 1
            else:
                break

    counter_market_no_catalyst = bool(counter_market_rise and sector_change > 2)

    # --- F004: 量价背离 ---
    vr = last_val(ind, "volume_ratio", 1.0)
    vol_vals = _last_n(ind, "volume", 5)
    close_vals = ind["close"].tail(3).tolist() if isinstance(ind, pd.DataFrame) and "close" in ind.columns else []

    # 量递减（近2日量下降但价格上涨）
    volume_declining = False
    if len(vol_vals) >= 3 and len(close_vals) >= 3:
        vol_decl = vol_vals[-1] < vol_vals[-2] and close_vals[-1] > close_vals[-2]
        volume_declining = bool(vol_decl)

    # 放量滞涨（量比>2但涨幅<1%）
    volume_stagnation = bool(vr > 2 and abs(sector_change) < 1)

    # 缩量加速上涨（量比<0.7但今日涨幅>昨日涨幅）
    prev_change = 0.0
    if len(close_vals) >= 3:
        prev_change = (close_vals[-2] - close_vals[-3]) / close_vals[-3] * 100 if close_vals[-3] else 0
    shrinking_acceleration = bool(vr < 0.7 and sector_change > prev_change and sector_change > 0)

    # --- F007: 技术顶背离 ---
    is_20d_high = bool(last_val(ind, "is_20day_high", 0.0))
    macd_dif_val = last_val(ind, "macd_dif", 0.0)

    # 价格创20日新高但 MACD DIF 未同步创新高
    price_high_dif_not = False
    if is_20d_high and isinstance(ind, pd.DataFrame) and "macd_dif" in ind.columns:
        dif_20_max = ind["macd_dif"].tail(20).max()
        price_high_dif_not = bool(macd_dif_val < dif_20_max * 0.9)

    # 价格创20日新高但成交量未同步放大
    price_high_vol_not = False
    if is_20d_high and isinstance(ind, pd.DataFrame) and "volume" in ind.columns:
        vol_20_max = ind["volume"].tail(20).max()
        cur_vol = ind["volume"].iloc[-1]
        price_high_vol_not = bool(cur_vol < vol_20_max * 0.7)

    # 双顶背离
    dual_top_divergence = bool(price_high_dif_not and price_high_vol_not)

    # ================================================================
    # 构建 fields 字典
    # ================================================================

    fields = {
        # F001: 从指标直接读取
        "consecutive_up_days": int(last_val(ind, "consecutive_up_days", 0)),
        "macd_dif": macd_dif_val,
        "daily_return": sector_change,
        "volume_ratio": vr,
        "is_20day_high": is_20d_high,

        # F002: 高标情绪（需外部数据：连板高度）
        "max_board_height": market_context.get("max_board_height", 0),
        "board_standalone": market_context.get("board_standalone", False),
        "max_board_in_sector": market_context.get("max_board_in_sector", {}).get(sector_name, False),

        # F003: 逆市补跌（硬编码计算）
        "counter_market_rise": counter_market_rise,
        "consecutive_counter_market": cons_counter,
        "counter_market_no_catalyst": counter_market_no_catalyst,

        # F004: 量价背离（硬编码计算）
        "volume_declining": volume_declining,
        "volume_stagnation": volume_stagnation,
        "shrinking_acceleration": shrinking_acceleration,

        # F005: 板块轮动（需排行榜对比）
        "top3_rotation": market_context.get("top3_rotation", False),
        "rotation_3day": market_context.get("rotation_3day", False),
        "intraday_fade": market_context.get("intraday_fade", {}).get(sector_name, False),

        # F006: 资金流向（需外部数据：北向/融资）
        "northbound_outflow": market_context.get("northbound_outflow", {}).get(sector_name, False),
        "margin_decline": market_context.get("margin_decline", {}).get(sector_name, False),
        "capital_dual_divergence": market_context.get("capital_dual_divergence", {}).get(sector_name, False),

        # F007: 技术顶背离（硬编码计算）
        "price_high_dif_not": price_high_dif_not,
        "price_high_vol_not": price_high_vol_not,
        "dual_top_divergence": dual_top_divergence,

        # F008: 利好兑现（需外部数据：新闻/事件）
        "gap_up_no_catalyst": market_context.get("gap_up_no_catalyst", {}).get(sector_name, False),
        "news_leaked": market_context.get("news_leaked", {}).get(sector_name, False),
        "earnings_priced_in": market_context.get("earnings_priced_in", {}).get(sector_name, False),
    }

    # 逐因子评估
    scores: dict[str, int] = {}
    total = 0

    for factor in rules.get("factors", []):
        if not factor.get("active", True):
            continue

        fid = factor["id"]
        factor_score = 0
        max_score = factor.get("max_score", 999)

        for cond in factor.get("conditions", []):
            field_name = cond["field"]
            field_value = fields.get(field_name, 0)
            if _eval_condition(field_value, cond["op"], cond["value"]):
                factor_score += cond["score"] * factor.get("weight", 1.0)

        # 封顶
        factor_score = min(int(factor_score), max_score)
        scores[fid] = factor_score
        total += factor_score

    # 风险等级
    level = "未知"
    for tier in rules.get("scoring_tiers", []):
        lo, hi = tier["range"]
        if lo <= total <= hi:
            level = tier["label"]
            break

    return {
        **scores,
        "total": total,
        "level": level,
    }


def score_all_sectors(
    sector_data: dict[str, list[dict] | None],
    market_context: dict | None = None,
    precomputed_indicators: dict[str, "pd.DataFrame"] | None = None,
) -> dict[str, dict]:
    """对所有板块批量评分。

    Args:
        sector_data: {sector_name: ohlcv_list}
        market_context: 市场上下文
        precomputed_indicators: {sector_name: indicators_df} 预计算结果

    Returns:
        {sector_name: scoring_dict}
    """
    precomputed = precomputed_indicators or {}
    results = {}
    for name, ohlcv in sector_data.items():
        results[name] = score_sector(name, ohlcv, market_context,
                                     indicators=precomputed.get(name))
    return results


# ============================================================
# 生命周期管理
# ============================================================

def update_factor_stats(factor_id: str, was_correct: bool) -> None:
    """更新因子的命中/有效统计。"""
    rules = _load_rules()
    for f in rules.get("factors", []):
        if f["id"] != factor_id:
            continue
        stats = f.setdefault("stats", {"hit_count": 0, "valid_count": 0, "consecutive_invalid": 0})
        stats["hit_count"] += 1
        if was_correct:
            stats["valid_count"] += 1
            stats["consecutive_invalid"] = 0
        else:
            stats["consecutive_invalid"] += 1
        break
    _save_rules(rules)


def apply_lifecycle() -> list[str]:
    """应用生命周期规则，返回变更日志。"""
    rules = _load_rules()
    changes = []
    to_remove = []

    for f in rules.get("factors", []):
        stats = f.get("stats", {})
        ci = stats.get("consecutive_invalid", 0)
        vc = stats.get("valid_count", 0)

        if ci >= 8:
            f["active"] = False
            changes.append(f"[废弃] {f['id']} {f['name']}: 连续{ci}次无效")
        elif ci >= 3:
            old_w = f.get("weight", 1.0)
            f["weight"] = max(0.5, old_w - 1)
            changes.append(f"[降权] {f['id']} {f['name']}: weight {old_w} → {f['weight']}")
        elif vc >= 5 and ci == 0:
            f["weight"] = f.get("weight", 1.0) + 1
            changes.append(f"[升级] {f['id']} {f['name']}: weight +1")
            f["stats"]["valid_count"] = 0  # 重置计数器

    # 废弃的因子移到末尾
    active = [f for f in rules["factors"] if f.get("active", True)]
    inactive = [f for f in rules["factors"] if not f.get("active", True)]
    rules["factors"] = active + inactive

    # 封顶 30 条
    if len(rules["factors"]) > rules.get("max_factors", 30):
        # 按 valid_count/hit_count 比率排序，淘汰末尾
        def _ratio(f):
            s = f.get("stats", {})
            h = max(s.get("hit_count", 0), 1)
            return s.get("valid_count", 0) / h
        rules["factors"].sort(key=_ratio, reverse=True)
        removed = rules["factors"][rules["max_factors"]:]
        rules["factors"] = rules["factors"][: rules["max_factors"]]
        for f in removed:
            changes.append(f"[淘汰] {f['id']} {f['name']}: 因子数超限")

    rules["last_updated"] = str(__import__("datetime").date.today())
    _save_rules(rules)
    return changes


def add_factor(factor_dict: dict) -> None:
    """新增风险因子。"""
    rules = _load_rules()
    rules["factors"].append(factor_dict)
    _save_rules(rules)

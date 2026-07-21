"""历史数据集构建 —— 拉取过去两年全量数据，生成训练/验证集。

流程:
    1. 拉取全量 K 线
    2. 数据质量检查（日期连续性、最新性、异常值）
    3. 构建特征 + 干支 + 标签
    4. 输出 state/training/sector_dataset.csv + quality_report.json

用法:
    python -m data.build_dataset           # 全量构建
    python -m data.build_dataset --check   # 仅检查数据质量
"""
from __future__ import annotations
import argparse
import json
import csv
import time
from datetime import date, timedelta
from pathlib import Path
from config import SECTOR_INDICES, STATE_DIR

TRAINING_DIR = STATE_DIR / "training"
DATASET_PATH = TRAINING_DIR / "sector_dataset.csv"
META_PATH = TRAINING_DIR / "dataset_meta.json"
QUALITY_PATH = TRAINING_DIR / "quality_report.json"

SECTOR_CODES = list(SECTOR_INDICES.keys())
SECTOR_NAMES = list(SECTOR_INDICES.values())

# Sina API 部分代码返回旧数据 → 替代代码
_CODE_OVERRIDES: dict[str, str] = {
    "sh000811": "sz399608",  # 中证通信
    "sh000820": "sz399440",  # 中证钢铁
    "sh000805": "sz399439",  # 中证油气
    "sh000812": "sz399987",  # 中证建材（原代码 2016 年过期）
}


# ============================================================
# 干支计算
# ============================================================

_GAN = "甲乙丙丁戊己庚辛壬癸"
_ZHI = "子丑寅卯辰巳午未申酉戌亥"
_ELEMENTS = {
    "甲": "木", "乙": "木", "丙": "火", "丁": "火", "戊": "土",
    "己": "土", "庚": "金", "辛": "金", "壬": "水", "癸": "水",
    "子": "水", "丑": "土", "寅": "木", "卯": "木", "辰": "土", "巳": "火",
    "午": "火", "未": "土", "申": "金", "酉": "金", "戌": "土", "亥": "水",
}
_BASE_DATE = date(1900, 1, 1)
_BASE_DAY_IDX = 10  # 1900-01-01 = 甲戌日


def _day_ganzhi(d: date) -> tuple[str, str, str]:
    days = (d - _BASE_DATE).days
    idx = (_BASE_DAY_IDX + days) % 60
    return f"{_GAN[idx % 10]}{_ZHI[idx % 12]}", _GAN[idx % 10], _ZHI[idx % 12]


def _month_ganzhi(d: date) -> tuple[str, str, str]:
    y = d.year
    if y == 2026:
        m = {1: "庚寅", 2: "辛卯", 3: "壬辰", 4: "癸巳", 5: "甲午", 6: "乙未",
             7: "丙申", 8: "丁酉", 9: "戊戌", 10: "己亥", 11: "庚子", 12: "辛丑"}
    elif y == 2025:
        m = {1: "戊寅", 2: "己卯", 3: "庚辰", 4: "辛巳", 5: "壬午", 6: "癸未",
             7: "甲申", 8: "乙酉", 9: "丙戌", 10: "丁亥", 11: "戊子", 12: "己丑"}
    elif y == 2024:
        m = {1: "丙寅", 2: "丁卯", 3: "戊辰", 4: "己巳", 5: "庚午", 6: "辛未",
             7: "壬申", 8: "癸酉", 9: "甲戌", 10: "乙亥", 11: "丙子", 12: "丁丑"}
    else:
        return "??", "?", "?"
    gz = m.get(d.month, "??")
    return f"{gz[0]}{gz[1]}月", gz[0], gz[1]


def _year_ganzhi(d: date) -> tuple[str, str, str]:
    y = d.year
    return f"{_GAN[(y - 4) % 10]}{_ZHI[(y - 4) % 12]}年", _GAN[(y - 4) % 10], _ZHI[(y - 4) % 12]


def ganzhi_features(d: date) -> dict[str, float]:
    _, yg, yz = _year_ganzhi(d)
    _, mg, mz = _month_ganzhi(d)
    _, dg, dz = _day_ganzhi(d)
    feats = {}
    for el in ["金", "木", "水", "火", "土"]:
        feats[f"gz_yg_{el}"] = 1.0 if _ELEMENTS.get(yg) == el else 0.0
        feats[f"gz_yz_{el}"] = 1.0 if _ELEMENTS.get(yz) == el else 0.0
        feats[f"gz_mg_{el}"] = 1.0 if _ELEMENTS.get(mg) == el else 0.0
        feats[f"gz_mz_{el}"] = 1.0 if _ELEMENTS.get(mz) == el else 0.0
        feats[f"gz_dg_{el}"] = 1.0 if _ELEMENTS.get(dg) == el else 0.0
        feats[f"gz_dz_{el}"] = 1.0 if _ELEMENTS.get(dz) == el else 0.0
    ti = _ELEMENTS.get(dg, "?")
    yong = _ELEMENTS.get(mz, "?")
    feats["gz_ti_yong_same"] = 1.0 if ti == yong else 0.0
    feats["gz_yong_sheng_ti"] = 1.0 if _generates(yong, ti) else 0.0
    feats["gz_ti_sheng_yong"] = 1.0 if _generates(ti, yong) else 0.0
    feats["gz_ti_ke_yong"] = 1.0 if _restricts(ti, yong) else 0.0
    feats["gz_yong_ke_ti"] = 1.0 if _restricts(yong, ti) else 0.0
    return feats


def _generates(a, b):
    return {"金": "水", "水": "木", "木": "火", "火": "土", "土": "金"}.get(a) == b


def _restricts(a, b):
    return {"金": "木", "木": "土", "土": "水", "水": "火", "火": "金"}.get(a) == b


# ============================================================
# 数据质量检查
# ============================================================

def check_quality(sector_data: dict[str, list[dict]], today: str) -> list[dict]:
    """对每个板块数据做质量检查，返回问题报告。"""
    issues = []
    for name, data in sector_data.items():
        if not data:
            issues.append({"sector": name, "severity": "error", "issue": "无数据"})
            continue

        dates = [d["date"] for d in data]
        last_date = dates[-1]
        first_date = dates[0]
        n_rows = len(data)

        # 1. 最新性：最后日期是否在最近 3 天内
        if last_date < today:
            days_behind = (date.fromisoformat(today) - date.fromisoformat(last_date)).days
            if days_behind > 3:
                issues.append({"sector": name, "severity": "error",
                               "issue": f"数据滞后 {days_behind} 天 (last={last_date}, today={today})"})

        # 2. 日期范围是否合理（应该覆盖近 2 年）
        expected_start = str(date.today().replace(year=date.today().year - 2))
        if first_date > expected_start:
            issues.append({"sector": name, "severity": "warn",
                           "issue": f"数据起始较晚 (first={first_date}, expected<={expected_start})"})

        # 3. 检查是否有跳空（超过 7 天的间隔）
        for i in range(1, len(dates)):
            try:
                d1 = date.fromisoformat(dates[i - 1])
                d2 = date.fromisoformat(dates[i])
                gap = (d2 - d1).days
                if gap > 7:
                    issues.append({"sector": name, "severity": "warn",
                                   "issue": f"日期跳空 {gap} 天 ({dates[i-1]} → {dates[i]})"})
                    break  # 每个板块只报告第一处跳空
            except ValueError:
                pass

        # 4. 检查异常价格（日涨跌超过 15%）
        for i in range(1, len(data)):
            prev_close = data[i - 1].get("close", 0)
            cur_close = data[i].get("close", 0)
            if prev_close > 0:
                chg = abs((cur_close - prev_close) / prev_close * 100)
                if chg > 15:
                    issues.append({"sector": name, "severity": "warn",
                                   "issue": f"异常涨跌 {chg:.1f}% ({dates[i]})"})
                    break

    return issues


# ============================================================
# 特征提取
# ============================================================

def _flatten(summary: dict) -> dict[str, float]:
    from data.training_log import flatten_summary
    return flatten_summary(summary)


# ============================================================
# 主流程
# ============================================================

def build_dataset(days: int = 500, min_lookback: int = 60) -> Path:
    from data.fetcher import fetch_kline
    from data.indicators import compute_all, compute_summary

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    today_str = str(date.today())

    # ================================================================
    # Phase 1: 拉取全量数据
    # ================================================================
    print("=" * 60)
    print("Phase 1: 拉取全量 K 线数据")
    print("=" * 60)
    t0 = time.time()

    sector_ohlcv: dict[str, list[dict]] = {}
    valid_sectors: list[str] = []
    target_cutoff = str(date.today().replace(year=date.today().year - 2))

    for code, name in zip(SECTOR_CODES, SECTOR_NAMES):
        fetch_code = _CODE_OVERRIDES.get(code, code)
        data = fetch_kline(fetch_code, count=days + min_lookback)
        if data:
            data.sort(key=lambda x: x["date"])
            last_date = data[-1]["date"]
            if last_date >= target_cutoff:
                sector_ohlcv[name] = data
                valid_sectors.append(name)
                status = "✓"
            else:
                status = f"✗ 过期 ({data[0]['date']}~{last_date})"
        else:
            status = "✗ 无数据"
        print(f"  {name:6s} ({fetch_code:10s}): {len(data):4d} 条  {status}")

    print(f"\n拉取: {len(valid_sectors)}/{len(SECTOR_NAMES)} 板块有效 ({time.time() - t0:.1f}s)")

    # ================================================================
    # Phase 2: 数据质量检查
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Phase 2: 数据质量检查")
    print("=" * 60)

    quality_issues = check_quality(sector_ohlcv, today_str)
    if quality_issues:
        errors = [i for i in quality_issues if i["severity"] == "error"]
        warns = [i for i in quality_issues if i["severity"] == "warn"]
        print(f"  ❌ {len(errors)} 个错误")
        for e in errors:
            print(f"     {e['sector']}: {e['issue']}")
        print(f"  ⚠️  {len(warns)} 个警告")
        for w in warns:
            print(f"     {w['sector']}: {w['issue']}")
    else:
        print("  ✅ 全部通过")

    with open(QUALITY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "checked_at": today_str,
            "total_sectors": len(sector_ohlcv),
            "valid_sectors": len(valid_sectors),
            "issues": quality_issues,
        }, f, ensure_ascii=False, indent=2)

    if not valid_sectors:
        print("\n❌ 无有效板块，退出")
        return DATASET_PATH

    # ================================================================
    # Phase 3: 构建特征 + 标签
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Phase 3: 构建特征 + 干支 + 标签")
    print("=" * 60)
    t0 = time.time()

    # 建立日期索引
    date_index: dict[str, dict[str, int]] = {}
    for name in valid_sectors:
        data = sector_ohlcv[name]
        date_index[name] = {d["date"]: i for i, d in enumerate(data)}

    # 合并所有日期
    all_dates = sorted(set(d for data in sector_ohlcv.values() for d in date_index.get(name, {})))
    all_dates = sorted(set().union(*[set(date_index[n].keys()) for n in valid_sectors]))

    all_samples: list[dict] = []
    feature_keys: set[str] = set()
    processed = 0
    skipped_missing = 0
    skipped_error = 0

    for i, dt in enumerate(all_dates):
        if i < min_lookback:
            continue
        if i >= len(all_dates) - 1:
            break

        next_dt = all_dates[i + 1]
        day_samples = []

        for name in valid_sectors:
            ohlcv = sector_ohlcv[name]
            idx_map = date_index[name]
            pos = idx_map.get(dt, -1)
            if pos < min_lookback:
                continue

            past = ohlcv[:pos + 1]
            try:
                ind_df = compute_all(past)
                summary = compute_summary(past, indicators=ind_df)
                if "error" in summary:
                    continue
            except Exception:
                continue

            features = _flatten(summary)
            try:
                dt_date = date.fromisoformat(dt)
                features.update(ganzhi_features(dt_date))
            except ValueError:
                pass

            next_pos = idx_map.get(next_dt, -1)
            if next_pos < 0:
                continue

            today_close = ohlcv[pos]["close"]
            tomorrow_close = ohlcv[next_pos]["close"]
            if today_close <= 0:
                continue
            next_return = (tomorrow_close - today_close) / today_close * 100

            day_samples.append({
                "name": name,
                "features": features,
                "next_return": next_return,
            })

        # 当天所有板块数据完整才保留
        if len(day_samples) < len(valid_sectors):
            skipped_missing += 1
            continue

        # 标签：次日涨幅 TOP3
        day_samples.sort(key=lambda x: x["next_return"], reverse=True)
        for rank, s in enumerate(day_samples):
            row = {
                "date": dt,
                "sector": s["name"],
                "label_top3": 1 if rank < 3 else 0,
                "next_return": round(s["next_return"], 2),
            }
            row.update(s["features"])
            feature_keys.update(s["features"].keys())
            all_samples.append(row)

        processed += 1
        if processed % 100 == 0:
            print(f"  {processed} 天完成, {len(all_samples)} 条...")

    print(f"特征构建: {len(all_samples)} 条, 跳过 {skipped_missing} 天(数据不全), {time.time() - t0:.1f}s")

    # ================================================================
    # Phase 4: 写入
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Phase 4: 写入文件")
    print("=" * 60)

    all_features = sorted(feature_keys)
    fieldnames = ["date", "sector", "label_top3", "next_return"] + all_features

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_samples)

    dates = sorted(set(s["date"] for s in all_samples))
    positives = sum(1 for s in all_samples if s["label_top3"] == 1)
    meta = {
        "total_samples": len(all_samples),
        "positive_ratio": round(positives / len(all_samples), 3) if all_samples else 0,
        "date_range": [dates[0], dates[-1]] if dates else [],
        "num_features": len(feature_keys),
        "features": all_features,
        "sectors": valid_sectors,
        "excluded_sectors": [n for n in SECTOR_NAMES if n not in valid_sectors],
        "created": today_str,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  {DATASET_PATH}")
    print(f"  样本: {len(all_samples)} 条 | 特征: {len(feature_keys)} 维")
    print(f"  正样本: {positives} ({meta['positive_ratio']:.1%})")
    print(f"  日期: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")
    print(f"  板块: {len(valid_sectors)} 个 ({', '.join(valid_sectors)})")
    if meta["excluded_sectors"]:
        print(f"  排除: {', '.join(meta['excluded_sectors'])}")

    return DATASET_PATH


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建历史训练数据集")
    parser.add_argument("--days", type=int, default=500, help="回溯天数")
    parser.add_argument("--check", action="store_true", help="仅质量检查")
    args = parser.parse_args()

    if args.check:
        from data.fetcher import fetch_kline
        sector_ohlcv = {}
        for code, name in zip(SECTOR_CODES, SECTOR_NAMES):
            fc = _CODE_OVERRIDES.get(code, code)
            data = fetch_kline(fc, count=560)
            if data:
                data.sort(key=lambda x: x["date"])
                sector_ohlcv[name] = data
        issues = check_quality(sector_ohlcv, str(date.today()))
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    else:
        build_dataset(days=args.days)

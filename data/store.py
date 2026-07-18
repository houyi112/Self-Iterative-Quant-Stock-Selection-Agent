"""JSON 文件缓存读写 —— 每个标的独立存储 OHLCV 数据。"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
from config import CACHE_DIR


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.json"


def load(symbol: str) -> list[dict]:
    """读取缓存的 OHLCV 数据，不存在返回空列表。"""
    p = _cache_path(symbol)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError):
        return []


def save(symbol: str, data: list[dict]) -> None:
    """写入 OHLCV 数据到缓存文件。"""
    p = _cache_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 按日期去重排序
    unique = {d["date"]: d for d in data if "date" in d}
    sorted_data = sorted(unique.values(), key=lambda x: str(x["date"]))
    p.write_text(json.dumps(sorted_data, ensure_ascii=False, indent=2), encoding="utf-8")


def last_date(symbol: str) -> str | None:
    """返回缓存中最新日期，无缓存返回 None。"""
    data = load(symbol)
    if not data:
        return None
    dates = sorted(d.get("date", "") for d in data)
    return dates[-1] if dates else None


def merge_incremental(symbol: str, new_data: list[dict]) -> list[dict]:
    """增量合并新数据到缓存，返回完整数据集。"""
    existing = load(symbol)
    existing_dates = {d.get("date") for d in existing}
    fresh = [d for d in new_data if d.get("date") not in existing_dates]
    if fresh:
        merged = existing + fresh
        save(symbol, merged)
        return merged
    return existing

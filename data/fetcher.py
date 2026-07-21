"""双源 K 线数据获取 —— AKShare（主）+ Sina API（备），增量拉取 + 缓存管理。"""
from __future__ import annotations
import json
import re
import time
import urllib.request
import urllib.error
from config import SINA_API_BASE, CACHE_DIR
from data.store import load, merge_incremental, last_date


# ---- AKShare 数据源 ----


def _fetch_kline_akshare(symbol: str, count: int = 800) -> list[dict]:
    """通过 AKShare 获取个股/指数 K 线（前复权）。

    Args:
        symbol: 纯数字代码，如 "000001" (平安银行), "000001" (上证指数)
        count: 拉取条数
    """
    try:
        from data.universe import get_stock_kline_akshare

        # 处理 symbol: sh600xxx → 600xxx, sz000xxx → 000xxx
        clean = symbol
        if clean.startswith("sh"):
            clean = clean[2:]
        elif clean.startswith("sz"):
            clean = clean[2:]

        result = get_stock_kline_akshare(clean, days=count)
        if result:
            return result
    except Exception as e:
        print(f"  [fetcher] AKShare fallback for {symbol}: {e}")

    return []


# ---- Sina API 数据源 ----


def _clean_jsonp(text: str) -> str:
    """去除 Sina API 返回的 JSONP 包装，提取纯 JSON 数组。"""
    # 格式: /* xxx */ callback_name([...])
    # 找到第一个 '[' 和最后一个 ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return text
    return text[start : end + 1]


def fetch_kline(symbol: str, count: int = 800, prefer: str = "sina") -> list[dict]:
    """从 Sina API 拉取 K 线数据（AKShare 作为备选）。

    Args:
        symbol: 标的代码，如 sh000001, sz399001
        count: 拉取数据条数
        prefer: "sina" (默认) 或 "akshare"
    """
    url = f"{SINA_API_BASE}?symbol={symbol}&scale=240&ma=no&datalen={count}"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            json_str = _clean_jsonp(raw)
            data = json.loads(json_str)

            if not isinstance(data, list):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return []

            result = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                day = item.get("day", "")
                try:
                    result.append({
                        "date": day,
                        "open": float(item.get("open", 0)),
                        "high": float(item.get("high", 0)),
                        "low": float(item.get("low", 0)),
                        "close": float(item.get("close", 0)),
                        "volume": float(item.get("volume", 0)),
                    })
                except (ValueError, TypeError):
                    continue

            return result

        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[fetcher] Sina failed for {symbol}, trying AKShare...")
            # Sina 失败后尝试 AKShare
            fallback = _fetch_kline_akshare(symbol, count)
            if fallback:
                print(f"  [fetcher] AKShare succeeded for {symbol} ({len(fallback)} rows)")
                return fallback
            print(f"[fetcher] Failed to fetch {symbol}: {e}")
            return []

    return []


def incremental_update(symbol: str) -> list[dict]:
    """增量拉取：比对缓存最新日期，只拉新增数据，合并后返回完整数据集。

    如果本地无缓存 → 拉800条全量
    如果本地有缓存 → 计算需要的新天数，拉取后合并
    """
    cached = load(symbol)
    if not cached:
        # 首次拉取
        fresh = fetch_kline(symbol, count=800)
        if fresh:
            merge_incremental(symbol, fresh)
        return load(symbol)

    cached_last = last_date(symbol)
    if cached_last is None:
        return cached

    # 只拉从缓存最后一天之后的新数据
    # 多拉几天以防有数据滞后
    new_data = fetch_kline(symbol, count=20)
    merged = merge_incremental(symbol, new_data)
    return merged


def batch_update(symbols: list[str], max_workers: int = 6) -> dict[str, list[dict]]:
    """批量增量更新多个标的（并行拉取）。

    Returns:
        {symbol: ohlcv_list}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(incremental_update, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                result[sym] = future.result()
            except Exception as e:
                print(f"  [fetcher] {sym} 拉取失败: {e}")
                result[sym] = load(sym)  # 保留缓存数据，不丢

    # 保持原始顺序输出
    return {sym: result.get(sym, []) for sym in symbols}


def refresh_stale(max_age_days: int = 2, max_workers: int = 6) -> int:
    """后台追更：拉取所有超过 N 天未更新的股票数据。

    Args:
        max_age_days: 缓存超过此天数则强制刷新
        max_workers: 并行数

    Returns:
        追更的股票数
    """
    from datetime import date, timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cutoff = str(date.today() - timedelta(days=max_age_days))
    stale_codes = []

    for f in sorted(CACHE_DIR.glob("*.json")):
        code = f.stem
        cached = load(code)
        if not cached:
            stale_codes.append(code)
            continue
        last = cached[-1].get("date", "0000-00-00")
        if last < cutoff:
            stale_codes.append(code)

    if not stale_codes:
        return 0

    print(f"  [fetcher] 后台追更 {len(stale_codes)} 只过期股票（>{max_age_days}天）...")
    updated = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(incremental_update, c): c for c in stale_codes}
        for future in as_completed(futures):
            try:
                future.result()
                updated += 1
            except Exception:
                pass

    print(f"  [fetcher] 追更完成: {updated}/{len(stale_codes)}")
    return updated

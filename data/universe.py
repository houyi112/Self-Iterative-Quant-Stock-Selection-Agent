"""动态全市场发现 —— AKShare 实时拉取全 A 股和行业板块，零硬编码。

每次运行时从东方财富（AKShare 后端）实时获取全市场数据。
不预设股票池，不预设板块列表。市场里有什么就分析什么。
"""
from __future__ import annotations
import pandas as pd


# ---- 股票 ----

def get_all_stocks() -> pd.DataFrame:
    """获取全 A 股实时行情（~5400 只）。

    Returns:
        DataFrame with columns: 代码, 名称, 最新价, 涨跌幅, 成交量, 成交额,
        换手率, 量比, 市盈率, 市净率, 总市值, 流通市值, 60日涨跌幅, 年初至今涨跌幅
    """
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[universe] AKShare stock_zh_a_spot_em 失败: {e}")

    # Fallback: 返回空 DataFrame（调用方自行处理）
    return pd.DataFrame()


def get_all_industries() -> list[dict]:
    """获取全市场申万行业板块（~80 个），按涨跌幅排序。

    Returns:
        [{name: "医药生物", code: "BK0438", change: 2.35, ...}, ...]
    """
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            # AKShare 返回中文列名
            results = []
            for _, row in df.iterrows():
                results.append({
                    "name": str(row.get("板块名称", "")),
                    "code": str(row.get("板块代码", "")),
                    "close": float(row.get("最新价", 0) or 0),
                    "change": float(row.get("涨跌幅", 0) or 0),
                    "total_cap": float(row.get("总市值", 0) or 0),
                    "turnover": float(row.get("换手率", 0) or 0),
                    "up_count": int(row.get("上涨家数", 0) or 0),
                    "down_count": int(row.get("下跌家数", 0) or 0),
                })
            # 按涨跌幅降序
            results.sort(key=lambda x: x["change"], reverse=True)
            return results
    except Exception as e:
        print(f"[universe] AKShare stock_board_industry_name_em 失败: {e}")

    return []


def get_industry_kline(code: str, days: int = 800) -> list[dict]:
    """获取行业板块指数的历史 K 线。

    Args:
        code: 板块代码，如 "BK0438"
        days: 拉取天数

    Returns:
        [{date, open, high, low, close, volume}, ...] 与 fetcher 接口一致
    """
    try:
        import akshare as ak
        df = ak.stock_board_industry_hist_em(
            symbol=code,
            period="日k",
            adjust="",
        )
        if df is None or df.empty:
            return []

        # 标准化列名
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # 只保留需要的列
        needed = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in needed if c in df.columns]]

        # 按日期排序，取最近 days 条
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
            df = df.sort_values("date").tail(days)

        return df.to_dict("records")
    except Exception as e:
        print(f"[universe] AKShare industry kline '{code}' 失败: {e}")
        return []


def get_stock_kline_akshare(symbol: str, days: int = 800) -> list[dict]:
    """从 AKShare 获取个股 K 线（前复权），作为 Sina API 的备选。

    Args:
        symbol: 股票代码，如 "sz000858"
        days: 拉取天数
    """
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            adjust="qfq",
        )
        if df is None or df.empty:
            return []

        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        needed = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in needed if c in df.columns]]

        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
            df = df.sort_values("date").tail(days)

        return df.to_dict("records")
    except Exception as e:
        print(f"[universe] AKShare kline '{symbol}' 失败: {e}")
        return []


def filter_stocks(
    max_price: float = 15.0,
    min_cap: float = 50.0,
    max_cap: float = 500.0,
) -> list[dict]:
    """从全市场动态筛选符合条件的股票。

    Args:
        max_price: 最高价格（元）
        min_cap: 最低流通市值（亿）
        max_cap: 最高流通市值（亿）

    Returns:
        [{code, name, close, change, cap, turnover, pe, pb, vol_ratio}, ...]
    """
    df = get_all_stocks()
    if df.empty:
        return []

    try:
        # 过滤条件
        price = pd.to_numeric(df.get("最新价", 0), errors="coerce")
        cap = pd.to_numeric(df.get("流通市值", 0), errors="coerce") / 1e8  # 元→亿
        turnover = pd.to_numeric(df.get("换手率", 0), errors="coerce")
        name = df.get("名称", "")

        mask = (
            (price > 1) &
            (price <= max_price) &
            (cap >= min_cap) &
            (cap <= max_cap) &
            (turnover > 0) &  # 排除停牌
            (~name.str.contains("ST|退|N", na=True))  # 排除 ST/退市/新股
        )
        filtered = df[mask].copy()

        results = []
        for _, row in filtered.iterrows():
            results.append({
                "code": str(row.get("代码", "")),
                "name": str(row.get("名称", "")),
                "close": float(row.get("最新价", 0) or 0),
                "change": float(row.get("涨跌幅", 0) or 0),
                "cap": round(float(row.get("流通市值", 0) or 0) / 1e8, 1),
                "turnover": float(row.get("换手率", 0) or 0),
                "pe": float(row.get("市盈率", 0) or 0),
                "pb": float(row.get("市净率", 0) or 0),
                "vol_ratio": float(row.get("量比", 0) or 0),
            })

        return results
    except Exception as e:
        print(f"[universe] filter_stocks 失败: {e}")
        return []

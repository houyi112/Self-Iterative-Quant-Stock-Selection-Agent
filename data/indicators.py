"""专业量化指标引擎 —— 纯 pandas + numpy 实现，零 LLM 参与。

7 大类 35 个指标，全部从 OHLCV 原始数据自算。
输入: list[dict] (按日期升序)，输出: pd.DataFrame (原数据 + 全部指标列) + 结构化摘要 dict。
"""
from __future__ import annotations
import math
import pandas as pd
import numpy as np


# ============================================================
# 工具函数
# ============================================================

def _to_df(data: list[dict]) -> pd.DataFrame:
    """将 OHLCV list 转为 DataFrame。"""
    if not data:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ============================================================
# 趋势类指标 (8)
# ============================================================

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    # SMA
    for p in [5, 10, 20, 60, 120]:
        df[f"sma_{p}"] = c.rolling(p).mean()

    # EMA
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()

    # MACD
    df["macd_dif"] = df["ema_12"] - df["ema_26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ADX
    df = _add_adx(df, h, l, c)

    # Aroon
    df = _add_aroon(df, h, l)

    # PSAR
    df = _add_psar(df, h, l)

    return df


def _add_adx(df: pd.DataFrame, h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.DataFrame:
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
    df["adx_plus_di"] = plus_di
    df["adx_minus_di"] = minus_di
    return df


def _add_aroon(df: pd.DataFrame, h: pd.Series, l: pd.Series, period: int = 14) -> pd.DataFrame:
    aroon_up, aroon_down = [], []
    for i in range(len(df)):
        if i < period:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            window_h = h.iloc[i - period : i + 1]
            window_l = l.iloc[i - period : i + 1]
            aroon_up.append(100 * (period - (period - window_h.values.argmax())) / period)
            aroon_down.append(100 * (period - (period - window_l.values.argmin())) / period)
    df["aroon_up"] = aroon_up
    df["aroon_down"] = aroon_down
    return df


def _add_psar(df: pd.DataFrame, h: pd.Series, l: pd.Series,
              af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    n = len(df)
    psar = [np.nan] * n
    trend = 1  # 1 = uptrend, -1 = downtrend
    ep = float(h.iloc[0])  # extreme point
    af = af_start
    psar[0] = float(l.iloc[0])

    for i in range(1, n):
        prev_psar = psar[i - 1]
        if trend == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            if float(l.iloc[i]) < psar[i]:
                trend = -1
                psar[i] = ep
                ep = float(l.iloc[i])
                af = af_start
            else:
                if float(h.iloc[i]) > ep:
                    ep = float(h.iloc[i])
                    af = min(af + af_step, af_max)
                psar[i] = min(psar[i], float(l.iloc[i - 1]), float(l.iloc[i - 2]) if i >= 2 else float(l.iloc[i - 1]))
        else:
            psar[i] = prev_psar - af * (prev_psar - ep)
            if float(h.iloc[i]) > psar[i]:
                trend = 1
                psar[i] = ep
                ep = float(h.iloc[i])
                af = af_start
            else:
                if float(l.iloc[i]) < ep:
                    ep = float(l.iloc[i])
                    af = min(af + af_step, af_max)
                psar[i] = max(psar[i], float(h.iloc[i - 1]), float(h.iloc[i - 2]) if i >= 2 else float(h.iloc[i - 1]))

    df["psar"] = psar
    return df


# ============================================================
# 动量类指标 (8)
# ============================================================

def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # RSI
    for p in [6, 14]:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # Stochastic (KDJ)
    low_n = l.rolling(14).min()
    high_n = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low_n) / (high_n - low_n + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]

    # CCI
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["cci_14"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # Williams %R
    df["willr_14"] = -100 * (high_n - c) / (high_n - low_n + 1e-10)

    # MFI (Money Flow Index)
    typical = (h + l + c) / 3
    money_flow = typical * v
    pos_flow = money_flow.where(typical > typical.shift(1), 0)
    neg_flow = money_flow.where(typical < typical.shift(1), 0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    mfr = pos_sum / (neg_sum + 1e-10)
    df["mfi_14"] = 100 - (100 / (1 + mfr))

    # Ultimate Oscillator
    df = _add_uo(df, h, l, c)

    return df


def _add_uo(df: pd.DataFrame, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.DataFrame:
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

    df["uo"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return df


# ============================================================
# 成交量类指标 (5)
# ============================================================

def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]

    # OBV
    obv = [0.0]
    for i in range(1, len(df)):
        if c.iloc[i] > c.iloc[i - 1]:
            obv.append(obv[-1] + v.iloc[i])
        elif c.iloc[i] < c.iloc[i - 1]:
            obv.append(obv[-1] - v.iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # Force Index
    df["force_idx"] = c.diff(1) * v

    # Ease of Movement
    half_range = (df["high"] - df["low"]) / 2
    box_ratio = (v / 1e8) / (half_range + 1e-10)
    df["eom"] = half_range.diff(1) / (box_ratio + 1e-10)

    # Price Volume Trend
    df["pvt"] = (v * c.pct_change().fillna(0)).cumsum()

    # Volume Ratio (量比)
    df["vol_ratio"] = v / v.rolling(20).mean()

    return df


# ============================================================
# 波动率类指标 (5)
# ============================================================

def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)

    # Keltner Channels
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    df["kc_upper"] = ema20 + 2 * atr
    df["kc_lower"] = ema20 - 2 * atr

    # Donchian Channels
    df["dc_upper"] = h.rolling(20).max()
    df["dc_lower"] = l.rolling(20).min()

    # ATR
    df["atr_14"] = tr.rolling(14).mean()

    # Historical Volatility (annualized)
    df["hist_vol_20"] = c.pct_change().rolling(20).std() * math.sqrt(252)

    return df


# ============================================================
# 统计类指标 (3)
# ============================================================

def _add_statistical(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    df["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)
    df["skew_20"] = c.rolling(20).skew()
    df["kurt_20"] = c.rolling(20).kurt()

    return df


# ============================================================
# 主入口
# ============================================================

def compute_all(data: list[dict]) -> pd.DataFrame:
    """计算全部 35+ 个指标，返回带所有列的 DataFrame。

    用法：
        df = compute_all(ohlcv_list)
        last_row = df.iloc[-1]  # 最新一行包含所有指标值

    同时提供向后兼容的列别名（ma5, rsi, volume_ratio 等），
    方便老代码用 .get() 访问。
    """
    if not data:
        return pd.DataFrame()

    df = _to_df(data)
    if df.empty or "close" not in df.columns:
        return df

    # 按顺序添加各指标族
    df = _add_trend(df)
    df = _add_momentum(df)
    df = _add_volume(df)
    df = _add_volatility(df)
    df = _add_statistical(df)

    # === 向后兼容：老接口字段 ===

    # 日涨跌幅
    df["daily_return"] = df["close"].pct_change() * 100

    # 连涨天数
    def _consec_up(series):
        up = series.diff() > 0
        result = [0] * len(series)
        cnt = 0
        for i in range(len(series)):
            if i == 0:
                continue
            if up.iloc[i]:
                cnt += 1
            else:
                cnt = 0
            result[i] = cnt
        return result
    df["consecutive_up_days"] = _consec_up(df["close"])

    # N日新高
    df["is_20day_high"] = df["close"] == df["close"].rolling(20).max()
    df["is_60day_high"] = df["close"] == df["close"].rolling(60).max()

    # 列名别名（消费者用老列名访问时能找到）
    _ALIASES = {
        "ma5": "sma_5", "ma10": "sma_10", "ma20": "sma_20",
        "ma60": "sma_60", "ma120": "sma_120",
        "ema12": "ema_12", "ema26": "ema_26",
        "rsi": "rsi_14", "rsi6": "rsi_6",
        "kdj_k": "stoch_k", "kdj_d": "stoch_d", "kdj_j": "stoch_j",
        "volume_ratio": "vol_ratio",
        "boll_upper": "bb_upper", "boll_mid": "bb_mid", "boll_lower": "bb_lower",
        "atr": "atr_14",
    }
    for alias, src in _ALIASES.items():
        if src in df.columns:
            df[alias] = df[src]

    # macd 嵌套结构兼容：老代码会做 indicators.get("macd", {}).get("dif", [None])[-1]
    # 直接在行级提供 macd_dif/macd_dea/macd_hist 列即可，无需嵌套

    return df


def compute_summary(data: list[dict]) -> dict:
    """计算所有指标后输出结构化摘要（供 LLM 消费，减少 token）。

    Returns:
        {
            "price": {...},
            "trend": {...},
            "momentum": {...},
            "volume": {...},
            "volatility": {...},
            "signal": {...}
        }
    """
    df = compute_all(data)
    if df.empty or len(df) < 20:
        return {"error": "数据不足（<20 条）"}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    c, h, l, v = last["close"], last["high"], last["low"], last["volume"]

    # 辅助函数：安全取值
    def _f(col, default=0.0):
        val = last.get(col, default)
        return round(float(val), 2) if pd.notna(val) else default

    def _b(col, default=False):
        val = last.get(col, 0)
        return bool(val) if pd.notna(val) else default

    return {
        "price": {
            "close": round(float(c), 2),
            "change": round(float(last.get("vol_ratio", 1) or 1), 2),
        },
        "trend": {
            "ma5": _f("sma_5"), "ma10": _f("sma_10"),
            "ma20": _f("sma_20"), "ma60": _f("sma_60"),
            "ma_alignment": (
                "多头" if _f("sma_5") > _f("sma_20") > _f("sma_60")
                else "空头" if _f("sma_5") < _f("sma_20") < _f("sma_60")
                else "交织"
            ),
            "macd_dif": _f("macd_dif"), "macd_dea": _f("macd_dea"),
            "macd_signal": "金叉" if last.get("macd_dif", 0) > last.get("macd_dea", 0) else "死叉",
            "adx": _f("adx"),
            "adx_regime": "趋势" if _f("adx") > 25 else "震荡",
            "aroon_up": _f("aroon_up"), "aroon_down": _f("aroon_down"),
            "psar": _f("psar"),
            "psar_signal": "多头" if float(c) > _f("psar", float(c) + 1) else "空头",
        },
        "momentum": {
            "rsi_6": _f("rsi_6"), "rsi_14": _f("rsi_14"),
            "rsi_regime": "超买" if _f("rsi_14") > 70 else "超卖" if _f("rsi_14") < 30 else "中性",
            "stoch_k": _f("stoch_k"), "stoch_d": _f("stoch_d"), "stoch_j": _f("stoch_j"),
            "stoch_signal": "超买" if _f("stoch_k") > 80 else "超卖" if _f("stoch_k") < 20 else "中性",
            "cci_14": _f("cci_14"),
            "willr_14": _f("willr_14"),
            "mfi_14": _f("mfi_14"),
            "uo": _f("uo"),
        },
        "volume": {
            "vol_ratio": _f("vol_ratio"),
            "obv_trend": "上升" if last.get("obv", 0) > (df.iloc[-5].get("obv", 0) if len(df) >= 5 else 0) else "下降",
            "force_idx": _f("force_idx"),
            "mfi_14": _f("mfi_14"),
        },
        "volatility": {
            "bb_upper": _f("bb_upper"), "bb_lower": _f("bb_lower"),
            "bb_position": round((float(c) - _f("bb_lower")) / (_f("bb_upper") - _f("bb_lower") + 1e-10), 2),
            "bb_squeeze": _f("bb_width") < (df["bb_width"].iloc[-20:].mean() * 0.7 if len(df) >= 20 else 1),
            "atr_14": _f("atr_14"),
            "atr_pct": round(_f("atr_14") / float(c) * 100, 2) if c else 0,
            "hist_vol_20": _f("hist_vol_20"),
        },
        "signal": {
            "trend_bias": "偏多" if _f("sma_5") > _f("sma_20") else "偏空",
            "momentum_bias": "偏多" if 30 < _f("rsi_14") < 70 and _f("macd_dif") > _f("macd_dea") else "偏空",
            "volume_quality": (
                "放量上涨" if _f("vol_ratio") > 1.2 and float(c) > float(prev.get("close", c))
                else "放量下跌" if _f("vol_ratio") > 1.2 and float(c) < float(prev.get("close", c))
                else "正常"
            ),
            "risk_warning": (
                "超买" if _f("rsi_14") > 70 or _f("stoch_k") > 80
                else "超卖" if _f("rsi_14") < 30 or _f("stoch_k") < 20
                else "无"
            ),
        },
    }


# ============================================================
# 便捷函数（兼容旧接口）
# ============================================================

def ma(data: list[dict], period: int) -> list:
    df = _to_df(data)
    sma = df["close"].rolling(period).mean()
    return [None if pd.isna(x) else float(x) for x in sma.tolist()]


def ma5(data): return ma(data, 5)
def ma10(data): return ma(data, 10)
def ma20(data): return ma(data, 20)
def ma60(data): return ma(data, 60)


def daily_return(data: list[dict]) -> float | None:
    if len(data) < 2:
        return None
    c0, c1 = data[-2].get("close", 0), data[-1].get("close", 0)
    if not c0:
        return None
    return (c1 - c0) / c0 * 100


def consecutive_up_days(data: list[dict]) -> int:
    closes = [d.get("close", 0) for d in data]
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            count += 1
        else:
            break
    return count


def is_n_day_high(data: list[dict], n: int = 20) -> bool:
    closes = [d.get("close", 0) for d in data]
    if len(closes) < n:
        return False
    return closes[-1] >= max(closes[-n:])


def volume_ratio(data: list[dict], period: int = 20) -> list:
    volumes = [d.get("volume", 0) for d in data]
    result = [None] * len(volumes)
    for i in range(period - 1, len(volumes)):
        avg = sum(volumes[i - period + 1 : i + 1]) / period
        result[i] = volumes[i] / avg if avg > 0 else 1.0
    return result

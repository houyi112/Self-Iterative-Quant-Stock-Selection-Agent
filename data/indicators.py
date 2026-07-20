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
# 新增：一目均衡表 (Ichimoku Cloud)
# ============================================================

def _add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    df["ichi_tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df["ichi_kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(26)
    df["ichi_senkou_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    df["ichi_chikou"] = c.shift(-26)
    df["ichi_cloud_top"] = df[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
    df["ichi_cloud_bottom"] = df[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)
    return df


# ============================================================
# 新增：Chaikin Money Flow (CMF)
# ============================================================

def _add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    hl_range = h - l
    mfm = ((c - l) - (h - c)) / (hl_range + 1e-10)
    mfv = mfm * v
    df["cmf"] = mfv.rolling(period).sum() / v.rolling(period).sum()
    return df


# ============================================================
# 新增：VWAP 偏离度
# ============================================================

def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    typical = (h + l + c) / 3
    cum_vp = (typical * v).cumsum()
    cum_v = v.cumsum()
    df["vwap"] = cum_vp / (cum_v + 1e-10)
    df["vwap_dev"] = (c - df["vwap"]) / (df["vwap"] + 1e-10) * 100
    return df


# ============================================================
# 新增：振幅趋势（筹码集中度近似）
# ============================================================

def _add_amplitude_trend(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    df["daily_amp"] = (h - l) / (c + 1e-10) * 100
    df["amp_ma5"] = df["daily_amp"].rolling(5).mean()
    df["amp_ma20"] = df["daily_amp"].rolling(20).mean()
    return df


# ============================================================
# 新增：量价背离
# ============================================================

def _add_volume_divergence(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    df["price_dir_5d"] = (c - c.shift(5)) / (c.shift(5) + 1e-10)
    df["vol_dir_5d"] = (v - v.rolling(5).mean()) / (v.rolling(5).mean() + 1e-10)
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
    df = _add_ichimoku(df)
    df = _add_cmf(df)
    df = _add_vwap(df)
    df = _add_amplitude_trend(df)
    df = _add_volume_divergence(df)

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


def compute_summary(data: list[dict], indicators=None) -> dict:
    """计算所有指标后输出结构化摘要（供 LLM 消费，减少 token）。

    Args:
        data: OHLCV 原始数据
        indicators: 预计算的指标 DataFrame（可选，避免重复 compute_all）

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
    import pandas as pd
    df = indicators if indicators is not None else compute_all(data)
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

    # 振幅趋势
    amp_now = _f("daily_amp")
    amp_5 = _f("amp_ma5")
    amp_20 = _f("amp_ma20")
    amp_signal = "收窄→集中" if amp_5 < amp_20 else "放大→分散"

    # Ichimoku
    cloud_top = _f("ichi_cloud_top")
    cloud_bot = _f("ichi_cloud_bottom")
    if float(c) > cloud_top:
        cloud_pos = "云上方（强势）"
    elif float(c) < cloud_bot:
        cloud_pos = "云下方（弱势）"
    elif cloud_top > cloud_bot:
        cloud_pos = "云中（震荡）"
    else:
        cloud_pos = "无云区"
    tk_cross = "金叉" if _f("ichi_tenkan") > _f("ichi_kijun") else "死叉"

    # CMF 方向
    cmf_val = _f("cmf")
    cmf_dir = "资金流入" if cmf_val > 0.05 else "资金流出" if cmf_val < -0.05 else "中性"

    # 量价背离
    p_dir = _f("price_dir_5d")
    v_dir = _f("vol_dir_5d")
    if p_dir > 0 and v_dir < 0:
        divergence = "上涨缩量（警示）"
    elif p_dir < 0 and v_dir > 0:
        divergence = "下跌放量（警示）"
    elif p_dir > 0 and v_dir > 0:
        divergence = "量价配合（健康）"
    else:
        divergence = "量价齐缩（观望）"

    return {
        "price": {
            "close": round(float(c), 2),
            "daily_return": round(float(last.get("daily_return", 0) or 0), 2),
            "ma5": _f("sma_5"),
            "ma10": _f("sma_10"),
            "ma20": _f("sma_20"),
            "ma60": _f("sma_60"),
            "ma120": _f("sma_120"),
        },
        "trend": {
            "ma5": _f("sma_5"), "ma10": _f("sma_10"),
            "ma20": _f("sma_20"), "ma60": _f("sma_60"), "ma120": _f("sma_120"),
            "ma_alignment": (
                "多头" if _f("sma_5") > _f("sma_20") > _f("sma_60")
                else "空头" if _f("sma_5") < _f("sma_20") < _f("sma_60")
                else "交织"
            ),
            "macd_dif": _f("macd_dif"), "macd_dea": _f("macd_dea"),
            "macd_hist": _f("macd_hist"),
            "macd_signal": "金叉" if _f("macd_dif") > _f("macd_dea") else "死叉",
            "adx": _f("adx"),
            "adx_regime": "趋势" if _f("adx") > 25 else "震荡",
            "adx_di_diff": round(_f("adx_plus_di") - _f("adx_minus_di"), 2),
            "aroon_up": _f("aroon_up"), "aroon_down": _f("aroon_down"),
            "aroon_signal": "上攻" if _f("aroon_up") > _f("aroon_down") else "下攻",
            "psar_signal": "多头" if float(c) > _f("psar", float(c) + 1) else "空头",
            "ichi_cloud": cloud_pos,
            "ichi_tk_cross": tk_cross,
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
            "eom": _f("eom"),
            "pvt": _f("pvt"),
            "pvt_trend": "上升" if _f("pvt") > (df.iloc[-5].get("pvt", 0) if len(df) >= 5 else _f("pvt")) else "下降",
            "cmf": round(cmf_val, 3),
            "cmf_signal": cmf_dir,
            "vwap_dev": round(_f("vwap_dev"), 2),
            "divergence": divergence,
        },
        "volatility": {
            "bb_position": round((float(c) - _f("bb_lower")) / (_f("bb_upper") - _f("bb_lower") + 1e-10), 2),
            "bb_squeeze": _f("bb_width") < (df["bb_width"].iloc[-20:].mean() * 0.7 if len(df) >= 20 else 1),
            "atr_pct": round(_f("atr_14") / float(c) * 100, 2) if c else 0,
            "hist_vol_20": _f("hist_vol_20"),
            "kc_position": round((float(c) - _f("kc_lower")) / (_f("kc_upper") - _f("kc_lower") + 1e-10), 2),
            "dc_position": round((float(c) - _f("dc_lower")) / (_f("dc_upper") - _f("dc_lower") + 1e-10), 2),
            "amp_now": round(amp_now, 2),
            "amp_signal": amp_signal,
        },
        "statistical": {
            "zscore_20": _f("zscore_20"),
            "skew_20": _f("skew_20"),
            "kurt_20": _f("kurt_20"),
            "is_20day_high": _b("is_20day_high"),
            "is_60day_high": _b("is_60day_high"),
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
            "money_flow": cmf_dir,
            "divergence": divergence,
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


def last_val(ind, col: str, default=0.0):
    """从 DataFrame 或 dict 中安全提取最后一个标量值。

    被 risk_scorer、report1_sectors 等多个模块共享使用。
    """
    import pandas as pd
    if isinstance(ind, pd.DataFrame) and col in ind.columns:
        val = ind[col].iloc[-1]
        return default if pd.isna(val) else float(val)
    if isinstance(ind, dict):
        val = ind.get(col, [None])
        if isinstance(val, list):
            v = val[-1] if val else None
            return default if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
        return default if val is None else val
    return default


def format_summary_for_llm(summary: dict) -> str:
    """将 compute_summary() 输出格式化为 LLM prompt 用的单行指标文本。

    供 report1/2/3 共享，避免重复的格式化代码。
    """
    trend = summary.get("trend", {})
    mom = summary.get("momentum", {})
    vol = summary.get("volume", {})
    vola = summary.get("volatility", {})
    stat = summary.get("statistical", {})
    sig = summary.get("signal", {})

    return " | ".join([
        # 趋势
        f"MA={trend.get('ma_alignment', '?')} "
        f"MA5/10/20/60/120={trend.get('ma5', 0):.1f}/{trend.get('ma10', 0):.1f}/{trend.get('ma20', 0):.1f}/{trend.get('ma60', 0):.1f}/{trend.get('ma120', 0):.1f}",
        f"MACD={trend.get('macd_signal', '?')} DIF={trend.get('macd_dif', 0):.3f} DEA={trend.get('macd_dea', 0):.3f} HIST={trend.get('macd_hist', 0):.3f}",
        f"ADX={trend.get('adx', 0):.0f}({trend.get('adx_regime', '?')}) DI_diff={trend.get('adx_di_diff', 0):.1f} PSAR={trend.get('psar_signal', '?')}",
        f"Aroon={trend.get('aroon_signal', '?')}(↑{trend.get('aroon_up', 0):.0f} ↓{trend.get('aroon_down', 0):.0f}) Ichimoku={trend.get('ichi_cloud', '?')} TK={trend.get('ichi_tk_cross', '?')}",
        # 动量
        f"RSI6/14={mom.get('rsi_6', 0):.0f}/{mom.get('rsi_14', 0):.0f}({mom.get('rsi_regime', '?')})",
        f"KDJ=K{mom.get('stoch_k', 0):.0f} D{mom.get('stoch_d', 0):.0f} J{mom.get('stoch_j', 0):.0f}({mom.get('stoch_signal', '?')})",
        f"CCI={mom.get('cci_14', 0):.0f} MFI={mom.get('mfi_14', 0):.0f} WillR={mom.get('willr_14', 0):.0f} UO={mom.get('uo', 0):.0f}",
        # 成交量
        f"量比={vol.get('vol_ratio', 1):.2f} OBV={vol.get('obv_trend', '?')} Force={vol.get('force_idx', 0):.0f}",
        f"EOM={vol.get('eom', 0):.1f} PVT={vol.get('pvt_trend', '?')} CMF={vol.get('cmf', 0):.3f}({vol.get('cmf_signal', '?')}) VWAP偏离={vol.get('vwap_dev', 0):.1f}%",
        f"量价={vol.get('divergence', '?')}",
        # 波动率
        f"BOLL pos={vola.get('bb_position', 0.5):.2f}{'收窄' if vola.get('bb_squeeze') else ''} KC={vola.get('kc_position', 0.5):.2f} DC={vola.get('dc_position', 0.5):.2f}",
        f"ATR%={vola.get('atr_pct', 0):.1f}% HVOL={vola.get('hist_vol_20', 0):.2f} 振幅={vola.get('amp_now', 0):.1f}%({vola.get('amp_signal', '?')})",
        # 统计
        f"Z={stat.get('zscore_20', 0):.1f} Skew={stat.get('skew_20', 0):.2f} Kurt={stat.get('kurt_20', 0):.2f} 20d高={stat.get('is_20day_high', False)} 60d高={stat.get('is_60day_high', False)}",
        # 综合信号
        f"趋势偏向={sig.get('trend_bias', '?')} 动量偏向={sig.get('momentum_bias', '?')} 量质={sig.get('volume_quality', '?')} 风险={sig.get('risk_warning', '?')} 资金={sig.get('money_flow', '?')} 量价={sig.get('divergence', '?')}",
    ])

"""梅花易数 + 天干地支 —— 双轨制。

梅花易数：起卦（纯计算）+ LLM 解读（保留兼容）
天干地支：直接交 LLM 推算干支 + 五行分析（主力）
"""
from __future__ import annotations
from datetime import datetime, date
from iching.hexagram import calculate_hexagram
from llm.client import chat_json
from llm.prompts import (
    ICHING_SYSTEM, ICHING_USER_TEMPLATE,
    GANZHI_SYSTEM, GANZHI_USER_TEMPLATE,
)


# ============================================================
# 梅花易数（保留兼容）
# ============================================================

def generate_iching_report(d: date = None, llm_enabled: bool = True) -> dict:
    """生成梅花易数报告。"""
    if d is None:
        d = date.today()

    hexagram = calculate_hexagram(d.year, d.month, d.day)

    if llm_enabled:
        interpretation = _get_llm_interpretation(hexagram)
        hexagram.update(interpretation)
    else:
        hexagram.update(_empty_iching_fields())

    hexagram["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return hexagram


def _get_llm_interpretation(hexagram: dict) -> dict:
    prompt = ICHING_USER_TEMPLATE.format(
        year=date.today().year,
        month=date.today().month,
        day=date.today().day,
        ben_gua_name=hexagram["ben_gua_name"],
        ben_gua_symbol=hexagram["ben_gua_symbol"],
        bian_gua_name=hexagram["bian_gua_name"],
        bian_gua_symbol=hexagram["bian_gua_symbol"],
        hu_gua_name=hexagram["hu_gua_name"],
        hu_gua_symbol=hexagram["hu_gua_symbol"],
        ti_gua_name=hexagram["ti_gua_name"],
        ti_element=hexagram["ti_element"],
        yong_gua_name=hexagram["yong_gua_name"],
        yong_element=hexagram["yong_element"],
        dong_yao=hexagram["dong_yao"],
        ti_yong_relation=hexagram["ti_yong_relation"],
    )
    return chat_json(ICHING_SYSTEM, prompt)


def _empty_iching_fields() -> dict:
    return {
        "hexagram_interpretation": "（易学模块未启用）",
        "dong_yao_revelation": "（易学模块未启用）",
        "market_judgment": "（易学模块未启用）",
        "five_element_table": [
            {"element": e, "rating": "—", "sectors": "", "reason": "未启用"}
            for e in ["金", "木", "水", "火", "土"]
        ],
        "one_sentence": "（易学模块未启用）",
    }


# ============================================================
# 天干地支（主力 —— LLM 直接推算干支 + 五行分析）
# ============================================================

def generate_ganzhi_report(d: date = None, llm_enabled: bool = True) -> dict:
    """生成干支分析报告。

    不做任何本地计算 —— 直接把公历日期交给 LLM，
    LLM 自行推算干支纪年，分析体用生克，输出板块吉凶。

    Args:
        d: 日期，默认今天
        llm_enabled: 是否启用 LLM

    Returns:
        结构化报告数据（与梅花易数同构，兼容 _compare_iching）
    """
    if d is None:
        d = date.today()

    if llm_enabled:
        result = _get_llm_ganzhi(d)
    else:
        result = _empty_ganzhi_fields()

    result["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["date"] = d.strftime("%Y-%m-%d")
    return result


def _search_ganzhi(d: date) -> str:
    """从万年历网站查询干支纪年，返回如 '丙午年 乙未月 丁酉日'。"""
    import urllib.request
    import re
    url = f"https://my.8s8s.com/wannianli/{d.year}/{d.year}-{d.month}-{d.day}.html"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            m = re.search(r"干支日期[：:]\s*(\S{2}年\s*\S{2}月\s*\S{2}日)", html)
            if m:
                return m.group(1)
            # 备用匹配
            m = re.search(r"(\S{2})年\s*(\S{2})月\s*(\S{2})日", html)
            if m:
                return f"{m.group(1)}年 {m.group(2)}月 {m.group(3)}日"
    except Exception as e:
        print(f"  [ganzhi] 干支查询失败: {e}")
    return ""


def _get_llm_ganzhi(d: date) -> dict:
    """调 LLM 推算干支 + 五行分析（先查询万年历确保干支正确）。"""
    ganzhi_lookup = _search_ganzhi(d)
    prompt = GANZHI_USER_TEMPLATE.format(
        year=d.year,
        month=d.month,
        day=d.day,
        ganzhi_lookup=ganzhi_lookup or "（查询失败，请根据日期自行确认干支）",
    )
    return chat_json(GANZHI_SYSTEM, prompt)


def _empty_ganzhi_fields() -> dict:
    return {
        "day_ganzhi": "（干支模块未启用）",
        "ti": "—",
        "yong": "—",
        "ti_yong_relation": "—",
        "key_relations": [],
        "market_judgment": "（干支模块未启用）",
        "five_element_table": [
            {"element": e, "rating": "—", "sectors": "", "reason": "未启用"}
            for e in ["金", "木", "水", "火", "土"]
        ],
        "one_sentence": "（干支模块未启用）",
    }

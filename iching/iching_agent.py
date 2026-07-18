"""梅花易数解读 —— 起卦（纯计算）+ LLM 解读。"""
from __future__ import annotations
from datetime import datetime, date
from iching.hexagram import calculate_hexagram
from llm.client import chat_json
from llm.prompts import ICHING_SYSTEM, ICHING_USER_TEMPLATE


def generate_iching_report(d: date = None, llm_enabled: bool = True) -> dict:
    """生成梅花易数报告。

    Args:
        d: 日期，默认今天
        llm_enabled: 是否启用 LLM 解读

    Returns:
        结构化报告数据
    """
    if d is None:
        d = date.today()

    # 1. 纯计算：起卦
    hexagram = calculate_hexagram(d.year, d.month, d.day)

    # 2. LLM：解读
    if llm_enabled:
        interpretation = _get_llm_interpretation(hexagram)
        hexagram.update(interpretation)
    else:
        hexagram.update({
            "hexagram_interpretation": "（易学模块未启用）",
            "dong_yao_revelation": "（易学模块未启用）",
            "market_judgment": "（易学模块未启用）",
            "five_element_table": [
                {"element": e, "rating": "—", "sectors": "", "reason": "未启用"}
                for e in ["金", "木", "水", "火", "土"]
            ],
            "one_sentence": "（易学模块未启用）",
        })

    hexagram["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return hexagram


def _get_llm_interpretation(hexagram: dict) -> dict:
    """调 LLM 做卦象解读。"""
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

    result = chat_json(ICHING_SYSTEM, prompt)
    return result

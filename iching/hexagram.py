"""梅花易数起卦引擎 —— 纯数学计算，零 LLM 参与。

基于公历日期起卦：年%8→上卦，月%8→下卦，日%6→动爻。
"""
from __future__ import annotations
from datetime import date


# 八卦编号 1-8
TRIGRAMS: dict[int, dict] = {
    1: {"name": "乾", "symbol": "☰", "element": "金", "direction": "天", "nature": "健"},
    2: {"name": "兑", "symbol": "☱", "element": "金", "direction": "泽", "nature": "悦"},
    3: {"name": "离", "symbol": "☲", "element": "火", "direction": "火", "nature": "丽"},
    4: {"name": "震", "symbol": "☳", "element": "木", "direction": "雷", "nature": "动"},
    5: {"name": "巽", "symbol": "☴", "element": "木", "direction": "风", "nature": "入"},
    6: {"name": "坎", "symbol": "☵", "element": "水", "direction": "水", "nature": "陷"},
    7: {"name": "艮", "symbol": "☶", "element": "土", "direction": "山", "nature": "止"},
    8: {"name": "坤", "symbol": "☷", "element": "土", "direction": "地", "nature": "顺"},
}

# 余数为0对应8
def _num_to_trigram(n: int) -> int:
    r = n % 8
    return r if r != 0 else 8

def _num_to_line(n: int) -> int:
    r = n % 6
    return r if r != 0 else 6


# 八卦 → 3位二进制（从上到下）: 1乾☰111, 2兑☱110, 3离☲101, 4震☳100,
#                            5巽☴011, 6坎☵010, 7艮☶001, 8坤☷000
_TRIGRAM_BITS: dict[int, tuple[int, int, int]] = {
    1: (1, 1, 1),  # 乾 ☰
    2: (1, 1, 0),  # 兑 ☱
    3: (1, 0, 1),  # 离 ☲
    4: (1, 0, 0),  # 震 ☳
    5: (0, 1, 1),  # 巽 ☴
    6: (0, 1, 0),  # 坎 ☵
    7: (0, 0, 1),  # 艮 ☶
    8: (0, 0, 0),  # 坤 ☷
}

# 二进制 → 卦编号 的反向映射
_BITS_TO_NUM: dict[tuple, int] = {v: k for k, v in _TRIGRAM_BITS.items()}


def _flip_line(trigram_num: int, line_pos: int) -> int:
    """翻转卦的某一位（line_pos: 1=上爻, 2=中爻, 3=下爻）。"""
    bits = list(_TRIGRAM_BITS[trigram_num])
    idx = line_pos - 1  # 1-based to 0-based
    bits[idx] = 1 - bits[idx]
    return _BITS_TO_NUM[tuple(bits)]


def _get_hexagram_lines(upper: int, lower: int) -> list[int]:
    """返回六爻的二进制数组（从下到上: 下卦[下中上] + 上卦[下中上]）。"""
    ub = list(_TRIGRAM_BITS[upper])  # 上中下
    lb = list(_TRIGRAM_BITS[lower])  # 上中下
    # 返回 1爻(下卦下) 到 6爻(上卦上)
    return [lb[2], lb[1], lb[0], ub[2], ub[1], ub[0]]


def _lines_to_hexagram(lines: list[int]) -> tuple[int, int]:
    """从六爻数组转回 (upper, lower) 卦编号。"""
    # lines[0..2] = 下卦的下中上 → lower bits (上中下): lines[2], lines[1], lines[0]
    lower_bits = (lines[2], lines[1], lines[0])
    # lines[3..5] = 上卦的下中上 → upper bits (上中下): lines[5], lines[4], lines[3]
    upper_bits = (lines[5], lines[4], lines[3])
    return _BITS_TO_NUM[upper_bits], _BITS_TO_NUM[lower_bits]


# 六十四卦简表（上卦×下卦）
# 索引 = (upper-1) * 8 + (lower-1)
_64_GUA = [
    "乾为天䷀", "天泽履䷉", "天火同人䷌", "天雷无妄䷘", "天风姤䷫", "天水讼䷅", "天山遁䷠", "天地否䷋",
    "泽天夬䷪", "兑为泽䷹", "泽火革䷰", "泽雷随䷐", "泽风大过䷛", "泽水困䷮", "泽山咸䷞", "泽地萃䷬",
    "火天大有䷍", "火泽睽䷥", "离为火䷝", "火雷噬嗑䷔", "火风鼎䷱", "火水未济䷿", "火山旅䷷", "火地晋䷢",
    "雷天大壮䷡", "雷泽归妹䷵", "雷火丰䷶", "震为雷䷲", "雷风恒䷟", "雷水解䷧", "雷山小过䷽", "雷地豫䷏",
    "风天小畜䷈", "风泽中孚䷼", "风火家人䷤", "风雷益䷩", "巽为风䷸", "风水涣䷺", "风山渐䷴", "风地观䷓",
    "水天需䷄", "水泽节䷻", "水火既济䷾", "水雷屯䷂", "水风井䷯", "坎为水䷜", "水山蹇䷦", "水地比䷇",
    "山天大畜䷙", "山泽损䷨", "山火贲䷕", "山雷颐䷚", "山风蛊䷑", "山水蒙䷃", "艮为山䷳", "山地剥䷖",
    "地天泰䷊", "地泽临䷒", "地火明夷䷣", "地雷复䷗", "地风升䷭", "地水师䷆", "地山谦䷎", "坤为地䷁",
]


def _get_gua_name(upper: int, lower: int) -> tuple[str, str]:
    """返回（卦名, 卦符号）。"""
    idx = (upper - 1) * 8 + (lower - 1)
    full = _64_GUA[idx]
    # 提取符号（最后一个字符）
    name = full[:-1]
    symbol = full[-1]
    return name, symbol


# 五行生克
FIVE_ELEMENTS = ["金", "木", "水", "火", "土"]

_ELEMENT_GENERATES = {
    "金": "水", "水": "木", "木": "火", "火": "土", "土": "金",
}
_ELEMENT_RESTRICTS = {
    "金": "木", "木": "土", "土": "水", "水": "火", "火": "金",
}

# 五行 → 板块映射
ELEMENT_SECTORS = {
    "金": "金融、有色金属、钢铁",
    "木": "医药、农业、环保",
    "水": "食品饮料、白酒、公用事业",
    "火": "科技、半导体、新能源、军工",
    "土": "房地产、基建、煤炭、化工",
}


def calculate_hexagram(year: int = None, month: int = None, day: int = None) -> dict:
    """梅花易数起卦。

    Args:
        year, month, day: 公历日期，默认当天

    Returns:
        结构化卦象数据
    """
    if year is None:
        today = date.today()
        year, month, day = today.year, today.month, today.day

    upper_num = _num_to_trigram(year)
    lower_num = _num_to_trigram(month)
    moving_line = _num_to_line(day)

    upper = TRIGRAMS[upper_num]
    lower = TRIGRAMS[lower_num]

    # 本卦
    ben_gua_name, ben_gua_symbol = _get_gua_name(upper_num, lower_num)

    # 变卦：动爻翻转对应位
    # 六爻排布（从下到上）：下卦[下,中,上] + 上卦[下,中,上] = 1爻..6爻
    # 下卦: 下/中/上 对应 1/2/3爻；上卦: 下/中/上 对应 4/5/6爻
    lines = _get_hexagram_lines(upper_num, lower_num)
    # 翻转动爻
    lines[moving_line - 1] = 1 - lines[moving_line - 1]
    new_upper_num, new_lower_num = _lines_to_hexagram(lines)
    bian_gua_name, bian_gua_symbol = _get_gua_name(new_upper_num, new_lower_num)

    # 互卦：2-3-4爻为下卦，3-4-5爻为上卦
    hu_lines = [
        lines[1], lines[2], lines[3],  # 2,3,4爻 → 互卦下卦的下中上
        lines[2], lines[3], lines[4],  # 3,4,5爻 → 互卦上卦的下中上 (unused for calc)
    ]
    # 互卦: upper from lines[2],lines[3],lines[4], lower from lines[1],lines[2],lines[3]
    hu_upper_num, hu_lower_num = _lines_to_hexagram([
        lines[1], lines[2], lines[3],  # 互卦下卦的三爻
        lines[2], lines[3], lines[4],  # 互卦上卦的三爻 → 按_lines_to_hexagram解析
    ])
    # _lines_to_hexagram expects [lo_lo, lo_mid, lo_hi, up_lo, up_mid, up_hi]
    # 互卦下卦来自原卦2-3-4爻: lines[1], lines[2], lines[3]
    # 互卦上卦来自原卦3-4-5爻: lines[2], lines[3], lines[4]
    hu_upper_num, hu_lower_num = _lines_to_hexagram([
        lines[1], lines[2], lines[3],   # 互下卦: 原2,3,4爻
        lines[2], lines[3], lines[4],   # 互上卦: 原3,4,5爻
    ])
    hu_gua_name, hu_gua_symbol = _get_gua_name(hu_upper_num, hu_lower_num)

    # 体用：动爻所在的卦为用卦，另一个为体卦
    if moving_line <= 3:
        # 动爻在下卦 → 下卦为用
        ti_trigram = upper
        yong_trigram = lower
    else:
        # 动爻在上卦 → 上卦为用
        ti_trigram = lower
        yong_trigram = upper

    ti_element = ti_trigram["element"]
    yong_element = yong_trigram["element"]

    # 体用生克关系
    relation = _get_ti_yong_relation(ti_element, yong_element)

    return {
        "ben_gua_name": ben_gua_name,
        "ben_gua_symbol": ben_gua_symbol,
        "bian_gua_name": bian_gua_name,
        "bian_gua_symbol": bian_gua_symbol,
        "hu_gua_name": hu_gua_name,
        "hu_gua_symbol": hu_gua_symbol,
        "ti_gua_name": ti_trigram["name"],
        "ti_element": ti_element,
        "yong_gua_name": yong_trigram["name"],
        "yong_element": yong_element,
        "dong_yao": moving_line,
        "ti_yong_relation": relation,
        "element_sectors": ELEMENT_SECTORS,
    }


def _get_ti_yong_relation(ti_element: str, yong_element: str) -> str:
    """判定体用生克关系。"""
    if ti_element == yong_element:
        return "比和（体用同属，和谐共振，吉）"

    if _ELEMENT_GENERATES.get(yong_element) == ti_element:
        return f"用生体（{yong_element}生{ti_element}，环境滋养主体，大吉）"

    if _ELEMENT_GENERATES.get(ti_element) == yong_element:
        return f"体生用（{ti_element}生{yong_element}，能量外泄，小凶）"

    if _ELEMENT_RESTRICTS.get(ti_element) == yong_element:
        return f"体克用（{ti_element}克{yong_element}，主体克制环境，小吉）"

    if _ELEMENT_RESTRICTS.get(yong_element) == ti_element:
        return f"用克体（{yong_element}克{ti_element}，环境压制主体，大凶）"

    return "未知"

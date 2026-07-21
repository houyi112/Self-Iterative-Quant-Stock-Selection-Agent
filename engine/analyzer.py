"""分析器抽象层 —— 所有 LLM 决策点统一接口，换模型改一行配置。

用法:
    from engine.analyzer import get_analyzer
    analyzer = get_analyzer()           # 从 ANALYZER 环境变量读取
    analyzer = get_analyzer("llm")      # 强制 LLM
    analyzer = get_analyzer("noop")     # 纯硬编码
    analyzer = get_analyzer("xgboost")  # 未来
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from datetime import date


class BaseAnalyzer(ABC):
    """分析器基类 —— 定义所有决策点接口。"""

    @abstractmethod
    def rank_sectors(self, sectors: list[dict], sector_data: dict) -> dict:
        """报告一：板块排序 + 领涨概率 + 叙事。

        Args:
            sectors: 板块列表 [{name, daily_return, consecutive_up, risk_level, risk_total, risk_scores, summary, indicators}, ...]
            sector_data: {sector_name: ohlcv_list} 原始数据（供大盘计算）

        Returns:
            {"rankings": [{name, probability, driver_logic, sustainability, key_risk}, ...],
             "market_narrative": "..."}
        """
        ...

    @abstractmethod
    def pick_stocks(self, leading_sectors: list[str], candidates: list[dict]) -> list[dict]:
        """报告二：从候选股中选出涨停潜力股。

        Args:
            leading_sectors: 领涨板块名称列表
            candidates: 候选股列表 [{code, name, sector, close, market_cap, is_20cm, summary, ...}, ...]

        Returns:
            [{code, thesis, limit_up_probability}, ...]
        """
        ...

    @abstractmethod
    def audit_shorts(self, picks: list[dict]) -> dict:
        """报告三：对选股做做空审查。

        Args:
            picks: 报告二的选股列表 [{code, name, close, limit_up_price, market_cap, is_20cm, summary, daily_return, consecutive_up, ...}, ...]

        Returns:
            {"stocks": [{code, name, signals, signal_count, level, verdict, remove_recommendation}, ...],
             "systemic_risk": "..."}
        """
        ...

    @abstractmethod
    def generate_insights(self, cumulative_stats: str, todays_deviation: str,
                          ganzhi_deviation: str, recent_insights: str) -> list[dict]:
        """报告四：从复盘数据中产出增量洞察。

        Returns:
            [{type, content, confidence}, ...]
        """
        ...

    @abstractmethod
    def analyze_ganzhi(self, d: date) -> dict:
        """天干地支五行分析。

        Returns:
            {day_ganzhi, ti, yong, ti_yong_relation, key_relations, market_judgment,
             five_element_table: [{element, rating, sectors, reason}, ...], one_sentence}
        """
        ...


# ============================================================
# LLM 实现
# ============================================================

class LLMAnalyzer(BaseAnalyzer):
    """DeepSeek LLM 分析器 —— 当前主实现。"""

    def rank_sectors(self, sectors: list[dict], sector_data: dict) -> dict:
        from reports.report1_sectors import _get_llm_analysis
        return _get_llm_analysis(sectors, sector_data)

    def pick_stocks(self, leading_sectors: list[str], candidates: list[dict]) -> list[dict]:
        from reports.report2_picks import _get_llm_picks
        return _get_llm_picks(leading_sectors, candidates)

    def audit_shorts(self, picks: list[dict]) -> dict:
        from reports.report3_short import _get_llm_audit
        return _get_llm_audit(picks)

    def generate_insights(self, cumulative_stats: str, todays_deviation: str,
                          ganzhi_deviation: str, recent_insights: str) -> list[dict]:
        from llm.client import chat_json
        from llm.prompts import REPORT4_SYSTEM, REPORT4_USER_TEMPLATE
        prompt = REPORT4_USER_TEMPLATE.format(
            cumulative_stats=cumulative_stats,
            todays_deviation=todays_deviation,
            ganzhi_deviation=ganzhi_deviation,
            recent_insights=recent_insights,
        )
        result = chat_json(REPORT4_SYSTEM, prompt, temperature=0.3)
        return result.get("insights", [])

    def analyze_ganzhi(self, d: date) -> dict:
        from iching.iching_agent import _get_llm_ganzhi
        return _get_llm_ganzhi(d)


# ============================================================
# 空实现（--no-llm 模式）
# ============================================================

class NoopAnalyzer(BaseAnalyzer):
    """纯硬编码分析器 —— 不调任何外部模型。"""

    def rank_sectors(self, sectors: list[dict], sector_data: dict) -> dict:
        return {"rankings": [], "market_narrative": ""}

    def pick_stocks(self, leading_sectors: list[str], candidates: list[dict]) -> list[dict]:
        return []

    def audit_shorts(self, picks: list[dict]) -> dict:
        result = {"stocks": [], "systemic_risk": ""}
        for p in picks:
            result["stocks"].append({
                "code": p["code"],
                "name": p.get("name", p["code"]),
                "signals": [],
                "signal_count": 0,
                "level": "—",
                "verdict": "LLM 未启用，跳过做空审查",
                "remove_recommendation": False,
            })
        return result

    def generate_insights(self, cumulative_stats: str, todays_deviation: str,
                          ganzhi_deviation: str, recent_insights: str) -> list[dict]:
        return []

    def analyze_ganzhi(self, d: date) -> dict:
        from iching.iching_agent import _empty_ganzhi_fields
        return _empty_ganzhi_fields()


# ============================================================
# 工厂
# ============================================================

_ANALYZERS: dict[str, type[BaseAnalyzer]] = {
    "llm": LLMAnalyzer,
    "noop": NoopAnalyzer,
}

_analyzer_instance: BaseAnalyzer | None = None


def get_analyzer(name: str | None = None) -> BaseAnalyzer:
    """获取分析器实例（单例）。

    Args:
        name: "llm" | "noop" | "xgboost"(未来)。默认从 ANALYZER 环境变量读取，未设置则为 "llm"。
    """
    global _analyzer_instance
    if name is None:
        name = os.getenv("ANALYZER", "llm")
    if _analyzer_instance is None or not isinstance(_analyzer_instance, _ANALYZERS.get(name, LLMAnalyzer)):
        cls = _ANALYZERS.get(name, LLMAnalyzer)
        _analyzer_instance = cls()
    return _analyzer_instance

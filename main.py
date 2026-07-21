#!/usr/bin/env python3
"""Stock Agent 主入口 —— 量化选股管线编排 + 定时调度。

Usage:
    python main.py --once              立即执行完整管线
    python main.py --once --date 2026-07-17  指定日期回溯
    python main.py --once --no-llm     纯硬编码模式（不调 LLM）
    python main.py --report 1          只执行报告一
    python main.py                     启动调度器（生产模式）
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRACKED_INDICES, SECTOR_STOCKS, ALL_STOCK_CODES,
    ENABLE_ICHING, STATE_DIR, SECTOR_ELEMENT,
)
from data.fetcher import batch_update
from data.store import load as load_cache
from data.indicators import daily_return
from engine.risk_scorer import score_all_sectors
from reports.report1_sectors import generate_report1
from reports.report2_picks import generate_report2
from reports.report3_short import generate_report3
from reports.report4_review import generate_report4
from reports.writer import write_report1, write_report2, write_report3, write_report4, write_iching, write_ganzhi
from data.backtest import (
    load_yesterdays_prediction, append_daily, load_log, load_insights,
)


def is_trading_day(d: date) -> bool:
    """判断是否为 A 股交易日（周一到周五且非节假日）。"""
    if d.weekday() >= 5:  # Sat=5, Sun=6
        return False
    holidays_path = STATE_DIR / "holidays.json"
    if holidays_path.exists():
        holidays_data = json.loads(holidays_path.read_text(encoding="utf-8"))
        all_holidays = holidays_data.get("holidays_2026", []) + holidays_data.get("holidays_2027", [])
        if str(d) in all_holidays:
            return False
    return True


def next_trading_day(d: date) -> date:
    """返回下一个 A 股交易日。"""
    nxt = d + __import__("datetime").timedelta(days=1)
    while not is_trading_day(nxt):
        nxt = nxt + __import__("datetime").timedelta(days=1)
    return nxt


def _save_last_ganzhi(d: date, data: dict) -> None:
    """保存今日干支预测，供下次复盘对比。"""
    import json as _json
    path = STATE_DIR / "last_ganzhi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({"date": str(d), "ganzhi": data}, ensure_ascii=False), encoding="utf-8")


def _load_yesterdays_ganzhi(run_date: date) -> dict | None:
    """加载最近一次干支预测（不含今天）。"""
    import json as _json
    path = STATE_DIR / "last_ganzhi.json"
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        if data.get("date") != str(run_date):
            return data.get("ganzhi")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def run_pipeline(run_date: date = None, llm_enabled: bool = True) -> dict:
    """执行完整选股管线。

    Args:
        run_date: 运行日期，默认今天
        llm_enabled: 是否启用分析器（LLM 或未来 XGBoost/微调模型）

    Returns:
        执行摘要
    """
    if run_date is None:
        run_date = date.today()
    date_str = str(run_date)

    # 分析器：统一决策入口（LLM / Noop / XGBoost / FineTuned）
    from engine.analyzer import get_analyzer
    analyzer = get_analyzer("noop" if not llm_enabled else None)

    print(f"\n{'='*60}")
    print(f"  Stock Agent Pipeline — {date_str}")
    print(f"  Analyzer: {type(analyzer).__name__}")
    print(f"  干支: {'启用' if (llm_enabled and ENABLE_ICHING) else '关闭'}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ============================================
    # Step 1: 数据更新（只拉指数，股票按需拉取）
    # ============================================
    print("[1/6] 拉取指数数据...")

    # 1a. 尝试 AKShare 动态发现板块
    dynamic_industries = []
    try:
        from data.universe import get_all_industries
        dynamic_industries = get_all_industries()
        if dynamic_industries:
            print(f"  ↳ AKShare: 发现 {len(dynamic_industries)} 个板块")
    except Exception as e:
        print(f"  ↳ AKShare 不可用，使用兜底板块: {e}")

    # 1b. 拉取所有指数数据
    index_data = {}
    index_codes = list(TRACKED_INDICES.keys())
    print(f"  ↳ 拉取 {len(index_codes)} 个指数...")
    for code, name in TRACKED_INDICES.items():
        data = batch_update([code])
        index_data[name] = data.get(code, [])

    print(f"  ✓ 指数拉取完成 ({time.time()-t0:.1f}s)\n")

    # ============================================
    # Step 2: 风险评分
    # ============================================
    print("[2/6] 风险评分...")

    # 预计算所有板块指标（后续步骤复用，避免重复 compute_all）
    from data.indicators import compute_all as _compute_all
    precomputed_indicators: dict[str, "pd.DataFrame"] = {}
    for name, ohlcv in index_data.items():
        if ohlcv and len(ohlcv) > 20:
            precomputed_indicators[name] = _compute_all(ohlcv)

    # 加载昨日预测（供 F005 板块轮动检测 + Step 6 复盘使用）
    yesterdays_pred = load_yesterdays_prediction()

    # 构建 market_context：传入大盘涨跌 → F003 计算逆市补跌
    market_context: dict = {}
    sh_ohlcv = index_data.get("上证指数", [])
    sz_ohlcv = index_data.get("深证成指", [])
    market_changes = {}
    for m_name, m_ohlcv in [("上证指数", sh_ohlcv), ("深证成指", sz_ohlcv)]:
        if m_ohlcv and len(m_ohlcv) >= 2:
            market_changes[m_name] = (m_ohlcv[-1]["close"] - m_ohlcv[-2]["close"]) / m_ohlcv[-2]["close"] * 100
    market_context["_market_changes"] = market_changes

    # 板块轮动检测：对比昨日排行榜 → F005
    yesterdays_rankings = []
    if yesterdays_pred and yesterdays_pred.get("rankings"):
        yesterdays_rankings = [r.get("name", "") for r in yesterdays_pred["rankings"][:3]]
    if yesterdays_rankings and len(index_data) >= 3:
        # 今日排行榜（从 sector_data 中取板块指数跌幅最小 = 相对最强）
        today_sectors = {name: ohlv for name, ohlv in index_data.items()
                         if name not in ["上证指数", "深证成指", "创业板指"]}
        today_ranked = sorted(
            [(n, (o[-1]["close"] - o[-2]["close"]) / o[-2]["close"] * 100 if o and len(o) >= 2 else -999)
             for n, o in today_sectors.items()],
            key=lambda x: x[1], reverse=True,
        )
        today_top3 = [n for n, _ in today_ranked[:3]]
        market_context["top3_rotation"] = len(set(today_top3) & set(yesterdays_rankings)) == 0
        # rotation_3day: 检查 backtest.json 中前两天的排行榜
        backtest_log = load_log()
        daily = backtest_log.get("daily", [])
        if len(daily) >= 3:
            prev3_top = []
            for entry in daily[-3:]:
                pred = entry.get("todays_prediction")
                if pred and pred.get("rankings"):
                    prev3_top.extend([r.get("name", "") for r in pred["rankings"][:3]])
            all_tops = set(today_top3 + prev3_top)
            market_context["rotation_3day"] = len(all_tops) >= 5  # 3天至少5个不同板块 = 高速轮动

    risk_scores = score_all_sectors(index_data, market_context,
                                    precomputed_indicators=precomputed_indicators)
    for name, scores in sorted(risk_scores.items(), key=lambda x: x[1].get("total", 0), reverse=True):
        if scores.get("total", 0) > 0:
            print(f"  ⚠️ {name}: {scores['total']}分 [{scores['level']}]")
    print(f"  ✓ 风险评分完成\n")

    # ============================================
    # Step 3: 报告一 + 干支分析（并行）
    # ============================================
    print("[3/6] 生成报告一 + 干支分析...")

    ganzhi_result = None
    if llm_enabled and ENABLE_ICHING:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_r1 = executor.submit(generate_report1, index_data, market_context, llm_enabled, precomputed_indicators, analyzer)
            next_td = next_trading_day(run_date)
            future_ganzhi = executor.submit(
                __import__("iching.iching_agent", fromlist=["generate_ganzhi_report"]).generate_ganzhi_report,
                next_td, llm_enabled, analyzer,
            )
            try:
                report1 = future_r1.result()
            except Exception as e:
                print(f"  ✗ 报告一生成失败: {e}")
                import traceback
                traceback.print_exc()
                # 报告一是核心，失败则降级为纯涨幅排序
                report1 = generate_report1(index_data, market_context, False, precomputed_indicators, analyzer)
            try:
                ganzhi_result = future_ganzhi.result()
            except Exception as e:
                print(f"  ✗ 干支分析失败: {e}")
                ganzhi_result = None
        print("  ↳ 报告一 + 干支分析 并行完成")
    else:
        report1 = generate_report1(index_data, market_context, llm_enabled,
                                  precomputed_indicators=precomputed_indicators, analyzer=analyzer)

    # 量化+干支共振对比
    if ganzhi_result:
        report1["iching_comparison"] = _compare_iching(report1, ganzhi_result)

    report1_path = write_report1(date_str, report1)
    print(f"  ✓ 报告一 → {report1_path}")

    if ganzhi_result:
        ganzhi_path = write_ganzhi(date_str, ganzhi_result)
        print(f"  ✓ 干支分析 → {ganzhi_path}")
        # 持久化干支预测（供下次复盘使用）
        _save_last_ganzhi(run_date, ganzhi_result)

    print()

    # ============================================
    # Step 4: 报告二（按需拉取领涨板块股票）
    # ============================================
    print("[4/6] 生成报告二...")
    leading_sectors = [s["name"] for s in report1.get("rankings", [])[:6]]
    print(f"  ↳ 领涨板块: {', '.join(leading_sectors)}")

    # 只拉取领涨板块对应的股票（而非全量 184 只）
    from config import SECTOR_STOCKS, INDEX_TO_SECTOR
    stock_codes_to_fetch: list[str] = []
    for sec_name in leading_sectors:
        pool_name = INDEX_TO_SECTOR.get(sec_name, sec_name)
        codes = SECTOR_STOCKS.get(pool_name, [])
        stock_codes_to_fetch.extend(codes)
    # 去重
    stock_codes_to_fetch = sorted(set(stock_codes_to_fetch))
    print(f"  ↳ 拉取 {len(stock_codes_to_fetch)} 只候选股...")
    stock_data = batch_update(stock_codes_to_fetch)

    report2 = generate_report2(leading_sectors, stock_data, llm_enabled, analyzer=analyzer)
    report2_path = write_report2(date_str, report2)
    n_picks = len(report2.get("picks", []))
    print(f"  ✓ 报告二 → {report2_path} ({n_picks} 只候选)\n")

    # ============================================
    # Step 5: 报告三
    # ============================================
    print("[5/6] 生成报告三（做空审查）...")
    report3 = generate_report3(report2.get("picks", []), llm_enabled, analyzer=analyzer)
    report3_path = write_report3(date_str, report3)

    n_removes = sum(1 for s in report3.get("stocks", []) if s.get("remove_recommendation"))
    print(f"  ✓ 报告三 → {report3_path} ({n_removes} 只建议剔除)\n")

    # ============================================
    # Step 6: 报告四（复盘）
    # ============================================
    print("[6/6] 生成报告四（复盘）...")

    # 构建今日实际数据（yesterdays_pred 已在 Step 2 加载）
    todays_actual = {}
    for name in index_data:
        ohlcv = index_data.get(name, [])
        if ohlcv and len(ohlcv) >= 2:
            close = ohlcv[-1]["close"]
            prev = ohlcv[-2]["close"]
            todays_actual[name] = {
                "close": close,
                "change": round((close - prev) / prev * 100, 2),
            }

    yesterdays_ganzhi = _load_yesterdays_ganzhi(run_date)
    # 清洗 rankings：只保留复盘需要的字段
    clean_rankings = []
    for s in report1.get("rankings", []):
        clean_rankings.append({
            "name": s.get("name", ""),
            "probability": s.get("probability", 0),
            "daily_return": s.get("daily_return", 0),
            "driver_logic": s.get("narrative", {}).get("driver_logic", ""),
            "sustainability": s.get("narrative", {}).get("sustainability", ""),
            "key_risk": s.get("narrative", {}).get("key_risk", ""),
        })

    todays_pred = {"rankings": clean_rankings}
    report4 = generate_report4(
        yesterdays_pred, todays_actual, llm_enabled, yesterdays_ganzhi,
        todays_prediction=todays_pred, analyzer=analyzer,
    )
    report4_path = write_report4(date_str, report4)
    n_insights = len(report4.get("new_insights", []))
    print(f"  ✓ 报告四 → {report4_path} ({n_insights} 条新洞察)")

    # 统一日志：增量写入 backtest.json（替代旧 review_log + backtest_csv + ganzhi_log）
    append_daily(date_str, yesterdays_pred, todays_pred, todays_actual,
                 report4.get("deviation", {}), report4.get("ganzhi_deviation"))
    print(f"  ✓ 回测日志 → state/backtest.json")

    # ============================================
    # 汇总
    # ============================================
    elapsed = time.time() - t0
    print(f"{'='*60}")
    print(f"  管线完成 — 耗时 {elapsed:.1f}s")
    print(f"  输出目录: output/{date_str}/")
    print(f"  领涨板块: {', '.join(leading_sectors)}")
    print(f"  候选股: {n_picks} 只")
    print(f"  剔除建议: {n_removes} 只")
    print(f"{'='*60}\n")

    return {
        "date": date_str,
        "elapsed": round(elapsed, 1),
        "leading_sectors": leading_sectors,
        "n_picks": n_picks,
        "n_removes": n_removes,
        "n_insights": n_insights,
    }


def _compare_iching(report1: dict, iching: dict) -> str:
    """对比量化排名与易学吉凶，生成共振/分歧描述。"""
    rankings = report1.get("rankings", [])
    element_table = iching.get("five_element_table", [])

    if not rankings or not element_table:
        return ""

    # 五行 → 吉凶 的映射
    element_rating = {e.get("element", ""): e.get("rating", "") for e in element_table}

    lines = []
    for s in rankings[:5]:
        name = s.get("name", "")
        el = SECTOR_ELEMENT.get(name, "")
        rating = element_rating.get(el, "—")
        prob = s.get("probability", 0)

        if rating in ("吉", "大吉") and prob >= 30:
            lines.append(f"- **{name}**：量化{prob}% + 易学{rating} → ✅ 量化+易学共振")
        elif rating in ("凶", "大凶") and prob >= 30:
            lines.append(f"- **{name}**：量化{prob}% + 易学{rating} → ⚠️ 分歧，建议谨慎")
        elif rating in ("凶", "大凶") and prob < 30:
            lines.append(f"- **{name}**：量化{prob}% + 易学{rating} → 双双看空，回避")
        else:
            lines.append(f"- **{name}**：量化{prob}% + 易学{rating}")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stock Agent Pipeline")
    parser.add_argument("--once", action="store_true", help="立即执行一次（不启动调度）")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM（纯硬编码模式）")
    parser.add_argument("--report", type=int, choices=[1, 2, 3, 4], default=None, help="只执行指定报告")
    args = parser.parse_args()

    run_date = date.today()
    if args.date:
        run_date = date.fromisoformat(args.date)

    if args.once:
        # 单次执行
        if not is_trading_day(run_date):
            print(f"[warning] {run_date} 不是交易日，继续执行...")
        llm_enabled = not args.no_llm

        if args.report:
            _run_single_report(args.report, run_date, llm_enabled)
        else:
            run_pipeline(run_date, llm_enabled)
    else:
        # 调度模式
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        print("启动调度器 — 每个交易日 18:00")
        scheduler = BackgroundScheduler()

        def scheduled_run():
            today = date.today()
            if is_trading_day(today):
                try:
                    run_pipeline(today, llm_enabled=True)
                except Exception as e:
                    print(f"[error] 管线执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[skip] {today} 非交易日")

        scheduler.add_job(
            scheduled_run,
            CronTrigger(hour=18, minute=0, day_of_week="mon-fri"),
            id="evening_pipeline",
            replace_existing=True,
        )
        scheduler.start()
        print("按 Ctrl+C 停止")

        try:
            import signal
            signal.pause()
        except (KeyboardInterrupt, SystemExit):
            print("\n停止调度器")
            scheduler.shutdown()


def _run_single_report(report_num: int, run_date: date, llm_enabled: bool):
    """单独执行某个报告（调试用）。"""
    date_str = str(run_date)

    if report_num == 1:
        index_data = {}
        for code, name in TRACKED_INDICES.items():
            data = batch_update([code])
            index_data[name] = data.get(code, [])
        r = generate_report1(index_data, {}, llm_enabled)
        write_report1(date_str, r)

    elif report_num == 2:
        stock_data = batch_update(ALL_STOCK_CODES)
        r = generate_report2(["医药", "TMT", "消费"], stock_data, llm_enabled)
        write_report2(date_str, r)

    elif report_num == 3:
        # 需要先跑报告二
        stock_data = batch_update(ALL_STOCK_CODES)
        r2 = generate_report2(["医药", "TMT", "消费"], stock_data, llm_enabled)
        r = generate_report3(r2.get("picks", []), llm_enabled)
        write_report3(date_str, r)

    elif report_num == 4:
        pred = load_yesterdays_prediction()
        actual = {}
        for code, name in TRACKED_INDICES.items():
            ohlcv = batch_update([code]).get(code, [])
            if ohlcv and len(ohlcv) >= 2:
                actual[name] = {
                    "close": ohlcv[-1]["close"],
                    "change": round((ohlcv[-1]["close"] - ohlcv[-2]["close"]) / ohlcv[-2]["close"] * 100, 2),
                }
        r = generate_report4(pred, actual, llm_enabled)
        write_report4(date_str, r)

    print(f"报告 {report_num} 已生成 → output/{date_str}/")


if __name__ == "__main__":
    main()

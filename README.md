# Stock Agent — A 股量化选股管线

全自动 A 股量化选股系统：19 个板块指数 × 26 个赛道 × 295 只股票，43 个技术指标，DeepSeek LLM 多维分析，梅花易数可选。

## 架构

```
18:00 每日管线 (crontab + APScheduler 双保险)
├── Step 1  数据更新 — 23 个指数 Sina 增量拉取（6 线程并行）
├── Step 2  风险评分 — F001-F008 因子引擎（F003/F004/F007 硬编码计算）
├── Step 3  报告一 + 干支 — LLM 多维预判明日领涨板块 + 天干地支分析（并行）
├── Step 4  报告二 — 硬过滤（价格<15 + 市值 50-500 亿）→ LLM 全量判断涨停潜力
├── Step 5  报告三 — 做空审查（隔离输入，五维尽调）
└── Step 6  报告四 — 硬编码偏差 + 累积统计 + LLM 增量洞察
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 DeepSeek API Key
python main.py --once         # 立即执行
python main.py --once --no-llm  # 纯硬编码模式（不调 LLM）
python main.py                  # 启动调度器（18:00 交易日）
```

## 项目结构

```
stock/
├── main.py                  # 管线编排 + CLI + 调度
├── config.py                # 环境配置 + 股票池（支持 JSON 外置）
├── requirements.txt
├── .env.example
│
├── data/                    # 数据层
│   ├── fetcher.py           # Sina/AKShare 双源 K 线拉取（并行）
│   ├── indicators.py        # 43 个技术指标 + compute_summary + 格式化
│   ├── store.py             # JSON 缓存读写
│   ├── universe.py          # AKShare 动态全市场发现
│   ├── backtest.py          # 统一复盘日志 + 累积统计 + 洞察
│   ├── sector_stocks.json   # 板块→成分股（可编辑）
│   ├── stock_names.json     # 代码→名称
│   └── fallback_caps.json   # 兜底市值
│
├── engine/                  # 引擎层
│   ├── risk_scorer.py       # F001-F008 风险因子评分 + 生命周期
│   └── screener.py          # 硬过滤筛选器（价格/市值/数据量）
│
├── reports/                 # 报告层
│   ├── report1_sectors.py   # 报告一：领涨板块预判
│   ├── report2_picks.py     # 报告二：涨停潜力股
│   ├── report3_short.py     # 报告三：做空审查
│   ├── report4_review.py    # 报告四：复盘 + 统计 + 洞察
│   └── writer.py            # Markdown 输出
│
├── llm/                     # LLM 层
│   ├── client.py            # DeepSeek API 客户端（错误分类 + JSON 提取）
│   └── prompts.py           # 全部 Prompt 模板
│
├── iching/                  # 易学模块
│   ├── hexagram.py          # 梅花易数起卦（纯计算）
│   └── iching_agent.py      # 干支分析（LLM 推算）
│
├── output/                  # 每日报告输出
├── cache/                   # K 线缓存
└── state/                   # 持久化状态
    ├── backtest.json        # 复盘日志 + 累积统计
    ├── insights.jsonl       # LLM 增量洞察
    ├── rules.json           # 风险因子规则库
    └── holidays.json        # 交易日历
```

## 技术指标（5 大类 43 个）

| 类别 | 指标 |
|------|------|
| 趋势 | MA(5/10/20/60/120), EMA(12/26), MACD, ADX, PSAR, Aroon, Ichimoku |
| 动量 | RSI(6/14), KDJ, CCI, MFI, Williams %R, Ultimate Oscillator |
| 成交量 | 量比, OBV, Force Index, EOM, PVT, CMF, VWAP |
| 波动率 | BOLL, Keltner, Donchian, ATR%, 历史波动率, 振幅 |
| 统计 | Z-Score, Skewness, Kurtosis, N日高点 |

## 风险因子（F001-F008）

| 因子 | 名称 | 数据源 |
|------|------|--------|
| F001 | 过热/超买 | 指标数据 |
| F002 | 高标情绪 | 外部数据（暂无） |
| F003 | 逆市补跌 | 硬编码计算 |
| F004 | 量价背离 | 硬编码计算 |
| F005 | 板块轮动 | 对比近 3 日排行榜 |
| F006 | 资金流向 | 外部数据（暂无） |
| F007 | 技术顶背离 | 硬编码计算 |
| F008 | 利好兑现 | 外部数据（暂无） |

因子生命周期：连续 3 次无效 → 降权，连续 8 次无效 → 废弃。

## 复盘系统

```
state/backtest.json  ← 每日日志 + 硬编码累积统计（命中率、板块偏差、干支准确率）
state/insights.jsonl  ← LLM 增量洞察（JSONL 追加，不覆盖）
```

报告四的 LLM 基于**硬编码聚合统计**产出增量洞察，不再全量重写知识库。

# CLAUDE.md

## 当前状态

A 股选股管线已补全为专业量化系统（2026-07-18 完成集成修复）：

1. ✅ **动态发现**：`data/universe.py` — AKShare 全市场股票列表 + 申万行业分类 + 兜底硬编码池。AKShare 在此 Mac 上网络不通，国内服务器部署后自动生效。
2. ✅ **专业指标**：`data/indicators.py` — 43 个指标（趋势/动量/成交量/波动率/统计 5 大类），纯 pandas+numpy，向后兼容老列名。
3. ✅ **实时数据**：`engine/screener.py` — AKShare 实时市值 + 硬编码兜底（graceful degradation）。
4. ✅ **风险引擎**：`engine/risk_scorer.py` — F003(逆市补跌)、F004(量价背离)、F007(技术顶背离) 从指标数据硬编码计算。F002/F006/F008 需外部数据（连板高度/北向/新闻），暂无数据时不触发。

## 项目概述

A 股量化选股管线，363 只股票 × 26 赛道 × 19 板块指数。Sina/AKShare 双源数据，DeepSeek LLM 定性分析，梅花易数可选。

- **板块指数**: 19 个（医疗/消费/计算机/通信/军工/传媒/有色/银行/证券/地产/新能/煤炭/钢铁/基建/油气/环保/建材/游戏/旅游）
- **股票池**: 26 个赛道 363 只（医药/计算机/通信/消费/金融/新能源/半导体/军工/有色/化工/汽车/电力/地产/煤炭/钢铁/传媒/家电/农业/机械/油气/交通/游戏/环保/建材/纺织服装/旅游）
- **指标**: 43 个技术指标（趋势/动量/成交量/波动率/统计）
- **风险**: F001-F008 打分引擎，F003/F004/F007 硬编码计算
- **调度**: 交易日 18:07 APScheduler

```bash
python main.py --once              # 立即执行
python main.py --once --no-llm     # 纯硬编码模式
python main.py --report 2          # 单独执行报告二
python main.py                     # 启动调度器（18:07 交易日）
```

---

## 自动执行（无需确认）

### 所有 Bash 命令
- 所有 `Bash` 命令默认自动执行
- 包括：curl、python3、cat、ls、grep、awk、jq、sort、head、tail、iconv、echo、for、while 等

### 所有搜索和抓取
- `WebSearch` 所有搜索请求
- `WebFetch` 所有网页抓取

### 所有 Agent
- `Agent` 启动子 Agent 执行任务

### 文件读写（安全路径）
- `Write` / `Edit` 记忆文件：当前项目的 `.claude/memory/`
- `Write` / `Edit` 当前项目下 `.md`、`.json`、`.txt` 文件

---

## 必须确认（每次弹框）

### 删除
- rm、rmdir、git rm
- 任何文件删除

### Git 高危
- git push、git push --force
- git reset --hard、git clean
- git rebase、git commit --amend
- 修改 .gitconfig

### 系统
- sudo
- chmod、chown
- brew install、npm install -g
- 修改 .bashrc、.zshrc、.profile

### 安全
- 向外部服务发送数据
- 操作账户密码、Token、密钥

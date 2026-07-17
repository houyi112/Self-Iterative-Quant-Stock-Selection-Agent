# Self-Iterative Quant Stock Selection Agent

A self-evolving multi-agent pipeline that hunts A-share market leaders before consensus forms.

## Architecture

```
18:07 Daily Pipeline
├── Report I   — Tomorrow's Leading Sector Prediction (8-dimension cross-validation)
├── Report II  — Limit-Up Candidate Screening (<¥15, ≤8 picks)
├── Report III — Short-Selling Audit (isolated sub-agent, 5-dimension DD)
└── Report IV  — Nightly Review → Risk Factor Extraction → Rules Evolution
```

## How It Works

| Component | Role |
|-----------|------|
| **Main Agent** | Predicts tomorrow's leaders, screens high-conviction picks |
| **Short-Selling Agent** | Stress-tests every pick from the bear side. Receives only ticker + price, no bullish context |
| **Review Agent** | Compares predictions vs reality. Extracts hard risk-factor rules from errors. Feeds them back |

## Rules Engine

Instead of fuzzy "reference weight," the system uses a **risk-factor scoring model**:

- Each sector gets scored across 5+ factors (overbought, sentiment fragility, counter-trend pullback risk, etc.)
- Scores are accumulated, not mechanically applied — the main agent decides
- Factors self-correct: 3 consecutive invalid signals → auto downgrade
- Contradictory signals → merged via root cause analysis

## Data Rules

- All OHLCV from Sina Finance API, self-computed indicators
- No borrowed analysis from third-party platforms
- Daily independent analysis — no cached judgments
- K-line data incremental pull, indicators always full recalculation

## Project Structure

```
stock/
├── CLAUDE.md              # Global config & permissions
├── README.md
└── .claude/
    ├── settings.local.json # Auto-execute rules
    └── memory/
        ├── a股资金流向预警.md # Workflow definition
        ├── rules.md           # Risk factor knowledge base
        └── 复盘总结.md         # Daily review log
```

## Quick Start

Open Claude Code in this directory. The pipeline triggers daily at 18:07.

```bash
cd "/path/to/stock"
claude
```

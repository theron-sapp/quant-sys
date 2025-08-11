# 🚀 Quant-Sys Quick Start Reference

## Current Development Status

```
✅ M0: Repo Foundations     [COMPLETE]
✅ M1: Data Pipeline        [COMPLETE]
✅ M2: Regime Detection     [COMPLETE]
🔜 M3: Advanced Features    [IN PROGRESS] <-- CURRENT
📋 M4: Fundamental Analysis [PENDING]
📋 M5: Strategy Implementation
📋 M6: Machine Learning
📋 M7: Portfolio Management
📋 M8: Options Overlay
📋 M9: Backtesting
📋 M10: Production System
```

## Essential Commands

```bash
# Setup (if needed)
pip install -e .

# Data Pipeline
quant ingest --top-n 50     # Get data
quant transform              # Weekly bars
quant check-data            # Verify

# Analysis
quant detect-regime         # Current regime

# Testing
python test_regime_detection.py
```

## Files to Update Between Sessions

| File                 | When to Update          | Location       |
| -------------------- | ----------------------- | -------------- |
| PROJECT_TRACKER.md   | After each session      | Root directory |
| Module Mapping Guide | If architecture changes | Artifacts/Docs |
| settings.yaml        | If config changes       | config/        |
| Chat Template        | Before new session      | Artifacts/Docs |

## New Session Checklist

```markdown
□ Update PROJECT_TRACKER.md with latest progress
□ Save any code artifacts locally
□ Note any errors or decisions
□ Fill in chat template with current files
□ Start new chat with complete template
□ Verify assistant understands context
□ Continue from "Next Task" in tracker
```

## Current Module Focus (M3)

```
features/
  ├── technical.py           # Basic indicators
  ├── high_quality_momentum.py # HQM from course
  ├── vol_models.py          # EWMA, GARCH
  └── factor_betas.py        # Fama-French
```

## Key Project Constraints

- Capital: $10,000 (max $5,000 exposure)
- Max Drawdown: 15%
- Rebalance: Weekly
- Max Position: 5%
- Max Sector: 25%
- Target Vol: 10-12% annual

## Regime Allocations

| Regime        | Growth % | Dividend % | Risk Scale |
| ------------- | -------- | ---------- | ---------- |
| Strong Growth | 80%      | 20%        | 100%       |
| Growth        | 60%      | 40%        | 90%        |
| Neutral       | 50%      | 50%        | 70%        |
| Dividend      | 40%      | 60%        | 50%        |
| Crisis        | 20%      | 80%        | 30%        |

## Quick Verification

Run these to verify system state:

```bash
quant check-data    # Should show 5,183 days
quant detect-regime # Should show current regime
```

## Next Implementation Priority

**M3 - Advanced Features**

1. Create features/ directory
2. Implement technical.py
3. Add high_quality_momentum.py
4. Build vol_models.py
5. Test feature calculation

---

_Keep this card handy for quick reference during development_

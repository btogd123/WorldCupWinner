# 2026 World Cup Prediction Model

Deep learning model predicting 2026 FIFA World Cup match outcomes.

**Model**: PyTorch neural network with self-attention, multi-task learning
**Accuracy**: 62.2% on 1,021 World Cup qualifiers (2024-2026)

---

## Quick Start

```bash
# Install uv (if not already)
pip install uv

# Clone & install
git clone https://github.com/btogd123/WorldCupWinner.git
cd WorldCupWinner
uv sync

# Predict all 72 World Cup matches
uv run python src/predict_wc2026.py

# Simulate full tournament
uv run python src/tournament_sim.py

# Betting analysis
uv run python src/betting.py --sample          # generate sample odds
uv run python src/betting.py --all             # analyze with real odds
uv run python src/betting.py --match "France" "Brazil" 2.50 3.20 2.80
```

## GPU Support

To enable CUDA (much faster training):

```bash
uv sync --override "torch @ https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp313-cp313-win_amd64.whl"
```

Check: `uv run python -c "import torch; print(torch.cuda.is_available())"`

## Model Architecture

- **Team Embedding** (64-dim) with home/away indicators
- **Multi-head Self-Attention** (4 heads) for team interactions
- **Residual Dense Blocks** [256, 256, 128]
- **Multi-task**: classification (win/draw/loss) + goal prediction
- **Focal Loss** with class weights for draw handling

## Features (21 dimensions)

| Category | Features |
|----------|----------|
| Elo (5) | elo_advantage, elo_quality, elo_diff_norm, elo_ratio, elo_gap |
| Form (3) | form_advantage, form_quality, win_rate_advantage |
| Goals (3) | goals_scored_adv, goals_conceded_adv, goal_diff_adv |
| Strength (2) | strength_advantage, match_quality |
| H2H (2) | h2h_dominance, has_h2h |
| Context (6) | is_neutral, year_norm, is_wc, is_wcq, is_continental, is_friendly |

## Top 10 Elo Rankings

1. 🇪🇸 Spain (2235)
2. 🇦🇷 Argentina (2203)
3. 🇫🇷 France (2148)
4. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England (2103)
5. 🇧🇷 Brazil (2095)
6. 🇨🇴 Colombia (2076)
7. 🇵🇹 Portugal (2071)
8. 🇪🇨 Ecuador (2051)
9. 🇩🇪 Germany (2034)
10. 🇳🇱 Netherlands (2024)

## Data

49,477 international matches (1872-2026) from [martj42/international_results](https://github.com/martj42/international_results).

## License

MIT

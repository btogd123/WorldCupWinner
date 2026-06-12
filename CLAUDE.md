# CLAUDE.md — Agent Handbook for World Cup 2026 Prediction

## Project Goal

Build a deep learning model to predict 2026 FIFA World Cup match winners. The model must use historical data with recent data (qualifiers), validated on 2024-2026 World Cup qualifiers.

## Quick Context

- **Current best model**: `ImprovedMatchPredictor` (PyTorch NN with Attention)
- **Performance**: 62.2% accuracy on 1,021 World Cup qualifiers, 56.9% overall test
- **Top Elo teams**: Spain (2235), Argentina (2203), France (2148), England (2103), Brazil (2095)
- **Tournament simulation winner**: France beats Belgium in final (penalties)

## Environment

```
Python: uv (Python 3.13) — just run `uv sync` to set up
GPU: CUDA available (downgrade torch to CPU if no GPU)
Working directory: D:/WorldCupWinner
PYTHONPATH MUST include D:/WorldCupWinner for imports to work
```

## Project Structure

```
D:/WorldCupWinner/
├── CLAUDE.md                          ← This file: agent handbook
├── README.md                          ← Human readme
├── pyproject.toml                     ← uv dependencies
├── uv.lock                            ← Locked deps
├── .gitignore
├── docs/
│   ├── literature_review.md           ← Survey of features/architectures to improve
│   └── data_sources.md                ← All APIs and data sources (free + paid)
├── data/
│   ├── results.csv                    ← Raw: 49,477 matches (martj42/international_results)
│   ├── processed_matches.csv          ← After preprocessing: + Elo + form + H2H + features
│   └── elo_ratings.csv                ← Final Elo per team
├── models/
│   ├── match_predictor.pt             ← Trained NN model (checkpoint)
│   ├── feature_scaler.pkl             ← StandardScaler for NN features
│   └── team_encoder.pkl               ← LabelEncoder for team IDs
├── results/
│   ├── improved_results.json          ← Training results: accuracy, F1, WCQ metrics
│   ├── wc2026_predictions.json/csv    ← All 72 WC group match predictions
│   ├── tournament_simulation.json     ← Full tournament simulation output
│   ├── training_history.json          ← Loss/acc curves
│   ├── final_report.txt               ← Text summary
│   ├── model_explanation.html         ← Deep dive explainer
│   └── odds.csv / odds_sample.csv     ← Betting odds (sample + real)
└── src/
    ├── __init__.py                    ← Package marker
    ├── config.py                      ← All paths, hyperparameters, constants
    ├── data_processor.py              ← Data pipeline: load → Elo → form → H2H → save
    ├── improved_model.py              ← NN architecture (ImprovedMatchPredictor)
    ├── train_improved.py              ← Training script for NN
    ├── predict_wc2026.py              ← Predict WC 2026 group matches
    ├── tournament_sim.py              ← Full 48-team tournament simulation
    ├── betting.py                     ← EV analysis + Kelly criterion
    └── generate_report.py            ← Final report generator
```

## How to Run Each Script

```bash
# Activate environment
uv sync

# Train the model (takes ~10min on GPU)
PYTHONPATH=D:/WorldCupWinner uv run python src/train_improved.py

# Predict all WC 2026 matches
PYTHONPATH=D:/WorldCupWinner PYTHONIOENCODING=utf-8 uv run python src/predict_wc2026.py

# Simulate full tournament (group + knockout)
PYTHONPATH=D:/WorldCupWinner PYTHONIOENCODING=utf-8 uv run python src/tournament_sim.py

# Betting analysis (single match)
PYTHONPATH=D:/WorldCupWinner uv run python src/betting.py --match "France" "Brazil" 2.50 3.20 2.80

# Betting analysis (batch, needs odds.csv)
PYTHONPATH=D:/WorldCupWinner uv run python src/betting.py --all
```

**Important**: Always use `PYTHONIOENCODING=utf-8` on Windows. The GBK console encoding causes Unicode errors with special characters.

## Model Architecture

### ImprovedMatchPredictor (459K params)
```
Inputs:
  ├── Home Team ID → Embedding(64-dim) + Home Indicator
  ├── Away Team ID → Embedding(64-dim) + Away Indicator
  └── Match Features (21-dim) → Encoder(128-dim)

Team Interaction: Multi-Head Self-Attention (4 heads)
  → Home + Away embeddings attend to each other

Combined (64+64+128=256-dim) → DenseBlock(256) → DenseBlock(256) → DenseBlock(128)

Outputs:
  ├── Classification Head: 128→64→32→3 (Away/Draw/Home, Focal Loss γ=2.0)
  └── Goal Prediction Head: 128→64→2 (Home goals, Away goals, MSE auxiliary)
```

### Features (21-dimensional)
| Category | Features | Source |
|----------|----------|--------|
| Elo (5) | elo_advantage_home, elo_quality, elo_diff_norm, elo_ratio, elo_gap | Dynamic Elo calculation |
| Form (3) | form_advantage, form_quality, wr_advantage | 10-match sliding window |
| Goals (3) | gs_advantage, gc_advantage, goal_diff_advantage | Rolling averages |
| Strength (2) | strength_advantage, match_quality | Elo + form composite |
| H2H (2) | h2h_dominance, has_h2h | Historical matchup lookup |
| Context (6) | is_neutral, year_norm, is_wc, is_wcq, is_continental, is_friendly | Match metadata |

### Training Config
- Train: 20,775 matches (2000-2021)
- Val: 2,025 matches (2022-2023)
- Test: 2,544 matches (2024-2026.6), including 1,021 WCQ
- Optimizer: AdamW (lr=0.001), CosineAnnealingWarmRestarts
- Loss: FocalLoss(γ=2.0) + 0.15 × MSE(goals)
- Class weights: [1.17, 1.43, 0.69] (away/draw/home)
- Early stopping: patience=25 on val F1
- Batch size: 64

## Critical Implementation Notes

### 1. Elo NaN bug (FIXED — do NOT reintroduce)
When calculating Elo, future matches (WC 2026) have NaN scores. The condition `NaN > NaN` returns False, which treated them as 0-0 draws and pulled all Elo toward the mean. **Always skip Elo update when `pd.isna(score)`**.

### 2. Time-series split (NOT random split)
Football is temporal. Training on 2024 to predict 2010 is cheating. Always use chronological split: train < val < test by date.

### 3. Data goes to 2026-06-27
The dataset includes WC 2026 fixtures (72 matches with team names but NaN scores). These must be excluded from training but CAN be predicted.

### 4. Windows encoding
All scripts print non-ASCII chars. Use `PYTHONIOENCODING=utf-8` or avoid emoji in print statements.

### 5. Elo ratings in processed_matches.csv
The `home_elo` column in processed_matches.csv is the Elo BEFORE that match. The Elo AFTER is `home_elo_after`. Only the pre-match Elo should be used as a feature (no look-ahead).

## Previous Iterations (What We Tried)

### Attempt 1: Basic NN
- File: `src/model.py` (DELETED — superseded)
- 14 features, simple ResidualBlock architecture
- Bug: NaN scores corrupted Elo calculation
- Result: 43% accuracy

### Attempt 2: Improved NN (CURRENT)
- File: `src/improved_model.py`
- 21 features, Attention, Focal Loss, Multi-task
- Elo bug fixed
- Result: 56.9% acc, 62.2% WCQ acc

### Attempt 3: Kaggle GBDT Ensemble (DELETED — inferior overall)
- Files: `src/train_kaggle.py`, `src/hybrid_ensemble.py` (DELETED)
- XGBoost + LightGBM, 30 features, draw-specific features
- Result: 60.3% acc, but **zero draw prediction** → useless for real use
- Lesson: GBDT kills minority class in 3-way classification

### Attempt 4: Literature Review
- File: `docs/literature_review.md`
- Comprehensive survey of GitHub projects + 2024-2025 papers
- Actionable improvement roadmap

## Data Sources & APIs

Full catalog in `docs/data_sources.md`. Quick reference:

**Already using**: `martj42/international_results` (GitHub raw CSV, 49K matches)

**Not yet tested (priority order):**
1. `football-data.org` — free, 10 req/min, WC 2026 covered
2. `API-Football` — free 100 req/day, odds + player stats
3. `football-data.co.uk` — free odds bulk download, no API key needed
4. `FBref` — xG, shot data (StatsBomb), free
5. `penaltyblog` — all-in-one Python package (Dixon-Coles, Bayesian, scrapers)
6. `StatsBomb Open Data` — event-level data, but latest WC is 2018 only
7. `SofaScore` / `Transfermarkt` — player ratings / market values, need scraping

**Paid options**: Sportmonks (€89/mo), Stats Perform/Opta (enterprise), API-Football Pro (~$120/yr)

## Improvement Roadmap (Next Steps)

From `docs/literature_review.md`, ranked by ROI:

| Priority | Item | Expected Gain | Difficulty |
|----------|------|--------------|------------|
| 🥇 | Add betting odds features | +3-5% Acc | Low |
| 🥈 | Pi-Rating instead of pure Elo | +1-3% Acc | Low |
| 🥉 | Draw-specific features | +5-10% Draw Recall | Low |
| 4 | Positional power (atk/def/mid) | +2-3% Acc | Medium |
| 5 | Player-level features (market value, age) | +2-4% Acc | Medium |
| 6 | Multi-Headed LSTM | +5-10% Acc | High |
| 7 | Stacking ensemble (precision-weighted) | +2-4% Acc | Low |

## Key File Dependencies
```
data/results.csv
  → src/data_processor.py → data/processed_matches.csv + data/elo_ratings.csv
    → src/train_improved.py → models/match_predictor.pt (+ scaler, encoder)
      → src/predict_wc2026.py → results/wc2026_predictions.json
      → src/tournament_sim.py → results/tournament_simulation.json
      → src/betting.py → results/betting_analysis.json
```

## Git & GitHub

- Remote: `https://github.com/btogd123/WorldCupWinner.git`
- Branch: `main`
- Commit message template: `Co-Authored-By: Claude <noreply@anthropic.com>`

## Related Memory

- [[world-cup-prediction-model]] — Project memory with architecture, performance, and key findings

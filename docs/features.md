# Feature Documentation — World Cup 2026 Match Predictor

26 features organized in 7 groups. All features are **pre-match** (no look-ahead bias).

---

## 1. Elo  (4 features)

Elo ratings are computed by `data_processor.py:calculate_elo_ratings()` using a dynamic system:
- **Initial Elo**: 1500 for all teams
- **K-factor**: 32 (base), scaled by tournament importance and goal difference
- **Home advantage**: +100 Elo points for non-neutral matches
- **Expected result**: `1 / (1 + 10^(-elo_diff / 400))`
- **NaN guard**: future matches (unplayed) do not trigger Elo updates

The `home_elo` / `away_elo` stored in `processed_matches.csv` are the **pre-match** Elo ratings, not the post-match values.

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 1 | `elo_diff` | 主客Elo原始差 | `home_elo - away_elo` | 西班牙2235 vs 巴西2087 → `+148` |
| 2 | `elo_quality` | 比赛质量（两队Elo水平） | `(home_elo + away_elo) / 2` | 西班牙2235 vs 巴西2087 → `2161` |
| 3 | `elo_ratio` | 主队Elo相对比值 | `home_elo / away_elo` | 2000 vs 1600 → `1.25` |
| 4 | `abs_elo_diff` | 两队绝对Elo差距 | `abs(home_elo - away_elo)` | 差200分 → `200` |

---

## 2. Form  (3 features)

Recent form is computed by `data_processor.py:calculate_recent_form()` using a **10-match sliding window**.
Each match in the window is weighted by recency (越近权重越大): weight = `(position / window_size)`.

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 5 | `form_advantage` | 主队状态优势 | `home_form - away_form` | 主队最近10场表现好 → `+1.2` ; 主队状态差 → `-0.8` |
| 6 | `form_quality` | 比赛状态水平 | `(home_form + away_form) / 2` | 两队都状态火热 → `2.5` ; 都低迷 → `0.3` |
| 7 | `wr_advantage` | 主队胜率优势 | `home_win_rate - away_win_rate` | 主队近10场胜率60% vs 客队30% → `0.30` |

**Form score** per match: 赢=3分, 平=1分, 输=0分, 加权平均后归一化到 [0, 3].

---

## 3. Goal  (3 features)

Rolling averages from the same 10-match window.

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 8 | `gs_advantage` | 进球能力优势 | `home_goals_scored_avg - away_goals_scored_avg` | 主队场均2.1球 vs 客队1.3球 → `0.80` |
| 9 | `gc_advantage` | 防守能力优势（失球少=好） | `home_goals_conceded_avg - away_goals_conceded_avg` | 主队场均失0.8 vs 客队失1.5 → `-0.70`（负数说明主队防守更好） |
| 10 | `goal_diff_advantage` | 净胜球优势 | `(GS_home - GC_home) - (GS_away - GC_away)` | 主队净胜球+1.3 vs 客队-0.2 → `+1.50` |

---

## 4. Head-to-Head  (2 features)

H2H records are computed by `data_processor.py:calculate_h2h_features()` — all past meetings between the two teams, regardless of venue.

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 11 | `h2h_dominance` | 历史交锋优势 | `(h2h_home_wins - h2h_away_wins) / h2h_count` (需要≥3场, 否则为0) | 10次交锋,主队赢6场客队赢2场 → `(6-2)/10 = 0.40` |
| 12 | `has_h2h` | 有历史交锋记录 | 二值: 1 = 有交锋记录, 0 = 无 | 首次碰面 → `0` |

---

## 5. Elo-Similarity  (3 features)

Computed by `data_processor.py:calculate_elo_similarity_features()`.

核心思想：找历史上 "Elo水平类似的两队" 的比赛, 看类似强队对类似弱队的表现如何。

**匹配规则**: 找到历史比赛中, `home_elo ≈ current_home_elo` (±75) 且 `away_elo ≈ current_away_elo` (±75) 的场次（或交换主客场）, 最多取最近10场。

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 13 | `sim_wr_advantage` | 相似Elo胜率优势 | `home_sim_win_rate - away_sim_win_rate` | 主队类似球队胜率60% vs 客队类似球队胜率30% → `0.30` |
| 14 | `sim_gs_advantage` | 相似Elo进球优势 | `home_sim_gs - away_sim_gs` | 类似主队场均1.8球 vs 类似客队场均1.2球 → `0.60` |
| 15 | `sim_dr_quality` | 相似Elo平局率水平 | `(home_sim_draw_rate + away_sim_draw_rate) / 2` | 平均平局率25% → `0.25` |

---

## 6. Draw-Specific  (6 features) — Hvattum 2017

来自论文 *"Forecasting Football Match Results — Are there any Draw-specific Characteristics?"* (Hvattum, 2017)。

研究发现以下因素会增加平局概率：两队实力接近、防守型球队对决、低进球倾向。

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 16 | `draw_rate_home` | 主队历史平局倾向 | 主队近10场平局比例 | 10场平了3场 → `0.30` |
| 17 | `draw_rate_away` | 客队历史平局倾向 | 客队近10场平局比例 | 10场平了4场 → `0.40` |
| 18 | `both_draw_prone` | 两队都容易平局 | `min(home_draw_rate, away_draw_rate)` | 主队30% vs 客队40% → `0.30` |
| 19 | `strength_parity` | 实力接近度 | `1 / (1 + |elo_diff| / 100)` | 差0分 → `1.00` ; 差200分 → `0.33` ; 差400分 → `0.20` |
| 20 | `defensive_similarity` | 防守相似度 | `1 / (1 + |home_GC_avg - away_GC_avg|)` | 都场均失1.0球 → `1.00` ; 一个失0.5一个失2.0 → `0.40` |
| 21 | `low_scoring_tendency` | 低进球倾向 | 二值: 两队近10场场均总进球都 < 2.5 则为1 | 主队场均总进球2.0 + 客队1.8 → `1` |

---

## 7. Context — Venue & Tournament  (5 features)

| # | Feature | Meaning | Calculation | Example |
|---|---------|---------|-------------|---------|
| 22 | `is_neutral` | 中立场地 | 1 = 中立场地, 0 = 有主客场 | 世界杯决赛 → `1` ; 世预赛主场 → `0` |
| 23 | `is_wc` | 世界杯正赛 | 1 = FIFA World Cup (不含预选赛) | 世界杯决赛圈 → `1` |
| 24 | `is_wcq` | 世界杯预选赛 | 1 = qualification | 亚洲区40强赛 → `1` |
| 25 | `is_continental` | 洲际锦标赛 | 1 = UEFA Euro / Copa América / AFC Asian Cup / AFCON / Gold Cup | 欧洲杯 → `1` |
| 26 | `is_friendly` | 友谊赛 | 1 = Friendly | 热身赛 → `1` |

---

## Appendix A: 原始数据列（Raw Columns from results.csv）

这些列直接来自数据源，不经过任何计算。

| # | Column | Meaning | Example |
|---|--------|---------|---------|
| R1 | `date` | 比赛日期 | `2018-07-15` |
| R2 | `home_team` | 主队名称 | `France` |
| R3 | `away_team` | 客队名称 | `Croatia` |
| R4 | `home_score` | 主队进球数（未赛=NaN） | `4` |
| R5 | `away_score` | 客队进球数（未赛=NaN） | `2` |
| R6 | `tournament` | 赛事名称 | `FIFA World Cup` |
| R7 | `city` | 比赛城市 | `Moscow` |
| R8 | `country` | 比赛国家 | `Russia` |
| R9 | `neutral` | 是否中立场地（TRUE/FALSE） | `TRUE`（世界杯正赛）/ `FALSE`（预选赛主场） |

---

## Appendix B: 中间特征（Intermediate Features — 不直接入模，但被26特征依赖）

以下列出现在 `processed_matches.csv` 中，但不在 26 个 `FEATURE_COLS` 中。它们是计算最终特征的中间产物，理解它们有助于理解最终特征的含义。

### B.1 Elo 中间列（calculate_elo_ratings）

| # | Column | Meaning | Calculation | Example |
|---|--------|---------|-------------|---------|
| I1 | `home_elo` | 主队**赛前** Elo 值 | 动态 Elo 系统，初始 1500，K=32，主场+100 | 西班牙 2235.2 |
| I2 | `away_elo` | 客队**赛前** Elo 值 | 同上 | 克罗地亚 2050.8 |
| I3 | `home_elo_after` | 主队**赛后** Elo 值 | 赛前 Elo + K × 重要性 × 进球系数 × (实际 - 预期) | 西班牙赢球后 2242.1 |
| I4 | `away_elo_after` | 客队**赛后** Elo 值 | 同上 | 克罗地亚输球后 2043.9 |
| I5 | `elo_diff` | Elo 差值 | `home_elo - away_elo` | 2235.2 - 2050.8 = `+184.4` |

> **关键约定**: `home_elo` / `away_elo` 是**赛前**值，避免 look-ahead bias。`home_elo_after` / `away_elo_after` 仅用于更新后续比赛的 Elo，不作特征。

**被以下入模特征依赖**: `elo_diff`, `elo_quality`, `elo_ratio`, `abs_elo_diff`, `strength_parity`

---

### B.2 Form 中间列（calculate_recent_form）

所有 Form 列基于**最近 10 场比赛**的滑动窗口，按时间远近加权（越近权重越大）。

**Form Score 计算**: 每场 赢=3, 平=1, 输=0，加权平均后归一化到 [0, 3]。

| # | Column | Meaning | Calculation | Example |
|---|--------|---------|-------------|---------|
| I6 | `home_form` | 主队近期状态分 | 10场加权平均 form score | 西班牙近10场7胜2平1负 → `(7×3+2×1)/10 = 2.30` |
| I7 | `away_form` | 客队近期状态分 | 同上 | 克罗地亚近10场5胜3平2负 → `(5×3+3×1)/10 = 1.80` |
| I8 | `home_win_rate` | 主队近期胜率 | 10场加权胜率 | 7/10 = `0.70` |
| I9 | `away_win_rate` | 客队近期胜率 | 同上 | 5/10 = `0.50` |
| I10 | `home_draw_rate` | 主队近期平局率 | 10场加权平局率 | 1/10 = `0.10` |
| I11 | `away_draw_rate` | 客队近期平局率 | 同上 | 3/10 = `0.30` |
| I12 | `home_goals_scored_avg` | 主队场均进球 | 10场加权平均进球 | 2.1 |
| I13 | `away_goals_scored_avg` | 客队场均进球 | 同上 | 1.5 |
| I14 | `home_goals_conceded_avg` | 主队场均失球 | 10场加权平均失球 | 0.8 |
| I15 | `away_goals_conceded_avg` | 客队场均失球 | 同上 | 1.0 |

> **NaN 处理**: 如果球队历史不足 10 场，用已有场次计算。

**被以下入模特征依赖**: `form_advantage`, `form_quality`, `wr_advantage`, `gs_advantage`, `gc_advantage`, `goal_diff_advantage`, `draw_rate_home`, `draw_rate_away`, `both_draw_prone`, `defensive_similarity`, `low_scoring_tendency`

---

### B.3 Head-to-Head 中间列（calculate_h2h_features）

H2H 统计两队**历史上所有交锋**（不限年份，不限赛事）。

| # | Column | Meaning | Calculation | Example |
|---|--------|---------|-------------|---------|
| I16 | `h2h_count` | 历史交锋总次数 | 统计两队所有历史碰面 | 西班牙 vs 克罗地亚历史交锋 8 次 → `8` |
| I17 | `h2h_home_wins` | 当前主队在历史交锋中的胜场 | 从当前主队视角统计 | 西班牙作为主队方赢了 3 次 → `3` |
| I18 | `h2h_away_wins` | 当前客队在历史交锋中的胜场 | 从当前客队视角统计 | 克罗地亚赢了 2 次 → `2` |
| I19 | `h2h_draws` | 历史交锋平局场次 | `h2h_count - h2h_home_wins - h2h_away_wins` | 8 - 3 - 2 = `3` |

**被以下入模特征依赖**: `h2h_dominance`, `has_h2h`

---

### B.4 Elo-Similarity 中间列（calculate_elo_similarity_features）

核心思想：找历史上 "Elo 水平类似的两队" 的比赛，看类似强队对类似弱队的表现如何。

**匹配规则**: 对每场比赛，在之前的所有比赛中找满足以下条件的场次（最多取最近 10 场）:
- `|历史主队Elo - 当前主队Elo| ≤ 75` **且** `|历史客队Elo - 当前客队Elo| ≤ 75`（mask1）
- 或主客场 Elo 互换匹配（mask2，即历史主队 ≈ 当前客队，且历史客队 ≈ 当前主队）

每列都有 `home_sim_*`（从当前主队视角）和 `away_sim_*`（从当前客队视角）两个版本。

| # | Column | Meaning | Calculation | Example |
|---|--------|---------|-------------|---------|
| I20 | `home_sim_win_rate` | 类似 Elo 的主队历史胜率 | 匹配场次中"类似主队"的胜率 | 类似西班牙(2235)的强队在类似对决中胜率 60% → `0.60` |
| I21 | `home_sim_draw_rate` | 类似 Elo 的主队历史平局率 | 匹配场次中"类似主队"的平局率 | 25% → `0.25` |
| I22 | `home_sim_gs` | 类似 Elo 的主队历史场均进球 | 匹配场次中"类似主队"的场均进球 | 1.8 |
| I23 | `home_sim_gc` | 类似 Elo 的主队历史场均失球 | 匹配场次中"类似主队"的场均失球 | 0.9 |
| I24 | `away_sim_win_rate` | 类似 Elo 的客队历史胜率 | 匹配场次中"类似客队"的胜率 | 类似克罗地亚(2050)的弱队在类似对决中胜率 30% → `0.30` |
| I25 | `away_sim_draw_rate` | 类似 Elo 的客队历史平局率 | 匹配场次中"类似客队"的平局率 | 25% → `0.25` |
| I26 | `away_sim_gs` | 类似 Elo 的客队历史场均进球 | 匹配场次中"类似客队"的场均进球 | 1.2 |
| I27 | `away_sim_gc` | 类似 Elo 的客队历史场均失球 | 匹配场次中"类似客队"的场均失球 | 1.3 |

**被以下入模特征依赖**: `sim_wr_advantage`, `sim_gs_advantage`, `sim_dr_quality`

---

### B.5 其他中间列（feature_engineering_v2）

这些列在 `feature_engineering_v2()` 中计算，用于派生出最终入模特征。

| # | Column | Meaning | Calculation | Used By |
|---|--------|---------|-------------|---------|
| I28 | `home_goal_diff_avg` | 主队场均净胜球 | `home_GS_avg - home_GC_avg` | `goal_diff_advantage` |
| I29 | `away_goal_diff_avg` | 客队场均净胜球 | `away_GS_avg - away_GC_avg` | `goal_diff_advantage` |
| I30 | `neutral` | 中立场地（原始bool） | 来自 results.csv | `is_neutral` |
| I31 | `tournament_importance` | 赛事重要性（0-3） | 世界杯正赛=3, 预选赛/洲际赛=2, 友谊赛=0 | 未入模（Elo 更新使用） |
| I32 | `home_team_id` | 主队整数编码 | LabelEncoder 编码 | 模型 Embedding 输入 |
| I33 | `away_team_id` | 客队整数编码 | 同上 | 模型 Embedding 输入 |
| I34 | `result` | 比赛结果编码 | 0=Away Win, 1=Draw, 2=Home Win | 训练标签（y） |

### B.6 未使用列（Not Used Anywhere）

以下列在 `processed_matches.csv` 中存在但**既不入模也不被任何入模特征依赖**：

| # | Column | Source | Remark |
|---|--------|--------|--------|
| N1 | `home_elo_after` | calculate_elo_ratings | 仅用于后续比赛的 Elo 计算 |
| N2 | `away_elo_after` | calculate_elo_ratings | 同上 |
| N3 | `group_round` | calculate_group_context | 曾入模，ablation 显示有害后被移除 |
| N4 | `is_already_qualified_home` | calculate_group_context | 同上 |
| N5 | `is_already_qualified_away` | calculate_group_context | 同上 |
| N6 | `must_win_to_survive_home` | calculate_group_context | 同上 |
| N7 | `must_win_to_survive_away` | calculate_group_context | 同上 |
| N8 | `h2h_draws` | calculate_h2h_features | 未被 h2h_dominance 使用 |
| N9 | `city` | results.csv | 仅用于元数据 |
| N10 | `country` | results.csv | 仅用于元数据 |
| N11 | `year` | engineer_features | 未直接入模 |
| N12 | `month` | engineer_features | 保留以备探索 |
| N13 | `days_since_first` | engineer_features | 保留以备探索 |
| N14 | `tournament_importance` | engineer_features | 仅用于 Elo K-factor 调整 |

---

## Preprocessing Pipeline

特征在送入模型前经过 `preprocessing.py:prepare_enhanced_data()` 处理：

1. **缺失值填充**: 所有 NaN 填充为 0
2. **StandardScaler**: z-score 标准化（均值0，标准差1）
3. **Binary passthrough**: 7 个二值特征不参与标准化，保持原始 0/1 值：
   - `is_neutral`, `is_wc`, `is_wcq`, `is_continental`, `is_friendly`
   - `has_h2h`, `low_scoring_tendency`
4. **is_neutral 特殊处理**: 即使不在 binary 列表中，也必须保持原始 0/1 值 — 标准化会破坏 neutral venue gating 的架构逻辑

---

## Feature Flow Summary

```
data/results.csv (49K raw matches)
    │
    ▼
data_processor.py
    ├── calculate_elo_ratings()          → home_elo, away_elo, elo_diff
    ├── calculate_recent_form()          → home_form, home_win_rate, home_draw_rate, GS/GC avg
    ├── calculate_h2h_features()         → h2h_count, h2h_home_wins, h2h_away_wins
    ├── calculate_elo_similarity_features() → home_sim_win_rate, home_sim_gs, etc.
    └── calculate_group_context()        → group_round, qualification status (不在26特征中)
    │
    ▼
data/processed_matches.csv (49K rows + all intermediate features)
    │
    ▼
features.py
    └── feature_engineering_v2()         → 完整26特征 (派生 + 归一化 + 编码)
    │
    ▼
preprocessing.py
    └── prepare_enhanced_data()          → StandardScaler + 转 tensor (binary特征保持原始0/1)
    │
    ▼
model.py                                → TeamAttentionNet 消费 26-dim 特征向量
```

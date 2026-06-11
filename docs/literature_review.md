# 足球比赛预测模型改进综述

> 基于 GitHub 开源项目、2024-2025 学术论文、Kaggle 竞赛方案的全面调研  
> 目标：找出可应用于我们 2026 世界杯预测模型的具体改进方向  

---

## 一、新的有用特征

### 1.1 博彩赔率特征（单项最强）

**来源**：[Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor)、[Atta Mills et al. (2024)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-01008-2)

赔率是多项研究公认的**最强单特征**，因为博彩市场已经消化了大量信息。

| 特征 | 计算方式 | 效果 |
|------|---------|------|
| `implied_home_prob` | 1/主胜赔率 | 直接反映市场判断 |
| `implied_draw_prob` | 1/平局赔率 | 帮助识别平局可能性 |
| `implied_away_prob` | 1/客胜赔率 | 同上 |
| `odds_margin` | Σ(1/赔率) − 1 | 反映比赛不确定性 |
| `odds_movement` | 赔率变化幅度 | 反映新信息冲击 |

**获取方式**：
- [football-data.co.uk](http://football-data.co.uk) — 免费，含 Bet365、William Hill 等多家赔率
- [OddsPortal](https://www.oddsportal.com) — 可爬取
- API-Football（免费层 100 次/天）

---

### 1.2 球员级特征（超越纯队伍级）

**来源**：[Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor)、[Unal-The-Engineer/Premier-League-Estimator](https://github.com/Unal-The-Engineer/Premier-League-Estimator)

| 特征 | 含义 | 实现难度 |
|------|------|---------|
| `player_elo_avg` | 首发 11 人平均 Elo | 中 |
| `player_elo_std` | 首发阵容 Elo 方差（默契度） | 中 |
| `key_player_missing` | 核心球员是否缺阵 | 中 |
| `squad_market_value` | 全队身价（Transfermarkt） | 低 |
| `avg_age` | 首发平均年龄 | 低 |
| `minutes_continuity` | 首发阵容连续性（与上场相同的球员数） | 中 |

**获取方式**：
- [Transfermarkt](https://www.transfermarkt.com) — 球员身价，可爬取
- [SofaScore](https://www.sofascore.com) — 球员评分，有 API
- [FBref](https://fbref.com) — StatsBomb 数据，免费
- [Understat](https://understat.com) — xG 数据，免费

---

### 1.3 高级队伍评分系统（替代纯 Elo）

**来源**：[calvinyeungck/Soccer-Prediction-Challenge-2023](https://github.com/calvinyeungck/Soccer-Prediction-Challenge-2023)、[andreyshelopugin/GlickoSoccer](https://github.com/andreyshelopugin/GlickoSoccer)

我们目前用 Elo。以下是更先进的评分系统：

| 系统 | 优势 | 来源 |
|------|------|------|
| **Pi-Rating** | 区分主/客场评分、指数衰减权重、日志化进球差 | [Constantinou & Fenton (2013)](https://doi.org/10.1007/s10994-013-5386-z) |
| **Berrar Rating** | 从数据中自动学习攻击/防守强度、每轮动态更新 | [Berrar et al. (2017)](https://doi.org/10.1007/s10994-018-5775-4) |
| **Glicko-2** | 评分波动性建模（σ²），不确定性量化 | [andreyshelopugin/GlickoSoccer](https://github.com/andreyshelopugin/GlickoSoccer) |
| **TrueSkill** | 贝叶斯推断，天然支持多人博弈 | 微软 Research |

**Pi-Rating 核心公式**：
```
r_home_new = r_home_old + λ × γ × (actual − expected)
r_away_new = r_away_old + λ × γ × (expected − actual)

其中:
  λ = 指数时间衰减权重
  γ = log(1 + |goal_diff|)  # 日志化进球差
  expected = 1 / (1 + 10^(r_away − r_home − home_adv) / 400)
```

---

### 1.4 球队实力分解特征（位置线强度）

**来源**：[Luiz, Fialho & Teixeira (2024)](https://www.mdpi.com/2571-9394/6/4/57)

将球队实力分解为三条线的相对强度：

| 特征 | 含义 |
|------|------|
| `attack_power_rel` | 相对进攻力 = 本队进攻 / 对手防守 |
| `defense_power_rel` | 相对防守力 = 本队防守 / 对手进攻 |
| `midfield_power_rel` | 相对中场力 = 本队中场 / 对手中场 |

**论文结论**：这三个特征的组合**超越了 Pi-Rating** 的预测能力。

**获取方式**：从 FIFA/EA Sports 球员属性、WhoScored 评分、或 xG 数据中提取。

---

### 1.5 比赛背景特征

**来源**：[TheDrawCode](https://github.com/ronyka77/TheDrawCode)、[Unal-The-Engineer/Premier-League-Estimator](https://github.com/Unal-The-Engineer/Premier-League-Estimator)

| 特征 | 含义 | 对预测的影响 |
|------|------|-------------|
| `days_since_last_match` | 距上一场天数 | 休息不足 → 主队优势减小 |
| `travel_distance` | 客场旅行距离（km） | 长途旅行 → 客队劣势增大 |
| `derby_match` | 是否德比/死敌战 | 德比 → 平局概率↑ |
| `match_importance` | 出线关键战/保级战/决赛 | 高重要性 → 主队更拼 |
| `temperature` / `weather` | 比赛天气 | 极端天气 → 进球数↓、不确定性↑ |
| `altitude` | 比赛地海拔 | 高原主场 → 主队优势显著 |
| `head_coach_days` | 教练在任天数 | 新教练 → 不确定性↑ |

---

### 1.6 竞技状态细粒度特征

**来源**：[Goal-Prediction-Model](https://github.com/iampreetdave/Goal-Prediction-Model)、[Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor)

| 特征 | 计算方式 |
|------|---------|
| `form_momentum` | 近 5 场积分趋势（线性回归斜率） |
| `goals_conceded_streak` | 连续不失球场次数 |
| `clean_sheet_rate` | 零封率（近 10 场） |
| `scoring_streak` | 连续进球场次数 |
| `comeback_ability` | 先丢球后拿分的比例 |
| `late_goal_tendency` | 75分钟后进球占比 |
| `xg_overperformance` | 实际进球 − xG（反映终结能力/运气） |
| `xga_underperformance` | 实际失球 − xGA（反映门将/防守运气） |

---

### 1.7 平局专用特征

**来源**：[TheDrawCode](https://github.com/ronyka77/TheDrawCode)、Hvattum (2017)

| 特征 | 计算 |
|------|------|
| `draw_rate_home` | 主队近 10 场平局率 |
| `draw_rate_away` | 客队近 10 场平局率 |
| `both_teams_draw_prone` | min(draw_rate_home, draw_rate_away) |
| `strength_parity` | 1 / (1 + |Elo_home − Elo_away| / 100) |
| `defensive_similarity` | 1 / (1 + |def_power_home − def_power_away|) |
| `low_scoring_tendency` | 两队场均总进球均 < 2.5 的概率 |

---

### 1.8 博彩市场衍生特征

**来源**：多个项目验证

| 特征 | 含义 |
|------|------|
| `over_under_2_5_prob` | 大小球隐含概率 |
| `btts_prob` | 双方进球（Both Teams To Score）隐含概率 |
| `asian_handicap` | 亚盘让球数 |
| `odds_volatility` | 赔率在赛前的变化幅度 |

---

## 二、如何获得这些特征

### 2.1 免费数据源汇总

| 数据源 | 覆盖 | 数据类型 | 难度 |
|--------|------|---------|------|
| [**football-data.co.uk**](http://football-data.co.uk) | 欧洲主要联赛 | 比赛结果 + 多家赔率 | ⭐ 极低 |
| [**FBref**](https://fbref.com) | 全球联赛/杯赛 | StatsBomb 数据：xG、射门、传球 | ⭐⭐ 低 |
| [**Understat**](https://understat.com) | 欧洲五大联赛 | xG、xGA、射门位置 | ⭐⭐ 低 |
| [**Transfermarkt**](https://transfermarkt.com) | 全球 | 球员身价、年龄、伤停 | ⭐⭐ 低 |
| [**SofaScore**](https://sofascore.com) | 全球 | 球员评分、技术统计 | ⭐⭐⭐ 中 |
| [**Club Elo**](http://clubelo.com) | 全球俱乐部 | Elo 评分（可直接下载） | ⭐ 极低 |
| [**World Football Elo**](https://www.eloratings.net) | 国家队 | Elo 评分 | ⭐ 极低 |
| [**penaltyblog**](https://github.com/martineastwood/penaltyblog) | — | **一站式 Python 工具包** | ⭐ 极低 |

### 2.2 penaltyblog —— 最佳一站式方案

```bash
pip install penaltyblog
```

这个包直接提供了：
- FBref、Understat、Club Elo、ESPN、FPL 的爬虫
- Dixon-Coles、Bivariate Poisson、贝叶斯层次模型
- Elo / Massey / Colley / Pi 评分系统
- 赔率隐含概率计算
- Opta / StatsBomb API 直连

### 2.3 球员数据获取路径

**国家队层面（适合我们）**：

1. **SofaScore API**：每场比赛的球员评分和统计数据
   - 免费层：基本评分
   - 可爬取赛后报告页
2. **Transfermarkt**：国家队大名单 + 球员身价
   - 直接爬取 HTML（robots.txt 宽松）
3. **FIFA 官方**：世界杯大名单、球员基本信息
4. **FBref**：世界杯预选赛的球员出场和基础统计

### 2.4 自动化数据管道

参考 [Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor) 的做法：

```
每日定时任务
  ├── 抓取前一日比赛结果 → 追加到数据库
  ├── 更新 Elo / Pi-Rating
  ├── 重新计算滑动窗口特征
  ├── 抓取即将进行的比赛赛程
  └── 运行预测 → 输出 CSV
```

---

## 三、模型架构 (Architecture)

### 3.1 2024-2025 年架构能力排名

从简单到复杂，按**已验证效果**排序：

#### Tier 1：梯度提升树（强大基线）

**项目**：[TheDrawCode](https://github.com/ronyka77/TheDrawCode)、多个 Kaggle 方案

```
XGBoost + LightGBM + CatBoost → Stacking Ensemble
  ├── 第一层：5 个异构模型（XGB, LGB, CatBoost, RF, KNN）
  ├── 第二层：Ridge 回归 / TabNet 作为元学习器
  └── 精度加权融合（对平局类赋予高精度权重）
```

**效果**：Home/Away 二分类 ~65-70%，三分类 ~52-55%  
**优势**：训练快、特征交互强、不需大量数据

#### Tier 2：多任务神经网络

**项目**：我们的现有模型、[Luiz et al. (2024)](https://www.mdpi.com/2571-9394/6/4/57)

```
Team Embedding (64-dim) + Attention
  ├── 分类头：Win / Draw / Loss（Focal Loss）
  └── 回归头：进球数预测（MSE，辅助任务）
```

**改进方向**：加入 Convolutional 层处理 Dixon-Coles 概率矩阵

#### Tier 3：Multi-Headed LSTM（已验证最佳公开方案）

**项目**：[Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor)

```
为每支球队创建独立的 LSTM Head：
  ├── 每场比赛拆成两行（主队视角 + 客队视角）
  ├── 训练时：冻结对手 Head，只训练当前队伍 Head
  ├── 每轮后：仅保留验证集表现最好的 Head
  └── 结合 Dixon-Coles 概率矩阵（Conv 层处理）
       + League Embedding（Attention 层）
       + Team Embedding
       + 球员级 Elo 评分
```

**效果**：Home/Away 79%，Home/Draw/Away 60%  
**这是目前 GitHub 上公开的足球预测最高准确率**

```
输入层:
  ├── 38场历史序列 → Bi-LSTM (per-team heads)
  ├── Dixon-Coles 矩阵 (10×10) → Conv2D
  ├── Power Ratings (Elo/Glicko2) → Dense
  ├── League Embedding → Attention
  ├── Team Embedding → Dense
  └── Player Features → Dense

  Concatenate → Dense(256) → Dropout → Dense(128) → Dense(3)
```

#### Tier 4：GNN + Transformer（学术 SOTA，工业级）

**项目**：[HIGFormer (KDD 2025)](https://dl.acm.org/doi/10.1145/3711896.3737082)、Stats Perform Axial Transformer

```
HIGFormer 架构:
  ├── Player Interaction Network（异质图神经网络）
  │   ├── 节点：球员（含 Laplacian 位置编码）
  │   ├── 边类型：10 种（传球、防守、争顶、拦截等）
  │   └── Heterogeneous GCN + Graph-Augmented Transformer
  ├── Team Interaction Network
  │   └── 队伍间的关系编码（市值、排名、历史）
  └── Match Comparison Transformer
      └── Mixture-of-Experts (MoE) 融合层

输出：预赛胜负预测（H/D/A）
```

**效果**：在所有 WyScout 数据集上超越之前方法  
**局限**：需要逐事件级别的球员数据（WyScout/StatsBomb），国家队层面难以获取

#### Tier 5：Axial Transformer（商业级，Stats Perform 专利）

**来源**：[WO/2024/145648 专利](https://sumobrain.com/patents/wipo/Systems-methods-transformer-neural-network/WO2024145648A1.html)

```
轴向注意力交替作用于：
  ├── 时间维度（比赛时间线）
  └── Agent 维度（球员/队伍）

输入粒度：
  ├── 球员级（位置、速度、动作类型）
  ├── 队伍级（阵型、控球率）
  └── 比赛级（比分、时间、红黄牌）

输出：75,000+ 个实时预测/场（12+ 种动作类型 × 所有球员）
```

---

### 3.2 对我们模型的具体改进建议

按**投入产出比**排序：

| 优先级 | 改进项 | 预期提升 | 实现难度 |
|--------|--------|---------|---------|
| 🥇 | **加入赔率特征**（implied_prob, odds_margin） | +3-5% Acc | ⭐ 低 |
| 🥈 | **Pi-Rating 替代纯 Elo**（区分主客、指数衰减） | +1-3% Acc | ⭐⭐ 低 |
| 🥉 | **进攻/防守/中场三条线分解特征** | +2-3% Acc | ⭐⭐⭐ 中 |
| 4 | **平局专用特征**（draw_rate、强度对等度） | +5-10% Draw Recall | ⭐ 低 |
| 5 | **球队状态动量特征**（form_momentum、xG overperformance） | +1-2% Acc | ⭐⭐ 低 |
| 6 | **球员级特征**（首发身价、年龄、核心缺阵） | +2-4% Acc | ⭐⭐⭐ 中 |
| 7 | **Multi-Headed LSTM** 序列建模 | +5-10% Acc | ⭐⭐⭐⭐ 高 |
| 8 | **Stacking Ensemble**（多模型 + 精度加权） | +2-4% Acc | ⭐⭐ 低 |

---

### 3.3 推荐的中期目标架构

结合可行性（国家队数据有限）和效果，建议演进至：

```
输入层:
  ├── Pi-Rating (主+客, 区分主客场) → Dense(32)
  ├── 赔率隐含概率 (3-dim) → Dense(16)
  ├── 位置线强度 (3-dim: 攻/防/中) → Dense(16)
  ├── 近期状态向量 (10场加权) → Dense(32)
  ├── 平局专用特征 (6-dim) → Dense(16)
  ├── 球员级特征 (首发身价/年龄/连续性) → Dense(16)
  ├── 比赛背景 (重要性/休息天数/中立场地) → Dense(16)
  └── H2H 历史 (5场加权) → Dense(16)

  Concatenate → ResBlock(256) → ResBlock(128) → ResBlock(64)
  
  多任务输出:
  ├── Win/Draw/Loss (Focal Loss, γ=2.0)
  └── Home Goals + Away Goals (Poisson NLL Loss)

后处理: Isotonic Calibration → 校准概率
```

### 3.4 训练技巧（来自各顶级方案）

| 技巧 | 来源 | 效果 |
|------|------|------|
| **TimeSeriesSplit CV**（非随机K-fold） | Kafrawy31, TheDrawCode | 防止时间泄露 |
| **精度加权融合**（非简单平均） | TheDrawCode | 提升少数类（平局） |
| **SMOTE-NC 过采样** | Atta Mills et al. | 平局 Recall +5-10% |
| **Isotonic Calibration** | 多个方案 | 概率更准确 |
| **1/4 Kelly 资金管理** | penaltyblog, 我们 | 控制风险 |
| **冻结对手 Head 训练** | Kafrawy31 | 防止过拟合 |
| **指数时间衰减**（近期比赛权重更高） | 多个方案 | +1-2% Acc |

---

## 四、参考资源

### GitHub 仓库

| 仓库 | Stars | 核心贡献 |
|------|-------|---------|
| [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) | ~140 | 一站式足球分析：Dixon-Coles、贝叶斯、Pi-Rating、爬虫 |
| [Kafrawy31/football-result-predictor](https://github.com/Kafrawy31/football-result-predictor) | — | **最佳公开准确率**：Multi-Headed LSTM + 球员 Elo + Dixon-Cole Conv |
| [ronyka77/TheDrawCode](https://github.com/ronyka77/TheDrawCode) | — | 平局预测专用：精度加权 Stacking + TabNet |
| [calvinyeungck/Soccer-Prediction-Challenge-2023](https://github.com/calvinyeungck/Soccer-Prediction-Challenge-2023) | — | Pi-Rating + Berrar Rating 实现 |
| [andreyshelopugin/GlickoSoccer](https://github.com/andreyshelopugin/GlickoSoccer) | — | Glicko-2 足球评分系统，Log Loss 0.583 |
| [Bowhza/euro2024-predictor](https://github.com/Bowhza/euro2024-predictor) | — | EURO 2024 预测：Elo + XGBoost |
| [mikedouzinas/euros-predictor](https://github.com/mikedouzinas/euros-predictor) | — | EURO 2024: RandomForest + FBref + FIFA Ranking |

### 论文

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| [HIGFormer (KDD 2025)](https://dl.acm.org/doi/10.1145/3711896.3737082) | 2025 | 异质图 Transformer → SOTA |
| [Luiz, Fialho & Teixeira (Forecasting)](https://www.mdpi.com/2571-9394/6/4/57) | 2024 | 位置线强度特征 > Pi-Rating |
| [Atta Mills et al. (J. Big Data)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-01008-2) | 2024 | SMOTE + Voting Ensemble |
| [Hvattum (IJCS Sports)](https://doi.org/10.1515/ijcss-2017-0004) | 2017 | 序数回归 vs 名义回归，平局预测困难 |
| [Constantinou & Fenton (MLJ)](https://doi.org/10.1007/s10994-013-5386-z) | 2013 | Pi-Rating 原始论文 |
| [Berrar et al. (MLJ)](https://doi.org/10.1007/s10994-018-5775-4) | 2018 | Berrar Rating：从数据学习评分 |
| [Dixon & Coles (JRSS)](https://doi.org/10.1111/1467-9876.00065) | 1997 | 经典模型：低比分相关性修正 |

---

## 五、总结

### 立即可做（本周）

1. ✅ 加入平局专用特征（draw_rate, parity）—— 提 Draw Recall
2. ✅ 加入博彩赔率特征—— 单特征最强，提 3-5% Acc
3. ✅ 用 Pi-Rating 或 Glicko-2 替换纯 Elo
4. ✅ 添加 form_momentum（状态趋势，非仅均值）

### 短期（有球员数据后）

5. 球员级特征：首发身价、平均年龄、核心缺阵
6. 攻/防/中三条线分解特征
7. Stacking Ensemble 精度加权融合

### 中期（充足数据）

8. Multi-Headed LSTM 序列建模
9. Dixon-Coles 矩阵 → Conv 层
10. HIGFormer 异质图网络（需 WyScout 事件数据）

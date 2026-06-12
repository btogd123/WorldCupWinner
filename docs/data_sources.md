# 足球数据 API 与数据源参考

> 记录我们在项目中调研过的所有数据获取渠道。  
> 标注了免费/付费、覆盖范围、接入难度，方便后续选择。

---

## 一、比赛结果与赛程

### 1. football-data.org
```
https://www.football-data.org/
```

| 属性 | 说明 |
|------|------|
| **免费额度** | 10 请求/分钟（需注册 API key） |
| **延迟** | ~5 分钟（实时比分有延迟） |
| **覆盖** | 欧洲主要联赛 + 世界杯 + 欧洲杯 |
| **数据类型** | 赛程、比分、积分榜、球队信息 |
| **Python 库** | 有 R 包 `worldcup26`（封装了 v4 API） |
| **备注** | 2026 世界杯在免费层覆盖范围内。最易上手。 |

```python
import requests
headers = {"X-Auth-Token": "YOUR_API_KEY"}
r = requests.get("https://api.football-data.org/v4/competitions/WC/matches", headers=headers)
```

---

### 2. API-Football
```
https://www.api-football.com/
```

| 属性 | 说明 |
|------|------|
| **免费额度** | 100 请求/天 |
| **延迟** | ~15 秒 |
| **覆盖** | 全球 ~900 联赛/杯赛 |
| **数据类型** | 赛程、比分、球员统计、阵容、赔率 |
| **备注** | 免费层覆盖广但日限额低。Pro 计划 ~$120/年。 |

```python
import requests
url = "https://v3.football.api-sports.io/fixtures"
headers = {"x-apisports-key": "YOUR_KEY"}
params = {"league": 1, "season": 2026}  # 1 = World Cup
```

---

### 3. TheSportsDB
```
https://www.thesportsdb.com/
```

| 属性 | 说明 |
|------|------|
| **免费额度** | ~100 请求/分钟（无需注册） |
| **延迟** | ~2 分钟 |
| **覆盖** | 全球联赛/杯赛 |
| **数据类型** | 赛程、比分、球队 Logo/图片 |

```python
# 无需 API key
r = requests.get("https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=Arsenal")
```

---

### 4. Apify Football Fixtures
```
https://apify.com/martin_sabo/football-fixtures-api
```

| 属性 | 说明 |
|------|------|
| **免费额度** | 免费层可用 |
| **付费** | $1.25 / 1000 场比赛 |
| **备注** | 封装了 football-data.org，提供 MCP server 集成 |

---

### 5. martj42/international_results（👈 我们正在使用）
```
https://github.com/martj42/international_results
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费 |
| **格式** | 单个 CSV 文件，直接下载 |
| **覆盖** | 1872–2026 所有国际 A 级赛事 |
| **字段** | date, home_team, away_team, home_score, away_score, tournament, city, country, neutral |
| **更新** | 社区维护，不定期更新 |
| **备注** | 2026 世界杯赛程已预填（72 场），比分列留空 |

```bash
curl -O https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

---

## 二、博彩赔率

### 6. football-data.co.uk
```
https://www.football-data.co.uk/
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费（Excel/CSV 下载） |
| **覆盖** | 欧洲主要联赛（英超到希腊超） |
| **赔率来源** | Bet365, William Hill, Pinnacle 等 20+ 家公司 |
| **备注** | 历史赔率批量下载，不需要 API。国家队比赛有限。 |

---

### 7. OddsPortal
```
https://www.oddsportal.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 免费浏览，需爬虫 |
| **覆盖** | 全球几乎所有比赛的赔率 |
| **历史** | 可追溯到多年前 |
| **备注** | 反爬较严，建议用 API-Football 替代 |

---

## 三、进阶数据（xG、球员、事件）

### 8. FBref（StatsBomb 数据）
```
https://fbref.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费 |
| **覆盖** | 全球 ~50 联赛 + 国际比赛 |
| **数据类型** | xG、xA、射门位置、传球、防守、门将、球员赛季统计 |
| **爬虫** | `pip install soccerdata`（Python 库，封装了 FBref） |
| **备注** | 国家队数据相对少，但世界杯/预选赛有覆盖 |

```python
import soccerdata as sd
fbref = sd.FBref(leagues="INTL World Cup", seasons=2026)
```

---

### 9. Understat
```
https://understat.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费 |
| **覆盖** | 欧洲五大联赛 + 俄超 |
| **数据类型** | xG、xA、射门位置图、球员/球队赛季统计 |
| **Python 库** | `pip install understat` 或 penaltyblog 内置爬虫 |

---

### 10. SofaScore
```
https://www.sofascore.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 免费浏览，有 API（需逆向） |
| **覆盖** | 全球联赛/杯赛 |
| **数据类型** | 球员评分（1-10）、热力图、技术统计（争顶、抢断等） |
| **备注** | API 非官方公开，需爬虫或逆向。Kafrawy31 等项目成功使用 |

---

### 11. StatsBomb（免费层）
```
https://github.com/statsbomb/open-data
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费（开放数据） |
| **覆盖** | 部分历史比赛（2018 世界杯、欧洲杯、女足世界杯、部分联赛） |
| **数据类型** | 逐事件数据：每脚传球、射门、抢断的坐标和时间 |
| **格式** | JSON，有 Python 库 `statsbombpy` |

```bash
pip install statsbombpy
```

```python
from statsbombpy import sb
matches = sb.matches(competition_id=43, season_id=3)  # 2018 World Cup
events = sb.events(match_id=matches.iloc[0].match_id)   # all events for a match
```

---

### 12. WyScout
```
https://wyscout.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 收费（学术研究可申请免费访问） |
| **覆盖** | 全球 200+ 联赛 |
| **数据类型** | 与 StatsBomb 同级：逐事件、球员追踪 |
| **备注** | HIGFormer 等学术 SOTA 模型的训练数据 |

---

## 四、球队/球员身价与属性

### 13. Transfermarkt
```
https://www.transfermarkt.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 免费浏览，需爬虫 |
| **覆盖** | 全球所有职业球队和球员 |
| **数据类型** | 球员身价、年龄、位置、伤停、合同到期日 |
| **备注** | 反爬较严，但有第三方 API 封装 |

```python
# 非官方 Python 库
# pip install transfermarkt-api  (社区维护，不一定稳定)
```

---

### 14. FIFA 官方排名 / 数据
```
https://inside.fifa.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 排名免费浏览 |
| **类型** | 男女足国家队排名、世界杯历史数据 |
| **API** | 无公开开发者 API（数据版权卖给 Opta） |

---

## 五、评分系统（直接可用）

### 15. World Football Elo Ratings
```
https://www.eloratings.net/
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费 |
| **覆盖** | 所有国家队，每日更新 |
| **格式** | 网页表格 + CSV 下载 |
| **备注** | 比 FIFA 排名更准确的实力评估 |

---

### 16. Club Elo
```
http://clubelo.com/
```

| 属性 | 说明 |
|------|------|
| **费用** | 完全免费 |
| **覆盖** | 全球 ~50 国家联赛的俱乐部 |
| **格式** | CSV 下载，含历史每日 Elo |
| **备注** | penaltyblog 内置了爬虫 |

---

## 六、一体化工具包

### 17. penaltyblog ⭐
```
https://github.com/martineastwood/penaltyblog
pip install penaltyblog
```

**内置数据源**：
- FBref 爬虫
- Understat 爬虫
- Club Elo 爬虫
- football-data.co.uk 爬虫
- ESPN 爬虫
- Opta / StatsBomb API 直连

**内置模型**：
- Poisson / Bivariate Poisson / Dixon-Coles / Negative Binomial
- 贝叶斯层次模型（原生 MCMC 采样器）
- Elo / Massey / Colley / Pi 评分

```python
from penaltyblog.models import DixonColesGoalsModel
from penaltyblog.scrapers import FBref

# 一步到位：爬数据 → 建模 → 预测
```

---

## 七、收费 API（参考）

### 18. Sportmonks
```
https://www.sportmonks.com/
```
- 起步价 ~€89/月
- 实时比分、球员统计、赔率
- 有 2026 世界杯专门套餐

### 19. Stats Perform (Opta)
```
https://www.statsperform.com/
```
- FIFA 官方数据合作商
- 企业定制价格（天价）
- 数据最全面最权威

### 20. API-Football (Pro)
```
https://www.api-football.com/pricing
```
- Pro 计划 ~$120/年
- 实时比分 + 历史全量 + 阵容详情
- 性价比最高的付费方案

---

## 八、数据源选择建议（按场景）

### 场景 1：延续当前模型，最小改动

| 数据 | 来源 | 用途 |
|------|------|------|
| 比赛结果 | `martj42/international_results` | 已有，继续用 |
| 赔率（新特征）| `football-data.co.uk` 或 `API-Football` 免费层 | 加入模型特征 |

### 场景 2：加入球员级特征

| 数据 | 来源 |
|------|------|
| 球员身价、年龄 | `Transfermarkt`（爬虫） |
| 球员评分 | `SofaScore`（爬虫） 或 `FBref`（免费） |

### 场景 3：全面升级（xG + 事件级数据）

| 数据 | 来源 |
|------|------|
| xG / 射门数据 | `FBref` + `Understat` |
| 事件级数据 | `StatsBomb Open Data`（免费，但世界杯只到 2018） |

### 场景 4：最高准确率（对齐公开 SOTA）

| 数据 | 来源 |
|------|------|
| 全部 | `penaltyblog`（一站式） + `API-Football (Pro)`（赔率 + 球员） |

---

## 九、自动化数据管道参考

来自 Kafrawy31/football-result-predictor 的做法：

```
┌─────────────────────────────────────────┐
│              每日定时任务                  │
├─────────────────────────────────────────┤
│ 1. 抓取前一日比赛结果 (API-Football)       │
│ 2. 追加到本地数据库 (SQLite / CSV)         │
│ 3. 重新计算 Elo / Pi-Rating             │
│ 4. 重新计算滑动窗口特征（近 N 场）         │
│ 5. 抓取即将进行的比赛赛程                  │
│ 6. 抓取最新赔率                           │
│ 7. 构建特征 → 模型预测 → 输出结果          │
└─────────────────────────────────────────┘
```

---

## 十、尚未测试的 API

以下是我们讨论过但还没接入的，优先级排序：

| 优先级 | API | 原因 |
|--------|-----|------|
| 🔴 最高 | **football-data.org** | 免费、简单、直接有世界杯数据 |
| 🟡 高 | **API-Football** | 免费 100 次/天，赔率 + 球员 |
| 🟡 高 | **football-data.co.uk** | 免费赔率批量下载，不需要 API key |
| 🟢 中 | **FBref** | xG 数据丰富，但国家队覆盖有限 |
| 🟢 中 | **penaltyblog** | 一站式，但暂不清楚国家队支持如何 |
| ⚪ 低 | **StatsBomb Open Data** | 事件级数据，但最新世界杯只有 2018 |
| ⚪ 低 | **SofaScore / Transfermarkt** | 需要爬虫，反爬风险 |

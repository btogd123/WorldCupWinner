"""
Betting analysis: combine model probabilities with odds to find value bets.
Kelly criterion for position sizing.
"""
import pandas as pd
import numpy as np
import json
import os
import sys

from config import RESULTS_DIR

# ============================================================
# 1. CORE MATH
# ============================================================

def implied_probability(odds_home, odds_draw, odds_away):
    """
    Convert decimal odds to implied probabilities, removing overround (vig).

    Bookmakers build a margin into their odds. For a 3-outcome market:
      raw probability = 1 / odds
      overround = sum(raw_probs) - 1  (typically 5-8%)

    We remove this by proportional normalization:
      true probability = raw_prob / sum(raw_probs)
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    overround = raw.sum() - 1.0  # bookie margin
    fair = raw / raw.sum()       # remove overround
    return fair, overround       # [home, draw, away], vig


def expected_value(model_prob, odds):
    """
    EV = model_prob * odds - 1
    Positive EV = long-term profitable bet.
    """
    return model_prob * odds - 1.0


def kelly_fraction(edge, odds):
    """
    Kelly criterion: optimal bet size as fraction of bankroll.

    Full Kelly:  f = edge / (odds - 1)
    We use 1/4 Kelly to reduce volatility (standard practice).
    """
    if odds <= 1.0 or edge <= 0:
        return 0.0
    full_kelly = edge / (odds - 1.0)
    # 1/4 Kelly: aggressive enough to grow, conservative enough to survive
    return max(0.0, full_kelly * 0.25)


def kelly_fraction_three_way(edges, odds):
    """
    For simultaneous mutually exclusive bets (3-way market),
    adjust Kelly so total stake <= 1.
    """
    raw = np.array([kelly_fraction(e, o) for e, o in zip(edges, odds)])
    total = raw.sum()
    if total > 1.0:
        raw = raw / total  # scale down to total 100%
    return raw


# ============================================================
# 2. SINGLE MATCH ANALYSIS
# ============================================================

LABELS = ["Away Win", "Draw", "Home Win"]
LABELS_SHORT = ["客胜", "平局", "主胜"]

def analyze_match(home_team, away_team,
                  odds_home, odds_draw, odds_away,
                  model_probs):
    """Analyze one match and print recommendation."""
    odds = np.array([odds_away, odds_draw, odds_home])
    implied, vig = implied_probability(odds_home, odds_draw, odds_away)
    # implied is [home, draw, away], reorder to [away, draw, home]
    implied = implied[::-1]

    evs = np.array([expected_value(model_probs[i], odds[i]) for i in range(3)])
    edges = model_probs - implied

    # Kelly stakes
    stakes = kelly_fraction_three_way(edges, odds)

    # Output
    print(f"\n{'='*65}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*65}")
    print(f"{'':>15s} {'赔率':>8s} {'隐含%':>8s} {'模型%':>8s} {'Edge':>8s} {'EV%':>8s} {'Kelly':>8s}")
    print(f"{'─'*65}")

    for i, label in enumerate(LABELS):
        arrow = " ← BET" if stakes[i] > 0.001 else ""
        print(
            f"  {label:>12s}  {odds[i]:>8.2f} {implied[i]*100:>7.1f}% "
            f"{model_probs[i]*100:>7.1f}% {edges[i]*100:>+7.1f}% "
            f"{evs[i]*100:>+7.1f}% {stakes[i]*100:>7.1f}%{arrow}"
        )

    print(f"{'─'*65}")
    print(f"  博彩公司抽水: {vig*100:.1f}%")

    # Recommendation
    best_idx = np.argmax(evs)
    if max(evs) > 0.02:
        print(f"\n  >>> 推荐: 买 {LABELS_SHORT[best_idx]}")
        print(f"      期望收益: {evs[best_idx]*100:+.1f}%")
        print(f"      建议投入: 资金的 {stakes[best_idx]*100:.1f}%")
    elif max(evs) > -0.02:
        print(f"\n  >>> 边缘不明确，观望")
    else:
        print(f"\n  >>> 无正期望，跳过")

    return {
        "evs": evs, "edges": edges, "stakes": stakes,
        "implied": implied, "vig": vig, "best_bet": best_idx if max(evs) > 0 else None,
    }


# ============================================================
# 3. BULK ANALYSIS FROM PREDICTIONS + ODDS CSV
# ============================================================

def load_odds(path):
    """
    Load odds from CSV. Expected columns:
      date, home_team, away_team, odds_away, odds_draw, odds_home

    Or if only team names given without date, match by team names.
    """
    df = pd.read_csv(path)
    required = {"home_team", "away_team", "odds_away", "odds_draw", "odds_home"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def analyze_all():
    """
    Analyze all WC 2026 matches with odds data.
    Reads predictions from our model, odds from user-provided CSV.
    """
    # Load predictions
    preds_path = os.path.join(RESULTS_DIR, "wc2026_predictions.json")
    if not os.path.exists(preds_path):
        print("Predictions not found. Run predict_wc2026.py first.")
        return

    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Look for odds file
    odds_path = os.path.join(RESULTS_DIR, "odds.csv")
    if not os.path.exists(odds_path):
        print("=" * 65)
        print("  赔率文件不存在")
        print("=" * 65)
        print(f"\n请创建 {odds_path}，格式如下：\n")
        print("  date,home_team,away_team,odds_away,odds_draw,odds_home")
        print("  2026-06-11,South Korea,Czech Republic,2.80,3.20,2.50")
        print("  ...\n")
        print("赔率可在以下网站获取：")
        print("  - oddsportal.com")
        print("  - flashscore.com")
        print("  - 各大博彩公司官网\n")
        return

    odds_df = load_odds(odds_path)
    print(f"加载 {len(odds_df)} 条赔率数据\n")

    results = []
    total_ev = 0
    bets_found = 0

    for _, row in odds_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        oa, od, oh = row["odds_away"], row["odds_draw"], row["odds_home"]

        # Find matching prediction
        match_pred = None
        for p in preds:
            if p["home_team"] == home and p["away_team"] == away:
                match_pred = p
                break

        if match_pred is None:
            # Try swapped
            for p in preds:
                if p["home_team"] == away and p["away_team"] == home:
                    match_pred = {
                        "home_team": p["home_team"],
                        "away_team": p["away_team"],
                        "pred_away_win": p["pred_home_win"],
                        "pred_draw": p["pred_draw"],
                        "pred_home_win": p["pred_away_win"],
                        "confidence": p["confidence"],
                        "date": p["date"],
                    }
                    home, away = away, home
                    break

        if match_pred is None:
            print(f"  ⚠ 未找到 {home} vs {away} 的预测，跳过")
            continue

        probs = np.array([
            match_pred["pred_away_win"],
            match_pred["pred_draw"],
            match_pred["pred_home_win"],
        ])

        r = analyze_match(home, away, oh, od, oa, probs)
        r["home_team"] = home
        r["away_team"] = away
        r["date"] = match_pred.get("date", "")
        results.append(r)

        if r["best_bet"] is not None:
            total_ev += r["evs"][r["best_bet"]]
            bets_found += 1

    # Summary
    print("\n" + "=" * 65)
    print("  总览")
    print("=" * 65)
    print(f"  分析场次: {len(results)}")
    print(f"  可投注场次: {bets_found}")
    if bets_found > 0:
        print(f"  平均期望收益: {total_ev/bets_found*100:+.2f}%")
        print(f"  总期望收益: {total_ev*100:+.2f}% (if all bets placed)")

    # Top picks
    if bets_found > 0:
        print(f"\n  {'─'*55}")
        print(f"  最佳投注机会 (按 EV 排序)")
        print(f"  {'─'*55}")
        sorted_results = sorted(
            [r for r in results if r["best_bet"] is not None],
            key=lambda r: r["evs"][r["best_bet"]],
            reverse=True,
        )
        for i, r in enumerate(sorted_results[:10]):
            best = r["best_bet"]
            print(
                f"  {i+1:2d}. {r['home_team']:20s} vs {r['away_team']:20s}  "
                f"→ {LABELS_SHORT[best]}  "
                f"EV={r['evs'][best]*100:+.1f}%  "
                f"投{r['stakes'][best]*100:.1f}%"
            )

    # Save detailed results
    out = []
    for r in results:
        out.append({
            "date": r.get("date", ""),
            "match": f"{r['home_team']} vs {r['away_team']}",
            "has_bet": r["best_bet"] is not None,
            "recommended_bet": LABELS[r["best_bet"]] if r["best_bet"] is not None else "None",
            "best_ev_pct": round(float(r["evs"].max()) * 100, 1),
            "kelly_stake_pct": round(float(r["stakes"][r["best_bet"] if r["best_bet"] is not None else 0]) * 100, 1),
            "vig_pct": round(float(r["vig"]) * 100, 1),
        })

    out_path = os.path.join(RESULTS_DIR, "betting_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {out_path}")


# ============================================================
# 4. SINGLE MATCH QUICK CHECK
# ============================================================

def quick_check(home_team, away_team, odds_home, odds_draw, odds_away):
    """Quick single-match check using model predictions."""
    preds_path = os.path.join(RESULTS_DIR, "wc2026_predictions.json")
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Find the match
    probs = None
    for p in preds:
        if p["home_team"] == home_team and p["away_team"] == away_team:
            probs = np.array([p["pred_away_win"], p["pred_draw"], p["pred_home_win"]])
            break

    if probs is None:
        # Load model directly
        print(f"Match not in predictions, loading model...")
        from predict_wc2026 import load_model_and_assets
        import torch

        model, team_encoder, scaler, feature_cols, device = load_model_and_assets()

        elo_df = pd.read_csv("data/elo_ratings.csv")
        elo_dict = dict(zip(elo_df["team"], elo_df["elo_rating"]))
        helo = elo_dict.get(home_team, 1500)
        aelo = elo_dict.get(away_team, 1500)

        feat = {c: 0.0 for c in feature_cols}
        feat.update({
            "elo_advantage_home": (helo - aelo) / 400.0,
            "elo_quality": (helo + aelo) / 3000.0,
            "elo_diff_norm": (helo - aelo) / 400.0,
            "elo_ratio": (helo / max(aelo, 1000)) - 1.0,
            "elo_gap": abs(helo - aelo) / 400.0,
            "strength_advantage": (helo - aelo) / 1500.0,
            "match_quality": (helo + aelo) / 3000.0,
            "is_neutral": 1,
            "year_norm": (2026 - 1950) / 80.0,
            "is_wc": 1,
        })

        X = np.array([[feat[c] for c in feature_cols]], dtype=np.float32)
        X = scaler.transform(X)
        hid = team_encoder.transform([home_team])[0] + 1
        aid = team_encoder.transform([away_team])[0] + 1

        model.eval()
        with torch.no_grad():
            logits = model(
                torch.LongTensor([hid]).to(device),
                torch.LongTensor([aid]).to(device),
                torch.FloatTensor(X).to(device),
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    analyze_match(home_team, away_team, odds_home, odds_draw, odds_away, probs)


# ============================================================
# 5. SAMPLE ODDS GENERATOR
# ============================================================

def generate_sample_odds():
    """
    Generate a sample odds.csv from our predictions, using
    a simple model: convert probabilities to odds with a 6% overround.
    This is for demonstration only — real odds come from bookmakers.
    """
    preds_path = os.path.join(RESULTS_DIR, "wc2026_predictions.json")
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    rows = []
    for p in preds:
        probs = np.array([p["pred_away_win"], p["pred_draw"], p["pred_home_win"]])
        # Add 6% overround
        raw_odds = 1.0 / probs
        fair_sum = raw_odds.sum()
        target_sum = fair_sum * 1.06  # 6% vig
        scale = target_sum / fair_sum
        odds = raw_odds * scale

        rows.append({
            "date": p["date"],
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "odds_away": round(odds[0], 2),
            "odds_draw": round(odds[1], 2),
            "odds_home": round(odds[2], 2),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "odds_sample.csv")
    df.to_csv(path, index=False)
    print(f"示例赔率已生成: {path}")
    print(f"格式: 模型概率 + 6% 抽水 → 赔率")
    print(f"前 5 行:")
    print(df.head().to_string(index=False))
    print(f"\n用真实赔率替换此文件后运行: python src/betting.py --all")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "--all":
            # Bulk analysis
            analyze_all()

        elif cmd == "--sample":
            # Generate sample odds
            generate_sample_odds()
            print("\n然后运行: python src/betting.py --all")

        elif cmd == "--match" and len(sys.argv) >= 7:
            # Quick single match: home away odds_h odds_d odds_a
            home = sys.argv[2]
            away = sys.argv[3]
            oh = float(sys.argv[4])
            od = float(sys.argv[5])
            oa = float(sys.argv[6])
            quick_check(home, away, oh, od, oa)

        else:
            print("用法:")
            print("  python src/betting.py --all                  批量分析(需odds.csv)")
            print("  python src/betting.py --sample               生成示例赔率")
            print("  python src/betting.py --match 法国 巴西 2.50 3.20 2.80  单场分析")
    else:
        # Default: show usage and generate sample
        print("=" * 65)
        print("  2026 世界杯博彩分析工具")
        print("=" * 65)
        print()
        print("  用法:")
        print("    python src/betting.py --sample              生成示例赔率文件")
        print("    python src/betting.py --all                 批量分析(需先准备odds.csv)")
        print("    python src/betting.py --match <主队> <客队> <主胜赔率> <平局赔率> <客胜赔率>")
        print()
        print("  示例:")
        print("    python src/betting.py --match 法国 巴西 2.50 3.20 2.80")
        print()
        # Auto-generate sample
        generate_sample_odds()

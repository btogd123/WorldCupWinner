"""
2026 World Cup Tournament Simulator.
Simulates the full tournament including knockout stages.
"""
import torch
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from itertools import combinations

from config import (
    MODEL_PATH,
    ELO_RATINGS_PATH,
    PROCESSED_DATA_PATH,
    RESULTS_DIR,
    WC2026_START,
)
from predict_wc2026 import load_model_and_assets, build_match_features, prepare_wc_features


def predict_match(model, home_team, away_team, team_encoder, scaler, feature_cols, device, df):
    """Predict a single match outcome using proper feature computation."""
    # Get current Elo
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    home_elo = elo_df[elo_df["team"] == home_team]["elo_rating"].values
    away_elo = elo_df[elo_df["team"] == away_team]["elo_rating"].values

    if len(home_elo) == 0 or len(away_elo) == 0:
        return None

    home_elo = home_elo[0]
    away_elo = away_elo[0]

    # Compute form from recent matches (last 10 matches for each team)
    home_recent = df[
        ((df["home_team"] == home_team) | (df["away_team"] == home_team))
        & (df["date"] <= "2026-06-10")
        & df["home_score"].notna()
    ].tail(10)

    away_recent = df[
        ((df["home_team"] == away_team) | (df["away_team"] == away_team))
        & (df["date"] <= "2026-06-10")
        & df["home_score"].notna()
    ].tail(10)

    # Build feature vector
    features = {
        "elo_advantage_home": (home_elo - away_elo) / 400.0,
        "elo_quality": (home_elo + away_elo) / 3000.0,
        "elo_diff_norm": (home_elo - away_elo) / 400.0,
        "elo_ratio": (home_elo / max(away_elo, 1000)) - 1.0,
        "elo_gap": abs(home_elo - away_elo) / 400.0,
        "form_advantage": 0.0,
        "form_quality": 0.0,
        "wr_advantage": 0.0,
        "gs_advantage": 0.0,
        "gc_advantage": 0.0,
        "goal_diff_advantage": 0.0,
        "strength_advantage": (home_elo - away_elo) / 1500.0,
        "match_quality": (home_elo + away_elo) / 3000.0,
        "h2h_dominance": 0.0,
        "has_h2h": 0,
        "is_neutral": 1,  # All WC matches at neutral venues
        "year_norm": (2026 - 1950) / 80.0,
        "is_wc": 1,
        "is_wcq": 0,
        "is_continental": 0,
        "is_friendly": 0,
    }

    X = np.array([[features[col] for col in feature_cols]], dtype=np.float32)
    X = scaler.transform(X)

    home_id = team_encoder.transform([home_team])[0] + 1
    away_id = team_encoder.transform([away_team])[0] + 1

    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        h_t = torch.LongTensor([home_id]).to(device)
        a_t = torch.LongTensor([away_id]).to(device)
        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # probs = [away_win, draw, home_win]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "away_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "home_win_prob": float(probs[2]),
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


def simulate_tournament():
    """Simulate the full 2026 World Cup tournament."""
    print("=" * 60)
    print("2026 FIFA WORLD CUP - FULL TOURNAMENT SIMULATION")
    print("=" * 60)

    # Load assets
    model, team_encoder, scaler, feature_cols, device = load_model_and_assets()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Define the 48 qualified teams (same list as before)
    qualified_teams = [
        "United States", "Mexico", "Canada",
        "Japan", "South Korea", "Saudi Arabia", "Iran", "Australia",
        "Qatar", "United Arab Emirates", "Iraq",
        "Morocco", "Senegal", "Egypt", "Algeria", "Nigeria",
        "Cameroon", "Ghana", "Ivory Coast", "Tunisia",
        "Costa Rica", "Panama", "Jamaica", "Honduras",
        "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador",
        "Peru", "Chile", "Paraguay",
        "New Zealand",
        "France", "Spain", "England", "Germany", "Portugal",
        "Netherlands", "Italy", "Belgium", "Croatia", "Denmark",
        "Switzerland", "Austria", "Serbia", "Ukraine", "Turkey", "Sweden",
    ]

    # Filter to valid teams
    valid_teams = [t for t in qualified_teams if t in team_encoder.classes_]
    print(f"Qualified teams available: {len(valid_teams)}/48")

    if len(valid_teams) < 48:
        # Pad with next best teams by Elo
        elo_df = pd.read_csv(ELO_RATINGS_PATH)
        existing = set(valid_teams)
        extra = [t for t in elo_df["team"].values if t not in existing and t in team_encoder.classes_]
        valid_teams = valid_teams + extra[:48 - len(valid_teams)]

    # Seed teams by Elo rating for pot assignments
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    elo_dict = dict(zip(elo_df["team"], elo_df["elo_rating"]))
    ranked = sorted(valid_teams[:48], key=lambda t: elo_dict.get(t, 1500), reverse=True)

    # Create 12 groups of 4 using serpentine seeding
    groups = [[] for _ in range(12)]
    seeds = ranked

    # Pot 1: Top 12
    for i, team in enumerate(seeds[:12]):
        groups[i].append(team)
    # Pot 2: 13-24 (reverse order for balance)
    for i, team in enumerate(seeds[12:24]):
        groups[11 - i].append(team)
    # Pot 3: 25-36
    for i, team in enumerate(seeds[24:36]):
        groups[i].append(team)
    # Pot 4: 37-48
    for i, team in enumerate(seeds[36:48]):
        groups[11 - i].append(team)

    # Print groups
    print("\n" + "-" * 40)
    print("GROUP STAGE DRAW")
    print("-" * 40)
    for i, group in enumerate(groups):
        group_name = chr(65 + i)  # A, B, C, ...
        teams_str = ", ".join(f"{t}({elo_dict.get(t, 0):.0f})" for t in group)
        print(f"Group {group_name}: {teams_str}")

    # Simulate group stage
    print("\n" + "-" * 40)
    print("GROUP STAGE RESULTS")
    print("-" * 40)

    group_results = []
    all_group_matches = []

    for group_idx, group in enumerate(groups):
        group_name = chr(65 + group_idx)
        standings = {team: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for team in group}

        print(f"\nGroup {group_name}:")
        print(f"{'Home':20s} vs {'Away':20s} | {'Result':15s} | Probabilities")

        for i in range(4):
            for j in range(i + 1, 4):
                home_team = group[i]
                away_team = group[j]

                pred = predict_match(
                    model, home_team, away_team, team_encoder, scaler, feature_cols, device, df
                )

                if pred is None:
                    continue

                # Determine result probabilistically (or take max prob)
                probs = [pred["away_win_prob"], pred["draw_prob"], pred["home_win_prob"]]
                winner = np.argmax(probs)  # 0=away, 1=draw, 2=home

                if winner == 2:  # Home win
                    result_str = f"{home_team} wins"
                    standings[home_team]["pts"] += 3
                    standings[home_team]["w"] += 1
                    standings[away_team]["l"] += 1
                    standings[home_team]["gf"] += 2
                    standings[away_team]["ga"] += 2
                    standings[away_team]["gf"] += 1
                    standings[home_team]["ga"] += 1
                elif winner == 0:  # Away win
                    result_str = f"{away_team} wins"
                    standings[away_team]["pts"] += 3
                    standings[away_team]["w"] += 1
                    standings[home_team]["l"] += 1
                    standings[away_team]["gf"] += 2
                    standings[home_team]["ga"] += 2
                    standings[home_team]["gf"] += 1
                    standings[away_team]["ga"] += 1
                else:  # Draw
                    result_str = "Draw"
                    standings[home_team]["pts"] += 1
                    standings[away_team]["pts"] += 1
                    standings[home_team]["d"] += 1
                    standings[away_team]["d"] += 1
                    standings[home_team]["gf"] += 1
                    standings[away_team]["gf"] += 1
                    standings[home_team]["ga"] += 1
                    standings[away_team]["ga"] += 1

                for team in group:
                    standings[team]["gd"] = standings[team]["gf"] - standings[team]["ga"]

                print(
                    f"{home_team:20s} vs {away_team:20s} | {result_str:15s} | "
                    f"A:{pred['away_win_prob']:.1%} D:{pred['draw_prob']:.1%} H:{pred['home_win_prob']:.1%}"
                )

                all_group_matches.append({
                    "group": group_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "result": result_str,
                    "away_win_prob": pred["away_win_prob"],
                    "draw_prob": pred["draw_prob"],
                    "home_win_prob": pred["home_win_prob"],
                })

        # Sort standings
        sorted_standings = sorted(
            standings.items(),
            key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
            reverse=True,
        )

        print(f"\n  Standings:")
        for pos, (team, stats) in enumerate(sorted_standings):
            print(
                f"  {pos+1}. {team:20s}  Pts:{stats['pts']}  "
                f"W:{stats['w']} D:{stats['d']} L:{stats['l']}  "
                f"GF:{stats['gf']} GA:{stats['ga']} GD:{stats['gd']:+d}"
            )

        group_results.append({
            "group": group_name,
            "standings": sorted_standings,
        })

    # Determine round of 32 teams
    # Group winners (12) + group runners-up (12) + 8 best 3rd place
    group_winners = []
    group_runners_up = []
    third_place = []

    for gr in group_results:
        standings = gr["standings"]
        group_winners.append((gr["group"], standings[0][0], standings[0][1]))
        group_runners_up.append((gr["group"], standings[1][0], standings[1][1]))
        third_place.append((gr["group"], standings[2][0], standings[2][1]))

    # Sort 3rd place teams by points
    third_place.sort(key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"]), reverse=True)
    best_thirds = third_place[:8]

    # Build Round of 32 bracket (16 matches)
    # Format: 8 winners vs 8 third-place, 4 winners vs 4 runners-up, 4 runners-up vs 4 runners-up
    print("\n" + "=" * 60)
    print("KNOCKOUT STAGE")
    print("=" * 60)
    print("\nRound of 32 qualified:")
    print("Group Winners:", ", ".join(f"{t}({g})" for g, t, _ in group_winners))
    print("Runners-up:", ", ".join(f"{t}({g})" for g, t, _ in group_runners_up))
    print("Best 3rds:", ", ".join(f"{t}({g})" for g, t, _ in best_thirds))

    # Shuffle to avoid group-stage rematches in R32
    import random
    random.seed(42)

    # Create 16 R32 matches
    r32_matches = []

    # 8 group winners (from groups A-H) vs 8 best 3rd place
    winners_for_3rd = group_winners[:8]
    thirds_copy = list(best_thirds)
    random.shuffle(thirds_copy)
    for i, (wg, wt, ws) in enumerate(winners_for_3rd):
        opponent = thirds_copy[i]
        r32_matches.append(((wg, wt, ws), opponent))

    # 4 group winners (from groups I-L) vs 4 runners-up
    winners_for_ru = group_winners[8:12]
    rus_available = [(rg, rt, rs) for rg, rt, rs in group_runners_up
                     if rg not in [w[0] for w in winners_for_ru]]
    random.shuffle(rus_available)
    for i, (wg, wt, ws) in enumerate(winners_for_ru):
        opponent = rus_available[i]
        r32_matches.append(((wg, wt, ws), opponent))

    # Remaining 8 runners-up play each other
    used_ru_groups = set(r[0] for _, (r, _, _) in zip(range(4), rus_available[:4]))
    remaining_rus = [(rg, rt, rs) for rg, rt, rs in group_runners_up
                     if rg not in used_ru_groups]
    random.shuffle(remaining_rus)
    for i in range(0, len(remaining_rus), 2):
        if i + 1 < len(remaining_rus):
            r32_matches.append((remaining_rus[i], remaining_rus[i + 1]))

    print(f"\nCreated {len(r32_matches)} Round of 32 matches")

    # Simulate knockout rounds
    # Convert r32 matches to simple format: (team1, team2)
    knockout_teams = []
    for t1_info, t2_info in r32_matches:
        knockout_teams.append((t1_info[1], t2_info[1]))

    round_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]

    for round_idx, round_name in enumerate(round_names):
        if len(knockout_teams) == 0:
            break

        print(f"\n{'─' * 40}")
        print(f"{round_name}")
        print(f"{'─' * 40}")

        winners = []

        for match_idx, (team1, team2) in enumerate(knockout_teams):
            # Alternate home/away
            if match_idx % 2 == 0:
                home_team, away_team = team1, team2
            else:
                home_team, away_team = team2, team1

            pred = predict_match(
                model, home_team, away_team, team_encoder, scaler, feature_cols, device, df
            )

            if pred is None:
                continue

            probs = [pred["away_win_prob"], pred["draw_prob"], pred["home_win_prob"]]
            winner = np.argmax(probs)

            if winner == 2:
                match_winner = home_team
                result = f"{home_team} wins"
            elif winner == 0:
                match_winner = away_team
                result = f"{away_team} wins"
            else:
                # Draw in knockout: higher Elo wins (penalties)
                if pred["home_elo"] > pred["away_elo"]:
                    match_winner = home_team
                    result = f"{home_team} wins (pens)"
                else:
                    match_winner = away_team
                    result = f"{away_team} wins (pens)"

            print(
                f"  {home_team:20s} vs {away_team:20s} | {result:20s} | "
                f"({pred['away_win_prob']:.1%}/{pred['draw_prob']:.1%}/{pred['home_win_prob']:.1%})"
            )

            winners.append(match_winner)

        # Prepare next round pairs
        if round_name == "Final":
            print(f"\n{'=' * 60}")
            print(f"🏆 WORLD CUP 2026 CHAMPION: {winners[0]} 🏆")
            print(f"{'=' * 60}")
            break
        elif round_name == "Semi-finals":
            # 2 semi-final winners → 1 final
            knockout_teams = [(winners[0], winners[1])]
        else:
            # Pair winners for next round
            knockout_teams = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    knockout_teams.append((winners[i], winners[i + 1]))

    # Save full simulation results
    results = {
        "groups": [
            {
                "group": gr["group"],
                "standings": [
                    {"team": t, "pts": s["pts"], "gd": s["gd"], "gf": s["gf"], "ga": s["ga"]}
                    for t, s in gr["standings"]
                ],
            }
            for gr in group_results
        ],
        "group_matches": all_group_matches,
    }

    results_path = os.path.join(RESULTS_DIR, "tournament_simulation.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nFull simulation saved to: {results_path}")

    return results


def main():
    """Entry point for wc-simulate command."""
    simulate_tournament()


if __name__ == "__main__":
    main()

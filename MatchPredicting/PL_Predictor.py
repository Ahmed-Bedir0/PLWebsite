## Ahmed Bedir

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path(".")
SEASON_FILES = sorted(DATA_DIR.glob("PL_*.csv"))

TEST_SEASON = "2024-2025"

ROLLING_WINDOW = 5
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


TEAM_NAME_MAPPING = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
}


def parse_score(score: str) -> Tuple[int, int]:
    try:
        home_goals, away_goals = score.split("-")
        return int(home_goals), int(away_goals)

    except (ValueError, AttributeError):
        return 0, 0


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["Team 1"] = df["Team 1"].replace(TEAM_NAME_MAPPING)
    df["Team 2"] = df["Team 2"].replace(TEAM_NAME_MAPPING)

    return df


def load_data() -> pd.DataFrame:

    if not SEASON_FILES:
        raise FileNotFoundError(
            "No PL_*.csv files found in the current directory."
        )

    all_seasons = []

    for file in SEASON_FILES:

        season = file.stem.replace("PL_", "")

        log.info("Loading %s", file.name)

        df = pd.read_csv(file)

        df = normalize_team_names(df)

        df[["home_ft", "away_ft"]] = (
            df["FT"]
            .apply(parse_score)
            .apply(pd.Series)
        )

        df[["home_ht", "away_ht"]] = (
            df["HT"]
            .apply(parse_score)
            .apply(pd.Series)
        )

        df["date"] = pd.to_datetime(df["Date"])

        df["season"] = season

        df["target"] = np.where(
            df["home_ft"] > df["away_ft"],
            2,
            np.where(
                df["home_ft"] < df["away_ft"],
                0,
                1,
            ),
        )

        all_seasons.append(df)

    matches = pd.concat(
        all_seasons,
        ignore_index=True,
    )

    matches = (
        matches
        .sort_values("date")
        .reset_index(drop=True)
    )

    log.info(
        "Loaded %d matches across %d seasons",
        len(matches),
        len(SEASON_FILES),
    )

    return matches


def build_team_features(
    matches: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, match in matches.iterrows():

        home_points = (
            3 if match["target"] == 2
            else 1 if match["target"] == 1
            else 0
        )

        away_points = (
            3 if match["target"] == 0
            else 1 if match["target"] == 1
            else 0
        )

        rows.append({
            "date": match["date"],
            "season": match["season"],
            "team": match["Team 1"],
            "opponent": match["Team 2"],
            "is_home": 1,
            "gf": match["home_ft"],
            "ga": match["away_ft"],
            "ht_gf": match["home_ht"],
            "ht_ga": match["away_ht"],
            "points": home_points,
        })

        rows.append({
            "date": match["date"],
            "season": match["season"],
            "team": match["Team 2"],
            "opponent": match["Team 1"],
            "is_home": 0,
            "gf": match["away_ft"],
            "ga": match["home_ft"],
            "ht_gf": match["away_ht"],
            "ht_ga": match["home_ht"],
            "points": away_points,
        })

    team_df = pd.DataFrame(rows)

    team_df = (
        team_df
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )

    grouped = team_df.groupby(
        "team",
        group_keys=False,
    )

    rolling_columns = [
        "gf",
        "ga",
        "ht_gf",
        "ht_ga",
        "points",
    ]

    for column in rolling_columns:

        team_df[f"{column}_roll"] = grouped[column].transform(
            lambda values: (
                values
                .shift(1)
                .rolling(
                    window=ROLLING_WINDOW,
                    min_periods=3,
                )
                .mean()
            )
        )

    required_columns = [
        f"{column}_roll"
        for column in rolling_columns
    ]

    team_df = (
        team_df
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )

    return team_df


def merge_features(
    matches: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:

    home_features = features[
        features["is_home"] == 1
    ].rename(columns={
        "team": "Team 1",
        "opponent": "Team 2",

        "gf_roll": "home_gf_roll",
        "ga_roll": "home_ga_roll",

        "ht_gf_roll": "home_ht_gf_roll",
        "ht_ga_roll": "home_ht_ga_roll",

        "points_roll": "home_points_roll",
    })

    away_features = features[
        features["is_home"] == 0
    ].rename(columns={
        "team": "Team 2",
        "opponent": "Team 1",

        "gf_roll": "away_gf_roll",
        "ga_roll": "away_ga_roll",

        "ht_gf_roll": "away_ht_gf_roll",
        "ht_ga_roll": "away_ht_ga_roll",

        "points_roll": "away_points_roll",
    })

    merged = matches.merge(
        home_features[[
            "date",
            "Team 1",
            "Team 2",

            "home_gf_roll",
            "home_ga_roll",

            "home_ht_gf_roll",
            "home_ht_ga_roll",

            "home_points_roll",
        ]],
        on=[
            "date",
            "Team 1",
            "Team 2",
        ],
        how="inner",
    )

    merged = merged.merge(
        away_features[[
            "date",
            "Team 1",
            "Team 2",

            "away_gf_roll",
            "away_ga_roll",

            "away_ht_gf_roll",
            "away_ht_ga_roll",

            "away_points_roll",
        ]],
        on=[
            "date",
            "Team 1",
            "Team 2",
        ],
        how="inner",
    )

    merged["home_team_code"] = (
        merged["Team 1"]
        .astype("category")
        .cat.codes
    )

    merged["away_team_code"] = (
        merged["Team 2"]
        .astype("category")
        .cat.codes
    )

    merged["day_of_week"] = (
        merged["date"]
        .dt.dayofweek
    )

    merged["month"] = (
        merged["date"]
        .dt.month
    )

    return merged


def train_and_predict(
    df: pd.DataFrame,
) -> None:

    predictors = [
        "home_team_code",
        "away_team_code",

        "day_of_week",
        "month",

        "home_gf_roll",
        "home_ga_roll",

        "home_ht_gf_roll",
        "home_ht_ga_roll",

        "home_points_roll",

        "away_gf_roll",
        "away_ga_roll",

        "away_ht_gf_roll",
        "away_ht_ga_roll",

        "away_points_roll",
    ]

    train = df[
        df["season"] != TEST_SEASON
    ]

    test = df[
        df["season"] == TEST_SEASON
    ]

    if test.empty:
        log.error(
            "No data found for test season: %s",
            TEST_SEASON,
        )
        return

    log.info(
        "Training matches: %d",
        len(train),
    )

    log.info(
        "Testing matches: %d",
        len(test),
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        train[predictors],
        train["target"],
    )

    predictions = model.predict(
        test[predictors]
    )

    probabilities = model.predict_proba(
        test[predictors]
    )

    confidence_scores = (
        probabilities.max(axis=1)
    )

    accuracy = accuracy_score(
        test["target"],
        predictions,
    )

    log.info(
        "Model accuracy: %.4f",
        accuracy,
    )

    prediction_results = pd.DataFrame({
        "Date": (
            test["date"]
            .dt.strftime("%Y-%m-%d")
        ),

        "Home Team": test["Team 1"],

        "Away Team": test["Team 2"],

        "Actual": test["target"],

        "Predicted": predictions,

        "Confidence": confidence_scores,
    })

    result_mapping = {
        0: "Away Win",
        1: "Draw",
        2: "Home Win",
    }

    prediction_results["Actual"] = (
        prediction_results["Actual"]
        .map(result_mapping)
    )

    prediction_results["Predicted"] = (
        prediction_results["Predicted"]
        .map(result_mapping)
    )

    print("\nTop 20 Highest Confidence Predictions")
    print("=" * 70)

    print(
        prediction_results
        .sort_values(
            "Confidence",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nClassification Report")
    print("=" * 70)

    print(
        classification_report(
            test["target"],
            predictions,
            target_names=[
                "Away Win",
                "Draw",
                "Home Win",
            ],
        )
    )

    feature_importance = pd.DataFrame({
        "Feature": predictors,
        "Importance": (
            model.feature_importances_
        ),
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
    )

    print("\nTop Feature Importance")
    print("=" * 70)

    print(
        feature_importance
        .head(10)
        .to_string(index=False)
    )


def main() -> None:

    matches = load_data()

    team_features = build_team_features(
        matches
    )

    final_dataset = merge_features(
        matches,
        team_features,
    )

    log.info(
        "Final dataset contains %d matches",
        len(final_dataset),
    )

    train_and_predict(
        final_dataset
    )

    log.info(
        "Prediction pipeline completed successfully."
    )


if __name__ == "__main__":
    main()

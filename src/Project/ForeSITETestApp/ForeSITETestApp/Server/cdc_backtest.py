from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import requests

CDC_DATASET_ENDPOINT = "https://data.cdc.gov/resource/r8kw-7aab.json"

SOURCE_TO_DEATH_COLUMN = {
    "COVID-19 Deaths": "covid_19_deaths",
    "Pneumonia Deaths": "pneumonia_deaths",
    "Flu Deaths": "influenza_deaths",
}


@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    weekly_detail: pd.DataFrame


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _normalize_data_as_of(df: pd.DataFrame) -> pd.DataFrame:
    if "data_as_of" not in df.columns:
        df["data_as_of"] = pd.NaT
        return df

    df["data_as_of"] = pd.to_datetime(df["data_as_of"], errors="coerce")
    return df


def _prepare_cdc_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "group" in df.columns:
        df = df[df["group"] == "By Week"].copy()

    df["week_ending_date"] = pd.to_datetime(df["week_ending_date"], errors="coerce")
    for death_col in ("covid_19_deaths", "pneumonia_deaths", "influenza_deaths"):
        if death_col in df.columns:
            df[death_col] = pd.to_numeric(df[death_col], errors="coerce").fillna(0.0)
        else:
            df[death_col] = 0.0
    if "percent_of_expected_deaths" in df.columns:
        df["percent_of_expected_deaths"] = pd.to_numeric(
            df["percent_of_expected_deaths"], errors="coerce"
        )
    else:
        df["percent_of_expected_deaths"] = pd.NA

    df = _normalize_data_as_of(df)
    df = df.dropna(subset=["week_ending_date"]).copy()
    df = df.sort_values(["week_ending_date", "data_as_of"]).copy()
    df = df.drop_duplicates(subset=["week_ending_date"], keep="last").copy()
    df = df.sort_values("week_ending_date").reset_index(drop=True)
    return df


def fetch_cdc_weekly_data(
    state: str,
    *,
    data_as_of: Optional[str] = None,
    app_token: Optional[str] = None,
    timeout_sec: int = 30,
) -> pd.DataFrame:
    """
    Pull CDC weekly provisional death data for one state/region.
    """
    escaped_state = state.replace("'", "''")
    where_parts = [f"state='{escaped_state}'"]
    if data_as_of:
        # API stores date-time; date literal works in SODA for date fields.
        where_parts.append(f"data_as_of='{data_as_of}'")

    params = {
        "$select": ",".join(
            [
                "data_as_of",
                "week_ending_date",
                "state",
                "covid_19_deaths",
                "pneumonia_deaths",
                "influenza_deaths",
                "total_deaths",
                "percent_of_expected_deaths",
            ]
        ),
        "$where": " AND ".join(where_parts),
        "$order": "week_ending_date ASC",
        "$limit": 50000,
    }

    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    resp = requests.get(CDC_DATASET_ENDPOINT, params=params, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        raise ValueError("No rows returned from CDC dataset for the provided filters.")

    return _prepare_cdc_frame(pd.DataFrame(rows))


def _build_truth_labels(
    df: pd.DataFrame,
    truth_rule: str,
    rule_params: Dict[str, Any],
    death_column: str,
) -> pd.Series:
    truth_rule = truth_rule.lower().strip()

    if truth_rule == "percent_expected":
        pct_threshold = float(rule_params.get("percent_threshold", 110.0))
        return (pd.to_numeric(df["percent_of_expected_deaths"], errors="coerce") >= pct_threshold).fillna(
            False
        )

    if truth_rule == "zscore":
        window = int(rule_params.get("window", 8))
        min_history = int(rule_params.get("min_history", window))
        z_k = float(rule_params.get("z_k", 2.0))

        rolling_mean = df[death_column].rolling(window=window, min_periods=min_history).mean().shift(1)
        rolling_std = df[death_column].rolling(window=window, min_periods=min_history).std(ddof=0).shift(1)
        z = (df[death_column] - rolling_mean) / rolling_std.replace(0, pd.NA)
        return (z >= z_k).fillna(False)

    if truth_rule == "combined":
        pct_threshold = float(rule_params.get("percent_threshold", 110.0))
        window = int(rule_params.get("window", 8))
        min_history = int(rule_params.get("min_history", window))
        z_k = float(rule_params.get("z_k", 2.0))

        by_pct = (pd.to_numeric(df["percent_of_expected_deaths"], errors="coerce") >= pct_threshold).fillna(False)
        rolling_mean = df[death_column].rolling(window=window, min_periods=min_history).mean().shift(1)
        rolling_std = df[death_column].rolling(window=window, min_periods=min_history).std(ddof=0).shift(1)
        z = (df[death_column] - rolling_mean) / rolling_std.replace(0, pd.NA)
        by_z = (z >= z_k).fillna(False)
        return by_pct | by_z

    if truth_rule == "provided_column":
        col = rule_params.get("column")
        if not col:
            raise ValueError("truth_rule='provided_column' requires rule_params['column'].")
        if col not in df.columns:
            raise ValueError(f"Provided truth column '{col}' not found in data.")
        return pd.to_numeric(df[col], errors="coerce").fillna(0) > 0

    raise ValueError(
        "Unsupported truth_rule. Use one of: percent_expected, zscore, combined, provided_column."
    )


def run_backtest(
    state: str,
    threshold: float,
    truth_rule: str,
    *,
    source: str = "COVID-19 Deaths",
    rule_params: Optional[Dict[str, Any]] = None,
    data_as_of: Optional[str] = None,
    app_token: Optional[str] = None,
) -> BacktestResult:
    """
    Inputs:
      - state: e.g. 'United States', 'California'
      - threshold: outbreak threshold on covid_19_deaths
      - truth_rule: percent_expected | zscore | combined | provided_column
      - rule_params: parameters for truth_rule
      - data_as_of: optional CDC snapshot date (YYYY-MM-DD)
      - app_token: optional CDC app token

    Outputs:
      - metrics: TP/FN/FP/Precision/Recall/F1 (+ TN and counts)
      - weekly_detail: week-level frame with y_true/y_pred
    """
    params = rule_params or {}
    death_column = SOURCE_TO_DEATH_COLUMN.get(source)
    if not death_column:
        raise ValueError(
            f"Unsupported source '{source}'. Use one of: {', '.join(SOURCE_TO_DEATH_COLUMN.keys())}."
        )

    df = fetch_cdc_weekly_data(state=state, data_as_of=data_as_of, app_token=app_token)

    df["outbreak_number"] = df[death_column] - float(threshold)
    df["y_pred"] = df["outbreak_number"] > 0
    df["y_true"] = _build_truth_labels(df, truth_rule, params, death_column)

    tp = int(((df["y_true"]) & (df["y_pred"])).sum())
    fn = int(((df["y_true"]) & (~df["y_pred"])).sum())
    fp = int(((~df["y_true"]) & (df["y_pred"])).sum())
    tn = int(((~df["y_true"]) & (~df["y_pred"])).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "state": state,
        "source": source,
        "death_column": death_column,
        "threshold": float(threshold),
        "truth_rule": truth_rule,
        "data_as_of": data_as_of,
        "weeks_total": int(len(df)),
        "weeks_true_outbreak": int(df["y_true"].sum()),
        "weeks_pred_alert": int(df["y_pred"].sum()),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    detail_cols = [
        "week_ending_date",
        "state",
        death_column,
        "percent_of_expected_deaths",
        "outbreak_number",
        "y_true",
        "y_pred",
    ]
    weekly_detail = df[detail_cols].copy()
    weekly_detail["week_ending_date"] = weekly_detail["week_ending_date"].dt.strftime("%Y-%m-%d")

    return BacktestResult(metrics=metrics, weekly_detail=weekly_detail)


def _parse_rule_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if args.percent_threshold is not None:
        params["percent_threshold"] = args.percent_threshold
    if args.window is not None:
        params["window"] = args.window
    if args.min_history is not None:
        params["min_history"] = args.min_history
    if args.z_k is not None:
        params["z_k"] = args.z_k
    if args.truth_column:
        params["column"] = args.truth_column
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="CDC weekly alert backtest with metrics and weekly details.")
    parser.add_argument("--state", required=True, help="State/region in CDC dataset, e.g. 'United States'")
    parser.add_argument(
        "--source",
        default="COVID-19 Deaths",
        choices=list(SOURCE_TO_DEATH_COLUMN.keys()),
        help="Data source name to select the death variable",
    )
    parser.add_argument("--threshold", type=float, required=True, help="Alert threshold on covid_19_deaths")
    parser.add_argument(
        "--truth-rule",
        default="percent_expected",
        choices=["percent_expected", "zscore", "combined", "provided_column"],
        help="Rule to build y_true labels",
    )
    parser.add_argument("--data-as-of", default=None, help="Optional CDC snapshot date YYYY-MM-DD")
    parser.add_argument("--app-token", default=None, help="Optional CDC app token")
    parser.add_argument("--percent-threshold", type=float, default=None, help="For percent_expected/combined")
    parser.add_argument("--window", type=int, default=None, help="For zscore/combined")
    parser.add_argument("--min-history", type=int, default=None, help="For zscore/combined")
    parser.add_argument("--z-k", type=float, default=None, help="For zscore/combined")
    parser.add_argument("--truth-column", default=None, help="For provided_column")
    parser.add_argument("--detail-out", default=None, help="Optional CSV path for weekly detail output")
    args = parser.parse_args()

    result = run_backtest(
        state=args.state,
        threshold=args.threshold,
        truth_rule=args.truth_rule,
        source=args.source,
        rule_params=_parse_rule_params(args),
        data_as_of=args.data_as_of,
        app_token=args.app_token,
    )

    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    if args.detail_out:
        result.weekly_detail.to_csv(args.detail_out, index=False)
        print(f"Weekly detail written to: {args.detail_out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Age-stratum matching of disease and control donors.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor-column", default="donorID")
    parser.add_argument("--age-column", default="age")
    parser.add_argument("--condition-column", default="condition")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--disease-label", action="append", required=True)
    parser.add_argument("--age-bin-width", type=float, required=True)
    parser.add_argument("--controls-per-case", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def donor_table(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    columns = [args.donor_column, args.age_column, args.condition_column]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Missing metadata columns: {', '.join(missing)}")
    clean = frame[columns].dropna().copy()
    clean[args.age_column] = pd.to_numeric(clean[args.age_column], errors="raise")
    counts = clean.groupby(args.donor_column)[[args.age_column, args.condition_column]].nunique()
    inconsistent = counts[(counts > 1).any(axis=1)]
    if not inconsistent.empty:
        raise ValueError("Age or condition is inconsistent within donor: " + ", ".join(inconsistent.index.astype(str)))
    return clean.drop_duplicates(args.donor_column).reset_index(drop=True)


def match_one(donors: pd.DataFrame, disease: str, args: argparse.Namespace, rng) -> pd.DataFrame:
    condition = args.condition_column
    age = args.age_column
    disease_frame = donors[donors[condition] == disease].copy()
    control_frame = donors[donors[condition] == args.control_label].copy()
    if disease_frame.empty or control_frame.empty:
        raise ValueError(f"Missing disease or control donors for {disease!r}")

    lower = max(disease_frame[age].min(), control_frame[age].min())
    upper = min(disease_frame[age].max(), control_frame[age].max())
    disease_frame = disease_frame[disease_frame[age].between(lower, upper)].copy()
    control_frame = control_frame[control_frame[age].between(lower, upper)].copy()
    anchor = np.floor(lower / args.age_bin_width) * args.age_bin_width
    edges = np.arange(anchor, upper + args.age_bin_width * 1.0001, args.age_bin_width)
    if len(edges) < 2:
        edges = np.array([lower, lower + args.age_bin_width])
    disease_frame["age_stratum"] = pd.cut(disease_frame[age], edges, right=False, include_lowest=True)
    control_frame["age_stratum"] = pd.cut(control_frame[age], edges, right=False, include_lowest=True)

    matched: list[pd.DataFrame] = []
    for stratum in disease_frame["age_stratum"].dropna().unique():
        cases = disease_frame[disease_frame["age_stratum"] == stratum]
        controls = control_frame[control_frame["age_stratum"] == stratum]
        n_cases = min(len(cases), len(controls) // args.controls_per_case)
        if n_cases == 0:
            continue
        case_index = rng.choice(cases.index.to_numpy(), n_cases, replace=False)
        control_index = rng.choice(
            controls.index.to_numpy(), n_cases * args.controls_per_case, replace=False
        )
        selected_cases = cases.loc[case_index].copy()
        selected_controls = controls.loc[control_index].copy()
        selected_cases["match_role"] = "disease"
        selected_controls["match_role"] = "control"
        matched.extend([selected_cases, selected_controls])
    if not matched:
        raise ValueError(f"No shared age strata contained both {disease!r} and controls")
    result = pd.concat(matched, ignore_index=True)
    result["comparison"] = disease
    result["overlap_age_min"] = lower
    result["overlap_age_max"] = upper
    return result


def main() -> int:
    args = parse_args()
    if args.age_bin_width <= 0 or args.controls_per_case < 1:
        raise ValueError("age-bin-width and controls-per-case must be positive")
    frame = pd.read_csv(args.metadata)
    donors = donor_table(frame, args)
    rng = np.random.default_rng(args.seed)
    matched = pd.concat(
        [match_one(donors, disease, args, rng) for disease in args.disease_label],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output, index=False)
    summary = {
        "metadata": str(args.metadata.resolve()),
        "age_bin_width": args.age_bin_width,
        "controls_per_case": args.controls_per_case,
        "seed": args.seed,
        "comparisons": matched.groupby(["comparison", "match_role"])[args.donor_column]
        .nunique()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

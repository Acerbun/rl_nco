"""Generate exact-benchmark figures and LaTeX table from CSV results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
HEIGHTS_CSV = ROOT / "heights_summary.csv"
USERS_CSV = ROOT / "users_summary.csv"

OUTPUT_DIR = ROOT / "exact_benchmark_figures"

METHOD_ORDER = [
    "Equal Allocation",
    "Sum-rate DDQN",
    "Proposed Q-Min",
    "Exact Max-Min",
]

METHOD_STYLES = {
    "Equal Allocation": {
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    "Sum-rate DDQN": {
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "Proposed Q-Min": {
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.2,
    },
    "Exact Max-Min": {
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
}

NUMERIC_FIELDS = {
    "x_value",
    "num_users",
    "height_m",
    "mean_terminal_rate_mbps",
    "std_across_seed_means_mbps",
    "std_all_values_mbps",
    "mean_optimality_gap_percent",
    "mean_optimality_ratio_percent",
}


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    """Read a benchmark CSV containing a UTF-8 BOM if present."""

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        for raw_row in reader:
            row: dict[str, Any] = dict(raw_row)

            for field in NUMERIC_FIELDS:
                row[field] = float(row[field])

            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    return rows


def validate_results(rows: list[dict[str, Any]]) -> None:
    """Check method completeness and exact-reference consistency."""

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["x_value"]].append(row)

    for x_value, group in grouped.items():
        methods = {row["method"] for row in group}

        missing = set(METHOD_ORDER) - methods
        if missing:
            raise ValueError(
                f"Configuration {x_value} is missing methods: {missing}"
            )

        exact_row = next(
            row for row in group
            if row["method"] == "Exact Max-Min"
        )

        if abs(exact_row["mean_optimality_gap_percent"]) > 1e-8:
            raise ValueError(
                f"Exact gap is not zero at x={x_value}."
            )

        if abs(
            exact_row["mean_optimality_ratio_percent"] - 100.0
        ) > 1e-8:
            raise ValueError(
                f"Exact ratio is not 100% at x={x_value}."
            )

        exact_rate = exact_row["mean_terminal_rate_mbps"]

        for row in group:
            if (
                row["method"] != "Exact Max-Min"
                and row["mean_terminal_rate_mbps"]
                > exact_rate + 1e-8
            ):
                raise ValueError(
                    f"{row['method']} exceeds the exact reference "
                    f"at x={x_value}."
                )


def extract_method_series(
    rows: list[dict[str, Any]],
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted x values and mean terminal rates."""

    selected = sorted(
        (
            row for row in rows
            if row["method"] == method
        ),
        key=lambda row: row["x_value"],
    )

    x_values = np.asarray(
        [row["x_value"] for row in selected],
        dtype=np.float64,
    )

    rates = np.asarray(
        [
            row["mean_terminal_rate_mbps"]
            for row in selected
        ],
        dtype=np.float64,
    )

    return x_values, rates


def plot_benchmark(
    rows: list[dict[str, Any]],
    x_label: str,
    output_stem: str,
    integer_ticks: bool = False,
) -> None:
    """Create a publication-style terminal-rate comparison figure."""

    fig, ax = plt.subplots(figsize=(5.2, 3.65))

    for method in METHOD_ORDER:
        x_values, rates = extract_method_series(
            rows=rows,
            method=method,
        )

        ax.plot(
            x_values,
            rates,
            label=method,
            markersize=5.5,
            **METHOD_STYLES[method],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Terminal Minimum Rate (Mbps)")

    if integer_ticks:
        all_x = sorted({int(row["x_value"]) for row in rows})
        ax.set_xticks(all_x)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.legend(
        frameon=True,
        fontsize=8.5,
    )

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    png_path = OUTPUT_DIR / f"{output_stem}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def find_representative_rows(
    rows: list[dict[str, Any]],
    x_value: float,
) -> list[dict[str, Any]]:
    """Select one complete benchmark configuration."""

    selected = [
        row for row in rows
        if abs(row["x_value"] - x_value) < 1e-9
    ]

    if len(selected) != len(METHOD_ORDER):
        raise ValueError(
            f"Expected {len(METHOD_ORDER)} rows at x={x_value}, "
            f"but found {len(selected)}."
        )

    lookup = {
        row["method"]: row
        for row in selected
    }

    return [lookup[method] for method in METHOD_ORDER]


def format_rate(row: dict[str, Any]) -> str:
    """Format rate and cross-seed standard deviation."""

    mean_rate = row["mean_terminal_rate_mbps"]
    seed_std = row["std_across_seed_means_mbps"]

    if row["method"] in {
        "Sum-rate DDQN",
        "Proposed Q-Min",
    }:
        return (
            f"${mean_rate:.4f}"
            f"\\mathbin{{\\pm}}{seed_std:.4f}$"
        )

    return f"${mean_rate:.4f}$"


def write_latex_table(
    rows: list[dict[str, Any]],
    representative_x: float = 100.0,
) -> None:
    """
    Generate the representative H=100 m, K=10 table from
    heights_summary.csv.
    """

    selected = find_representative_rows(
        rows=rows,
        x_value=representative_x,
    )

    lines = [
        r"\begin{table}[!t]",
        r"    \caption{Terminal Performance Relative to the Exact "
        r"Max-Min Solution ($K=10$, $H=100$~m)}",
        r"    \label{tab:optimality_gap}",
        r"    \centering",
        r"    \footnotesize",
        r"    \renewcommand{\arraystretch}{1.15}",
        r"    \setlength{\tabcolsep}{3.3pt}",
        r"    \begin{tabular}{lccc}",
        r"        \hline",
        r"        \textbf{Method} &",
        r"        \begin{tabular}[c]{@{}c@{}}\textbf{Terminal Rate}\\"
        r"\textbf{(Mbps)}\end{tabular} &",
        r"        \begin{tabular}[c]{@{}c@{}}\textbf{Opt. Gap}\\"
        r"\textbf{(\%)}\end{tabular} &",
        r"        \begin{tabular}[c]{@{}c@{}}\textbf{Opt. Ratio}\\"
        r"\textbf{(\%)}\end{tabular} \\",
        r"        \hline",
    ]

    latex_names = {
        "Equal Allocation": "Equal allocation",
        "Sum-rate DDQN": "Sum-rate DDQN",
        "Proposed Q-Min": r"\textbf{Proposed Q-Min}",
        "Exact Max-Min": "Exact max-min",
    }

    for row in selected:
        method = row["method"]
        rate_text = format_rate(row)

        gap = row["mean_optimality_gap_percent"]
        ratio = row["mean_optimality_ratio_percent"]

        if method == "Proposed Q-Min":
            rate_text = rf"\mathbf{{{rate_text[1:-1]}}}"
            rate_text = f"${rate_text}$"

        lines.append(
            f"        {latex_names[method]} & "
            f"{rate_text} & "
            f"${gap:.2f}$ & "
            f"${ratio:.2f}$ \\\\"
        )

    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )

    table_path = (
        OUTPUT_DIR
        / "table_optimality_gap_h100.tex"
    )

    table_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {table_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    height_rows = read_summary_csv(HEIGHTS_CSV)
    user_rows = read_summary_csv(USERS_CSV)

    validate_results(height_rows)
    validate_results(user_rows)

    plot_benchmark(
        rows=height_rows,
        x_label=r"UAV Hovering Altitude $H$ (m)",
        output_stem="fig_heights_exact",
        integer_ticks=True,
    )

    plot_benchmark(
        rows=user_rows,
        x_label=r"Number of Ground Users $K$",
        output_stem="fig_users_exact",
        integer_ticks=True,
    )

    # Use Models_h100 results for the representative table.
    write_latex_table(
        rows=height_rows,
        representative_x=100.0,
    )

    print("\nAll exact-benchmark outputs were generated successfully.")


if __name__ == "__main__":
    main()
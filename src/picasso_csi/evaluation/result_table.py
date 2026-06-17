"""Compact CSV result table utilities for PICASSO experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


RESULT_FIELDS = [
    "stage",
    "method",
    "pilot_ratio",
    "snr_db",
    "nmse",
    "mse",
    "mae",
    "pilot_consistency_error",
    "delay_sparsity_score",
    "epochs",
    "num_train_samples",
    "seed",
]

STAGE2BC_FIELDS = [
    "stage",
    "method",
    "loss_mode",
    "use_condition",
    "pilot_ratio",
    "snr_db",
    "seed",
    "nmse",
    "mse",
    "mae",
    "pilot_consistency_error",
    "delay_sparsity_score",
    "epochs",
    "num_train_samples",
    "runtime_seconds",
]


class ResultTable:
    """In-memory result accumulator with small CSV writers."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append(self, row: dict[str, object]) -> None:
        self.rows.append(dict(row))

    def save_raw_csv(self, path: str | Path) -> Path:
        return write_csv(self.rows, path, STAGE2BC_FIELDS)

    def grouped_summary(self) -> list[dict[str, object]]:
        grouped = _group_rows(self.rows, ("method",))
        summary = []
        for (method,), rows in sorted(grouped.items()):
            values = [_float(row["nmse"]) for row in rows]
            summary.append(
                {
                    "method": method,
                    "nmse_mean": _format_float(mean(values)),
                    "nmse_std": _format_float(_std(values)),
                    "count": len(values),
                }
            )
        return summary

    def save_summary_csv(self, path: str | Path) -> Path:
        return write_csv(self.grouped_summary(), path, ["method", "nmse_mean", "nmse_std", "count"])

    def paper_table(self) -> list[dict[str, object]]:
        methods = _ordered_methods(self.rows)
        grouped = _group_rows(self.rows, ("pilot_ratio", "snr_db"))
        output = []
        for (pilot_ratio, snr_db), rows in sorted(grouped.items(), key=lambda item: (float(item[0][0]), float(item[0][1]))):
            row_out: dict[str, object] = {"pilot_ratio": pilot_ratio, "snr_db": snr_db}
            best_method = ""
            best_value = math.inf
            for method in methods:
                values = [_float(row["nmse"]) for row in rows if row["method"] == method]
                row_out[method] = _format_mean_std(values) if values else ""
                if values and mean(values) < best_value:
                    best_value = mean(values)
                    best_method = method
            row_out["best_method"] = best_method
            output.append(row_out)
        return output

    def save_paper_table_csv(self, path: str | Path) -> Path:
        methods = _ordered_methods(self.rows)
        return write_csv(self.paper_table(), path, ["pilot_ratio", "snr_db", *methods, "best_method"])

    def diagnosis_rows(self) -> list[dict[str, object]]:
        rows = []
        summary_by_method = {row["method"]: _float(row["nmse_mean"]) for row in self.grouped_summary()}
        ls = summary_by_method.get("LS")
        dncnn = summary_by_method.get("DnCNN")
        enhanced = summary_by_method.get("Enhanced-DnCNN")
        for method, value in sorted(summary_by_method.items(), key=lambda item: item[1]):
            rows.append(
                {
                    "method": method,
                    "overall_nmse": _format_float(value),
                    "gap_vs_ls": _format_float(value - ls) if ls is not None else "",
                    "gap_vs_dncnn": _format_float(value - dncnn) if dncnn is not None else "",
                    "gap_vs_enhanced_dncnn": _format_float(value - enhanced) if enhanced is not None else "",
                }
            )
        return rows

    def save_diagnosis_csv(self, path: str | Path) -> Path:
        return write_csv(
            self.diagnosis_rows(),
            path,
            ["method", "overall_nmse", "gap_vs_ls", "gap_vs_dncnn", "gap_vs_enhanced_dncnn"],
        )

    def winner(self, predicate=None) -> str:
        rows = [row for row in self.rows if predicate is None or predicate(row)]
        if not rows:
            return ""
        grouped = _group_rows(rows, ("method",))
        scores = {method[0]: mean(_float(row["nmse"]) for row in method_rows) for method, method_rows in grouped.items()}
        return min(scores, key=scores.get)


def write_result_csv(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    """Write the legacy Stage 2A metrics CSV."""

    return write_csv(rows, path, RESULT_FIELDS)


def write_csv(rows: Iterable[dict[str, object]], path: str | Path, fields: list[str]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in row_list:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output_path


def _group_rows(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _ordered_methods(rows: Iterable[dict[str, object]]) -> list[str]:
    preferred = [
        "LS",
        "DnCNN",
        "Cond-DnCNN",
        "Enhanced-DnCNN",
        "PICASSO-rec",
        "PICASSO-rec-physics",
        "PICASSO-rec-adv",
        "PICASSO-full",
        "PICASSO-cond-full",
    ]
    present = {str(row["method"]) for row in rows}
    return [method for method in preferred if method in present] + sorted(present.difference(preferred))


def _format_mean_std(values: list[float]) -> str:
    return f"{mean(values):.6f}+/-{_std(values):.6f}"


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _float(value: object) -> float:
    return float(value)

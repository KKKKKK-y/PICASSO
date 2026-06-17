"""Small CSV result table utilities for Stage 2A."""

from __future__ import annotations

import csv
from pathlib import Path
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


def write_result_csv(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    """Write a compact metrics-only CSV result table."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in row_list:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    return output_path

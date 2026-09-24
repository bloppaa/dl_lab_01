"""
Junta los resultados.csv generados por scripts/run_all_experiments.sh
(layout results/<target>/<variante>/resultados.csv) en una sola tabla
comparativa para el informe.

Por que existe este script: save_experiment_reports() en src/reporting.py
sobreescribe resultados.csv en cada corrida de main.py (no acumula), por
eso cada variante se corre con su propio --output-dir y este script las
junta despues, sin tocar main.py ni src/.

Uso:
    python scripts/merge_results.py
    python scripts/merge_results.py --results-root results_quick
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VARIANT_LABELS = {
    "softmax_def": "Softmax (def.)",
    "softmax_grid": "Softmax (mejor HP)",
    "coral": "CORAL",
    "coral_weighted": "CORAL + pesos",
}

VARIANT_ORDER = list(VARIANT_LABELS.values())

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mae_ordinal",
    "qwk",
    "accuracy_pm1",
    "errores_graves",
]

DISPLAY_NAMES = {
    "accuracy": "Acc",
    "balanced_accuracy": "BalAcc",
    "precision_macro": "Prec_m",
    "recall_macro": "Rec_m",
    "f1_macro": "F1_m",
    "mae_ordinal": "MAE",
    "qwk": "QWK",
    "accuracy_pm1": "Acc±1",
    "errores_graves": "Err≥2",
}


def discover_runs(results_root: Path) -> list[tuple[Path, str, str]]:
    """
    Recorre results_root/<target>/<variante>/resultados.csv.
    Devuelve (csv_path, target_folder_name, variant_folder_name).
    """

    found = []
    if not results_root.exists():
        return found

    for target_dir in sorted(results_root.iterdir()):
        if not target_dir.is_dir():
            continue
        for variant_dir in sorted(target_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            csv_path = variant_dir / "resultados.csv"
            if csv_path.exists():
                found.append((csv_path, target_dir.name, variant_dir.name))

    return found


def load_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    runs = discover_runs(results_root)

    if not runs:
        print(f"Aviso: no se encontro ningun resultados.csv bajo {results_root}/")
        return rows

    for csv_path, target_folder, variant_folder in runs:
        label = VARIANT_LABELS.get(variant_folder, variant_folder)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["run_folder"] = str(csv_path.parent)
                row["variant"] = label
                rows.append(row)

    return rows


def variant_sort_key(row: dict) -> tuple:
    variant = row["variant"]
    order_index = (
        VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
    )
    return (row["target"], order_index)


def write_combined_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["target", "variant", "algorithm", "run_folder"] + METRIC_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_combined_markdown(rows: list[dict], path: Path) -> None:
    header = ["Target", "Variante"] + [DISPLAY_NAMES[m] for m in METRIC_COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    current_target = None
    for row in rows:
        if row["target"] != current_target:
            if current_target is not None:
                lines.append("| | | | | | | | | | |")
            current_target = row["target"]

        values = [row["target"], row["variant"]]
        for metric in METRIC_COLUMNS:
            std_key = f"{metric}_std"
            mean_val = float(row[metric])
            std_val = float(row[std_key]) if row.get(std_key) else None
            if std_val is not None:
                values.append(f"{mean_val:.4f} ± {std_val:.4f}")
            else:
                values.append(f"{mean_val:.4f}")
        lines.append("| " + " | ".join(values) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results",
        help="Carpeta raiz con <target>/<variante>/resultados.csv (default: results)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = load_rows(results_root)
    if not rows:
        return

    rows.sort(key=variant_sort_key)

    output_dir = results_root / "_combined"
    csv_path = output_dir / "resultados_combinados.csv"
    md_path = output_dir / "resultados_combinados.md"
    write_combined_csv(rows, csv_path)
    write_combined_markdown(rows, md_path)

    print(f"{len(rows)} filas combinadas desde {results_root}/")
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Corre las 4 variantes (Softmax def., Softmax mejor HP, CORAL, CORAL+pesos)
# para los 6 targets (GDS, GDS_R1..GDS_R5), en un layout reproducible:
#
#   results/<target>/<variante>/resultados.csv
#
# Uso:
#   bash scripts/run_all_experiments.sh            # corrida completa (20 epocas)
#   bash scripts/run_all_experiments.sh --quick     # depuracion rapida (5 epocas)
#
# Correr desde la raiz del repo (donde esta main.py).

set -euo pipefail

DATA_PATH="dataset/15_atributos_R0-R5.sav"
EPOCHS=20
OUTPUT_ROOT="results"

if [[ "${1:-}" == "--quick" ]]; then
    EPOCHS=5
    OUTPUT_ROOT="results_quick"
    echo "Modo rapido: --epochs ${EPOCHS}, salida en ${OUTPUT_ROOT}/"
fi

mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${OUTPUT_ROOT}/run_log.txt"
echo "Corrida iniciada: $(date)" > "${LOG_FILE}"

TARGETS=("GDS" "GDS_R1" "GDS_R2" "GDS_R3" "GDS_R4" "GDS_R5")

run_experiment () {
    local target="$1"
    local variant_name="$2"
    shift 2
    local extra_args=("$@")

    local out_dir="${OUTPUT_ROOT}/${target}/${variant_name}"
    local start_ts
    start_ts=$(date +%s)

    echo ""
    echo "=== ${target} / ${variant_name} ==="
    python main.py \
        --data-path "${DATA_PATH}" \
        --target-name "${target}" \
        --epochs "${EPOCHS}" \
        --output-dir "${out_dir}" \
        "${extra_args[@]}" 2>&1 | tee -a "${LOG_FILE}"

    local end_ts
    end_ts=$(date +%s)
    echo "  (${target}/${variant_name} tardo $((end_ts - start_ts))s)" | tee -a "${LOG_FILE}"
}

overall_start=$(date +%s)

for target in "${TARGETS[@]}"; do
    if [[ "${target}" == "GDS" ]]; then
        fold_args=(--outer-folds 2 --inner-folds 2)
    else
        fold_args=()
    fi

    run_experiment "${target}" "softmax_def"    "${fold_args[@]}" --no-grid
    run_experiment "${target}" "softmax_grid"   "${fold_args[@]}"
    run_experiment "${target}" "coral"          "${fold_args[@]}" --coral
    run_experiment "${target}" "coral_weighted" "${fold_args[@]}" --coral --use-weights
done

overall_end=$(date +%s)
echo ""
echo "Todos los experimentos terminaron en $(( (overall_end - overall_start) / 60 )) min."
echo "Log completo en: ${LOG_FILE}"
echo ""
echo "Ahora corre:"
echo "  python scripts/merge_results.py --results-root ${OUTPUT_ROOT}"
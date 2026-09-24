# Laboratorio 01: Clasificación Ordinal con Redes Neuronales (CORAL vs. Softmax)

Este repositorio contiene la implementación y evaluación experimental para la clasificación de etapas de deterioro cognitivo (escala GDS, objetivos `GDS` y `GDS_R1` a `GDS_R5`) a partir de 15 atributos binarios. Se compara una arquitectura base Softmax frente a una cabeza de regresión ordinal **CORAL** (_Consistent Rank Logits_) con soporte de ponderación por número efectivo de muestras.

## 1. Requisitos y Configuración del Entorno

El proyecto fue desarrollado usando Conda y Python 3.10.

```bash
# Clonar el repositorio
git clone
cd

# Crear y activar el entorno de Conda
conda env create -f environment.yml
conda activate lab_pytorch
```

Asegúrese de contar con el archivo de datos dentro de la carpeta `dataset/`:

- `dataset/15_atributos_R0-R5.sav`

## 2. Ejecución Rápida de Pruebas Unitarias

Para verificar de forma aislada las implementaciones matemáticas de `labels_to_levels`, `coral_loss`, `effective_number_weights`, `CoralLayer`, `MLPCoral` y el cálculo de predicciones ordinales:

```bash
python scratch_tests.py
```

## 3. Reproducción Completa de Experimentos

Se proveen scripts automatizados para ejecutar la validación cruzada anidada (Nested CV) sobre las 4 variantes metodológicas:

- `Softmax (def.)`: Configuración por defecto sin búsqueda interna.
- `Softmax (mejor HP)`: Selección interna guiada por menor MAE ordinal (desempate QWK).
- `CORAL`: Modelo ordinal con sesgos consistentes.
- `CORAL + pesos`: Modelo ordinal ponderado por muestras efectivas ($\beta$).

### Opción A: Modo Rápido / Verificación (5 épocas)

Ejecuta la batería de pruebas en modo depuración sobre los 6 objetivos:

```bash
chmod +x scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh --quick
python scripts/merge_results.py --results-root results_quick
```

Los resultados consolidados se guardan en `results_quick/\_combined/resultados_combinados.md` y `.csv`.

### Opción B: Ejecución Completa del Informe (20 épocas)

Ejecuta el protocolo experimental completo reportado en el informe:

```bash
bash scripts/run_all_experiments.sh
python scripts/merge_results.py
```

Los resultados consolidados se generarán en `results/_combined/resultados_combinados.md` y `.csv`.

**Nota metodológica sobre GDS (7 clases)**: El objetivo `GDS` ejecuta automáticamente 2 folds externos y 2 internos debido a la escasez extrema de instancias en clases severas (clase 7 con 2 casos), omitiendo la búsqueda interna estratificada por diseño. Los objetivos `GDS_R1` a `GDS_R5` emplean la configuración estándar de 5 folds externos y 3 internos.

## 4. Ejecución Puntual vía CLI

Para correr un experimento específico de manera individual:

```bash
# Softmax con Grid Search en GDS_R2
python main.py --data-path "dataset/15_atributos_R0-R5.sav" --target-name GDS_R2 --epochs 20

# Softmax sin Grid (configuración fija por defecto)
python main.py --data-path "dataset/15_atributos_R0-R5.sav" --target-name GDS_R2 --epochs 20 --no-grid

# CORAL con Grid Search
python main.py --data-path "dataset/15_atributos_R0-R5.sav" --target-name GDS_R2 --epochs 20 --coral

# CORAL con Grid Search y Pesos de Muestra Efectiva
python main.py --data-path "dataset/15_atributos_R0-R5.sav" --target-name GDS_R2 --epochs 20 --coral --use-weights
```

## 5. Estructura del Código

```text
├── dataset/                   # Archivo de datos (.sav)
├── scripts/
│   ├── run_all_experiments.sh # Automatización de las 4 variantes x 6 targets
│   └── merge_results.py       # Consolidación de tablas Markdown/CSV
├── src/
│   ├── data_loader.py         # Carga de datos (.sav / .csv)
│   ├── evaluation.py          # Métricas (QWK, MAE ordinal, BalAcc, etc.)
│   ├── losses.py              # coral_loss, labels_to_levels, effective_number_weights
│   ├── models.py              # ShallowMultiClassNet, CoralLayer, MLPCoral
│   ├── ordinal.py             # logits_to_ordinal_predictions
│   ├── preprocessing.py       # StratifiedKFold y encodings
│   └── reporting.py           # Exportación de reportes y figuras
├── config.py                  # HYPERPARAMETER_GRID y constantes globales
├── main.py                    # Flujo de ejecución y CLI
└── scratch_tests.py           # Pruebas unitarias de verificación matemática
```

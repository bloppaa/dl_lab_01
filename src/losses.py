"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch
import torch.nn.functional as F


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1)
    """

    thresholds = torch.arange(num_classes - 1, device=labels.device)  # [0, 1, ..., K-2]
    levels = (
        labels.unsqueeze(1) > thresholds.unsqueeze(0)
    ).float()  # broadcasting -> (B, K-1)
    return levels


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    BCE con logits sobre los K-1 umbrales ordinales.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    """

    levels = labels_to_levels(labels, num_classes)  # (B, K-1) target binario por umbral

    # BCE por umbral, sin reducir todavía
    per_threshold_loss = F.binary_cross_entropy_with_logits(
        logits, levels, reduction="none"
    )  # (B, K-1)

    # Sumar los K-1 umbrales -> una pérdida por muestra (así lo define CORAL)
    per_sample_loss = per_threshold_loss.sum(dim=1)  # (B,)

    if class_weights is not None:
        sample_weights = class_weights[
            labels
        ]  # (B,) -- peso según la clase real de cada muestra
        per_sample_loss = per_sample_loss * sample_weights

    return per_sample_loss.mean()


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """
    TODO(alumno):
    Pesos por numero efectivo de muestras:

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normalizar los pesos para que su media sea 1.

    Formas:
    - labels: (N,)
    - salida: (num_classes,)
    """

    raise NotImplementedError("TODO: implementar effective_number_weights().")

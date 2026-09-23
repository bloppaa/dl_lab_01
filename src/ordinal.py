"""Utilidades ordinales que los alumnos deben implementar."""

import torch


@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convierte logits CORAL en una clase entera.

    Pista:
    aplicar sigmoide, contar cuantos umbrales superan threshold
    y devolver ese conteo como y_hat.

    Formas:
    - logits: (batch_size, K-1)
    - salida: (batch_size,)
    """

    probs = torch.sigmoid(logits)  # (B, K-1)
    predictions = (probs > threshold).sum(
        dim=1
    )  # (B,) -- cuenta cuantos umbrales se superan
    return predictions.long()

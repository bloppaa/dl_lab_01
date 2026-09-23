import torch
import numpy as np
from src.losses import labels_to_levels, coral_loss, effective_number_weights


def test_labels_to_levels():
    labels = torch.tensor([2, 0, 4, 1])
    levels = labels_to_levels(labels, num_classes=5)
    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(levels, expected), f"got {levels}"
    print("labels_to_levels OK")


def test_coral_loss():
    logits = torch.tensor([[2.0, -1.0]])
    labels = torch.tensor([1])
    loss = coral_loss(logits, labels, num_classes=3)
    # BCE(2.0, 1) + BCE(-1.0, 0) ~= 0.1269 + 0.3133 = 0.4402
    assert torch.isclose(loss, torch.tensor(0.4402), atol=1e-3), f"got {loss.item()}"
    print("coral_loss (sin pesos) OK ->", loss.item())

    # Con class_weights: debe escalar el resultado
    weights = torch.tensor([1.0, 2.0, 1.0])  # clase 1 pesa el doble
    loss_weighted = coral_loss(logits, labels, num_classes=3, class_weights=weights)
    assert torch.isclose(loss_weighted, loss * 2), f"got {loss_weighted.item()}"
    print("coral_loss (con pesos) OK ->", loss_weighted.item())


def test_effective_number_weights():
    labels = np.array([0] * 649 + [1] * 298 + [2] * 172)
    weights = effective_number_weights(labels, num_classes=3, beta=0.99)
    assert torch.isclose(
        weights.mean(), torch.tensor(1.0), atol=1e-5
    ), f"media = {weights.mean()}"
    # clase minoritaria debe pesar mas que la mayoritaria
    assert weights[2] > weights[0], f"got {weights}"
    print("effective_number_weights OK ->", weights)


if __name__ == "__main__":
    test_labels_to_levels()
    test_coral_loss()
    test_effective_number_weights()

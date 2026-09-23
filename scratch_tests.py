import torch
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


if __name__ == "__main__":
    test_labels_to_levels()
    test_coral_loss()

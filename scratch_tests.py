import torch
import numpy as np
import torch.nn.functional as F

from src.losses import labels_to_levels, coral_loss, effective_number_weights
from src.models import CoralLayer, MLPCoral


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


def test_coral_layer_shape_and_order():
    torch.manual_seed(0)
    layer = CoralLayer(input_size=16, num_classes=5)
    x = torch.randn(8, 16)
    logits = layer(x)
    assert logits.shape == (8, 4), f"shape esperada (8,4), got {logits.shape}"

    biases = (
        layer.bias_base
        - torch.cat(
            [torch.zeros(1), torch.cumsum(F.softplus(layer.bias_deltas), dim=0)]
        )
    ).detach()
    assert torch.all(biases[:-1] >= biases[1:]), f"biases no ordenados: {biases}"
    print("CoralLayer OK -> shape", logits.shape, "biases", biases)


def test_mlp_coral_forward():
    torch.manual_seed(0)
    model = MLPCoral(num_features=15, num_classes=3, dropout=0.15)
    model.eval()  # evita el problema de BatchNorm con batch chico
    x = torch.randn(4, 15)
    logits = model(x)
    assert logits.shape == (4, 2), f"shape esperada (4,2), got {logits.shape}"
    print("MLPCoral OK -> shape", logits.shape)


if __name__ == "__main__":
    test_labels_to_levels()
    test_coral_loss()
    test_effective_number_weights()
    test_coral_layer_shape_and_order()
    test_mlp_coral_forward()

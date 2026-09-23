# scratch_tests.py
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


if __name__ == "__main__":
    test_labels_to_levels()

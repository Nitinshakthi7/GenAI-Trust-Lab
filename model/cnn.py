# cnn.py - SmallCNN, the simple "real or fake?" baseline classifier.
# It looks at raw pixels of a 3x32x32 image and gives ONE number (a "logit").
# A big positive logit means "fake", a big negative logit means "real".
# Turn the logit into a 0..1 score with torch.sigmoid(logit).
# Labels used when training: real = 0, fake = 1 (use BCEWithLogitsLoss).
# It is NOT trained in this file: training needs the fake images, which come later.
# Run the self-check with:  python -m model.cnn

import torch
import torch.nn as nn

from model.utils import count_params


class SmallCNN(nn.Module):
    """3 convolution layers (32 -> 16 -> 8 -> 4) and a small head with one output."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8x8 -> 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),                                               # ONE logit: real vs fake
        )

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    model = SmallCNN()
    x = torch.randn(8, 3, 32, 32)
    assert model(x).shape == (8, 1), "should give one logit per image"
    print("parameters:", f"{count_params(model):,}")

    # One training step on random data, just to prove the code runs.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    labels = torch.randint(0, 2, (8, 1)).float()
    loss = loss_fn(model(x), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("one training step ran, loss:", round(loss.item(), 4))
    print("self-check passed")

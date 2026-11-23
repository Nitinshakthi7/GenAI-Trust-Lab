# vae.py - a small convolutional VAE (Variational Auto-Encoder).
# A VAE squeezes an image down to a short list of numbers (the "latent" vector)
# and then rebuilds the image from it. We use it in two ways:
#   1. As a GENERATOR: pick random latent numbers -> decode -> a new fake image.
#   2. As an ANOMALY SCORER: a VAE trained on real images rebuilds real images
#      well and odd images badly, so a high score = "looks less like the training images".
#
# Loss (used for training AND for the anomaly score, so they match):
#   reconstruction error = sum of squared pixel differences for ONE image
#   KL term              = how far the latent numbers are from a plain bell curve
#   loss = reconstruction error + KL, then averaged over the images in the batch.
#
# Images are 32x32 with pixel values in [-1, 1] (1 channel or 3 channels).
# Run the self-check with:  python -m model.vae

import torch
import torch.nn as nn

from model.utils import get_device, count_params


class ConvVAE(nn.Module):
    """Encoder 32 -> 16 -> 8 -> 4 (strided convolutions), decoder mirrors it.

    Note: sample(), interpolate() and anomaly_score() switch the model to eval
    mode. The training loop calls model.train() again at the start of each epoch.
    """

    def __init__(self, channels=3, latent=128):
        super().__init__()
        self.latent = latent
        self.channels = channels
        # Encoder: each conv halves the width and height (32 -> 16 -> 8 -> 4).
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(128 * 4 * 4, latent)       # the "centre" of the latent
        self.to_logvar = nn.Linear(128 * 4 * 4, latent)   # the "spread" (as a log)
        # Decoder: a mirror image of the encoder (4 -> 8 -> 16 -> 32).
        self.from_latent = nn.Linear(latent, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1), nn.Tanh(),  # tanh keeps pixels in [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        h = self.from_latent(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Random nudge around the centre (the "reparameterisation trick").
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def anomaly_score(self, x):
        """Score for each image: reconstruction error + KL. Higher = less like the training images.

        x is one image (C,32,32) or a batch (N,C,32,32). Returns a tensor of N numbers.
        No random nudge is used, so the same image always gets the same score.
        """
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        mu, logvar = self.encode(x)
        recon_err, kl = per_image_terms(self.decode(mu), x, mu, logvar)
        return recon_err + kl

    def _latent_from_seed(self, seed):
        """One repeatable random latent vector, shape (1, latent)."""
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(1, self.latent, generator=g)
        return z.to(next(self.parameters()).device)

    @torch.no_grad()
    def sample(self, n, seed):
        """Make n new images from random latents. Same seed = same images. Shape (n,C,32,32)."""
        self.eval()
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.latent, generator=g).to(next(self.parameters()).device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, seed_a, seed_b, frames):
        """Slide from the latent of seed_a to the latent of seed_b in `frames` steps. Shape (frames,C,32,32)."""
        self.eval()
        a, b = self._latent_from_seed(seed_a), self._latent_from_seed(seed_b)
        t = torch.linspace(0, 1, frames).view(-1, 1).to(a.device)
        return self.decode(a + t * (b - a))  # straight-line blend between the two latents


def per_image_terms(recon, x, mu, logvar):
    """Two numbers per image: (reconstruction error, KL). Both have shape (N,)."""
    recon_err = ((recon - x) ** 2).flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return recon_err, kl


def vae_loss(recon, x, mu, logvar):
    """Returns (total, reconstruction part, KL part), each averaged over the batch."""
    recon_err, kl = per_image_terms(recon, x, mu, logvar)
    return (recon_err + kl).mean(), recon_err.mean(), kl.mean()


def batch_loss(model, x):
    """Run the model on a batch and return just the total loss (one number)."""
    recon, mu, logvar = model(x)
    return vae_loss(recon, x, mu, logvar)[0]


if __name__ == "__main__":
    # Self-check on real CIFAR-10 images (needs the dataset downloaded first).
    from model.data import load_cifar10
    from model.utils import set_seed

    set_seed()
    device = get_device()
    images = load_cifar10("train")[:2000].to(device)
    model = ConvVAE(channels=3, latent=128).to(device)

    assert model(images[:16])[0].shape == images[:16].shape, "output shape must equal input shape"
    with torch.no_grad():
        loss_before = batch_loss(model, images[:512]).item()
    assert torch.isfinite(torch.tensor(loss_before)), "loss must be a finite number"
    n = count_params(model)
    print(f"parameters: {n:,}   weights size (float32): {n * 4 / 1e6:.2f} MB")

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for step in range(200):
        batch = images[torch.randint(0, 2000, (128,), device=device)]
        loss = batch_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss_after = batch_loss(model, images[:512]).item()
    print(f"loss at step 0: {loss_before:.1f}   after 200 steps: {loss_after:.1f}")
    assert loss_after < loss_before, "loss should go down while training"

    scores = model.anomaly_score(images[:5])
    assert scores.shape == (5,), "one score per image"
    assert model.anomaly_score(images[0]).shape == (1,), "a single image gives one score"
    assert model.sample(4, seed=1).shape == (4, 3, 32, 32)
    # The GPU can differ in the 8th decimal place, so "almost equal" is the right test.
    assert torch.allclose(model.sample(4, seed=1), model.sample(4, seed=1), atol=1e-5), "same seed, same images"
    assert not torch.allclose(model.sample(4, seed=1), model.sample(4, seed=2), atol=1e-5), "other seed, other images"
    assert model.interpolate(1, 2, 8).shape == (8, 3, 32, 32)
    print("anomaly scores of 5 real images:", [round(s, 1) for s in scores.tolist()])
    print("self-check passed")

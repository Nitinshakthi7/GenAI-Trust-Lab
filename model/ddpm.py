# ddpm.py - a small diffusion model (DDPM) for 32x32 images.
#
# The idea in plain words:
#   TRAINING: take a real image, mix in a random amount of noise (the "timestep" t
#             says how much: 0 = almost none, 999 = pure noise), and teach a
#             network to guess the noise that was added.
#   SAMPLING: start from pure noise and remove the guessed noise step by step
#             until an image appears. DDIM is a fast way to do this in 20-100
#             steps instead of 1000.
#
# What is in this file:
#   TinyUNet        the network (a small U-Net, about 2 million numbers to learn)
#   make_schedule   how much noise is added at each timestep (cosine or linear)
#   q_sample        add noise to an image up to timestep t
#   ddpm_loss       the training loss (mean squared error on the noise guess)
#   ddim_sample     make new images (can use "classifier-free guidance")
#   reconstruct     add some noise to a real image, then denoise it again, and
#                   return the per-pixel error (the Detector's DIRE-style idea,
#                   Wang et al., arXiv 2303.09295)
#   EMA             a slowly-moving average of the weights, which samples better
#
# Images are 32x32 with pixel values in [-1, 1] (1 channel or 3 channels).
# Every number here is "proposed, to tune".
# Run the self-check with:  python -m model.ddpm

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import count_params, get_device, set_seed


# ---------------------------------------------------------------- the network

def timestep_embedding(t, dim):
    """Turn each timestep number into `dim` sine/cosine numbers. Shape (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """Two convolutions with the timestep/class information added in the middle.
    The input is added back at the end (the "residual" connection)."""

    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]   # add the time/class info
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """A small U-Net. Sizes go 32 -> 16 -> 8 and back up to 32.

    in_channels: 1 (Fashion-MNIST) or 3 (CIFAR-10).
    num_classes: 0 = unconditional. 10 = the model also gets a class label.
                 Label number `num_classes` means "no label" (used for guidance).
    forward(x, t, class_ids=None): returns the guess of the noise, same shape as x.
    A conditional model with class_ids=None uses the "no label" label.
    An unconditional model ignores class_ids.
    """

    def __init__(self, in_channels=3, num_classes=0, base=64):
        super().__init__()
        self.in_channels, self.num_classes, self.base = in_channels, num_classes, base
        c1, c2, e = base, base * 2, base * 4        # channel counts and embedding size
        self.time_mlp = nn.Sequential(nn.Linear(base, e), nn.SiLU(), nn.Linear(e, e))
        self.class_emb = nn.Embedding(num_classes + 1, e) if num_classes > 0 else None

        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)
        # Going down: 32x32 -> 16x16 -> 8x8.
        self.down0 = ResBlock(c1, c1, e)
        self.pool0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.down1 = ResBlock(c1, c2, e)
        self.pool1 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.down2 = ResBlock(c2, c2, e)
        # The middle (8x8).
        self.mid1 = ResBlock(c2, c2, e)
        self.mid2 = ResBlock(c2, c2, e)
        # Going up: each block also receives the matching "skip" from the way down.
        self.up2 = ResBlock(c2 + c2, c2, e)
        self.upconv1 = nn.Conv2d(c2, c2, 3, padding=1)
        self.up1 = ResBlock(c2 + c2, c2, e)
        self.upconv0 = nn.Conv2d(c2, c1, 3, padding=1)
        self.up0 = ResBlock(c1 + c1, c1, e)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(),
                                 nn.Conv2d(c1, in_channels, 3, padding=1))

    def forward(self, x, t, class_ids=None):
        emb = self.time_mlp(timestep_embedding(t, self.base))
        if self.class_emb is not None:
            if class_ids is None:                                   # no label given
                class_ids = torch.full_like(t, self.num_classes)    # use the "no label" label
            emb = emb + self.class_emb(class_ids)

        h = self.conv_in(x)
        s0 = self.down0(h, emb)                   # 32x32
        s1 = self.down1(self.pool0(s0), emb)      # 16x16
        s2 = self.down2(self.pool1(s1), emb)      # 8x8
        h = self.mid2(self.mid1(s2, emb), emb)
        h = self.up2(torch.cat([h, s2], dim=1), emb)
        h = self.upconv1(F.interpolate(h, scale_factor=2, mode="nearest"))     # 8 -> 16
        h = self.up1(torch.cat([h, s1], dim=1), emb)
        h = self.upconv0(F.interpolate(h, scale_factor=2, mode="nearest"))     # 16 -> 32
        h = self.up0(torch.cat([h, s0], dim=1), emb)
        return self.out(h)


# ------------------------------------------------------- noise schedule + loss

def make_schedule(T=1000, kind="cosine", device="cpu"):
    """How much noise each timestep adds. Returns a dict of tensors of length T.

    "cosine": from Improved DDPM (s = 0.008). "linear": betas from 1e-4 to 0.02.
    alphas_cumprod[t] = how much of the ORIGINAL image is left at timestep t
    (close to 1 at t=0, close to 0 at t=T-1).
    """
    if kind == "linear":
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    elif kind == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    else:
        raise ValueError("kind must be 'cosine' or 'linear'")
    alphas = 1 - betas
    return {
        "T": T, "kind": kind,
        "betas": betas.float().to(device),
        "alphas": alphas.float().to(device),
        "alphas_cumprod": torch.cumprod(alphas, dim=0).float().to(device),
    }


def q_sample(x0, t, noise, schedule):
    """Add noise to the clean images x0 up to timesteps t (a tensor of N ints)."""
    ab = schedule["alphas_cumprod"].to(x0.device)[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def ddpm_loss(model, x0, schedule, class_ids=None, p_null=0.1):
    """Training loss: random timesteps, add noise, guess the noise, mean squared error.

    For a conditional model each label is replaced by the "no label" label with
    probability p_null (0.1, proposed), so one network learns both modes.
    """
    n = len(x0)
    t = torch.randint(0, schedule["T"], (n,), device=x0.device)
    noise = torch.randn_like(x0)
    if class_ids is not None:
        drop = torch.rand(n, device=x0.device) < p_null
        class_ids = torch.where(drop, torch.full_like(class_ids, model.num_classes), class_ids)
    return F.mse_loss(model(q_sample(x0, t, noise, schedule), t, class_ids), noise)


# ------------------------------------------------------------------ sampling

def _guess_noise(model, x, t, class_ids, guidance):
    """The network's noise guess. With guidance w > 0 (classifier-free guidance):
    eps = (1 + w) * eps_with_label - w * eps_without_label  (two network passes)."""
    tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
    eps = model(x, tt, class_ids)
    if guidance > 0:
        eps_plain = model(x, tt, None)                  # None = the "no label" label
        eps = (1 + guidance) * eps - guidance * eps_plain
    return eps


def _ddim_loop(model, x, timesteps, schedule, class_ids, guidance, eta, gen):
    """Walk down the list of timesteps (big to small), removing noise at each step."""
    ac = schedule["alphas_cumprod"].to(x.device)
    for i, t in enumerate(timesteps):
        ab_t = ac[t]
        ab_prev = ac[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.ones_like(ab_t)
        eps = _guess_noise(model, x, t, class_ids, guidance)
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)   # guess of the clean image
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()                  # noise that matches it
        sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0:                                     # eta = 0 is the deterministic DDIM
            x = x + sigma * torch.randn(x.shape, generator=gen).to(x.device)
    return x


@torch.no_grad()
def ddim_sample(model, shape, steps, seed, class_ids=None, guidance=0.0, eta=0.0, schedule=None):
    """Make new images with DDIM. Returns a tensor of `shape` with values in [-1, 1].

    steps: how many denoising steps (20, 50 or 100). Same seed = same images.
    class_ids: a tensor of N label numbers (conditional models only).
    guidance: w in the guidance formula. 0 = off, 3 is a typical strong value.
    eta: 0 = deterministic DDIM. schedule: the one the model was trained with
         (default: cosine, T=1000).
    """
    model.eval()
    device = next(model.parameters()).device
    if guidance > 0 and (class_ids is None or model.num_classes == 0):
        raise ValueError("guidance needs a conditional model and class_ids")
    schedule = schedule or make_schedule(device=device)
    if class_ids is not None:
        class_ids = class_ids.to(device)
    gen = torch.Generator().manual_seed(seed)           # CPU generator, like vae.py
    x = torch.randn(shape, generator=gen).to(device)    # start from pure noise
    timesteps = torch.linspace(schedule["T"] - 1, 0, steps).round().long().tolist()
    x = _ddim_loop(model, x, timesteps, schedule, class_ids, guidance, eta, gen)
    return x.clamp(-1, 1)


@torch.no_grad()
def reconstruct(model, x, t_start=200, steps=20, seed=0, schedule=None):
    """Noise the real images x up to timestep t_start, then denoise with DDIM.

    Returns (reconstruction, error_map). The error map is the absolute difference
    per pixel, same shape as x. Images the model knows well come back almost
    unchanged (small error); other images change more (bigger error).
    """
    model.eval()
    schedule = schedule or make_schedule(device=x.device)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=gen).to(x.device)
    t = torch.full((len(x),), t_start, device=x.device, dtype=torch.long)
    x_noisy = q_sample(x, t, noise, schedule)
    timesteps = torch.linspace(t_start, 0, steps).round().long().tolist()
    recon = _ddim_loop(model, x_noisy, timesteps, schedule, None, 0.0, 0.0, gen).clamp(-1, 1)
    return recon, (recon - x).abs()


# ----------------------------------------------------------------------- EMA

class EMA:
    """Keeps a slow-moving average copy of the model weights (decay 0.999, proposed).
    Train the normal model, call update() after every optimiser step, and
    sample from `.model` (the averaged copy), which gives smoother images.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.steps = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        # In the first steps use a smaller decay, so the average is not stuck on the random start.
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for avg, p in zip(self.model.parameters(), model.parameters()):
            avg.mul_(decay).add_(p.detach(), alpha=1 - decay)


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    # Self-check on real CIFAR-10 images (needs the dataset downloaded first).
    from model.data import load_cifar10

    set_seed()
    device = get_device()
    schedule = make_schedule(device=device)
    images = load_cifar10("train")[:2000].to(device)

    # 1. Output shape = input shape, with 3 and 1 channels, with and without class ids.
    for channels, classes in [(3, 0), (1, 0), (3, 10), (1, 10)]:
        net = TinyUNet(channels, classes).to(device)
        x = torch.randn(4, channels, 32, 32, device=device)
        t = torch.randint(0, 1000, (4,), device=device)
        ids = torch.randint(0, classes, (4,), device=device) if classes else None
        assert net(x, t, ids).shape == x.shape, f"bad shape for {channels} channels, {classes} classes"
        if classes:
            assert net(x, t).shape == x.shape, "conditional model must also work without labels"
        n = count_params(net)
        print(f"channels={channels} classes={classes:2d}  parameters: {n:,}  float32 size: {n * 4 / 1e6:.2f} MB")

    # 2. The loss is finite and goes down in 300 steps (measured on the same noise each time).
    model = TinyUNet(3, 0).to(device)

    def fixed_loss():
        torch.manual_seed(0)
        with torch.no_grad():
            return ddpm_loss(model, images[:512], schedule).item()

    loss_before = fixed_loss()
    assert math.isfinite(loss_before), "loss must be a finite number"
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for step in range(300):
        batch = images[torch.randint(0, 2000, (128,), device=device)]
        loss = ddpm_loss(model, batch, schedule)
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss_after = fixed_loss()
    print(f"loss at step 0: {loss_before:.4f}   after 300 steps: {loss_after:.4f}")
    assert loss_after < loss_before, "loss should go down while training"

    # 3. DDIM sampling: shape, range, same seed = same images, other seed = other images.
    for steps in (20, 50, 100):
        out = ddim_sample(model, (4, 3, 32, 32), steps, seed=1)
        assert out.shape == (4, 3, 32, 32) and out.min() >= -1 and out.max() <= 1, f"bad output for {steps} steps"
        print(f"ddim {steps:3d} steps: shape {tuple(out.shape)}  min {out.min():.3f}  max {out.max():.3f}")
    a, b = ddim_sample(model, (4, 3, 32, 32), 20, seed=1), ddim_sample(model, (4, 3, 32, 32), 20, seed=1)
    c = ddim_sample(model, (4, 3, 32, 32), 20, seed=2)
    assert torch.allclose(a, b, atol=1e-4), "same seed must give the same images"
    assert not torch.allclose(a, c, atol=1e-4), "other seed must give other images"
    print(f"same seed max difference: {(a - b).abs().max():.2e}   other seed mean difference: {(a - c).abs().mean():.3f}")

    # 4. Guidance changes the images (conditional model, same seed).
    cmodel = TinyUNet(3, 10).to(device)
    ids = torch.arange(4, device=device)
    g0 = ddim_sample(cmodel, (4, 3, 32, 32), 20, seed=1, class_ids=ids, guidance=0.0)
    g3 = ddim_sample(cmodel, (4, 3, 32, 32), 20, seed=1, class_ids=ids, guidance=3.0)
    assert not torch.allclose(g0, g3, atol=1e-4), "guidance 3 must differ from guidance 0"
    print(f"guidance 0 vs 3: mean difference {(g0 - g3).abs().mean():.3f}")

    # 5. Reconstruction: an image and an error map, both the same shape as the input.
    recon, err = reconstruct(model, images[:4], t_start=200, steps=20)
    assert recon.shape == images[:4].shape and err.shape == images[:4].shape
    assert err.min() >= 0, "error map is an absolute difference"
    print(f"reconstruct: image {tuple(recon.shape)}, error map {tuple(err.shape)}, mean error {err.mean():.3f}")

    # 6. Schedules: both kinds fall from about 1 to about 0.
    for kind in ("cosine", "linear"):
        ab = make_schedule(kind=kind)["alphas_cumprod"]
        print(f"{kind}: alphas_cumprod first {ab[0]:.4f}  last {ab[-1]:.6f}")
        assert ab[0] > 0.99 and ab[-1] < 0.01 and (ab[1:] < ab[:-1]).all()

    # 7. EMA: the averaged copy moves towards the trained model after an update.
    def distance(a, b):
        return sum((p - q).abs().sum().item() for p, q in zip(a.parameters(), b.parameters()))

    ema = EMA(TinyUNet(3, 0).to(device))              # starts from a different (random) model
    before = distance(ema.model, model)
    ema.update(model)
    after = distance(ema.model, model)
    print(f"EMA distance to the model: {before:.1f} -> {after:.1f}")
    assert after < before, "EMA must move towards the model"
    print("self-check passed")

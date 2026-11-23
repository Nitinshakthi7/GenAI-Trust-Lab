# gradcam.py - Grad-CAM for the small CNN baseline (model/cnn.py): "where did the CNN look?"
#
# Grad-CAM (Selvaraju et al., ICCV 2017, arXiv 1610.02391) turns the CNN's answer into a heat map:
#   1. Run the picture through the CNN. Save the "feature maps" of the LAST convolution layer
#      (128 small 8 x 8 pictures: each one lights up where one pattern was found).
#   2. Take the CNN's one output number (the logit; big and positive = "AI"). Walk backwards
#      and save the "gradients": how much does the output change if a feature-map pixel gets bigger?
#   3. Average each gradient picture to ONE number. That is the "weight" of that feature map
#      (a big positive weight = "when this pattern is strong, the CNN says AI more").
#   4. Add up the feature maps, each multiplied by its weight.
#   5. Keep only the positive part (ReLU): we only want places that push towards "AI".
#   6. Blow the 8 x 8 map up to 32 x 32 and scale it to 0..1 (0 = least, 1 = most).
# The two hooks (forward and backward) are how we "save" the feature maps and the gradients.
#
# HONEST MEANING: this explains the CNN CLASSIFIER, NOT the forgery. It shows where the CNN baseline
# looked when it produced its "AI" output. It does not show which part of the picture is fake,
# and it does not prove why a picture is real or AI-made. (EXPLANATION below says this in words.)
#
# Run the quick self-check with:  python -m model.gradcam

import base64
import io
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SIZE = 32                      # the map has the same size as the CNN's input picture
OVERLAY_SIZE = 128             # the overlay picture is shown at 128 x 128
MAX_RED = 0.7                  # where the map is 1, the picture is mixed 70% red + 30% picture

EXPLANATION = ("Grad-CAM explains the CNN classifier: it shows where the CNN baseline looked for its 'AI' "
               "output. It does NOT show the forgery, and it does not prove why an image is real or AI.")

# Hooks change the CNN for a moment, so only one Grad-CAM may run at a time.
_one_at_a_time = threading.Lock()


def last_conv(cnn):
    """The last convolution layer of the CNN."""
    return [layer for layer in cnn.modules() if isinstance(layer, nn.Conv2d)][-1]


def gradcam(cnn, x):
    """x: ONE picture, floats in [-1,1], shape (1,3,32,32). Returns a numpy map, shape (32, 32), values 0..1.
    A high value = the CNN looked there when it gave its 'AI' output."""
    saved = {}
    layer = last_conv(cnn)
    with _one_at_a_time, torch.enable_grad():                  # (gradients are needed here, even if the caller switched them off)
        # Hook 1 (forward): copy the feature maps when the picture goes through the layer.
        forward_hook = layer.register_forward_hook(lambda layer, inputs, output: saved.update(maps=output))
        # Hook 2 (backward): copy the gradients when we walk backwards through the layer.
        backward_hook = layer.register_full_backward_hook(lambda layer, grad_in, grad_out: saved.update(grads=grad_out[0]))
        try:
            cnn.zero_grad()
            logit = cnn(x)[0, 0]                               # the CNN's one output number
            logit.backward()                                   # walk backwards: this fills saved["grads"]
            cnn.zero_grad(set_to_none=True)                    # we do not train, so throw the gradients away
        finally:
            forward_hook.remove()
            backward_hook.remove()
    maps = saved["maps"].detach()                              # (1, 128, 8, 8)
    weights = saved["grads"].mean(dim=(2, 3), keepdim=True)    # one weight per feature map: (1, 128, 1, 1)
    cam = F.relu((weights * maps).sum(dim=1, keepdim=True))    # weighted sum, then ReLU: (1, 1, 8, 8)
    cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    low, high = cam.min().item(), cam.max().item()
    if high - low < 1e-12:
        return np.zeros((SIZE, SIZE), dtype=np.float32)        # the CNN found nothing that pushes towards "AI"
    return ((cam - low) / (high - low)).numpy().astype(np.float32)


def overlay_png_base64(cam, pixels):
    """cam: the 32 x 32 map (0..1). pixels: the picture as numbers 0..255, shape (32, 32, 3).
    Returns a 128 x 128 PNG as plain base64 text (no 'data:' prefix): the picture, tinted red where the map is high."""
    picture = Image.fromarray(pixels).resize((OVERLAY_SIZE, OVERLAY_SIZE), Image.Resampling.NEAREST)
    level = Image.fromarray((cam * 255).astype(np.uint8)).resize((OVERLAY_SIZE, OVERLAY_SIZE), Image.Resampling.BILINEAR)
    mix = (np.array(level, dtype=np.float32) / 255 * MAX_RED)[:, :, None]          # 0 = keep the picture, 0.7 = mostly red
    red = np.array([255, 0, 0], dtype=np.float32)
    blended = np.array(picture, dtype=np.float32) * (1 - mix) + red * mix
    buffer = io.BytesIO()
    Image.fromarray(blended.round().astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    from model.cnn import SmallCNN

    torch.manual_seed(0)
    cnn = SmallCNN().eval()                                    # untrained: enough to prove the code runs
    x = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
    cam = gradcam(cnn, x)
    assert cam.shape == (SIZE, SIZE), "the map should be 32 x 32"
    assert 0 <= cam.min() and cam.max() <= 1, "values should be 0..1"
    assert cam.max() > cam.min(), "the map should not be one flat colour"
    assert all(p.grad is None for p in cnn.parameters()), "gradients should be cleaned up"
    assert not cnn.features[6]._forward_hooks and not cnn.features[6]._backward_hooks, "hooks should be removed"
    pixels = ((x[0].permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()
    overlay = Image.open(io.BytesIO(base64.b64decode(overlay_png_base64(cam, pixels))))
    assert overlay.format == "PNG" and overlay.size == (OVERLAY_SIZE, OVERLAY_SIZE), "overlay should be a 128 x 128 PNG"
    print("map shape:", cam.shape, " min/max:", float(cam.min()), float(cam.max()))
    print(EXPLANATION)
    print("self-check passed")

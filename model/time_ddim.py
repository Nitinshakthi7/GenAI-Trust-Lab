# time_ddim.py - measures how long DDIM sampling takes, on the GPU and on the CPU.
# It makes 16 images with 20, 50 and 100 steps using an UNTRAINED network
# (speed does not depend on training) and saves the numbers to
# results/ddim_timing.json. Each time is the middle one of 3 tries.
#
# Run it from the project folder:  python -m model.time_ddim

import statistics
import time

import torch

from model.ddpm import TinyUNet, ddim_sample
from model.utils import RESULTS_DIR, count_params, save_json, set_seed

IMAGES = 16
STEPS = [20, 50, 100]
TRIES = 3


def seconds_for(model, steps, device):
    """Time one call of ddim_sample, waiting for the GPU to finish."""
    start = time.time()
    ddim_sample(model, (IMAGES, 3, 32, 32), steps, seed=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - start


def measure(device):
    """Median seconds for each step count on one device."""
    set_seed()
    model = TinyUNet(3, 0).to(device)
    ddim_sample(model, (IMAGES, 3, 32, 32), 2, seed=0)      # warm-up, not timed
    return {str(s): round(statistics.median(seconds_for(model, s, device) for _ in range(TRIES)), 3)
            for s in STEPS}


def main():
    result = {
        "images_per_call": IMAGES,
        "tries_per_number": TRIES,
        "parameter_count": count_params(TinyUNet(3, 0)),
        "cpu_threads": torch.get_num_threads(),
        "seconds_by_steps": {},
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["seconds_by_steps"]["gpu"] = measure(torch.device("cuda"))
        print("gpu:", result["seconds_by_steps"]["gpu"], flush=True)
    result["seconds_by_steps"]["cpu"] = measure(torch.device("cpu"))
    print("cpu:", result["seconds_by_steps"]["cpu"], flush=True)
    save_json(RESULTS_DIR / "ddim_timing.json", result)


if __name__ == "__main__":
    main()

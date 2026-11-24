# examples - the "Try an example" pictures for the Detector page

These are 20 tiny pictures (each exactly 32x32 pixels, the size the Detector is built for).
The website shows them as a strip, so a visitor can click one and send it to the Detector.

## What is in here

- `fake/` - 10 fake pictures: 5 from the diffusion model (`fake_ddpm_0.png` ... `fake_ddpm_4.png`)
  and 5 from the VAE (`fake_vae_0.png` ... `fake_vae_4.png`).
  They are our own samples, made by our own small models, so they **may be committed to git**.
  Each file carries a text note inside the PNG (key `provenance`) saying which model made it.
- `real/` - 10 real pictures from the CIFAR-10 dataset (`real_0.png` ... `real_9.png`).
  **These are NOT committed** (git ignores `examples/real/`). The CIFAR-10 licence is "to verify":
  the official page does not state one, so we do not publish the pictures.
- `examples.json` - a list of all 20 pictures: the file name (relative to this folder), `kind`
  (`real` or `fake`), where it came from (`source`) and its `index`. The website uses it to build the strip.
  For real pictures `index` is the picture's number in the CIFAR-10 test set; for fakes it is the
  position in the generator's calibration set.

## Only calibration pictures are used

The real pictures are the first 10 of the "calibration" part of CIFAR-10, and the fakes come from the
"calibration" part of the fakes files. The final test images are never used here.

## Sharing the real pictures

Because `examples/real/` is not in git, Nitin sends that folder to Tejash directly (a file transfer).
Tejash puts it in `web/public/examples/real/` (that folder is also ignored by git).
The `fake/` pictures and `examples.json` come through git as normal.

## Make the files again

From the project folder:

```
python -m model.export_examples
```

It uses fixed picture numbers, so running it again gives the same files. It needs
`data/cifar10/` and the two files `fakes/g-ddpm-cifar_fakes.npz` and `fakes/g-vae-cifar_fakes.npz`.

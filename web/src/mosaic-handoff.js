const desktopMotion = '(min-width: 840px) and (min-height: 620px) and (prefers-reduced-motion: no-preference)';
const imageWidth = 1536;
const imageHeight = 1024;
// These centers are measured in the original artwork, before each scene crops or scales it.
const tile = { x: 725, y: 209, size: 42 };
const lens = { x: 1090, y: 446 };

function blend(from, to, amount) {
  return from + (to - from) * amount;
}

function ease(start, end, value) {
  const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
}

export function prepareMosaicHandoff(pair) {
  if (!pair) return;
  const motion = window.matchMedia(desktopMotion);
  const generator = pair.querySelector('.generation-method');
  const stage = generator.firstElementChild;
  const painting = generator.querySelector('.seed-art');
  const detector = pair.querySelector('.detector-scene');
  const camera = detector.querySelector('.detector-camera');
  const layer = pair.querySelector('.sample-handoff-layer');
  const style = pair.style;

  return function updateMosaicHandoff(enabled) {
    if (!enabled || !motion.matches) {
      pair.removeAttribute('data-aperture');
      pair.removeAttribute('data-sample-in-flight');
      style.removeProperty('--sample-lift');
      style.removeProperty('--detector-arrival');
      return;
    }

    const bounds = generator.getBoundingClientRect();
    const travel = bounds.height - stage.offsetHeight;
    if (travel <= 0) return;
    // Reserve the first half of the pin for the seed, denoising, and finished image.
    const progress = (-bounds.top / travel - .56) / .44;
    const aperture = progress >= 1 ? 'open' : progress >= .58 ? 'opening' : null;
    if (aperture) pair.setAttribute('data-aperture', aperture);
    else pair.removeAttribute('data-aperture');
    style.setProperty('--mosaic-copy', 1 - ease(.42, .69, progress));
    style.setProperty('--detector-copy', ease(.75, 1, progress));

    // Bring the incoming stage to the viewport edge before opening the mask.
    if (aperture === 'opening') {
      style.setProperty('--detector-arrival', -Math.max(0, detector.getBoundingClientRect().top) + 'px');
    } else {
      style.removeProperty('--detector-arrival');
    }

    const active = progress > .31 && progress < 1.27;
    pair.toggleAttribute('data-sample-in-flight', active);
    if (!active) {
      style.removeProperty('--sample-lift');
      return;
    }

    const source = painting.getBoundingClientRect();
    const target = camera.getBoundingClientRect();
    const stageBounds = stage.getBoundingClientRect();
    const viewport = layer.getBoundingClientRect();
    const cover = Math.max(camera.clientWidth / imageWidth, camera.clientHeight / imageHeight);
    // The Detector image uses object-fit: cover and object-position: 68% 54%.
    const targetX = target.left + ((camera.clientWidth - imageWidth * cover) * .68 + lens.x * cover) * target.width / camera.clientWidth;
    const targetY = target.top + ((camera.clientHeight - imageHeight * cover) * .54 + lens.y * cover) * target.height / camera.clientHeight;
    const sourceX = source.left + source.width * tile.x / imageWidth;
    const sourceY = source.top + source.height * tile.y / imageHeight;

    const lift = ease(.31, .68, progress);
    const dock = ease(.62, 1.04, progress);
    const x = blend(blend(sourceX, targetX - viewport.width * .08, lift), targetX, dock);
    const y = blend(blend(sourceY, window.innerHeight * .29, lift), targetY, dock);
    const originalSize = source.width * tile.size / imageWidth;
    const sampleSize = originalSize * (1 + 3.3 * lift - 3.65 * ease(.81, 1.25, progress));
    const opacity = ease(.31, .4, progress) * (1 - ease(.98, 1.27, progress));

    const irisX = x - stageBounds.left;
    const irisY = y - stageBounds.top;
    const farthestCorner = Math.hypot(
      Math.max(irisX, stageBounds.width - irisX),
      Math.max(irisY, stageBounds.height - irisY),
    );

    style.setProperty('--sample-x', (x - viewport.left) + 'px');
    style.setProperty('--sample-y', (y - viewport.top) + 'px');
    style.setProperty('--sample-size', sampleSize + 'px');
    style.setProperty('--sample-angle', (lift * -28 + dock * 28) + 'deg');
    style.setProperty('--sample-opacity', opacity);
    style.setProperty('--sample-lift', ease(.33, .48, progress));
    style.setProperty('--iris-x', irisX + 'px');
    style.setProperty('--iris-y', irisY + 'px');
    style.setProperty('--iris-radius', farthestCorner * 1.15 * ease(.58, .98, progress) + 'px');
  };
}

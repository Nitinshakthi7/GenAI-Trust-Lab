import React, { useEffect, useRef, useState } from 'react';
import { sitePath } from './lab-data.js';
import { subscribeFrame } from './motion.js';
import './method-stories.css';

const desktopMotion = '(min-width: 840px) and (min-height: 620px) and (prefers-reduced-motion: no-preference)';
const coastFiles = ['gallery/coast-day-small.webp', 'gallery/coast-dusk-small.webp'].map(sitePath);

const generation = {
  limits: [.19, .43],
  stops: [.08, .30, .54],
  steps: [
    ['Set a seed', 'A seed sets the starting pattern of random noise. Reusing it with the same settings can reproduce that starting point.'],
    ['Refine the noise', 'A diffusion model gradually removes noise. Broad shapes emerge before finer details.'],
    ['Reveal the result', 'The final passes leave an image you can download, inspect in Detector, or compare with another.'],
  ],
};
const comparison = {
  limits: [.32, .65],
  stops: [.08, .46, .82],
  steps: [
    ['Line up your images', 'The lighthouse, path, and horizon should meet at the same positions. A shifted camera can make unchanged objects look different.'],
    ['Inspect what changed', 'Drag across the image to see how the sky and water darken while the lighthouse lantern gets brighter.'],
    ['Read the difference map', 'Brighter marks show where the two images have larger colour differences. Lighting and edits can both produce changes.'],
  ],
};

function useMethodSteps(config, motionDisabled) {
  const track = useRef(null);
  const stage = useRef(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const chapter = track.current;
    if (window.location.hash !== '#' + chapter.id) return;
    // The browser may resolve the fragment before React has rendered the landing page.
    const frame = window.requestAnimationFrame(() => chapter.scrollIntoView({ behavior: 'instant', block: 'start' }));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (motionDisabled) return;
    const motion = window.matchMedia(desktopMotion);
    let lastScrollStep = null;
    function update() {
      if (!motion.matches) {
        lastScrollStep = null;
        return;
      }
      const { top, bottom, height } = track.current.getBoundingClientRect();
      if (top > window.innerHeight || bottom < 0) return;
      const travel = height - stage.current.offsetHeight;
      if (travel <= 1) return;
      const progress = -top / travel;
      const next = config.limits.filter((limit) => progress >= limit).length;
      if (lastScrollStep !== next) {
        lastScrollStep = next;
        setStep(next);
      }
    }
    update();
    motion.addEventListener('change', update);
    const unsubscribe = subscribeFrame(update);
    return () => {
      motion.removeEventListener('change', update);
      unsubscribe();
    };
  }, [config, motionDisabled]);

  function select(index) {
    if (!motionDisabled && window.matchMedia(desktopMotion).matches) {
      const bounds = track.current.getBoundingClientRect();
      const travel = bounds.height - stage.current.offsetHeight;
      if (travel > 1) {
        window.scrollTo({ top: window.scrollY + bounds.top + travel * config.stops[index], behavior: 'smooth' });
        return;
      }
    }
    setStep(index);
  }

  return { track, stage, step, select };
}

function MethodSteps({ steps, step, select }) {
  return (
    <ol className="method-steps">
      {steps.map(([title, text], index) => (
        <li key={title}>
          <button type="button" aria-pressed={index === step} onClick={() => select(index)}>
            <strong>{title}</strong>
            <span>{text}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

function SeedGrain() {
  const canvas = useRef(null);
  useEffect(() => {
    const context = canvas.current.getContext('2d');
    if (!context) return;
    const width = canvas.current.width;
    const height = canvas.current.height;
    const pixels = context.createImageData(width, height);
    let seed = 42;
    for (let index = 0; index < pixels.data.length; index += 4) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      const value = (seed >>> 24) * .74 + 20;
      pixels.data[index] = value + 31;
      pixels.data[index + 1] = value * .8;
      pixels.data[index + 2] = value * .59;
      pixels.data[index + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
  }, []);
  return <canvas ref={canvas} className="seed-grain" aria-hidden="true" width="144" height="96" />;
}

const seedCaptions = [
  ['Seed 42 · Starting noise', 'A field of randomly patterned amber and dark pixels'],
  ['Denoising · Shapes begin to appear', 'A hazy mosaic landscape emerging from the noise'],
  ['Result · The image comes into view', 'A finished amber mosaic of mountains and a sun'],
];

export function GeneratorMethod({ motionDisabled }) {
  const { track, stage, step, select } = useMethodSteps(generation, motionDisabled);
  const [caption, description] = seedCaptions[step];

  return (
    <section id="generator-method" className="tool-spread method-story generation-method" ref={track} data-step={step} aria-labelledby="generator-method-title">
      <div className="tool-stage method-stage" ref={stage}>
        <div className="shell method-layout">
          <div className="method-copy">
            <header className="method-heading">
              <p className="eyebrow">How the Generator works</p>
              <h3 id="generator-method-title">From seed <em>to scene.</em></h3>
              <p className="generation-lead">Watch an image take shape, one refinement at a time.</p>
            </header>
            <MethodSteps steps={generation.steps} step={step} select={select} />
            <details className="method-detail">
              <summary>Diffusion and VAE</summary>
              <p>Diffusion removes predicted noise over several steps. A VAE takes a small set of sampled numbers and decodes them directly into an image.</p>
            </details>
          </div>
          <figure className="method-figure" data-step={step}>
            <div className="seed-frame-heading"><span>Image formation</span><span>Seed 42</span></div>
            <div className="seed-picture" role="img" aria-label={description}>
              <img className="seed-art" src={sitePath('gallery/seed-mosaic.webp')} alt="" width="1536" height="1024" loading="lazy" decoding="async" />
              <span className="mosaic-origin" />
              <SeedGrain />
            </div>
            <div className="seed-stage-strip" role="group" aria-label="Image formation stages">
              {['Noise', 'Refinement', 'Image'].map((label, index) => (
                <button type="button" key={label} aria-pressed={step === index} onClick={() => select(index)}>
                  <span aria-hidden="true" />{label}
                </button>
              ))}
            </div>
            <figcaption className="method-caption"><strong>{caption}</strong><span>Illustrative diffusion sequence</span></figcaption>
          </figure>
        </div>
      </div>
    </section>
  );
}

function PixelMap() {
  const canvas = useRef(null);
  const [status, setStatus] = useState('loading');

  useEffect(() => {
    const target = canvas.current;
    let cancelled = false;

    async function draw() {
      const images = coastFiles.map((file) => {
        const image = new Image();
        image.src = file;
        return image;
      });
      await Promise.all(images.map((image) => image.decode()));
      if (cancelled) return;
      const context = target.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Canvas is unavailable');
      const width = target.width;
      const height = target.height;
      const [day, dusk] = images.map((image) => {
        context.drawImage(image, 0, 0, width, height);
        return context.getImageData(0, 0, width, height).data;
      });
      const map = context.createImageData(width, height);
      for (let index = 0; index < map.data.length; index += 4) {
        const distance = (
          Math.abs(day[index] - dusk[index]) +
          Math.abs(day[index + 1] - dusk[index + 1]) +
          Math.abs(day[index + 2] - dusk[index + 2])
        ) / 3;
        const strength = Math.min(1, distance / 128);
        map.data[index] = 22 + 233 * strength;
        map.data[index + 1] = 26 + 166 * strength ** 1.8;
        map.data[index + 2] = 52 + 92 * strength ** .5 - 78 * strength ** 3;
        map.data[index + 3] = 255;
      }
      context.putImageData(map, 0, 0);
      setStatus('ready');
    }

    draw().catch(() => {
      if (!cancelled) setStatus('error');
    });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="coast-pixel-map">
      <canvas ref={canvas} width="360" height="305" role="img" aria-label="Colour map of pixel changes between the daylight and dusk illustrations" hidden={status !== 'ready'} />
      {status !== 'ready' && <p role="status">{status === 'loading' ? 'Reading the pixels…' : 'The difference map is unavailable.'}</p>}
    </div>
  );
}

function ComparePreview({ step, offset }) {
  const [split, setSplit] = useState(54);

  return (
    <figure className="method-figure compare-method-figure" data-step={step}
      data-active={offset === 0 || undefined} inert={offset !== 0}
      style={{ '--card-depth': Math.abs(offset), '--card-offset': offset }}>
      {step === 0 && (
        <div className="compare-aligned-pair">
          {['Daylight', 'Dusk'].map((label, index) => (
            <div className="compare-sample" key={label}>
              <img src={coastFiles[index]} alt={index === 0 ? 'Coastal lighthouse in daylight' : 'Coastal lighthouse at dusk'} width="780" height="661" loading="lazy" />
              <span>{label}</span>
            </div>
          ))}
        </div>
      )}
      {step === 1 && (
        <div className="compare-reveal">
          <div className="compare-focus">
            <div className="compare-reveal-image" style={{ '--method-split': split + '%' }} role="img" aria-label="The same lighthouse with daylight to the left of the marker and dusk to the right">
              <img src={coastFiles[1]} alt="" width="780" height="661" />
              <img src={coastFiles[0]} alt="" width="780" height="661" />
              <span className="compare-focus-handle" aria-hidden="true" />
            </div>
            <input className="compare-focus-range" type="range" aria-label="Daylight share" aria-valuetext={split + '% daylight'} min="0" max="100" value={split}
              onChange={(event) => setSplit(Number(event.target.value))} />
          </div>
          <div className="compare-reveal-labels" aria-hidden="true"><span>Daylight</span><span>Dusk</span></div>
        </div>
      )}
      {step === 2 && <PixelMap />}
      <figcaption className="method-caption">
        <strong>{['Same framing, two scenes', 'Drag to reveal the light changes', 'Pixel difference'][step]}</strong>
        {step === 2 ? <span className="compare-scale"><span>Smaller</span><span aria-hidden="true" /><span>Larger</span></span>
          : <span>Illustrative lighthouse pair</span>}
      </figcaption>
    </figure>
  );
}

export function CompareMethod({ motionDisabled }) {
  const { track, stage, step, select } = useMethodSteps(comparison, motionDisabled);
  return (
    <section id="compare-method" className="tool-spread method-story comparison-method" ref={track} aria-labelledby="compare-method-title">
      <div className="tool-stage method-stage" ref={stage}>
        <div className="shell method-layout">
          <div className="method-copy">
            <header className="method-heading">
              <p className="eyebrow">How Compare works</p>
              <h3 id="compare-method-title">Two views.<em> One coast.</em></h3>
            </header>
            <MethodSteps steps={comparison.steps} step={step} select={select} />
            <details className="method-detail">
              <summary>Pixel difference and SSIM</summary>
              <p>This preview shows the average absolute RGB difference at each pixel. The Compare tool uses SSIM to measure structural similarity and works best with nearly aligned images.</p>
            </details>
          </div>
          <div className="compare-deck">
            {comparison.steps.map(([title], index) => <ComparePreview key={title} step={index} offset={index - step} />)}
          </div>
        </div>
      </div>
    </section>
  );
}

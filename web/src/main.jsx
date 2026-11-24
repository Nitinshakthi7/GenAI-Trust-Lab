import React, { useContext, useEffect, useRef, useState, createContext } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/bricolage-grotesque/wght.css';
import '@fontsource/instrument-serif/latin-400-italic.css';
import '@fontsource-variable/instrument-sans';
import './styles.css';
import { subscribeFrame } from './motion.js';
import { prepareMosaicHandoff } from './mosaic-handoff.js';
import { DetectorStory } from './detector-story.jsx';
import { CompareScene } from './compare-scene.jsx';
import { CompareMethod, GeneratorMethod } from './method-stories.jsx';
import { HomeHero } from './home-hero.jsx';
import { currentPage, LabPage, routes, sitePath } from './pages.jsx';
import { auditWarning, honesty } from './lab-data.js';
import './home-transitions.css';

const MotionContext = createContext(false);

const asset = (path) => import.meta.env.BASE_URL + path;
const samples = [
  {
    label: 'Original',
    image: 'input.webp',
    alt: 'A dark mountain and a glacial river in an open valley.',
    caption: 'Start with the full image. Look at the river, the ridgeline, and the texture of the rocks.',
  },
  {
    label: 'Reconstructed',
    image: 'reconstruction.webp',
    alt: 'A softened version of the same mountain and river.',
    caption: 'A simplified reconstruction keeps the large shapes and loses some of the finer detail.',
  },
  {
    label: 'Differences',
    image: 'difference.webp',
    alt: 'A pink and yellow difference map tracing the landscape.',
    caption: 'Brighter areas show where the two versions differ most, especially along edges and detailed textures.',
  },
];

function earlyAccessLink(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:', 'mailto:'].includes(url.protocol) ? url.href : '';
  } catch {
    return '';
  }
}

function Brand() {
  return (
    <a className="brand" href={sitePath()}>
      <span className="brand-mark" aria-hidden="true" />
      GenAI Trust Lab
    </a>
  );
}

function ExampleDialog({ dialog }) {
  const disabled = useContext(MotionContext);
  const [split, setSplit] = useState(50);
  const views = [...samples, { label: 'Compare' }];
  const [selected, setSelected] = useState(0);
  const tabs = useRef([]);
  const figure = useRef(null);
  const sample = samples[selected];

  useEffect(() => {
    const target = figure.current;
    if (disabled || !dialog.current.open || !target.animate || !window.matchMedia('(prefers-reduced-motion: no-preference)').matches) return;
    const transition = target.animate(
      [{ opacity: 0.65, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }],
      { duration: 220, easing: 'ease-out' },
    );
    return () => transition.cancel();
  }, [selected, dialog, disabled]);

  function navigateTabs(event) {
    const offsets = { ArrowLeft: -1, ArrowRight: 1 };
    let next;
    if (event.key in offsets) next = (selected + offsets[event.key] + views.length) % views.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = views.length - 1;
    else return;
    event.preventDefault();
    setSelected(next);
    tabs.current[next].focus();
  }

  return (
    <dialog ref={dialog} className="example" aria-labelledby="example-title" onClose={() => { setSelected(0); setSplit(50); }}>
      <div className="example-header">
        <div>
          <p className="eyebrow">Illustrative sample</p>
          <h2 id="example-title">A closer look</h2>
        </div>
        <button className="close-button" type="button" autoFocus onClick={() => dialog.current.close()} aria-label="Close example">×</button>
      </div>
      <div className="sample-tabs" role="tablist" aria-label="Image view" onKeyDown={navigateTabs}>
        {views.map(({ label }, index) => (
          <button key={label} type="button" role="tab" id={'sample-tab-' + index}
            aria-selected={selected === index} aria-controls="sample-panel"
            tabIndex={selected === index ? 0 : -1}
            ref={(element) => { tabs.current[index] = element; }}
            onClick={() => setSelected(index)}>
            {label}
          </button>
        ))}
      </div>
      <div id="sample-panel" role="tabpanel" aria-labelledby={'sample-tab-' + selected} tabIndex={0}>
        <figure className="sample-figure" ref={figure}>
          {selected === 3 ? <>
            <div className="image-comparison" style={{ '--split': split + '%' }}>
              <img src={asset('illustration/reconstruction.webp')} alt="Reconstructed mountain landscape" width="960" height="640" />
              <img className="comparison-original" src={asset('illustration/input.webp')} alt="Original mountain landscape, revealed on the left" width="960" height="640" />
              <span className="comparison-divider" aria-hidden="true"><span>↔</span></span>
              <span className="comparison-label">Original</span><span className="comparison-label comparison-label-right">Reconstructed</span>
            </div>
            <label className="comparison-control" htmlFor="comparison-split">Compare image detail <output>{split}% original</output></label>
            <input id="comparison-split" className="comparison-range" type="range" min="0" max="100" value={split}
              aria-valuetext={`${split}% original, ${100 - split}% reconstructed`} onChange={(event) => setSplit(Number(event.target.value))} />
            <figcaption>Drag the slider to compare the original with its reconstruction. Look at the river and ridgeline.</figcaption>
          </> : <>
            <img src={asset('illustration/' + sample.image)} alt={sample.alt} width="960" height="640" />
            <figcaption>{sample.caption}</figcaption>
          </>}
        </figure>
      </div>
    </dialog>
  );
}

const auditorSteps = [
  { title: 'Pick a generated image', text: 'The Auditor starts with one image produced by a model.' },
  { title: 'Compare it with training examples', text: 'It measures how different the image is from each example in the set.' },
  { title: 'Bring the closest match into focus', text: 'An unusually close match can be flagged for review. The flag measures similarity.' },
];

function MatchStage({ step }) {
  const [featured, setFeatured] = useState(1);
  const candidates = [
    ['Example A', 'studies/charcoal-ribbon.webp', 'match-closest', 720, 900],
    ['Example B', 'studies/obsidian.webp', 'match-shard', 300, 450],
    ['Example C', 'studies/porcelain.webp', 'match-sphere', 360, 240],
  ];
  const caption = ['Generated image', 'Comparing three examples', 'Generated image and closest match'][step];
  function turn(direction) {
    setFeatured(index => (index + direction + candidates.length) % candidates.length);
  }

  return (
    <figure className="walkthrough-figure match-stage" data-step={step}>
      <div className="match-viewport" role="img" aria-label={[
        'An image of a terracotta ribbon sculpture.',
        'The terracotta sculpture sits beside a similar ceramic form, an obsidian shard, and a porcelain sphere.',
        'The terracotta sculpture sits beside its closest example, a charcoal ceramic form with a similar curled silhouette.',
      ][step]}>
        <div className="match-tile match-query">
          <img src={asset('gallery/ribbon-study.webp')} alt="" width="720" height="900" loading="lazy" />
          <span>Generated image</span>
        </div>
        {candidates.map(([label, image, className, width, height], index) => {
          const offset = (index - featured + candidates.length + 1) % candidates.length - 1;
          return (
            <div key={label} className={'match-tile match-candidate ' + className}
              data-arc-position={offset}>
              <img src={asset(image)} alt="" width={width} height={height} loading="lazy" />
              <span>{step === 2 && index === 0 ? 'Closest match' : label}</span>
            </div>
          );
        })}
      </div>
      <figcaption>
        <strong>{caption}</strong>
        {step === 1 ? (
          <div className="match-arc-controls" role="group" aria-label="Browse training examples">
            <button type="button" onClick={() => turn(-1)} aria-label="Previous example">←</button>
            <span aria-live="polite">{candidates[featured][0]}</span>
            <button type="button" onClick={() => turn(1)} aria-label="Next example">→</button>
          </div>
        ) : <span>Illustrative comparison</span>}
      </figcaption>
    </figure>
  );
}

function Walkthrough({ id, title, steps, Stage, children }) {
  const disabled = useContext(MotionContext);
  const track = useRef(null);
  const layout = useRef(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const motion = window.matchMedia('(min-width: 900px) and (min-height: 720px) and (prefers-reduced-motion: no-preference)');
    let frame = 0;
    let lastScrollStep = -1;

    function update() {
      frame = 0;
      if (disabled || !motion.matches) {
        lastScrollStep = -1;
        return;
      }

      const bounds = track.current.getBoundingClientRect();
      if (bounds.bottom < 0 || bounds.top > window.innerHeight) return;
      const travel = Math.max(1, bounds.height - layout.current.offsetHeight);
      const progress = Math.max(0, Math.min(1, -bounds.top / travel));
      const scrollStep = Math.min(steps.length - 1, Math.floor(progress * steps.length));
      // A selected step stays visible until scrolling reaches a different step.
      if (scrollStep !== lastScrollStep) {
        lastScrollStep = scrollStep;
        setStep(scrollStep);
      }
    }

    function queueUpdate() {
      if (!frame) frame = window.requestAnimationFrame(update);
    }

    update();
    const unsubscribe = subscribeFrame(update);
    motion.addEventListener('change', queueUpdate);
    return () => {
      unsubscribe();
      motion.removeEventListener('change', queueUpdate);
      window.cancelAnimationFrame(frame);
    };
  }, [steps.length, disabled]);

  return (
    <section className="walkthrough" id={id} aria-labelledby={id + '-title'}>
      <div className="walkthrough-track" ref={track}>
        <div className="walkthrough-layout shell" ref={layout}>
          <header className="walkthrough-heading">
            <p className="eyebrow">How it works</p>
            <h4 id={id + '-title'}>{title}</h4>
          </header>
          <Stage step={step} />
          <ol className="walkthrough-steps">
            {steps.map(({ title: stepTitle, text }, index) => (
              <li key={stepTitle}>
                <button type="button" aria-pressed={step === index} onClick={() => setStep(index)}>
                  <strong>{stepTitle}</strong>
                  <span>{text}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      </div>
      <details className="walkthrough-detail shell">
        <summary>Technical detail</summary>
        <p>{children}</p>
      </details>
    </section>
  );
}

function GeneratorScene() {
  return (
    <article className="tool-spread illustrated-tool generator-scene" id="generator" data-scene aria-labelledby="generator-title">
      <div className="tool-stage">
        <div className="tool-art" aria-hidden="true">
          <div className="scene-painting">
            <picture>
              <source media="(max-width: 740px)" srcSet={asset('gallery/seed-mosaic-small.webp')} />
              <img src={asset('gallery/seed-mosaic.webp')} alt="" width="1536" height="1024" loading="lazy" decoding="async" />
            </picture>
            <span className="scene-light" />
          </div>
        </div>
        <div className="shell tool-layout">
          <div className="tool-copy reveal-text" data-reveal>
            <p className="eyebrow">Image Generator</p>
            <h3 id="generator-title"><span className="reveal-line">Start with</span>{' '}<em className="reveal-line">a seed.</em></h3>
            <p>Try a small generative model, then send the result to Detector or Compare to look closer.</p>
            <a className="button" href={sitePath('generator')}>Open Generator</a>
            <a className="text-link walkthrough-link" href="#generator-method">How the Generator works</a>
          </div>
        </div>
      </div>
    </article>
  );
}

function DetectorScene() {
  return (
    <article className="tool-spread detector-scene" id="detector" data-scene aria-labelledby="detector-title">
      <div className="tool-stage">
        <div className="tool-art" aria-hidden="true">
          <div className="detector-camera">
            <img src={asset('gallery/inspection-lens.webp')} alt="" width="1536" height="1024" loading="lazy" />
            <img className="detector-grade" src={asset('gallery/inspection-lens.webp')} alt="" width="1536" height="1024" loading="lazy" />
            <span className="detector-sweep" />
          </div>
          <div className="detector-paper-handoff" />
        </div>
        <div className="shell tool-layout">
          <div className="tool-copy reveal-text" data-reveal>
            <p className="eyebrow">Image Detector</p>
            <h3 id="detector-title"><span className="reveal-line">Question</span>{' '}<em className="reveal-line">an image.</em></h3>
            <p>Explore clues that an image might be AI-made. The Detector is designed to show what it finds in an image you upload.</p>
            <a className="button button-coral" href={sitePath('detector')}>Open Detector</a>
            <a className="text-link walkthrough-link" href="#detector-method">How the Detector works</a>
          </div>
        </div>
      </div>
    </article>
  );
}

function Tools() {
  const motionDisabled = useContext(MotionContext);

  return (
    <section className="tools" id="tools" aria-labelledby="tools-title" tabIndex={-1}>
      <div className="tools-intro" data-scene>
        <h2 id="tools-title" className="shell">The <em>tools.</em></h2>
      </div>
      <GeneratorScene />
      <div className="mosaic-to-detector" style={{ '--mosaic-art': 'url(' + asset('gallery/seed-mosaic.webp') + ')' }}>
        <div className="sample-handoff-layer" aria-hidden="true"><span className="sample-handoff-tile" /></div>
        <GeneratorMethod motionDisabled={motionDisabled} />
        <DetectorScene />
      </div>
      <DetectorStory samples={samples} motionDisabled={motionDisabled} />
      <CompareScene motionDisabled={motionDisabled} />
      <CompareMethod motionDisabled={motionDisabled} />
      <article className="tool-spread auditor-scene" id="auditor" data-scene aria-labelledby="auditor-title">
        <div className="tool-stage">
          <div className="tool-art" aria-hidden="true">
            <div className="auditor-display">
              <div className="auditor-pedestal" />
              <span className="auditor-orbit" />
              <div className="auditor-paper-handoff" />
              <img className="auditor-object auditor-object-cool" src={asset('studies/charcoal-ribbon.webp')} alt="" width="720" height="900" loading="lazy" />
              <img className="auditor-object auditor-object-warm" src={asset('gallery/ribbon-study.webp')} alt="" width="720" height="900" loading="lazy" />
            </div>
          </div>
          <div className="shell tool-layout">
            <div className="tool-copy reveal-text" data-reveal>
              <p className="eyebrow">Memorization Auditor</p>
              <h3 id="auditor-title"><span className="reveal-line">Discover the</span>{' '}<em className="reveal-line">resemblance.</em></h3>
              <p>See how closely generated images can resemble training examples. The Auditor is designed to make those patterns easier to explore across experiments.</p>
              <a className="button button-coral" href={sitePath('auditor')}>Open Auditor</a>
              <a className="text-link walkthrough-link" href="#auditor-method">How the Auditor works</a>
            </div>
          </div>
        </div>
      </article>
      <Walkthrough id="auditor-method" title={<>Find the<br /><em>closest match.</em></>} steps={auditorSteps} Stage={MatchStage}>
        The planned Auditor ranks examples by pixel distance. It uses distances from held-out images to set a threshold for unusually close matches.
      </Walkthrough>
      <div className="shell home-tool-note" data-reveal>
        <p>{honesty} {auditWarning}</p>
        <a className="text-link" href={sitePath('limits')}>Limits and Methods</a>
      </div>
    </section>
  );
}

function Closing({ onExplore }) {
  const signup = earlyAccessLink(import.meta.env.VITE_EARLY_ACCESS_URL);

  return (
    <section className="closing" aria-labelledby="closing-title" data-reveal data-scene>
      <div className="shell closing-panel">
        <div className="closing-art" aria-hidden="true">
          <img src={asset('gallery/eclipse.webp')} alt="" width="1536" height="1024" loading="lazy" />
        </div>
        <div className="closing-inner reveal-text" data-reveal>
          <h2 id="closing-title"><span className="reveal-line">What do</span>{' '}<em className="reveal-line">you see now?</em></h2>
          <p>Compare an original image with its reconstruction and see which details change.</p>
          <div className="actions">
            <button type="button" className="button button-light" onClick={onExplore}>Explore the sample</button>
            {signup && <a className="text-link" href={signup}>Get early access</a>}
          </div>
        </div>
      </div>
    </section>
  );
}

function Footer({ paused, onToggleMotion }) {
  const footer = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      footer.current?.toggleAttribute('data-offscreen', !entry.isIntersecting);
    });
    observer.observe(footer.current);
    return () => observer.disconnect();
  }, []);

  return (
    <footer className="site-footer" id="about" ref={footer} data-reveal>
      <div className="footer-gallery" aria-hidden="true">
        {['gallery/eclipse.webp', 'illustration/difference.webp', 'gallery/ribbon-study.webp', 'illustration/input.webp'].map((image, index) => (
          <div className={'footer-study footer-study-' + index} key={image}>
            <img src={asset(image)} alt="" loading="lazy" width="300" height="360" />
          </div>
        ))}
      </div>
      <div className="footer-inner">
        <div className="footer-identity">
          <a className="footer-wordmark" href={sitePath()} aria-label="GenAI Trust Lab home">
            <span>GenAI</span><em>Trust Lab</em>
          </a>
          <p>A closer look at AI images.</p>
        </div>
        <div className="footer-links">
          <p>Atria University · Generative AI &amp; Diffusion Models</p>
          <nav aria-label="Footer navigation">
            {routes.map(([label, path]) => (
              <a key={path} href={sitePath(path)} id={path === 'limits' ? 'limits' : undefined}>{label}</a>
            ))}
          </nav>
          {onToggleMotion && (
            <button className="motion-control" type="button" aria-pressed={paused} onClick={onToggleMotion}>
              <span aria-hidden="true">{paused ? '▸' : 'Ⅱ'}</span>
              {paused ? 'Resume motion' : 'Pause motion'}
            </button>
          )}
        </div>
      </div>
    </footer>
  );
}

function App() {
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const disabled = paused || reduced;

  useEffect(() => {
    const label = routes.find(([, path]) => path === currentPage)?.[0];
    if (currentPage) document.title = (label || 'Page not found') + ' | GenAI Trust Lab';
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(preference.matches);
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);
  const dialog = useRef(null);
  const page = useRef(null);

  useEffect(() => {
    const motion = window.matchMedia('(prefers-reduced-motion: no-preference)');
    const scenes = [...page.current.querySelectorAll('[data-scene]')];
    const updateMosaicHandoff = prepareMosaicHandoff(page.current.querySelector('.mosaic-to-detector'));
    let frame = 0;

    function updateScenes() {
      frame = 0;
      if (disabled || !motion.matches) {
        for (const scene of scenes) {
          scene.style.removeProperty('--scene-progress');
          scene.style.removeProperty('--outgoing');
        }
        updateMosaicHandoff?.(false);
        return;
      }

      const height = window.innerHeight;
      for (const scene of scenes) {
        const bounds = scene.getBoundingClientRect();
        if (bounds.bottom < 0 || bounds.top > height) continue;
        const distance = Math.max(height * 0.78, bounds.height - height * 0.58);
        const progress = Math.max(0, Math.min(1, (height * 0.42 - bounds.top) / distance));
        scene.style.setProperty('--scene-progress', progress.toFixed(4));
        if (scene.matches('.tool-spread')) {
          // Use the final part of the pinned scene to bring in the next section.
          const travel = Math.max(1, bounds.height - scene.firstElementChild.offsetHeight);
          const exitStart = Number(scene.dataset.exitStart || .62);
          const phase = Math.max(0, Math.min(1, (-bounds.top / travel - exitStart) / (1 - exitStart)));
          scene.style.setProperty('--outgoing', (phase * phase * (3 - 2 * phase)).toFixed(4));
        }
      }
      updateMosaicHandoff?.(true);
    }

    function queueSceneUpdate() {
      if (!frame) frame = window.requestAnimationFrame(updateScenes);
    }

    updateScenes();
    const unsubscribe = subscribeFrame(updateScenes);
    motion.addEventListener('change', queueSceneUpdate);

    const observer = window.IntersectionObserver && new IntersectionObserver((entries) => {
      for (const { target, isIntersecting } of entries) {
        if (!isIntersecting) continue;
        target.classList.add('is-visible');
        observer.unobserve(target);
      }
    });
    if (observer) page.current.querySelectorAll('[data-reveal]').forEach((element) => observer.observe(element));
    return () => {
      observer?.disconnect();
      unsubscribe();
      motion.removeEventListener('change', queueSceneUpdate);
      window.cancelAnimationFrame(frame);
    };
  }, [disabled]);

  useEffect(() => {
    if (disabled) return;
    const fine = window.matchMedia('(hover: hover) and (pointer: fine)');
    const root = page.current;
    let frame = 0;
    let active;
    function move(event) {
      if (!fine.matches) return;
      const target = event.target.closest('.walkthrough-figure');
      if (active && active !== target) reset();
      if (!target) return;
      active = target;
      const { clientX, clientY } = event;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const box = target.getBoundingClientRect();
        const x = (clientX - box.left) / box.width - .5;
        const y = (clientY - box.top) / box.height - .5;
        target.style.setProperty('--tilt-x', `${-y * 3}deg`);
        target.style.setProperty('--tilt-y', `${x * 3}deg`);
      });
    }
    function reset() {
      cancelAnimationFrame(frame);
      ['--tilt-x', '--tilt-y'].forEach((name) => active?.style.removeProperty(name));
      active = null;
    }
    root.addEventListener('pointermove', move);
    root.addEventListener('pointerleave', reset);
    window.addEventListener('scroll', reset, { passive: true });
    return () => {
      reset();
      root.removeEventListener('pointermove', move);
      root.removeEventListener('pointerleave', reset);
      window.removeEventListener('scroll', reset);
    };
  }, [disabled]);

  function openExample() {
    if (!dialog.current.open) dialog.current.showModal();
  }

  return (
    <MotionContext.Provider value={disabled}>
    <div id="top" ref={page} data-motion-disabled={disabled || undefined}>
      <a className="skip-link" href="#main">Skip to content</a>
      {currentPage && (
        <header className="lab-header">
          <div className="shell lab-header-inner">
            <Brand />
            <nav aria-label="Main navigation">
              {routes.map(([label, path]) => <a key={path} href={sitePath(path)} aria-current={path === currentPage ? 'page' : undefined}>{label}</a>)}
            </nav>
          </div>
        </header>
      )}
      <main id="main" tabIndex={-1}>
        {currentPage ? <LabPage page={currentPage} /> : <>
          <HomeHero />
          <Tools />
          <Closing onExplore={openExample} />
        </>}
      </main>
      <Footer paused={paused} onToggleMotion={currentPage ? undefined : () => setPaused((value) => !value)} />
      {!currentPage && <ExampleDialog dialog={dialog} />}
    </div>
    </MotionContext.Provider>
  );
}

createRoot(document.getElementById('root')).render(<App />);

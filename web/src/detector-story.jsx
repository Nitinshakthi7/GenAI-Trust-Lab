import React, { useEffect, useRef, useState } from 'react';
import { sitePath } from './lab-data.js';
import { subscribeFrame } from './motion.js';
import './detector-story.css';

const desktopMotion = '(min-width: 900px) and (min-height: 720px) and (prefers-reduced-motion: no-preference)';
const previewStops = [.08, .5, .92];
const frames = [
  { id: 'original', label: 'Original', title: 'Start with an image', text: 'Look at the river, the ridgeline, and the texture the Detector will try to reconstruct.' },
  { id: 'reconstruction', label: 'Reconstruction', title: 'Rebuild the scene', text: 'The model keeps the broad shapes. Finer details, such as rock texture and ripples, can change.' },
  { id: 'difference', label: 'Difference map', title: 'See where pixels changed', text: 'Brighter areas mark larger reconstruction errors. They are one signal the Detector can use.' },
];

function ease(start, end, value) {
  const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
}

export function DetectorStory({ samples, motionDisabled }) {
  const track = useRef(null);
  const stage = useRef(null);
  const windowRef = useRef(null);
  const rail = useRef(null);
  const geometry = useRef(null);
  const [active, setActive] = useState(null);

  useEffect(() => {
    const section = track.current;
    const stageElement = stage.current;
    const windowElement = windowRef.current;
    const strip = rail.current;
    const cards = [...strip.children];
    const motion = window.matchMedia(desktopMotion);
    let currentFrame = null;

    function update() {
      const measures = geometry.current;
      if (!measures) return;
      const height = window.innerHeight;
      const { top, bottom } = section.getBoundingClientRect();
      if (bottom < 0 || top > height) return;
      const p = Math.max(0, Math.min(1, -top / measures.travel));
      const firstMove = ease(.15, .44, p);
      const secondMove = ease(.56, .85, p);
      const shift = measures.middle * firstMove + (measures.end - measures.middle) * secondMove;
      const sweep = ease(.27, .96, 1 - top / height);
      const frame = p >= .705 ? 2 : p >= .295 ? 1 : 0;

      strip.style.setProperty('--story-offset', -shift + 'px');
      cards.forEach((card, index) => {
        const distance = Math.max(-1, Math.min(1, (measures.centers[index] - shift) / measures.width));
        card.style.setProperty('--card-angle', -distance * 26 + 'deg');
        card.style.setProperty('--card-zoom', 1 - Math.abs(distance) * .22);
      });
      windowElement.style.setProperty('--story-reveal', sweep);
      section.style.setProperty('--story-progress', p);
      if (currentFrame !== frame) {
        currentFrame = frame;
        setActive(frame);
      }
    }

    function measure() {
      if (motionDisabled || !motion.matches) {
        geometry.current = null;
        section.removeAttribute('data-pinned');
        currentFrame = null;
        setActive(null);
        return;
      }

      section.setAttribute('data-pinned', '');
      const width = windowElement.clientWidth;
      const end = Math.max(0, strip.scrollWidth - width);
      const middleCard = cards[1];
      const middle = middleCard.offsetLeft - (width - middleCard.offsetWidth) / 2;
      const travel = Math.round(end * .9 + window.innerHeight * .28);
      const centers = cards.map(card => card.offsetLeft + (card.offsetWidth - width) / 2);
      // The first and last cards settle flush with the frame.
      centers[0] = 0;
      centers[centers.length - 1] = end;
      geometry.current = { end, middle, travel, centers, width: Math.max(1, width) };
      section.style.setProperty('--story-stage-height', stageElement.offsetHeight + 'px');
      section.style.setProperty('--story-travel', travel + 'px');
      update();
    }

    function followHash() {
      const index = frames.findIndex(({ id }) => window.location.hash === '#detector-view-' + id);
      if (index === -1 || !geometry.current) return;
      const top = section.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({ top: top + geometry.current.travel * previewStops[index], behavior: 'instant' });
    }

    measure();
    followHash();
    const observer = new ResizeObserver(measure);
    observer.observe(stageElement);
    window.addEventListener('resize', measure);
    window.addEventListener('hashchange', followHash);
    motion.addEventListener('change', measure);
    const unsubscribe = subscribeFrame(update);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
      window.removeEventListener('hashchange', followHash);
      motion.removeEventListener('change', measure);
      unsubscribe();
      geometry.current = null;
    };
  }, [motionDisabled]);

  function showFrame(event, index) {
    const measures = geometry.current;
    if (!measures || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const top = track.current.getBoundingClientRect().top + window.scrollY;
    window.scrollTo({ top: top + measures.travel * previewStops[index], behavior: 'smooth' });
  }

  return (
    <section className="detector-story" id="detector-method" aria-labelledby="detector-method-title">
      <div className="detector-story-track" ref={track}>
        <div className="detector-story-stage" ref={stage}>
          <header className="shell detector-story-heading">
            <div>
              <p className="eyebrow">How the Detector works</p>
              <h3 id="detector-method-title">One image, <em>three views.</em></h3>
            </div>
            <p>Follow one landscape from its original pixels to a map of what changed.</p>
          </header>

          <nav className="shell detector-story-nav" aria-label="Detector walkthrough">
            <div className="detector-story-links">
              {frames.map(({ id, label }, index) => (
                <a key={id} href={'#detector-view-' + id} aria-current={active === index ? 'step' : undefined}
                  onClick={(event) => showFrame(event, index)}>{label}</a>
              ))}
            </div>
            <span className="detector-story-note">Illustrative example</span>
            <span className="detector-story-progress" aria-hidden="true" />
          </nav>

          <div className="detector-story-window shell" ref={windowRef}>
            <div className="detector-story-rail" ref={rail}>
              {frames.map(({ id, title, text }, index) => (
                <figure key={id} className="detector-story-frame" id={'detector-view-' + id}
                  data-current={active === index || undefined}>
                  <div className="detector-story-photo">
                    <img src={sitePath('illustration/' + samples[index].image)} width="960" height="640" alt={samples[index].alt} loading="lazy" decoding="async" />
                  </div>
                  <figcaption><strong>{title}</strong><span>{text}</span></figcaption>
                </figure>
              ))}
            </div>
          </div>
        </div>
      </div>
      <details className="shell detector-story-detail">
        <summary>Technical detail</summary>
        <p>The Detector is designed for 32×32 CIFAR-10-style images. A reconstruction map shows where a model changes pixels as it rebuilds an image. That difference is one signal the Detector can use.</p>
      </details>
    </section>
  );
}

import React, { useEffect, useRef } from 'react';
import { sitePath } from './lab-data.js';
import { subscribeFrame } from './motion.js';
import './compare-scene.css';

function CoastPlate({ light, alt = '' }) {
  const name = 'gallery/coast-' + light;
  return (
    <picture className="coast-plate">
      <source media="(max-width: 740px)" srcSet={sitePath(name + '-small.webp')} />
      <img src={sitePath(name + '.webp')} alt={alt} width="1536" height="1024" loading="lazy" decoding="async" />
    </picture>
  );
}

function CoastView({ light }) {
  return (
    <div className={'coast-view coast-view-' + light}>
      <div className="coast-world"><CoastPlate light={light} /></div>
      <div className="coast-foreground">
        <img className="coast-reeds" src={sitePath('gallery/coast-reeds.webp')} alt="" width="900" height="450" loading="lazy" decoding="async" />
      </div>
    </div>
  );
}

export function CompareScene({ motionDisabled }) {
  const scene = useRef(null);
  const stage = useRef(null);

  useEffect(() => {
    const article = scene.current;
    const element = stage.current;
    if (motionDisabled) {
      article.style.removeProperty('--coast-reveal');
      article.removeAttribute('data-dusk-palette');
      return;
    }

    function update() {
      const bounds = article.getBoundingClientRect();
      const height = window.innerHeight;
      if (bounds.bottom < 0 || bounds.top > height) return;

      const pinnedTravel = bounds.height - element.offsetHeight;
      // Short viewports reveal dusk during regular scrolling instead of pinning the scene.
      const travel = pinnedTravel > 1 ? pinnedTravel : Math.min(bounds.height, height) * .48;
      const progress = Math.max(0, Math.min(1, (-bounds.top / travel - .10) / .51));
      const reveal = progress * progress * (3 - 2 * progress);
      article.style.setProperty('--coast-reveal', reveal.toFixed(4));
      article.toggleAttribute('data-dusk-palette', reveal >= .68);
    }

    const observer = window.IntersectionObserver && new IntersectionObserver(([entry]) => {
      element.toggleAttribute('data-coast-visible', entry.isIntersecting);
    }, { threshold: .16 });
    observer?.observe(element);
    update();
    const unsubscribe = subscribeFrame(update);
    return () => {
      observer?.disconnect();
      element.removeAttribute('data-coast-visible');
      unsubscribe();
    };
  }, [motionDisabled]);

  return (
    <article id="compare" className="tool-spread compare-scene" ref={scene} data-scene data-exit-start=".82" aria-labelledby="compare-title">
      <div className="tool-stage coast-stage" ref={stage}>
        <div className="coast-visual" role="img" aria-label="Painted lighthouse and coastal headland">
          <CoastView light="day" />
          <CoastView light="dusk" />
          <div className="coast-wash" />
          <div className="coast-exit" />
        </div>
        <div className="shell coast-layout">
          <div className="tool-copy coast-copy reveal-text" data-reveal>
            <p className="eyebrow">Image Compare</p>
            <h3 id="compare-title"><span className="reveal-line">See what</span>{' '}<em className="reveal-line">changed.</em></h3>
            <p>Scroll from daylight to dusk and see how the same coast changes. Compare two images to inspect their colour, brightness, and texture.</p>
            <a className="button" href={sitePath('compare')}>Open Compare</a>
            <a className="text-link walkthrough-link" href="#compare-method">How Compare works</a>
          </div>
        </div>
      </div>
      <div className="coast-still-dusk">
        <CoastPlate light="dusk" alt="The same lighthouse and headland at dusk, with the lantern glowing over darker water" />
      </div>
    </article>
  );
}


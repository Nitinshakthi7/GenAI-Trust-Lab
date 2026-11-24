import React, { useRef, useState } from 'react';
import { sitePath } from './lab-data.js';
import './home-hero.css';

export function HomeHero() {
  const imageRef = useRef(null);
  const [inspection, setInspection] = useState(false);

  function moveLens(event) {
    if (event.pointerType === 'touch') return;
    const image = imageRef.current;
    const bounds = image.getBoundingClientRect();
    image.style.setProperty('--lens-x', ((event.clientX - bounds.left) / bounds.width * 100) + '%');
    image.style.setProperty('--lens-y', ((event.clientY - bounds.top) / bounds.height * 100) + '%');
  }

  return (
    <section className="home-hero" aria-labelledby="hero-title">
      <header className="home-nav">
        <a className="home-logo" href={sitePath()} aria-label="GenAI Trust Lab home"><span aria-hidden="true" />GenAI Trust Lab</a>
        <nav aria-label="Homepage navigation">
          <a href="#tools">Explore the tools</a>
          <a href={sitePath('detector')}>Open Detector</a>
        </nav>
      </header>
      <div className="home-headline">
        <h1 id="hero-title">Looks real.<br /><em>Look again.</em></h1>
        <div className="home-introduction">
          <p>There’s more to an image<br />than meets the eye.</p>
          <p>Make small AI images, try a detector, compare two photos, and see where each tool works and where it fails.</p>
        </div>
      </div>
      <figure className="home-experiment">
        <div className="home-landscape" ref={imageRef} data-inspecting={inspection || undefined}
          onPointerMove={moveLens}>
          <img src={sitePath('illustration/input.webp')} alt="A green mountain ridge above a glacial river, beneath a clouded sky." width="960" height="640" fetchPriority="high" />
          <img className="home-difference" src={sitePath('illustration/difference.webp')} alt="" width="960" height="640" aria-hidden="true" />
          <span className="home-lens" aria-hidden="true"><span>Difference map</span></span>
          <span className="home-image-caption">An image. A closer look.</span>
        </div>
        <figcaption>
          <span>Explore an illustrative reconstruction</span>
          <button type="button" aria-pressed={inspection} onClick={() => setInspection(value => !value)}>
            {inspection ? 'Close inspection' : 'Inspect the image'}<span aria-hidden="true">{inspection ? '−' : '+'}</span>
          </button>
        </figcaption>
      </figure>
      <div className="home-bottom">
        <p>A space to question what you see.</p>
        <a href="#tools">Discover the lab <span aria-hidden="true">↓</span></a>
      </div>
    </section>
  );
}

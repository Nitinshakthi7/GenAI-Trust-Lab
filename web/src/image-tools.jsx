import React, { useCallback, useEffect, useState } from 'react';
import { alignmentWarning, displayValue, pngUrl, sendGeneratedImage, sitePath, stubMode, wordsEnabled, wordsPrivacy } from './lab-data.js';
import { formatNumber, PageIntro, RequestStatus, ResultImage, ResultNote, UploadField, useGeneratedHandoff, useLabRequest } from './lab-ui.jsx';

function GeneratorControls({ model, request }) {
  const [interpolate, setInterpolate] = useState(false);
  const controls = model.controls || [];
  const has = (name) => controls.includes(name);
  const maxSeed = 4294967295;

  function submit(event) {
    event.preventDefault();
    const settings = new FormData(event.currentTarget);
    const read = (name) => settings.has(name) ? Number(settings.get(name)) : null;
    const n = read('n') ?? 1;
    request.run({
      generator_id: model.id,
      seed: read('seed') ?? 42,
      n,
      steps: read('steps'),
      guidance: read('guidance'),
      class_id: read('class_id'),
      interpolate: interpolate ? { seed_b: read('seed_b'), frames: n } : null,
    });
  }

  return (
    <form className="lab-controls" onSubmit={submit}>
      <div className="control-grid">
        {has('seed') && <label>Seed<input name="seed" type="number" min="0" max={maxSeed} defaultValue="42" required /></label>}
        {has('n') && <label>{interpolate ? 'Number of frames' : 'Number of images'}<input name="n" type="number" min={interpolate ? 2 : 1} max="16" defaultValue="4" required /></label>}
        {has('steps') && <div className="select-control"><label htmlFor="generator-steps">DDIM steps</label><select id="generator-steps" name="steps" defaultValue="20">{[20, 50, 100].map((steps) => <option key={steps}>{steps}</option>)}</select></div>}
        {has('class_id') && model.classes?.length > 0 && (
          <div className="select-control"><label htmlFor="generator-class">Clothing class</label><select id="generator-class" name="class_id">{model.classes.map((name, index) => <option key={name} value={index}>{name}</option>)}</select></div>
        )}
        {has('guidance') && <label>Guidance<input name="guidance" type="number" min="0" step="0.1" defaultValue="3" required /></label>}
        {interpolate && <label>Second seed<input name="seed_b" type="number" min="0" max={maxSeed} defaultValue="7" required /></label>}
      </div>
      {has('interpolate') && <label className="check-control"><input type="checkbox" checked={interpolate} onChange={(event) => setInterpolate(event.target.checked)} /> Interpolate between two seeds</label>}
      <button type="submit" className="button lab-button" disabled={model.status !== 'ready' || request.loading}>Generate images</button>
      <RequestStatus request={request} loadingText="Generating images…" />
    </form>
  );
}

function GeneratedImages({ result, onSend }) {
  const images = result.images || [];
  const { model, dataset, label = 'small model, expect low quality' } = result.metadata || {};
  return (
    <div className="lab-result-body">
      <ResultNote result={result} />
      {images.length ? <div className="generated-grid">
        {images.map(({ png_base64: image, seed }, index) => {
          const name = (result.generator_id || 'generated') + '-' + (seed ?? 'image') + '-' + (index + 1) + '.png';
          return <article className="generated-card" key={name}>
            <figure>
              <img src={pngUrl(image)} alt={'Generated sample ' + (index + 1)} />
              <figcaption>{model} on {dataset}, {label}</figcaption>
            </figure>
            <a className="text-link" href={pngUrl(image)} download={name}>Download PNG</a>
            <button className="button lab-button" type="button" onClick={() => onSend('detector', image, name)}>Send to Detector</button>
            <details>
              <summary>Send to Compare</summary>
              <div className="handoff-actions">
                <button type="button" onClick={() => onSend('compare', image, name, 'a')}>Use as Image A</button>
                <button type="button" onClick={() => onSend('compare', image, name, 'b')}>Use as Image B</button>
              </div>
            </details>
          </article>;
        })}
      </div> : <p className="field-hint">The generator returned no images.</p>}
    </div>
  );
}

export function GeneratorPage() {
  const catalogue = useLabRequest('/generators');
  const request = useLabRequest('/generate');
  const [dataset, setDataset] = useState('');
  const [modelId, setModelId] = useState('');
  const [transferError, setTransferError] = useState('');
  const loadCatalogue = catalogue.run;
  useEffect(() => { loadCatalogue(); }, [loadCatalogue]);

  const models = (catalogue.data?.generators || []).filter((model) => model.id !== 'g-ddpm-cifar' || model.status === 'ready');
  const datasets = [...new Set(models.map((model) => model.dataset))];
  const activeDataset = dataset || models.find((model) => model.status === 'ready')?.dataset || datasets[0] || '';
  const available = models.filter((model) => model.dataset === activeDataset);
  const model = available.find((item) => item.id === modelId) || available.find((item) => item.status === 'ready') || available[0];

  function changeDataset(event) {
    request.reset();
    setTransferError('');
    setDataset(event.target.value);
    setModelId('');
  }

  function changeModel(event) {
    request.reset();
    setTransferError('');
    setModelId(event.target.value);
  }

  function send(destination, image, name, slot) {
    try {
      sendGeneratedImage(destination, image, name, slot);
    } catch (error) {
      setTransferError(error.message);
    }
  }

  return (
    <div className="lab-page generator-page">
      <PageIntro label="Image Generator" title="From seed" accent="to image." art="porcelain.webp" position="78% center">
        See how a small VAE or diffusion model turns a seed into a new image.
      </PageIntro>
      <section id="workspace" className="shell lab-section generator-workbench" aria-label="Image generation">
        <div className="lab-panel">
          <h2>Choose a model</h2>
          <RequestStatus request={catalogue} loadingText="Loading the model catalogue…" />
          {catalogue.error && <button type="button" className="button lab-button" onClick={() => loadCatalogue()}>Retry catalogue</button>}
          {catalogue.data && <ResultNote result={catalogue.data} demoLabel="Demo catalogue" />}
          {model ? <>
            <div className="lab-controls control-grid">
              <div className="select-control"><label htmlFor="generator-dataset">Dataset</label><select id="generator-dataset" value={activeDataset} onChange={changeDataset}>{datasets.map((name) => <option key={name}>{name}</option>)}</select></div>
              <div className="select-control"><label htmlFor="generator-model">Model</label><select id="generator-model" value={model.id} onChange={changeModel}>{available.map((item) => <option key={item.id} value={item.id}>{item.name}{item.status === 'ready' ? '' : ' · ' + (item.status || 'Unavailable')}</option>)}</select></div>
            </div>
            {model.status !== 'ready' && <p role="status" className="field-hint">This model is currently unavailable.</p>}
            <GeneratorControls key={model.id} model={model} request={request} />
          </> : catalogue.data && <p className="field-hint">No models are listed yet.</p>}
        </div>
        <section className="lab-panel lab-results" aria-live="polite" aria-busy={Boolean(request.loading)}>
          <h2>Image gallery</h2>
          {transferError && <p className="request-error" role="alert">{transferError}</p>}
          {request.data ? <GeneratedImages result={request.data} onSend={send} /> : <p className="field-hint">Your images will appear here after generation.</p>}
        </section>
      </section>
      <div className="shell lab-callout"><a className="text-link" href={sitePath('#generator-method')}>How the Generator works</a></div>
    </div>
  );
}

function CompareResults({ result }) {
  const features = result.feature_table || [];
  const differences = result.differences || [];
  return (
    <div className="lab-result-body">
      <ResultNote result={result} />
      <dl className="lab-scores">
        <div><dt>Structural similarity (SSIM)</dt><dd>{formatNumber(result.similarity?.ssim)}</dd></div>
        {result.similarity?.lpips != null && <div><dt>Perceptual distance (LPIPS)</dt><dd>{formatNumber(result.similarity.lpips)}</dd></div>}
      </dl>
      <ResultImage image={result.diff_heatmap_png_base64} label="SSIM difference map" />
      {result.regions_png_base64 && <ResultImage image={result.regions_png_base64} label="Matched regions" />}
      <h3>Image features</h3>
      {features.length ? (
        <div className="table-scroll" tabIndex="0" role="region" aria-label="Image features table">
          <table className="results-table">
            <thead><tr><th scope="col">Feature</th><th scope="col">Image A</th><th scope="col">Image B</th><th scope="col">What this doesn’t prove</th></tr></thead>
            <tbody>{features.map((row, index) => <tr key={index}><th scope="row">{row.feature}</th><td>{displayValue(row.image_a)}</td><td>{displayValue(row.image_b)}</td><td>{displayValue(row.does_not_prove)}</td></tr>)}</tbody>
          </table>
        </div>
      ) : <p className="field-hint">Image features are not available.</p>}
      <section className="words-results" aria-label="Words results">
        <h3>Words</h3>
        {result.source === 'vlm' ? <>
          {result.summary && <p>{result.summary}</p>}
          {differences.length > 0 && <div className="table-scroll" tabIndex="0" role="region" aria-label="Differences described in words">
            <table className="results-table">
              <thead><tr><th scope="col">Attribute or part</th><th scope="col">Image A</th><th scope="col">Image B</th></tr></thead>
              <tbody>{differences.map((row, index) => <tr key={index}><th scope="row">{row.part}</th><td>{displayValue(row.image_a)}</td><td>{displayValue(row.image_b)}</td></tr>)}</tbody>
            </table>
          </div>}
          {!result.summary && !differences.length && <p className="field-hint">No differences were described.</p>}
        </> : <p role="status">Words are off.</p>}
      </section>
    </div>
  );
}

export function ComparePage() {
  const request = useLabRequest('/compare');
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [focus, setFocus] = useState('');
  const [words, setWords] = useState(false);
  const placeGenerated = useCallback((image) => {
    if (image.slot === 'b') setImageB(image);
    else setImageA(image);
  }, []);
  const handoffError = useGeneratedHandoff('compare', placeGenerated);

  function submit(event) {
    event.preventDefault();
    if (!imageA || !imageB) return;
    const useWords = wordsEnabled && words;
    request.run({ image_a: imageA, image_b: imageB, use_words: useWords, focus: useWords ? focus.trim() || null : null });
  }

  return (
    <div className="lab-page compare-page">
      <PageIntro label="Image Compare · Distinguisher" title="See what" accent="changed." art="obsidian.webp" position="65% center">
        Put two images side by side and compare their colours, textures, and structure.
      </PageIntro>
      <section id="workspace" className="shell lab-section" aria-label="Compare two images">
        <div className="lab-panel">
          <h2>Add two photos</h2>
          {handoffError && <p className="request-error" role="alert">{handoffError}</p>}
          <form className="lab-upload" onSubmit={submit}>
            <div className="control-grid">
              <UploadField id="image-a" label="Image A" image={imageA} onImage={(image) => { request.reset(); setImageA(image); }} />
              <UploadField id="image-b" label="Image B" image={imageB} onImage={(image) => { request.reset(); setImageB(image); }} />
            </div>
            <label htmlFor="compare-focus">What should I compare?</label>
            <input id="compare-focus" type="text" maxLength={100} value={focus} placeholder="For example, fur colour and eyes" onChange={(event) => { request.reset(); setFocus(event.target.value); }} aria-describedby="focus-hint" />
            <p className="field-hint" id="focus-hint">Optional. Words uses this focus when enabled.</p>
            <label className="check-control"><input type="checkbox" checked={words} disabled={!wordsEnabled} onChange={(event) => { request.reset(); setWords(event.target.checked); }} aria-describedby="words-privacy" /> {stubMode ? 'Preview Words' : 'Use Words'}</label>
            <p id="words-privacy" className="field-hint">{wordsPrivacy} {stubMode ? 'The demo uses a fixed example without sending your photos.' : !wordsEnabled ? 'Words are off for this demo.' : ''}</p>
            <button type="submit" className="button lab-button" disabled={!imageA || !imageB || request.loading}>Compare images</button>
            <RequestStatus request={request} loadingText="Comparing images…" />
          </form>
        </div>
        <section className="lab-panel lab-results compare-results" aria-live="polite" aria-busy={Boolean(request.loading)}>
          <h2>Comparison</h2>
          <p className="field-hint">{alignmentWarning}</p>
          {request.data ? <CompareResults result={request.data} /> : <p className="field-hint">Add both images to see their comparison.</p>}
        </section>
      </section>
      <div className="shell lab-callout"><a className="text-link" href={sitePath('#compare-method')}>How Compare works</a></div>
    </div>
  );
}

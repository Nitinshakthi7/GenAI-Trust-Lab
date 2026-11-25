import React, { useCallback, useEffect, useRef, useState } from 'react';
import './pages.css';
import { auditWarning, displayValue, honesty, isDemo, placeholder, readImage, sitePath, sizeLimitation, sizeWarning, stubMode, unseenWarning, unavailable } from './lab-data.js';
import { formatNumber, PageIntro, RequestStatus, ResultImage, ResultNote, UploadField, useGeneratedHandoff, useLabRequest } from './lab-ui.jsx';
import { ComparePage, GeneratorPage } from './image-tools.jsx';
import { LimitsPage } from './limits.jsx';

export const routes = [
  ['Home', ''],
  ['Generator', 'generator'],
  ['Detector', 'detector'],
  ['Compare', 'compare'],
  ['Auditor', 'auditor'],
  ['Limits and Methods', 'limits'],
];

export { sitePath };
export const currentPage = window.location.pathname.slice(import.meta.env.BASE_URL.length).replace(/\/+$/, '');

const demoExamples = [
  { file: 'demo/ridge.png', label: 'Ridge · Demo', demo: true },
  { file: 'demo/valley.png', label: 'Valley · Demo', demo: true },
  { file: 'demo/dusk.png', label: 'Dusk · Demo', demo: true },
  { file: 'demo/larger-frame.png', label: 'Larger frame · Demo', demo: true },
];

function useProjectExamples() {
  const [examples, setExamples] = useState([]);
  useEffect(() => {
    const controller = new AbortController();
    fetch(sitePath('examples/index.json'), { signal: controller.signal })
      .then((response) => response.ok ? response.json() : [])
      .then((data) => {
        if (controller.signal.aborted || !Array.isArray(data)) return;
        setExamples(data.filter(({ file, label }) => /^(fake|real)\/[\w.-]+\.(png|jpe?g)$/i.test(file) && typeof label === 'string'));
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);
  return examples;
}

function DetectorExamples({ image, onSelect, onError, disabled, loading }) {
  const projectExamples = useProjectExamples();
  const [missing, setMissing] = useState([]);
  const project = projectExamples.filter(({ file }) => !missing.includes(file));
  const examples = project.length ? project : demoExamples;
  return (
    <div className="detector-examples">
      <h3>Try an example</h3>
      {!project.length && <p className="field-hint">{stubMode ? 'Illustrations for testing the interface, including the size warning.' : 'Project examples are unavailable. These illustrations can be used to exercise the upload.'}</p>}
      <ul className="example-strip">
        {examples.map((example) => (
          <li key={example.file}>
            <button type="button" disabled={disabled} aria-pressed={image?.example === example.file} onClick={() => onSelect(example)}>
              <img src={sitePath('examples/' + example.file)} alt="" width="48" height="48" loading="lazy" onError={() => {
                if (!example.demo) setMissing((files) => [...files, example.file]);
                else onError('A demo example image could not be loaded.');
              }} />
              <span>{example.label}</span>
            </button>
          </li>
        ))}
      </ul>
      {loading && <p role="status" className="field-hint">Opening the example…</p>}
    </div>
  );
}

const measured = (value) => value == null ? placeholder : formatNumber(value);

function FullCheck({ check, labels }) {
  const features = [...(check.features || [])].sort((a, b) => (b.push == null ? -1 : Math.abs(b.push)) - (a.push == null ? -1 : Math.abs(a.push)));
  const crops = check.crops || [];
  return (
    <section className="evidence-item full-check" aria-labelledby="full-check-heading">
      <h3 id="full-check-heading">Full-size image check</h3>
      <dl className="lab-scores">
        <div><dt>Verdict</dt><dd>{labels[check.label] || unavailable}</dd></div>
        <div><dt>Combined score</dt><dd>{check.probability_ai == null ? placeholder : check.probability_ai.toFixed(2)}</dd></div>
      </dl>
      <p>{check.decided_by === 'disagreement'
        ? 'Unclear because our CLIP model and Smaug disagree: one says AI-made, the other says real. When the two strongest detectors disagree, we do not give a verdict.'
        : check.decided_by === 'combiner'
          ? 'Decided by our combiner, a logistic regression that learned how much to trust each detector.' + (check.agree ? ' All detectors agree.' : ' Our measured-feature model disagrees, and it is the weakest detector.')
          : 'Our combiner is not available, so this is the plain average of the scores, cut at 0.5.'}</p>
      {check.band && <p className="field-hint">Score 0 to 1, higher means more AI-like. It is a score, not a calibrated probability. {check.band.low === check.band.high
        ? `AI-made if at least ${check.band.high.toFixed(2)}, otherwise Real. On the training images real and AI scores did not overlap, so the rule left no Unclear band.`
        : `AI-made if at least ${check.band.high.toFixed(2)}, Real if at most ${check.band.low.toFixed(2)}, otherwise Unclear.`} The cut-offs were chosen on training images, never on the test images.</p>}
      {check.vote_tested && <p className="field-hint">Tested: {check.vote_tested}.</p>}
      <h4>The three detectors</h4>
      <div className="table-scroll" tabIndex="0" role="region" aria-label="Full-size detectors table">
        <table className="results-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Ours?</th><th scope="col">AI score</th><th scope="col">Counted in the final answer</th><th scope="col">Tested on our images</th></tr></thead>
          <tbody>{(check.models || []).map((model) => <tr key={model.id}>
            <th scope="row">{model.name}</th><td>{model.ours ? 'Ours' : 'Not ours'}</td><td>{model.probability_ai == null ? placeholder : model.probability_ai.toFixed(2)}</td>
            <td>{model.counted == null ? placeholder : model.counted > 0 ? `towards AI (+${model.counted.toFixed(2)})` : `towards real (${model.counted.toFixed(2)})`}</td><td>{model.tested ?? placeholder}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <h4>Which features pushed the decision</h4>
      <div className="table-scroll" tabIndex="0" role="region" aria-label="Feature push table">
        <table className="results-table">
          <thead><tr><th scope="col">Feature</th><th scope="col">This image</th><th scope="col">Typical real photo</th><th scope="col">Typical AI image</th><th scope="col">Push</th><th scope="col">What it means</th></tr></thead>
          <tbody>{features.map((feature) => <tr key={feature.name}>
            <th scope="row">{feature.name}</th><td>{measured(feature.value)}</td><td>{measured(feature.typical_real)}</td><td>{measured(feature.typical_ai)}</td>
            <td>{feature.push == null ? placeholder : feature.push > 0 ? `towards AI (+${feature.push.toFixed(2)})` : `towards real (${feature.push.toFixed(2)})`}</td>
            <td>{feature.meaning}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p className="field-hint">Push comes from our measured-feature model (logistic regression). On images from AI tools it never saw it was close to chance, so read these pushes as a description of the image, not proof. Our CLIP model and Smaug decide from learned features, which cannot be listed.</p>
      {crops.length > 0 && <>
        <h4>Where Smaug looked</h4>
        <ul>{crops.map((crop) => <li key={crop.part}>{crop.part[0].toUpperCase() + crop.part.slice(1)} part: score {crop.probability_ai.toFixed(2)}</li>)}</ul>
        <p className="field-hint">Smaug scores three square crops of the image and averages them.</p>
      </>}
      <h4>File clues</h4>
      <dl className="evidence-fields">
        {(check.file_clues || []).map((clue) => <div key={clue.clue}><dt>{clue.clue}</dt><dd>{clue.found ? 'Found' + (clue.detail ? ': ' + clue.detail : '') : 'Not found'}</dd></div>)}
      </dl>
      <p className="field-hint">File clues are evidence, not proof: apps and screenshots can remove them, and they can be faked.</p>
      {check.note && <p className="field-hint">{check.note}</p>}
    </section>
  );
}

function ScoreResults({ result }) {
  const labels = { real: 'Real', ai: 'AI-made', unclear: 'Unclear' };
  const source = { vae: 'VAE error map', ddpm: 'DDPM error map' }[result.heatmap_source];
  return (
    <div className="lab-result-body">
      {result.full_check && <FullCheck check={result.full_check} labels={labels} />}
      <ResultNote result={result} demoLabel="Demo response · example scores" />
      {result.out_of_scope && <div className="scope-warning" role="status">
        <strong>Out of scope</strong>
        <p>{result.warning || sizeWarning}</p>
      </div>}
      <dl className="lab-scores">
        <div><dt>Classification</dt><dd>{labels[result.label] || unavailable}</dd></div>
        <div><dt>AI-made score</dt><dd>{formatNumber(result.score)}</dd></div>
        <div><dt>VAE score</dt><dd>{formatNumber(result.vae_score)}</dd></div>
        <div><dt>CNN baseline score</dt><dd>{formatNumber(result.baseline_score)}</dd></div>
      </dl>
      <p className="field-hint">VAE score: how badly the VAE rebuilds the image. In our tests it was lower for VAE-made fakes than for real photos, so a higher number does not mean 'more AI'.</p>
      <p className="field-hint">Input size {displayValue(result.input_size)} · Heatmap source {source || unavailable}</p>
      <ResultImage image={result.heatmap_png_base64} label="Reconstruction heatmap" />
      <ResultImage image={result.gradcam_png_base64} label="Grad-CAM" />
      {result.gradcam_png_base64 && <p className="field-hint">Grad-CAM explains the CNN classifier’s response. It cannot establish forgery.</p>}
    </div>
  );
}

function EvidenceResults({ result }) {
  const exif = result.exif;
  const c2pa = result.c2pa;
  return (
    <div className="lab-result-body">
      <ResultNote result={result} />
      <section className="evidence-item">
        <h3>EXIF metadata</h3>
        <p>{exif?.found == null ? unavailable : exif.found ? 'Metadata found' : 'No EXIF metadata found'}</p>
        {exif && <dl className="evidence-fields">
          {Object.keys(exif.fields || {}).length ? Object.entries(exif.fields).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(displayValue(value))}</dd></div>) : <div><dt>Fields</dt><dd>{unavailable}</dd></div>}
        </dl>}
        <p className="field-hint">{exif?.does_not_prove || 'Missing EXIF is weak evidence. Present EXIF does not mean real.'}</p>
      </section>
      <section className="evidence-item">
        <h3>C2PA provenance</h3>
        <p>Status {displayValue(c2pa?.status)}</p>
        <p>Summary {displayValue(c2pa?.summary)}</p>
        <p className="field-hint">{c2pa?.does_not_prove || 'A missing manifest means nothing.'}</p>
      </section>
      <section className="evidence-item">
        <h3>Image signals</h3>
        <ResultImage image={result.ela_png_base64} label="Error level analysis (ELA)" />
        <p className="field-hint">ELA is a teaching aid for lossy images. It cannot prove manipulation or AI generation.</p>
        <ResultImage image={result.fft_png_base64} label="Frequency spectrum (FFT)" />
        <p className="field-hint">A frequency pattern alone cannot establish whether an image is AI-made.</p>
      </section>
    </div>
  );
}

function DetectorPage() {
  const request = useLabRequest('/detect');
  const evidence = useLabRequest('/evidence');
  const [image, setImage] = useState(null);
  const [view, setView] = useState('analysis');
  const [exampleError, setExampleError] = useState('');
  const [loadingExample, setLoadingExample] = useState(false);
  const evidenceRequested = useRef(false);
  const exampleRequest = useRef(null);
  const handoffError = useGeneratedHandoff('detector', setImage);

  useEffect(() => () => exampleRequest.current?.abort(), []);

  function changeImage(selected) {
    exampleRequest.current?.abort();
    exampleRequest.current = null;
    setLoadingExample(false);
    setExampleError('');
    request.reset();
    evidence.reset();
    evidenceRequested.current = false;
    setView('analysis');
    setImage(selected);
  }

  async function selectExample(example) {
    changeImage(null);
    const controller = new AbortController();
    exampleRequest.current = controller;
    setLoadingExample(true);
    try {
      const response = await fetch(sitePath('examples/' + example.file), { signal: controller.signal });
      if (!response.ok) throw new Error();
      const file = new File([await response.blob()], example.file.split('/').at(-1));
      const selected = await readImage(file);
      if (!controller.signal.aborted) setImage({ ...selected, example: example.file, demo: example.demo });
    } catch {
      if (!controller.signal.aborted) setExampleError('This example could not be opened. Try another example or upload an image.');
    } finally {
      if (exampleRequest.current === controller) {
        exampleRequest.current = null;
        setLoadingExample(false);
      }
    }
  }

  function showEvidence() {
    setView('evidence');
    if (image && !evidenceRequested.current) {
      evidenceRequested.current = true;
      evidence.run({ image });
    }
  }

  return (
    <div className="lab-page detector-page">
      <PageIntro label="Image Detector" title="Question" accent="an image." art="prism.webp">
        Explore signals from a detector designed for 32×32 CIFAR-10-style images.
      </PageIntro>
      <section id="workspace" className="shell lab-section" aria-label="Image analysis">
        <div className="lab-panel">
          <h2>Analyze an image</h2>
          <DetectorExamples image={image} onSelect={selectExample} onError={setExampleError} disabled={request.loading} loading={loadingExample} />
          {(exampleError || handoffError) && <p role="alert" className="request-error">{exampleError || handoffError}</p>}
          <form className="lab-upload" onSubmit={(event) => {
            event.preventDefault();
            setView('analysis');
            if (image) request.run({ image });
          }}>
            <UploadField id="detector-upload" image={image} onImage={changeImage} />
            <button type="submit" className="button lab-button" disabled={!image || request.loading}>Analyze image</button>
          </form>
        </div>
        <section className="lab-panel lab-results detector-results" aria-label="Detector results">
          <div className="view-switch" role="group" aria-label="Detector result view">
            <button type="button" aria-pressed={view === 'analysis'} onClick={() => setView('analysis')}>Analysis</button>
            <button type="button" aria-pressed={view === 'evidence'} onClick={showEvidence}>Evidence</button>
          </div>
          {view === 'analysis' ? <div aria-live="polite" aria-busy={Boolean(request.loading)}>
            <h2>Analysis</h2>
            <RequestStatus request={request} loadingText="Analyzing the image…" />
            {request.data ? <ScoreResults result={request.data} /> : !request.loading && !request.error && <p className="field-hint">Your scores and heatmap will appear here after analysis.</p>}
          </div> : <div aria-live="polite" aria-busy={Boolean(evidence.loading)}>
            <h2>Evidence</h2>
            <RequestStatus request={evidence} loadingText="Checking available evidence…" />
            {evidence.data && <EvidenceResults result={evidence.data} />}
            {!image && <p className="field-hint">Choose an image to check what evidence is available.</p>}
            {evidence.error && <button type="button" className="button lab-button" onClick={() => evidence.run({ image })}>Retry evidence</button>}
          </div>}
        </section>
      </section>
      <div className="shell lab-callout">
        <p>{sizeLimitation}</p>
        <p>{unseenWarning}</p>
        <a className="text-link" href={sitePath('#detector-method')}>See how reconstruction works</a>
      </div>
    </div>
  );
}

const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 });

function validRate(rate) {
  return Number.isFinite(rate) && rate >= 0 && rate <= 1;
}

function ExperimentChart({ experiment, demo }) {
  const points = (experiment.points || []).filter((point) => Number.isFinite(point.x));
  const script = experiment.script || experiment.script_name;
  return (
    <article className="lab-panel experiment">
      <h3>{experiment.name}</h3>
      <p className="field-hint">Share of generated images flagged for review</p>
      {points.length ? <table className="results-table">
        <thead><tr><th scope="col">{experiment.x_label || 'Condition'}</th><th scope="col">Flagged share</th></tr></thead>
        <tbody>{points.map(({ x, memorization_rate: rate }, index) => (
          <tr key={index}>
            <th scope="row">{formatNumber(x)}</th>
            <td>{!demo && validRate(rate) ? <span className="run-rate">
              <span className="bar-track" aria-hidden="true"><span style={{ width: rate * 100 + '%' }} /></span>{percent.format(rate)}
            </span> : placeholder}</td>
          </tr>
        ))}</tbody>
      </table> : <p className="field-hint">{placeholder}</p>}
      {script && <p className="field-hint">Source script <code>{script}</code></p>}
    </article>
  );
}

function AuditExamples({ examples, demo }) {
  return (
    <div className="audit-pairs">
      {examples.map((example, index) => <article className="lab-panel" key={index}>
        {demo && <p className="result-mode">Illustrative pair</p>}
        <div className="control-grid">
          <ResultImage className="audit-image" image={example.generated_png_base64} label="Generated image" />
          <ResultImage className="audit-image" image={example.nearest_png_base64} label="Nearest training image" />
        </div>
        <p className="field-hint">Pixel distance {demo || !Number.isFinite(example.distance) ? placeholder : formatNumber(example.distance)}</p>
        {!demo && example.phash_distance != null && <p className="field-hint">pHash distance {formatNumber(example.phash_distance)}</p>}
        {example.diff_heatmap_png_base64 && <ResultImage image={example.diff_heatmap_png_base64} label="Difference map" />}
      </article>)}
    </div>
  );
}

function MatchResults({ result }) {
  const demo = isDemo(result);
  const nearest = result.nearest || [];
  return (
    <div className="lab-result-body">
      <ResultNote result={result} demoLabel="Demo comparison · illustrative matches" />
      {!demo && result.flag && <p className="result-mode">{result.flag === 'possible_copy' ? 'Possible copy' : 'Not flagged'}</p>}
      {nearest.length ? <ol className="lab-matches">
        {nearest.map((match) => <li key={match.rank}>
          <ResultImage className="audit-image" image={match.image_png_base64} label={'Match ' + match.rank} />
          <span>Distance {demo || match.distance == null ? placeholder : formatNumber(match.distance)}</span>
          {!demo && match.phash_distance != null && <span>pHash distance {formatNumber(match.phash_distance)}</span>}
        </li>)}
      </ol> : <p className="field-hint">No matches were returned.</p>}
    </div>
  );
}

function AuditorPage() {
  const results = useLabRequest('/audit/results');
  const match = useLabRequest('/audit/check');
  const [image, setImage] = useState(null);
  const [run, setRun] = useState('a-500');
  const loadResults = results.run;
  useEffect(() => { loadResults(); }, [loadResults]);

  const experiments = results.data?.experiments || [];
  const examples = results.data?.examples || [];
  const demo = stubMode || isDemo(results.data);

  return (
    <div className="lab-page auditor-page">
      <PageIntro label="Memorization Auditor" title="Discover the" accent="resemblance." art="echoes.webp">
        Explore how closely generated images resemble examples in a training set.
      </PageIntro>
      <section className="shell lab-section dashboard" aria-labelledby="dashboard-title">
        <div className="lab-section-head">
          <div><h2 id="dashboard-title">Experiment results</h2></div>
          {results.error && <button type="button" className="button lab-button" onClick={() => loadResults()}>Retry results</button>}
        </div>
        <RequestStatus request={results} loadingText="Loading experiments…" />
        {results.data && <ResultNote result={results.data} demoLabel="Demo dashboard" />}
        <div className="experiment-grid">
          {experiments.length ? experiments.map((experiment, index) => <ExperimentChart key={index} experiment={experiment} demo={demo} />) : <p className="lab-panel empty-result">{placeholder}</p>}
        </div>
        <div className="lab-section-head"><h2>Nearest training images</h2></div>
        {examples.length ? <AuditExamples examples={examples} demo={demo} /> : <p className="lab-panel empty-result">{placeholder}</p>}
      </section>
      {match.status !== 404 ? (
        <section className="shell lab-section lab-workbench" aria-label="Optional image similarity check">
          <div className="lab-panel">
            <h2>Check similarity</h2>
            <form className="lab-upload" onSubmit={(event) => {
              event.preventDefault();
              if (image) match.run({ image, run });
            }}>
              <UploadField id="auditor-upload" image={image} onImage={(selected) => { setImage(selected); match.reset(); }} />
              <div className="lab-controls">
                <div className="select-control"><label htmlFor="auditor-run">Compare with the training images of</label><select id="auditor-run" value={run} onChange={(event) => { setRun(event.target.value); match.reset(); }}>
                  <option value="a-500">500 images (run a-500)</option>
                  <option value="a-10000">10,000 images (run a-10000)</option>
                  <option value="a-500-dup">500 rows, 30% repeats (run a-500-dup)</option>
                </select></div>
              </div>
              <button type="submit" className="button lab-button" disabled={!image || match.loading}>Find similar images</button>
              <RequestStatus request={match} loadingText="Checking similarity…" hide404 />
            </form>
          </div>
          <section className="lab-panel lab-results" aria-live="polite" aria-busy={Boolean(match.loading)}>
            <h2>Closest matches</h2>
            {match.data ? <MatchResults result={match.data} /> : <p className="field-hint">Similar images and their distances will appear here.</p>}
          </section>
        </section>
      ) : <p className="shell lab-callout" role="status">The connected service has disabled the optional image similarity check.</p>}
      <div className="shell lab-callout">
        <p>{auditWarning}</p>
        <p>We say "similar" or "possible copy". We never say "copyright infringement".</p>
        <p>{honesty}</p>
        <a className="text-link" href={sitePath('#auditor-method')}>See how the comparison works</a>
      </div>
    </div>
  );
}

export function LabPage({ page }) {
  switch (page) {
    case 'generator': return <GeneratorPage />;
    case 'detector': return <DetectorPage />;
    case 'compare': return <ComparePage />;
    case 'auditor': return <AuditorPage />;
    case 'limits': return <LimitsPage />;
    default: return <div className="shell lab-section"><h1>Page not found</h1><a className="text-link" href={sitePath()}>Return home</a></div>;
  }
}


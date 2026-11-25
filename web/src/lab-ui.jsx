import React, { useCallback, useEffect, useRef, useState } from 'react';
import { isDemo, labRequest, pngUrl, readImage, receiveGeneratedImage, sitePath, stubMode, unavailable } from './lab-data.js';

export function useLabRequest(path) {
  const [state, setState] = useState({});
  const pending = useRef(null);

  function reset() {
    pending.current?.abort();
    pending.current = null;
    setState({});
  }

  useEffect(() => () => {
    const controller = pending.current;
    pending.current = null;
    controller?.abort();
  }, []);

  const run = useCallback(async (input) => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setState({ loading: true });
    const wakeTimer = window.setTimeout(() => {
      if (pending.current === controller) setState({ loading: true, delayed: true });
    }, 1500);
    const timeout = window.setTimeout(() => controller.abort(), 60000);

    try {
      const data = await labRequest(path, controller.signal, input);
      if (pending.current === controller) setState({ data });
    } catch (error) {
      if (pending.current !== controller) return;
      setState({
        error: controller.signal.aborted ? 'The request timed out. Try again.' :
          error instanceof TypeError ? 'The lab service is unreachable. Try again when it is online.' : error.message,
        status: error.status,
      });
    } finally {
      window.clearTimeout(wakeTimer);
      window.clearTimeout(timeout);
      if (pending.current === controller) pending.current = null;
    }
  }, [path]);

  return { ...state, run, reset };
}

export function RequestStatus({ request, loadingText, hide404 = false }) {
  return (
    <>
      {request.loading && <p className="request-status" role="status">{request.delayed ? 'Still waiting for the lab service…' : loadingText}</p>}
      {request.error && !(hide404 && request.status === 404) && <p className="request-error" role="alert">{request.error}</p>}
    </>
  );
}

export function useGeneratedHandoff(destination, onImage) {
  const [error, setError] = useState('');
  useEffect(() => {
    receiveGeneratedImage(destination).then((image) => {
      if (!image) return;
      onImage(image);
      document.getElementById('workspace')?.scrollIntoView();
    }).catch((error) => setError(error.message));
  }, [destination, onImage]);
  return error;
}

export function UploadField({ id, label = 'Choose an image', image, onImage }) {
  const [preview, setPreview] = useState('');
  const [error, setError] = useState('');
  const [inputFile, setInputFile] = useState(null);
  const input = useRef(null);
  const selection = useRef(0);
  useEffect(() => {
    if (!image) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(image.file);
    setPreview(url);
    if (image.file !== inputFile) {
      ++selection.current;
      input.current.value = '';
      setError('');
    }
    return () => URL.revokeObjectURL(url);
  }, [image, inputFile]);

  async function selectFile(event) {
    const file = event.target.files[0];
    const current = ++selection.current;
    setError('');
    setInputFile(file);
    onImage(null);
    if (!file) return;
    try {
      const selected = await readImage(file);
      if (selection.current === current) onImage(selected);
    } catch (error) {
      if (selection.current !== current) return;
      input.current.value = '';
      setError(error.message);
    }
  }

  return (
    <div className="upload-field">
      <label htmlFor={id}>{label}</label>
      <p id={id + '-hint'} className="field-hint">PNG or JPG, up to 20 MB.</p>
      <input ref={input} id={id} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" aria-describedby={id + '-hint'} onChange={selectFile} />
      {error && <p role="alert" className="request-error">{error}</p>}
      {preview && image && <figure className="selected-image">
        <img className="upload-preview" src={preview} alt={'Preview for ' + label.toLowerCase()} />
        <figcaption className="field-hint">{image.file.name} · {image.width}×{image.height}{image.demo ? ' · Demo illustration' : ''}</figcaption>
      </figure>}
    </div>
  );
}

export function PageIntro({ label, title, accent, children, art, position }) {
  return (
    <section className="lab-intro">
      <div className="shell lab-intro-inner">
        <div className="lab-intro-text">
          {stubMode && <p className="result-mode mode-banner">Demo mode · sample data</p>}
          <p className="eyebrow">{label}</p>
          <h1><span>{title}</span>{' '}<em>{accent}</em></h1>
          <p className="lab-intro-copy">{children}</p>
        </div>
        <div className="lab-intro-art" aria-hidden="true">
          <img src={sitePath('studies/' + art)} width="1440" height="960" alt="" style={{ objectPosition: position }} fetchPriority="high" />
        </div>
      </div>
    </section>
  );
}

export function ResultNote({ result, demoLabel = 'Demo response · illustrative data' }) {
  return (
    <>
      {isDemo(result) && <p className="result-mode">{demoLabel}</p>}
      {result.note && <p className="field-hint">{result.note.replace(/^stub:\s*/i, '')}</p>}
    </>
  );
}

export function ResultImage({ image, label, className = 'lab-heatmap' }) {
  return image ? (
    <figure className={className}>
      <img src={pngUrl(image)} alt={label} />
      <figcaption>{label}</figcaption>
    </figure>
  ) : <p className="field-hint">{label} · {unavailable}</p>;
}

export function formatNumber(value) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : unavailable;
}

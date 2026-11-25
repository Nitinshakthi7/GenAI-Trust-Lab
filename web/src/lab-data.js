const base = import.meta.env.BASE_URL;
const api = (import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '');
const maxBytes = 20 * 1024 * 1024;
const fixtures = {
  '/health': 'health',
  '/generators': 'generators',
  '/generate': 'generate',
  '/detect': 'detect',
  '/compare': 'compare',
  '/evidence': 'evidence',
  '/audit/results': 'audit_results',
  '/audit/check': 'audit_check',
};

export const stubMode = import.meta.env.VITE_USE_STUB !== 'false';
export const wordsEnabled = import.meta.env.VITE_ENABLE_COMPARE_WORDS === 'true';
export const placeholder = '[PLACEHOLDER - not measured yet]';
export const unavailable = 'not available';
export const sizeWarning = 'This detector was trained only on 32x32 CIFAR-10-style images. Results on other photos are not meaningful.';
export const sizeLimitation = 'The size check only catches image size. A 32x32 photo that is not CIFAR-10-style is not flagged.';
export const unseenWarning = 'Detecting images from a generator the detector has never seen is an open research problem. We will report the drop in performance as a finding. We do not promise the detector will work on unseen generators.';
export const auditWarning = 'The auditor measures similarity between images. It does not decide copyright infringement or "legal copying".';
export const honesty = 'Honest reporting: where a tool fails, we show the failure.';
export const alignmentWarning = 'Pixel tools (SSIM, difference map) only make sense for nearly aligned images, for example a generated image and its nearest training image, or an original and an edited copy. For two different photos, for example an orange cat and a black cat in different poses, the difference map will light up almost everywhere; the colour and brightness tables and the Words panel say more, and the Words panel can be wrong.';
export const wordsPrivacy = 'When Words is on, photos leave your computer; do not upload private photos or photos of people.';
export const pngUrl = (value) => 'data:image/png;base64,' + value;
export const isDemo = (data) => /^\s*(stub|demo)\b/i.test(data?.note || '');
export const displayValue = (value) => value ?? unavailable;
export const sitePath = (path = '') => base + path;

function stubResult(path, payload, input) {
  if (path === '/detect') {
    const { width, height } = input.image;
    const out_of_scope = width !== 32 || height !== 32;
    return { ...payload, input_size: width + 'x' + height, out_of_scope, warning: out_of_scope ? sizeWarning : null };
  }
  if (path === '/compare' && !input.use_words) {
    return { ...payload, source: 'heatmap_only', differences: [], summary: null };
  }
  if (path === '/generate') {
    const catalogue = payload[input.generator_id];
    const { seed, n, steps, guidance, class_id, interpolate } = input;
    const lastFrame = (interpolate?.frames || n) - 1;
    return {
      generator_id: input.generator_id,
      images: Array.from({ length: lastFrame + 1 }, (_, index) => ({
        png_base64: catalogue.images[index % catalogue.images.length],
        seed: interpolate && index === lastFrame ? interpolate.seed_b : interpolate ? seed : seed + index,
        class_id,
      })),
      metadata: { ...catalogue.metadata, steps, guidance },
      note: payload.note,
    };
  }
  return payload;
}

export async function labRequest(path, signal, input) {
  const url = stubMode ? sitePath('stub/' + fixtures[path] + '.json') : api + path;
  let body;
  let headers;
  if (!stubMode && input) {
    if (path === '/generate') {
      body = JSON.stringify(input);
      headers = { 'Content-Type': 'application/json' };
    } else {
      body = new FormData();
      for (const [key, value] of Object.entries(input)) {
        if (value != null) body.append(key, value.file || String(value));
      }
    }
  }
  const response = await fetch(url, { method: body ? 'POST' : 'GET', headers, body, signal });
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload) {
    const error = new Error(payload?.error || (stubMode ? 'The demo data could not be loaded.' : 'The lab service could not complete this request.'));
    error.status = response.status;
    throw error;
  }
  return stubMode ? stubResult(path, payload, input) : payload;
}

export async function readImage(file) {
  if (file.size > maxBytes) throw new Error('The image must be 20 MB or smaller.');
  const magic = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  const png = [137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => magic[index] === byte);
  const jpg = magic[0] === 255 && magic[1] === 216 && magic[2] === 255;
  if (!png && !jpg) throw new Error('Choose a readable PNG or JPG image.');
  const source = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = source;
    await image.decode();
    return { file, width: image.naturalWidth, height: image.naturalHeight };
  } catch {
    throw new Error('The file is not a readable PNG or JPG image.');
  } finally {
    URL.revokeObjectURL(source);
  }
}

const transferKey = 'trust-lab-generated-image';

export function sendGeneratedImage(destination, image, name, slot) {
  try {
    sessionStorage.setItem(transferKey, JSON.stringify({ destination, image, name, slot, demo: stubMode }));
  } catch {
    throw new Error('The browser could not pass this image. Download it and upload it on the next page.');
  }
  window.location.assign(sitePath(destination) + '?from=generator#workspace');
}

export async function receiveGeneratedImage(destination) {
  if (new URLSearchParams(window.location.search).get('from') !== 'generator') return null;
  try {
    const transfer = JSON.parse(sessionStorage.getItem(transferKey));
    if (transfer?.destination !== destination) return null;
    sessionStorage.removeItem(transferKey);
    const response = await fetch(pngUrl(transfer.image));
    const file = new File([await response.blob()], transfer.name, { type: 'image/png' });
    return { ...await readImage(file), slot: transfer.slot, demo: transfer.demo };
  } catch {
    throw new Error('The generated image could not be opened. Upload its downloaded PNG to continue.');
  }
}

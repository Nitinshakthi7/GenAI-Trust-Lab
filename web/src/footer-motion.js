import { createPaintingRenderer } from './painting-renderer.js';

// The rig uses coordinates on a 768 × 512 copy of the artwork.
const fragmentSource = `
  precision highp float;
  uniform sampler2D painting;
  uniform float seconds;
  varying vec2 uv;
  const vec2 size = vec2(768., 512.);

  float gesture(float start, float peak, float settle, float end) {
    return smoothstep(start, peak, seconds) * (1. - smoothstep(settle, end, seconds));
  }

  float falloff(vec2 point, vec2 center, vec2 radius) {
    return 1. - smoothstep(.45, 1., length((point - center) / radius));
  }

  vec2 turn(vec2 point, vec2 pivot, float angle) {
    vec2 offset = point - pivot;
    float s = sin(angle);
    float c = cos(angle);
    return vec2(c * offset.x - s * offset.y, s * offset.x + c * offset.y) - offset;
  }

  void main() {
    vec2 point = uv * size;
    vec2 motion = vec2(0.);

    if (point.x < 186.) {
      float glance = gesture(.12, .95, 1.85, 3.);
      float lift = gesture(.46, 1.56, 2.2, 3.72);

      // The head tilts at the neck. The plate and the gloved hand pivot together.
      motion += falloff(point, vec2(47., 106.), vec2(46., 49.)) * glance *
        (turn(point, vec2(48., 141.), .115 * glance) + vec2(1.2, -2.));
      motion += falloff(point, vec2(114., 109.), vec2(45., 91.)) *
        (turn(point, vec2(103., 165.), .12 * lift) + vec2(1.5, -5.) * lift);
      motion += falloff(point, vec2(97., 177.), vec2(48., 40.)) * vec2(1.5, -4.) * lift;
    } else if (point.x > 584.) {
      float glance = gesture(1.02, 1.96, 2.75, 4.15);
      float adjust = gesture(1.35, 2.42, 2.95, 4.52);
      float sway = gesture(1.72, 2.06, 3.65, 4.82);

      // Follow the film with the chin, then guide the strip through the lower hand.
      motion += falloff(point, vec2(717., 111.), vec2(46., 52.)) * glance *
        (turn(point, vec2(715., 140.), .14 * glance) + vec2(-2., 3.));
      motion += falloff(point, vec2(637., 85.), vec2(36., 48.)) * vec2(2., -5.) * adjust;
      motion += falloff(point, vec2(680., 190.), vec2(44., 64.)) * vec2(-5., 5.) * adjust;

      // The loose film bends below the hand while the ends stay anchored.
      float slack = smoothstep(207., 275., point.y) * (1. - smoothstep(409., 478., point.y));
      motion.x += falloff(point, vec2(703., 341.), vec2(61., 164.)) * slack * sway *
        15. * sin(seconds * 3.8 - point.y * .031);
    }

    gl_FragColor = texture2D(painting, (point - motion) / size);
  }
`;

export function animateFooterCharacters(canvas, image) {
  if (!window.IntersectionObserver) return;
  let renderer;
  let visible = false;
  let played = false;
  let frame = 0;

  function stop() {
    cancelAnimationFrame(frame);
    canvas.removeAttribute('data-playing');
  }

  function play() {
    if (!visible || played || !image.complete || !image.naturalWidth) return;
    played = true;
    renderer ??= createPaintingRenderer(canvas, fragmentSource);
    if (!renderer) return;

    const start = performance.now();
    let lastDraw = -Infinity;
    function tick(now) {
      const elapsed = now - start;
      if (elapsed >= 4850) {
        stop();
        return;
      }
      if (now - lastDraw >= 30) {
        renderer.draw(image, elapsed / 1000);
        canvas.setAttribute('data-playing', '');
        lastDraw = now;
      }
      frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
  }

  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.intersectionRatio >= .4;
    if (!entry.isIntersecting) {
      stop();
      played = false;
    }
    play();
  }, { threshold: [0, .4] });

  observer.observe(image);
  image.addEventListener('load', play);
  canvas.addEventListener('webglcontextlost', stop);
  return () => {
    stop();
    observer.disconnect();
    image.removeEventListener('load', play);
    canvas.removeEventListener('webglcontextlost', stop);
    renderer?.dispose();
  };
}

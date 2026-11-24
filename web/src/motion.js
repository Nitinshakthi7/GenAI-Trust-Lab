// All scroll scenes share one scheduled frame and passive window listeners.
const subscribers = new Set();
let frame = 0;
function schedule() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    subscribers.forEach((update) => update());
  });
}

export function subscribeFrame(update) {
  if (!subscribers.size) {
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
  }
  subscribers.add(update);
  return () => {
    subscribers.delete(update);
    if (!subscribers.size) {
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      cancelAnimationFrame(frame);
      frame = 0;
    }
  };
}

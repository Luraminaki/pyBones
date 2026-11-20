// Small DOM and formatting helpers shared by the page modules.

const TOAST_MS = 6000;
const SECONDS_PER_MINUTE = 60;
const PERCENT = 100;

// Create an element: el('li', {class: 'job', dataset: {state}}, child, 'text', ...).
export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== undefined && c !== null && c !== false));
  return node;
}

export function percent(fraction, digits = 0) {
  return `${(fraction * PERCENT).toFixed(digits)} %`;
}

// A fraction as a CSS length, e.g. a bar width.
export function cssPercent(fraction) {
  return `${fraction * PERCENT}%`;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return '';
  if (seconds < SECONDS_PER_MINUTE) return `${Math.round(seconds)} s`;
  return `${Math.floor(seconds / SECONDS_PER_MINUTE)} min ${Math.round(seconds % SECONDS_PER_MINUTE)} s`;
}

export function megabytes(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

// Configuration colours are OpenCV BGR triples; CSS and three.js want RGB.
export function cssColor([b, g, r]) {
  return `rgb(${r}, ${g}, ${b})`;
}

export function announce(text) {
  document.getElementById('announcer').textContent = text;
}

export function toast(text, { error = false } = {}) {
  const node = el('div', { class: error ? 'toast error' : 'toast', role: error ? 'alert' : 'status' }, text);
  document.getElementById('toasts').append(node);
  setTimeout(() => node.remove(), TOAST_MS);
}

// Small DOM and formatting helpers shared by the page modules.

const TOAST_MS = 6000;
const SECONDS_PER_MINUTE = 60;
const PERCENT = 100;
const BYTES_PER_MB = 1e6;

/**
 * The final labels a segmentation shows, in drawing order (background is the remainder).
 * Checked against Python's `FINAL_LABELS` by tests/server/test_page_js.py.
 * @type {string[]}
 */
export const LABELS = ['bone', 'vessel', 'burst'];

/**
 * Create an element, e.g. `el('li', {class: 'job', dataset: {state}}, child, 'text')`.
 *
 * Attributes named `on…` become event listeners, `class` the class name, `dataset` the
 * data attributes; `true` makes an empty attribute, and `false`, `null` or `undefined`
 * leave it out (so do such children).
 * @param {string} tag Element name.
 * @param {Object<string, *>} [attributes] Attributes, listeners and data attributes.
 * @param {...(Node|string|null|undefined|false)} children Child nodes or texts.
 * @returns {HTMLElement} The element.
 */
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

/**
 * Format a fraction as a percentage for reading, e.g. `50 %`.
 * @param {number} fraction Value in [0, 1].
 * @param {number} [digits] Decimals shown.
 * @returns {string} The text.
 */
export function percent(fraction, digits = 0) {
  return `${(fraction * PERCENT).toFixed(digits)} %`;
}

/**
 * Format a fraction as a CSS length, e.g. a bar width `25%`.
 * @param {number} fraction Value in [0, 1].
 * @returns {string} The CSS percentage.
 */
export function cssPercent(fraction) {
  return `${fraction * PERCENT}%`;
}

/**
 * Format a duration, e.g. `42 s` or `2 min 5 s`.
 * @param {?number} seconds Duration (`null` or `undefined` if unknown).
 * @returns {string} The text, empty if unknown.
 */
export function duration(seconds) {
  if (seconds === null || seconds === undefined) return '';
  // Rounded once, before splitting: 119.6 s is 2 min 0 s, not 1 min 60 s.
  const whole = Math.round(seconds);
  if (whole < SECONDS_PER_MINUTE) return `${whole} s`;
  return `${Math.floor(whole / SECONDS_PER_MINUTE)} min ${whole % SECONDS_PER_MINUTE} s`;
}

/**
 * Take the last part of a path written on any system, e.g. the stack of a segmentation.
 * @param {string} path POSIX or Windows path.
 * @returns {string} Its file or folder name.
 */
export function baseName(path) {
  return path.split(/[\\/]/).pop();
}

/**
 * Format a size in (decimal) megabytes, e.g. `2.5 MB`.
 * @param {number} bytes Size.
 * @returns {string} The text.
 */
export function megabytes(bytes) {
  return `${(bytes / BYTES_PER_MB).toFixed(1)} MB`;
}

/**
 * Convert a configuration colour (OpenCV's BGR order) to a CSS colour.
 * @param {number[]} bgr Blue, green and red, 0-255.
 * @returns {string} The CSS `rgb(…)` colour (three.js reads it too).
 */
export function cssColor([b, g, r]) {
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Read a value remembered in this browser.
 * @param {string} key Storage key.
 * @returns {?string} The value, or `null` if none (or the browser blocks storage).
 */
export function recall(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * Remember a value in this browser; a browser blocking storage simply forgets it.
 * @param {string} key Storage key.
 * @param {string} value Value.
 */
export function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // not remembered: still used for this visit
  }
}

/**
 * Tell screen readers about an event (the page's live region).
 * @param {string} text What happened.
 */
export function announce(text) {
  document.getElementById('announcer').textContent = text;
}

/**
 * Show a short message in a corner of the page; it goes away after a few seconds.
 * @param {string} text Message.
 * @param {{error?: boolean}} [options] `error`: shown as an alert.
 */
export function toast(text, { error = false } = {}) {
  const node = el('div', { class: error ? 'toast error' : 'toast', role: error ? 'alert' : 'status' }, text);
  document.getElementById('toasts').append(node);
  setTimeout(() => node.remove(), TOAST_MS);
}

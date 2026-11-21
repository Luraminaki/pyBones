// Shared state of the annotation page: constants, the canvases, and what is being painted.
// Data-dependent values (labels, colours, crop size, scale...) come from /api/session, which
// reads tools/annotation/config.json and the pybones configuration.

export const UNSURE_KEY = 'unsure';
export const UNSURE_HOTKEY = 'u';
export const DEFAULT_LABEL_KEY = 'vessel';
export const PERCENT = 100;
export const RGBA = 4;
export const OPAQUE = 255;
export const TRANSPARENT = 0;
// Brightness and contrast multipliers that leave the image unchanged.
export const NEUTRAL_FILTER = 1;
export const FILL_NEIGHBOURS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const LUT_ENTRIES = 256;

/**
 * Find an element of the page by id.
 * @param {string} id Element id.
 * @returns {HTMLElement} The element.
 */
export const $ = (id) => document.getElementById(id);

export const canvas = $('canvas');
export const ctx = canvas.getContext('2d');
// The labels, one pixel per image pixel, drawn scaled over the image.
export const overlay = document.createElement('canvas');
export const octx = overlay.getContext('2d');

export const state = {
  session: null,
  index: -1,
  labels: null, // Uint8Array(size * size), row-major
  images: new Map(), // z offset -> HTMLImageElement
  undo: [],
  redo: [],
  active: null, // label value painted with the left button
  tool: 'brush',
  radius: 2,
  zOffset: 0,
  alpha: 0.45,
  overlayHidden: false,
  dirty: false,
  stroke: null, // {value, last: [x, y]} while painting
  hover: null, // [x, y] in image pixels
  lut: new Uint8ClampedArray(LUT_ENTRIES * RGBA),
  zoom: 1, // view magnification on top of display.scale
  pan: [0, 0], // view offset, in canvas pixels
  panning: null, // {from: [clientX, clientY], pan: [x, y]} while dragging the view
  brightness: NEUTRAL_FILTER,
  contrast: NEUTRAL_FILTER,
  paintOver: null, // only this label value is repainted (null: every label)
  autosave: null, // pending autosave timer
  gray: null, // Uint8Array(size * size): grey levels of the centre slice (for the wand)
  wandTolerance: 0,
  outline: false, // draw only the borders between labels
  noteDirty: false,
};

/**
 * List the labels a pixel can take, the "unsure" one included.
 * @returns {{value: number, key: string, rgb: ?number[]}[]} The labels, in palette order.
 */
export function allLabels() {
  const { labels, unsure } = state.session;
  return [...labels, { value: unsure.value, key: UNSURE_KEY, rgb: unsure.rgb }];
}

/**
 * Create an element with a class and children.
 * @param {string} tag Element name.
 * @param {?string} className Its class (`null`: none).
 * @param {...(Node|string)} children Child nodes or texts.
 * @returns {HTMLElement} The element.
 */
export function element(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.append(...children);
  return node;
}

/**
 * Fetch a resource of the annotation server, failing on an error status.
 * @param {string} url Address.
 * @param {RequestInit} [init] Fetch options.
 * @returns {Promise<Response>} The successful response.
 * @throws {Error} If the server cannot be reached or answers an error (its message).
 */
export async function fetchOk(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response;
}

/**
 * Say that something failed, in the save indicator (the page has no other message area).
 * @param {string} what What was being done.
 * @param {Error} error Why it failed.
 */
export function reportError(what, error) {
  const node = $('save-state');
  node.textContent = `${what} failed: ${error.message}`;
  node.classList.add('dirty');
}

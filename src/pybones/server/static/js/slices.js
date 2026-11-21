// Slice viewer: one slice of the stack with its labels overlaid (rendered by the server).

import { fetchSlice } from './api.js';
import { cssColor, el, LABELS, percent, toast } from './ui.js';

// Image sizes offered, as fractions of the full slice (capped by the server's limit).
const SCALES = [0.25, 0.5, 1];
const DEFAULT_SCALE = 0.5;
// Wait for the slider to settle before asking for a slice.
const DEBOUNCE_MS = 120;

let segmentation = null;
let timer = null;
// One slice request at a time: a position asked for meanwhile is loaded right after
// (latest wins), so dragging the slider never piles renders up on the server.
let inFlight = false;
let pending = false;
let voxelZ = 0;
const nodes = {};

/** Show the slider's position, as an index and a depth in µm. */
function showPosition() {
  const z = Number(nodes.slider.value);
  nodes.value.textContent = `${z} / ${nodes.slider.max} · ${(z * voxelZ).toFixed(0)} µm`;
}

/**
 * Show the slice at the slider's position, without letting the browser keep it: fetch it
 * ourselves (`no-store`), show it through a temporary blob: URL, and revoke that URL once
 * the image is decoded. The <img> keeps the picture on screen; nothing else holds the data,
 * so scrolling through a stack never accumulates slices in memory, however long the page
 * stays open.
 */
async function load() {
  if (!segmentation) return;
  showPosition();
  if (inFlight) {
    pending = true;
    return;
  }
  inFlight = true;
  const id = segmentation;
  const z = Number(nodes.slider.value);
  nodes.loading.hidden = false;
  let url = null;
  try {
    url = URL.createObjectURL(await fetchSlice(id, z, nodes.scale.value));
    nodes.image.src = url;
    nodes.image.alt = `Slice ${z} of ${id} with its labels overlaid`;
    await nodes.image.decode();
  } catch (err) {
    // A newer position is waiting: this one no longer matters.
    if (!pending) toast(`Slice ${z} could not be shown: ${err.message}`, { error: true });
  } finally {
    if (url) URL.revokeObjectURL(url);
    inFlight = false;
    if (pending) {
      pending = false;
      void load();
    } else {
      nodes.loading.hidden = true;
    }
  }
}

/**
 * Show the slices of a segmentation (from the middle one, unless it is already shown).
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 */
export function showSegmentation(summary) {
  const sameStack = segmentation === summary.id;
  segmentation = summary.id;
  const [depth] = summary.shape_zyx;
  [voxelZ] = summary.voxel_size_um; // (z, y, x) µm
  nodes.slider.max = String(depth - 1);
  if (!sameStack) nodes.slider.value = String(Math.floor(depth / 2));
  void load();
}

/**
 * Set up the viewer: scales offered, legend, slider and scale listeners.
 * @param {{max_preview_scale: number}} server What the server describes of itself.
 * @param {Object<string, number[]>} colors BGR colour per label.
 */
export function initSlices(server, colors) {
  nodes.slider = document.getElementById('slice-z');
  nodes.value = document.getElementById('slice-z-value');
  nodes.scale = document.getElementById('slice-scale');
  nodes.image = document.getElementById('slice-image');
  nodes.loading = document.getElementById('slice-loading');

  const scales = SCALES.filter((s) => s <= server.max_preview_scale);
  nodes.scale.replaceChildren(
    ...scales.map((s) => el('option', { value: String(s), selected: s === DEFAULT_SCALE }, percent(s))),
  );
  document.getElementById('legend').replaceChildren(
    ...LABELS.map((label) =>
      el('li', {}, el('span', { class: 'swatch', style: `background: ${cssColor(colors[label])}` }), label),
    ),
  );

  nodes.slider.addEventListener('input', () => {
    showPosition();
    clearTimeout(timer);
    timer = setTimeout(load, DEBOUNCE_MS);
  });
  nodes.scale.addEventListener('change', () => void load());
}

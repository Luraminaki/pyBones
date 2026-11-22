// Slice viewer: one slice of the stack with its labels overlaid (rendered by the server).

import { sliceLoader } from './slice_loader.js';
import { cssColor, el, LABELS, percent } from './ui.js';

// Image sizes offered, as fractions of the full slice (capped by the server's limit).
const SCALES = [0.25, 0.5, 1];
const DEFAULT_SCALE = 0.5;
// Wait for the slider to settle before asking for a slice.
const DEBOUNCE_MS = 120;

let segmentation = null;
let timer = null;
let voxelZ = 0;
const nodes = {};

/** Show the slider's position, as an index and a depth in µm. */
function showPosition() {
  const z = Number(nodes.slider.value);
  nodes.value.textContent = `${z} / ${nodes.slider.max} · ${(z * voxelZ).toFixed(0)} µm`;
}

/** Show the slice at the slider's position (the loader keeps one request at a time). */
function load() {
  if (!segmentation) return;
  showPosition();
  nodes.show(segmentation, Number(nodes.slider.value), nodes.scale.value);
}

/**
 * Show the slices of a segmentation, from the middle one. Shown again (the lists are refreshed
 * after every job), it is left as it is: a segmentation's slices never change.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 */
export function showSegmentation(summary) {
  if (segmentation === summary.id) return;
  segmentation = summary.id;
  const [depth] = summary.shape_zyx;
  [voxelZ] = summary.voxel_size_um; // (z, y, x) µm
  nodes.slider.max = String(depth - 1);
  nodes.slider.value = String(Math.floor(depth / 2));
  load();
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
  const loading = document.getElementById('slice-loading');
  nodes.show = sliceLoader(document.getElementById('slice-image'), (busy) => { loading.hidden = !busy; });

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
  nodes.scale.addEventListener('change', load);
}

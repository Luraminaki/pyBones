// Centrelines in the 3D view: the measured vessel centre voxels, drawn as points over the meshes.
// Used by viewer3d.js, which hands over its scene; downloaded only when shown.

import * as THREE from 'three';

import { fetchCentrelines, JOB_KINDS } from './api.js';
import { JOB_FINISHED } from './jobs.js';
import { discard, releaseAfterUpload } from './mesh_memory.js';
import { el, toast } from './ui.js';

// Bright and unlike the label colours, so the line reads over the vessel mesh.
const POINT_COLOR = 0xffd23f;
// Points keep this size on screen at any zoom (in world units, a voxel is under a pixel when
// the whole stack is in view), and are drawn last, over the meshes that would hide them.
const POINT_PIXELS = 1;
const POINTS_ORDER = 2;
const XYZ = 3;
const KILO = 1e3;

let scene;
let changed;
let current = null; // {summary, checkbox, status, object, loading, generation}

/**
 * Remove the points from the scene and the GPU, and void any download still under way.
 */
function drop() {
  if (!current) return;
  current.generation += 1; // a download started before this point is now out of date
  current.loading = false;
  if (!current.object) return;
  discard(scene, current.object);
  current.object = null;
}

/**
 * Download the centrelines of the shown segmentation and add them to the scene.
 */
async function load() {
  const entry = current;
  const id = entry.summary.id;
  const generation = entry.generation;
  entry.loading = true;
  entry.status.textContent = 'Loading…';
  let positions;
  try {
    positions = await fetchCentrelines(id);
  } catch (err) {
    if (current !== entry || entry.generation !== generation) return; // no longer wanted
    entry.loading = false;
    entry.status.textContent = 'Could not load';
    toast(`The centrelines of ${id} could not be loaded: ${err.message}`, { error: true });
    return;
  }
  // Another segmentation is shown, or the points were dropped (measured again) meanwhile:
  // these are out of date, and nothing holds them once this function returns.
  if (current !== entry || entry.generation !== generation) return;
  entry.loading = false;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, XYZ));
  const material = new THREE.PointsMaterial({
    color: POINT_COLOR,
    size: POINT_PIXELS,
    sizeAttenuation: false,
    depthTest: false, // centrelines run inside the vessels: seen through them
  });
  entry.object = new THREE.Points(geometry, material);
  entry.object.renderOrder = POINTS_ORDER;
  releaseAfterUpload(entry.object);
  entry.object.visible = entry.checkbox.checked;
  scene.add(entry.object);
  entry.status.textContent = `${(positions.length / XYZ / KILO).toFixed(0)} k points`;
  changed();
}

/**
 * Show or hide the centrelines, downloading them the first time.
 * @param {boolean} visible Whether to show them.
 */
function show(visible) {
  if (current.object) {
    current.object.visible = visible;
    changed();
  } else if (visible && !current.loading) {
    void load();
  }
}

/**
 * Say whether the toggle can be used (only once the segmentation is quantified).
 */
function updateAvailability() {
  const ready = current.summary.quantified;
  current.checkbox.disabled = !ready;
  if (!ready) current.status.textContent = 'Quantify first (Network tab)';
  else if (!current.object && !current.loading) current.status.textContent = 'Ready';
}

/**
 * Tell whether centreline points are on screen.
 * @returns {boolean} Whether they are shown.
 */
export function centrelinesShown() {
  return Boolean(current?.object?.visible);
}

/**
 * Build the centreline toggle of a segmentation (forgetting the previous one's points).
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 * @returns {HTMLElement} The toggle, for the viewer's controls.
 */
export function centrelineToggle(summary) {
  drop();
  const checkbox = el('input', { type: 'checkbox', id: 'show-centrelines' });
  const status = el('span', { class: 'job-detail' });
  current = { summary, checkbox, status, object: null, loading: false, generation: 0 };
  checkbox.addEventListener('change', () => show(checkbox.checked));
  updateAvailability();
  const swatch = el('span', { class: 'swatch', style: `background: #${POINT_COLOR.toString(16).padStart(6, '0')}` });
  return el('div', { class: 'mesh-toggle' }, el('label', { for: 'show-centrelines' }, checkbox, swatch, 'centrelines'), status);
}

/**
 * Take a newer description of the shown segmentation (e.g. once it is quantified).
 * @param {import('./api.js').SegmentationSummary} summary The segmentation, same id.
 */
export function refreshCentrelines(summary) {
  current.summary = summary;
  updateAvailability();
}

/**
 * Remove the points of the shown segmentation, e.g. before another one is shown.
 */
export function clearCentrelines() {
  drop();
  current = null;
}

/**
 * Hand over the scene, and follow the quantification jobs: measured again, drawn again.
 * @param {THREE.Scene} viewerScene The viewer's scene.
 * @param {function(): void} onChange Called when the drawing changed (to redraw).
 */
export function initCentrelines(viewerScene, onChange) {
  scene = viewerScene;
  changed = onChange;
  document.addEventListener(JOB_FINISHED, (event) => {
    const status = event.detail;
    if (status.kind !== JOB_KINDS.QUANTIFICATION || status.segmentation !== current?.summary.id) return;
    if (status.state !== 'succeeded') return;
    current.summary = { ...current.summary, quantified: true };
    drop();
    updateAvailability();
    if (current.checkbox.checked) void load();
    changed();
  });
}

// 3D viewer: the meshes of a segmentation (GLB, coordinates in µm), drawn with three.js.
// Loaded on demand by app.js when the 3D tab first opens. A label whose mesh is not built
// yet is built by a job first. Renders only when something changes (no idle animation loop).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { api, meshUrl } from './api.js';
import { activeJob, JOB_FINISHED, JOB_UPDATED, track } from './jobs.js';
import { THEME_CHANGED } from './theme.js';
import { dispose, releaseAfterUpload, triangles } from './mesh_memory.js';
import { cssColor, duration, el, LABELS, megabytes, percent, toast } from './ui.js';

const FORMAT = 'glb';
// Shown when a segmentation opens: the vessel network, the point of pyBones.
const SHOWN_BY_DEFAULT = new Set(['vessel']);
// Bone is context around the vessels: drawn see-through, after the opaque meshes.
const TRANSLUCENT = { bone: 0.22 };
const TRANSLUCENT_ORDER = 1;
const ROUGHNESS = 0.55;
const FIELD_OF_VIEW = 40;
// Camera distance: the bounding box fills about 1 / FIT_MARGIN of the view.
const FIT_MARGIN = 1.1;
const NEAR_RATIO = 1 / 200;
const FAR_RATIO = 50;
const SKY_LIGHT = 1.4;
const KEY_LIGHT = 1.6;
const MEGA = 1e6;
// One click on + / − moves the camera this much closer / farther.
const ZOOM_STEP = 1.3;
// Image rows go down the screen, as in the slice viewer: "up" is -y.
const UP = new THREE.Vector3(0, -1, 0);
// What a plain left-drag (or one finger) does.
const DRAG_MODES = {
  rotate: { mouse: THREE.MOUSE.ROTATE, touch: THREE.TOUCH.ROTATE, label: 'Rotate' },
  move: { mouse: THREE.MOUSE.PAN, touch: THREE.TOUCH.PAN, label: 'Move' },
};

const loader = new GLTFLoader();
let renderer;
let scene;
let camera;
let controls;
let message;
let colors;
let current = null; // {summary, labels: Map(label -> {checkbox, status, object, loading})}
let fitted = false;

/** Draw the scene once (there is no animation loop: only changes redraw). */
function render() {
  renderer.render(scene, camera);
}

/**
 * Build the material of a label: its configuration colour, see-through for the bone.
 * @param {string} label Label key.
 * @returns {THREE.MeshStandardMaterial} The material.
 */
function material(label) {
  const opacity = TRANSLUCENT[label];
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(cssColor(colors[label])), // CSS colours are read as sRGB
    roughness: ROUGHNESS,
    metalness: 0,
    transparent: opacity !== undefined,
    opacity: opacity ?? 1,
    depthWrite: opacity === undefined,
  });
}

/** Paint the scene's background with the theme's colour (`--scene-bg`). */
function sceneBackground() {
  const value = getComputedStyle(document.documentElement).getPropertyValue('--scene-bg').trim();
  scene.background = new THREE.Color(value || '#000000');
}

/** Fit the drawing to its container and redraw. */
function resize() {
  const { clientWidth: width, clientHeight: height } = renderer.domElement.parentElement;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  render();
}

/** Frame the visible meshes, looking down the stack (+z) with image rows going down. */
export function fitView() {
  const box = new THREE.Box3();
  for (const entry of current?.labels.values() ?? []) {
    if (entry.object?.visible) box.expandByObject(entry.object);
  }
  if (box.isEmpty()) return;
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  // Seen along z, the box shows its x (width) and y (height): fit whichever is tighter.
  const visible = Math.max(size.y, size.x / camera.aspect);
  const halfTan = Math.tan(THREE.MathUtils.degToRad(FIELD_OF_VIEW / 2));
  const distance = (visible / 2 / halfTan) * FIT_MARGIN + size.z / 2;
  camera.up.copy(UP);
  camera.position.copy(centre).add(new THREE.Vector3(0, 0, -distance));
  camera.near = distance * NEAR_RATIO;
  camera.far = distance * FAR_RATIO;
  camera.updateProjectionMatrix();
  controls.target.copy(centre);
  controls.update();
  fitted = true;
  render();
}

/**
 * Move the camera towards or away from the point it orbits.
 * @param {number} factor Below 1: closer; above 1: farther.
 */
function zoom(factor) {
  const offset = camera.position.clone().sub(controls.target).multiplyScalar(factor);
  camera.position.copy(controls.target).add(offset);
  controls.update();
  render();
}

/**
 * Choose what a plain left-drag (or one finger) does.
 * @param {string} mode `rotate` or `move`.
 * @param {Object<string, HTMLButtonElement>} buttons The mode buttons (the chosen one is pressed).
 */
function setDragMode(mode, buttons) {
  controls.mouseButtons.LEFT = DRAG_MODES[mode].mouse;
  controls.touches.ONE = DRAG_MODES[mode].touch;
  for (const [name, button] of Object.entries(buttons)) {
    button.setAttribute('aria-pressed', String(name === mode));
  }
}

/**
 * Build the view buttons: drag mode, zoom in / out, reset.
 * @returns {HTMLElement} Their container.
 */
function viewActions() {
  const buttons = {};
  for (const [mode, { label }] of Object.entries(DRAG_MODES)) {
    buttons[mode] = el('button', {
      type: 'button',
      class: 'ghost small',
      title: mode === 'rotate' ? 'Drag to turn the view' : 'Drag to move the view',
      onclick: () => setDragMode(mode, buttons),
    }, label);
  }
  setDragMode(controls.mouseButtons.LEFT === THREE.MOUSE.PAN ? 'move' : 'rotate', buttons);
  return el(
    'div',
    { class: 'mesh-actions' },
    el('div', { class: 'segmented', role: 'group', 'aria-label': 'Left-drag' }, buttons.rotate, buttons.move),
    el('button', { type: 'button', class: 'ghost small', title: 'Zoom in', 'aria-label': 'Zoom in', onclick: () => zoom(1 / ZOOM_STEP) }, '+'),
    el('button', { type: 'button', class: 'ghost small', title: 'Zoom out', 'aria-label': 'Zoom out', onclick: () => zoom(ZOOM_STEP) }, '−'),
    el('button', { type: 'button', class: 'ghost small', onclick: fitView }, 'Reset view'),
  );
}

/** Show the hint over the scene while nothing is drawn (or "Loading…"). */
function updateMessage() {
  const shown = [...(current?.labels.values() ?? [])].some((e) => e.object?.visible);
  const busy = [...(current?.labels.values() ?? [])].some((e) => e.loading);
  message.hidden = shown;
  message.textContent = busy ? 'Loading…' : 'Choose the labels to show.';
}

/**
 * Download a built mesh and add it to the scene (dropped if another segmentation is shown
 * by the time it arrives).
 * @param {string} label Label key.
 */
function load(label) {
  const entry = current.labels.get(label);
  const id = current.summary.id;
  entry.loading = true;
  updateMessage();
  loader.load(
    meshUrl(id, label, FORMAT),
    (gltf) => {
      entry.loading = false;
      if (current?.summary.id !== id) {
        dispose(gltf.scene);
        return;
      }
      gltf.scene.traverse((child) => {
        if (!child.isMesh) return;
        child.material.dispose();
        child.material = material(label);
        if (TRANSLUCENT[label] !== undefined) child.renderOrder = TRANSLUCENT_ORDER;
      });
      releaseAfterUpload(gltf.scene);
      entry.object = gltf.scene;
      entry.object.visible = entry.checkbox.checked;
      scene.add(entry.object);
      entry.status.textContent = `${(triangles(entry.object) / MEGA).toFixed(2)} M triangles`;
      if (!fitted) fitView();
      updateMessage();
      render();
    },
    (event) => {
      if (event.lengthComputable) {
        entry.status.textContent = `Loading ${percent(event.loaded / event.total)} of ${megabytes(event.total)}`;
      }
    },
    (err) => {
      entry.loading = false;
      entry.status.textContent = 'Could not load';
      toast(`The ${label} mesh could not be loaded: ${err.message ?? err}`, { error: true });
      updateMessage();
    },
  );
}

/**
 * Have the server build a label's mesh (or join the job already doing it); onJob() loads it.
 * @param {string} label Label key.
 */
async function build(label) {
  const entry = current.labels.get(label);
  const id = current.summary.id;
  entry.loading = true;
  entry.status.textContent = 'Building the mesh…';
  updateMessage();
  let job = activeJob(id, label, FORMAT);
  if (!job) {
    try {
      job = await api.buildMesh(id, label, FORMAT);
    } catch (err) {
      entry.loading = false;
      entry.status.textContent = 'Could not build';
      toast(`The ${label} mesh could not be built: ${err.message}`, { error: true });
      updateMessage();
      return;
    }
  }
  void track(job);
}

/**
 * Show or hide a label: hidden meshes stay loaded; a mesh not loaded yet is loaded, or built.
 * @param {string} label Label key.
 * @param {boolean} visible Whether to show it.
 */
function show(label, visible) {
  const entry = current.labels.get(label);
  if (entry.object) {
    entry.object.visible = visible;
    updateMessage();
    render();
  } else if (visible && !entry.loading) {
    const built = (current.summary.meshes[label] ?? []).includes(FORMAT);
    if (built) load(label);
    else void build(label);
  }
}

/**
 * Follow the mesh jobs of the shown segmentation: progress next to the label, load on success.
 * @param {CustomEvent} event A JOB_UPDATED or JOB_FINISHED event (detail: the job status).
 * @param {boolean} finished Whether the job is over.
 */
function onJob(event, finished) {
  const status = event.detail;
  if (status.kind !== 'mesh' || status.mesh_format !== FORMAT || current?.summary.id !== status.segmentation) return;
  const entry = current.labels.get(status.label);
  if (!entry) return;
  if (!finished) {
    const p = status.progress;
    entry.status.textContent = status.state === 'queued' ? 'Waiting for a worker…' : `Building · ${duration(p?.elapsed_s ?? 0)}`;
    return;
  }
  entry.loading = false;
  if (status.state === 'succeeded') {
    current.summary.meshes[status.label] = [...new Set([...(current.summary.meshes[status.label] ?? []), FORMAT])];
    if (entry.object) {
      scene.remove(entry.object);
      dispose(entry.object);
      entry.object = null;
    }
    if (entry.checkbox.checked) load(status.label);
    else entry.status.textContent = 'Ready';
  } else {
    entry.status.textContent = status.state === 'cancelled' ? 'Cancelled' : 'Build failed';
    updateMessage();
  }
}

/** Remove every mesh of the shown segmentation from the scene and the GPU. */
function clear() {
  for (const entry of current?.labels.values() ?? []) {
    if (entry.object) {
      scene.remove(entry.object);
      dispose(entry.object);
    }
  }
  fitted = false;
}

/**
 * Show the meshes of a segmentation: one toggle per label, the default ones loaded.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 */
export function showSegmentation(summary) {
  if (current?.summary.id === summary.id) {
    current.summary = summary;
    return;
  }
  clear();
  const controlsNode = document.getElementById('mesh-controls');
  const labels = new Map();
  const toggles = LABELS.map((label) => {
    const checkbox = el('input', { type: 'checkbox', id: `show-${label}`, checked: SHOWN_BY_DEFAULT.has(label) });
    const status = el('span', { class: 'job-detail' });
    const built = (summary.meshes[label] ?? []).includes(FORMAT);
    status.textContent = built ? 'Ready' : 'Not built yet';
    labels.set(label, { checkbox, status, object: null, loading: false });
    checkbox.addEventListener('change', () => show(label, checkbox.checked));
    const swatch = el('span', { class: 'swatch', style: `background: ${cssColor(colors[label])}` });
    return el('div', { class: 'mesh-toggle' }, el('label', { for: `show-${label}` }, checkbox, swatch, label), status);
  });
  controlsNode.replaceChildren(...toggles, viewActions());
  current = { summary, labels };
  for (const label of LABELS) {
    if (labels.get(label).checkbox.checked) show(label, true);
  }
  updateMessage();
  render();
}

/**
 * Set up the scene once: renderer, camera, lights, controls and listeners.
 * @param {Object<string, number[]>} configColors BGR colour per label.
 * @returns {boolean} False if this browser cannot draw WebGL.
 */
export function initViewer(configColors) {
  colors = configColors;
  const container = document.getElementById('scene');
  message = document.getElementById('scene-message');
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch {
    message.textContent = 'This browser cannot draw 3D (WebGL is unavailable or disabled).';
    return false;
  }
  renderer.setPixelRatio(window.devicePixelRatio);
  container.prepend(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW, 1, 1, 10);
  camera.up.copy(UP); // before the controls: they read it once, when created
  scene.add(camera);
  const sky = new THREE.HemisphereLight(0xffffff, 0x404040, SKY_LIGHT);
  sky.position.set(0, -1, 0); // image rows go down: "up" is -y
  scene.add(sky);
  // A headlight: it rides with the camera and shines where the camera looks (camera -z).
  const key = new THREE.DirectionalLight(0xffffff, KEY_LIGHT);
  camera.add(key);
  camera.add(key.target);
  key.target.position.set(0, 0, -1);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.zoomToCursor = true; // the wheel zooms towards the point under the mouse
  // Arrow keys move the view once the scene has the focus (click it, or Tab to it).
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', '3D view: drag, wheel or arrow keys to explore');
  controls.listenToKeyEvents(renderer.domElement);
  controls.addEventListener('change', render);
  sceneBackground();
  new ResizeObserver(resize).observe(container);
  document.addEventListener(THEME_CHANGED, () => { sceneBackground(); render(); });
  document.addEventListener(JOB_UPDATED, (event) => onJob(event, false));
  document.addEventListener(JOB_FINISHED, (event) => onJob(event, true));
  return true;
}

// Drawing the crop: image, labels overlay, cursor; zoom, pan, display filters and the z offset.

import {
  $, allLabels, canvas, ctx, element, FILL_NEIGHBOURS, NEUTRAL_FILTER, octx, overlay, PERCENT, RGBA,
  state, TRANSPARENT,
} from './state.js';

const CURSOR_COLOR = 'rgba(255, 255, 255, 0.85)';
const PIXEL_CENTRE = 0.5;

/**
 * Tell whether a pixel touches another label (4-neighbours inside the crop).
 * @param {number} i Pixel index, row-major.
 * @param {number} size Crop side.
 * @returns {boolean} Whether it is on a border between labels.
 */
function onBorder(i, size) {
  const x = i % size;
  const y = (i - x) / size;
  const value = state.labels[i];
  for (const [dx, dy] of FILL_NEIGHBOURS) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx >= 0 && ny >= 0 && nx < size && ny < size && state.labels[ny * size + nx] !== value) return true;
  }
  return false;
}

/** Show the share of the crop each label covers. */
function updateCounts() {
  const counts = new Map();
  for (const v of state.labels) counts.set(v, (counts.get(v) ?? 0) + 1);
  const total = state.labels.length;
  $('counts').replaceChildren(...allLabels().flatMap((l) => [
    element('dt', null, l.key),
    element('dd', null, `${(((counts.get(l.value) ?? 0) / total) * PERCENT).toFixed(1)}%`),
  ]));
}

/** Draw the crop: the image at the current z offset, the labels, the brush cursor. */
export function render() {
  if (!state.labels) return;
  const scale = state.session.display.scale;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(state.zoom, 0, 0, state.zoom, state.pan[0], state.pan[1]);
  ctx.filter = `brightness(${state.brightness}) contrast(${state.contrast})`;
  ctx.drawImage(state.images.get(state.zOffset), 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none';
  if (!state.overlayHidden) {
    ctx.globalAlpha = state.alpha;
    ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
  }
  if (state.hover) {
    const [x, y] = state.hover;
    ctx.strokeStyle = CURSOR_COLOR;
    ctx.lineWidth = 1 / state.zoom;
    ctx.beginPath();
    if (state.tool === 'brush') {
      const r = (state.radius + PIXEL_CENTRE) * scale;
      ctx.arc((x + PIXEL_CENTRE) * scale, (y + PIXEL_CENTRE) * scale, r, 0, 2 * Math.PI);
    } else {
      ctx.rect(x * scale, y * scale, scale, scale);
    }
    ctx.stroke();
  }
}

/** Rebuild the labels overlay from the label array (after any edit), then redraw. */
export function refreshOverlay() {
  const size = state.session.crop_size;
  const image = octx.createImageData(size, size);
  const { data } = image;
  const { lut, labels } = state;
  for (let i = 0; i < labels.length; i++) {
    const src = labels[i] * RGBA;
    const dst = i * RGBA;
    data[dst] = lut[src];
    data[dst + 1] = lut[src + 1];
    data[dst + 2] = lut[src + 2];
    data[dst + 3] = state.outline && !onBorder(i, size) ? TRANSPARENT : lut[src + 3];
  }
  octx.putImageData(image, 0, 0);
  updateCounts();
  render();
}

/**
 * Set a display filter of the image.
 * @param {string} name `brightness` or `contrast` (also the ids of their sliders).
 * @param {number} value Multiplier (1: unchanged).
 */
export function setFilter(name, value) {
  state[name] = value;
  $(name).value = Math.round(value * PERCENT);
  $(`${name}-out`).textContent = `${Math.round(value * PERCENT)}%`;
  render();
}

/** Show the whole crop again (no zoom, no pan). */
export function resetZoom() {
  state.zoom = 1;
  state.pan = [0, 0];
  $('zoom-out').textContent = '';
}

/** Reset the zoom and the display filters. */
export function resetView() {
  resetZoom();
  setFilter('brightness', NEUTRAL_FILTER);
  setFilter('contrast', NEUTRAL_FILTER);
}

/** Keep the image covering the canvas: no empty band at a border. */
export function clampPan() {
  const lowX = canvas.width * (1 - state.zoom);
  const lowY = canvas.height * (1 - state.zoom);
  state.pan = [Math.min(0, Math.max(lowX, state.pan[0])), Math.min(0, Math.max(lowY, state.pan[1]))];
}

/**
 * Convert a mouse position to canvas pixels.
 * @param {MouseEvent} event Mouse event.
 * @returns {number[]} `[x, y]` in canvas pixels.
 */
export function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * canvas.width,
    ((event.clientY - rect.top) / rect.height) * canvas.height,
  ];
}

/**
 * Zoom around the mouse.
 * @param {MouseEvent} event Mouse event (its position stays put).
 * @param {number} factor Above 1: closer; below 1: farther.
 */
export function zoomAt(event, factor) {
  const display = state.session.display;
  const zoom = Math.max(1, Math.min(display.max_zoom, state.zoom * factor));
  const [cx, cy] = canvasPoint(event);
  const ratio = zoom / state.zoom;
  state.pan = [cx - (cx - state.pan[0]) * ratio, cy - (cy - state.pan[1]) * ratio];
  state.zoom = zoom;
  clampPan();
  $('zoom-out').textContent = zoom > 1 ? `zoom ×${zoom.toFixed(1)}` : '';
  render();
}

/** Show the current z offset on its slider and in the context notice. */
export function syncZ() {
  const z = state.zOffset;
  $('zoffset').value = z;
  $('zoffset-out').textContent = z > 0 ? `+${z}` : String(z);
  $('context-notice').hidden = z === 0;
  $('context-offset').textContent = `z ${z > 0 ? '+' : ''}${z}`;
}

/**
 * Show a neighbouring slice (labels are painted on the centre one only).
 * @param {number} offset Slices above (positive) or below the centre, clamped to the context.
 */
export function setZOffset(offset) {
  const limit = state.session.context_slices;
  state.zOffset = Math.max(-limit, Math.min(limit, offset));
  syncZ();
  render();
}

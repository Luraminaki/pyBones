// Painting the labels: brush strokes, bucket and wand fills, undo; the canvas mouse handling.

import { setDirty } from './session.js';
import { canvas, FILL_NEIGHBOURS, state } from './state.js';
import { canvasPoint, clampPan, refreshOverlay, render, zoomAt } from './view.js';

const MOUSE_LEFT = 0;
const MOUSE_MIDDLE = 1;
const MOUSE_RIGHT = 2;

/**
 * Find the image pixel under the mouse.
 * @param {MouseEvent} event Mouse event.
 * @returns {?number[]} `[x, y]` in image pixels, or `null` outside the crop.
 */
function pixelAt(event) {
  const size = state.session.crop_size;
  const pixel = canvas.width / size; // canvas pixels per image pixel at zoom 1
  const [cx, cy] = canvasPoint(event);
  const x = Math.floor((cx - state.pan[0]) / state.zoom / pixel);
  const y = Math.floor((cy - state.pan[1]) / state.zoom / pixel);
  return x >= 0 && y >= 0 && x < size && y < size ? [x, y] : null;
}

/**
 * Tell whether a pixel may be repainted ("paint over" restricts it to one label).
 * @param {number} i Pixel index, row-major.
 * @returns {boolean} Whether painting changes it.
 */
function paintable(i) {
  return state.paintOver === null || state.labels[i] === state.paintOver;
}

/** Remember the labels before an edit (bounded history); a new edit forgets the redo list. */
function pushUndo() {
  state.undo.push(state.labels.slice());
  if (state.undo.length > state.session.display.undo_depth) state.undo.shift();
  state.redo = [];
}

/**
 * Paint a disc of the brush's radius.
 * @param {number} cx Centre column.
 * @param {number} cy Centre row.
 * @param {number} value Label value.
 */
function paintDisc(cx, cy, value) {
  const size = state.session.crop_size;
  const r = state.radius;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = cx + dx;
      const y = cy + dy;
      if (dx * dx + dy * dy <= r * r && x >= 0 && y >= 0 && x < size && y < size && paintable(y * size + x)) {
        state.labels[y * size + x] = value;
      }
    }
  }
}

/**
 * Paint discs along a segment, so that a fast stroke leaves no gaps.
 * @param {number[]} from `[x, y]` start.
 * @param {number[]} to `[x, y]` end.
 * @param {number} value Label value.
 */
function paintLine([x0, y0], [x1, y1], value) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= steps; i++) {
    paintDisc(Math.round(x0 + ((x1 - x0) * i) / steps), Math.round(y0 + ((y1 - y0) * i) / steps), value);
  }
}

/**
 * Fill the 4-connected region of one label (bucket).
 * @param {number[]} seed `[x, y]` clicked pixel.
 * @param {number} value Label value.
 */
function floodFill([x, y], value) {
  const size = state.session.crop_size;
  const labels = state.labels;
  const target = labels[y * size + x];
  if (target === value || !paintable(y * size + x)) return;
  const stack = [[x, y]];
  labels[y * size + x] = value;
  while (stack.length) {
    const [px, py] = stack.pop();
    for (const [dx, dy] of FILL_NEIGHBOURS) {
      const nx = px + dx;
      const ny = py + dy;
      if (nx >= 0 && ny >= 0 && nx < size && ny < size && labels[ny * size + nx] === target) {
        labels[ny * size + nx] = value;
        stack.push([nx, ny]);
      }
    }
  }
}

/**
 * Fill the 4-connected pixels whose grey level is within the tolerance of the clicked one (wand).
 * @param {number[]} seed `[x, y]` clicked pixel.
 * @param {number} value Label value.
 */
function wandFill([x, y], value) {
  const size = state.session.crop_size;
  const { gray, labels } = state;
  const seed = y * size + x;
  const low = gray[seed] - state.wandTolerance;
  const high = gray[seed] + state.wandTolerance;
  const visited = new Uint8Array(size * size);
  const stack = [seed];
  visited[seed] = 1;
  while (stack.length) {
    const i = stack.pop();
    if (paintable(i)) labels[i] = value;
    const px = i % size;
    const py = (i - px) / size;
    for (const [dx, dy] of FILL_NEIGHBOURS) {
      const nx = px + dx;
      const ny = py + dy;
      const n = ny * size + nx;
      if (nx >= 0 && ny >= 0 && nx < size && ny < size && !visited[n] && gray[n] >= low && gray[n] <= high) {
        visited[n] = 1;
        stack.push(n);
      }
    }
  }
}

/**
 * Step through the edit history.
 * @param {Uint8Array[]} from History to take the labels from (undo or redo).
 * @param {Uint8Array[]} to History to keep the current labels in.
 */
export function undo(from, to) {
  if (!from.length) return;
  to.push(state.labels);
  state.labels = from.pop();
  setDirty(true);
  refreshOverlay();
}

/**
 * Start painting (or panning) where the mouse was pressed.
 * @param {MouseEvent} event The press.
 */
function onPress(event) {
  if (event.button === MOUSE_MIDDLE || (event.button === MOUSE_LEFT && event.altKey)) {
    event.preventDefault();
    state.panning = { from: [event.clientX, event.clientY], pan: [...state.pan] };
    return;
  }
  const p = pixelAt(event);
  if (!p || ![MOUSE_LEFT, MOUSE_RIGHT].includes(event.button) || state.zOffset !== 0) return;
  const value = event.button === MOUSE_RIGHT ? state.session.unsure.value : state.active;
  pushUndo();
  if (state.tool === 'fill') {
    floodFill(p, value);
  } else if (state.tool === 'wand') {
    wandFill(p, value);
  } else {
    paintDisc(p[0], p[1], value);
    state.stroke = { value, last: p };
  }
  setDirty(true);
  refreshOverlay();
}

/**
 * Pan, continue a stroke, or move the cursor.
 * @param {MouseEvent} event The move.
 */
function onMove(event) {
  if (state.panning) {
    const rect = canvas.getBoundingClientRect();
    const k = canvas.width / rect.width;
    const { from, pan } = state.panning;
    state.pan = [pan[0] + (event.clientX - from[0]) * k, pan[1] + (event.clientY - from[1]) * k];
    clampPan();
    render();
    return;
  }
  state.hover = pixelAt(event);
  if (state.stroke && state.hover) {
    paintLine(state.stroke.last, state.hover, state.stroke.value);
    state.stroke.last = state.hover;
    refreshOverlay();
  } else {
    render();
  }
}

/** Wire the canvas: paint with the buttons, pan with the middle one, zoom with the wheel. */
export function initPainting() {
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const step = state.session.display.zoom_step;
    zoomAt(event, event.deltaY < 0 ? step : 1 / step);
  }, { passive: false });
  canvas.addEventListener('mousedown', onPress);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseleave', () => {
    state.hover = null;
    render();
  });
  window.addEventListener('mouseup', () => {
    state.stroke = null;
    state.panning = null;
  });
}

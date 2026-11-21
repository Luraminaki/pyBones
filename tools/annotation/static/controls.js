// The controls: label palette, "paint over", tools, sliders, and the keyboard shortcuts.

import { undo } from './paint.js';
import { save, setDirty, showCrop, updateNoteCounter } from './session.js';
import { $, allLabels, element, PERCENT, state, UNSURE_HOTKEY, UNSURE_KEY } from './state.js';
import { refreshOverlay, render, resetView, setFilter, setZOffset } from './view.js';

/**
 * Choose the label the left button paints.
 * @param {number} value Label value.
 */
function selectLabel(value) {
  state.active = value;
  for (const b of $('palette').children) b.classList.toggle('active', Number(b.dataset.value) === value);
}

/** Build the palette: one button per label, with its colour and hotkey. */
function buildPalette() {
  $('palette').replaceChildren(...allLabels().map((l, i) => {
    const hotkey = l.key === UNSURE_KEY ? UNSURE_HOTKEY.toUpperCase() : String(i);
    const swatch = element('span', l.rgb ? 'swatch' : 'swatch none');
    if (l.rgb) swatch.style.background = `rgb(${l.rgb.join(',')})`;
    const button = element('button', null, swatch, l.key, element('kbd', 'key', hotkey));
    button.dataset.value = l.value;
    button.addEventListener('click', () => selectLabel(l.value));
    return button;
  }));
  selectLabel(state.active);
}

/**
 * Restrict painting to the pixels of one label.
 * @param {?number} value Label value (`null`: every label).
 */
function setPaintOver(value) {
  state.paintOver = value;
  $('paint-over').value = value === null ? '' : String(value);
}

/** Offer "paint over everything" or over one label. */
function buildPaintOver() {
  const select = $('paint-over');
  select.replaceChildren(new Option('everything', ''));
  for (const l of allLabels()) select.append(new Option(`${l.key} only`, String(l.value)));
  select.addEventListener('change', () => setPaintOver(select.value === '' ? null : Number(select.value)));
}

/** Move "paint over" to the next choice (keyboard). */
function cyclePaintOver() {
  const values = [null, ...allLabels().map((l) => l.value)];
  setPaintOver(values[(values.indexOf(state.paintOver) + 1) % values.length]);
}

/**
 * Choose the tool.
 * @param {string} tool `brush`, `fill` or `wand`.
 */
function selectTool(tool) {
  state.tool = tool;
  for (const b of document.querySelectorAll('[data-tool]')) b.classList.toggle('active', b.dataset.tool === tool);
}

/**
 * Set the brush radius (clamped to the settings' limit).
 * @param {number} radius Radius in pixels.
 */
function setRadius(radius) {
  state.radius = Math.max(0, Math.min(state.session.display.max_brush_radius, radius));
  $('brush').value = state.radius;
  $('brush-out').textContent = state.radius;
  render();
}

/**
 * Set how far from the clicked grey level the wand reaches.
 * @param {number} tolerance Grey levels either side (clamped to the settings' limit).
 */
function setWandTolerance(tolerance) {
  state.wandTolerance = Math.max(0, Math.min(state.session.display.max_wand_tolerance, tolerance));
  $('wand').value = state.wandTolerance;
  $('wand-out').textContent = `±${state.wandTolerance}`;
}

/**
 * Draw the labels whole, or only their borders.
 * @param {boolean} outline Whether to draw only the borders.
 */
function setOutline(outline) {
  state.outline = outline;
  $('outline').checked = outline;
  refreshOverlay();
}

/**
 * Set the opacity of the labels over the image.
 * @param {number} alpha Opacity in [0, 1].
 */
function setAlpha(alpha) {
  state.alpha = alpha;
  $('alpha').value = Math.round(alpha * PERCENT);
  $('alpha-out').textContent = `${Math.round(alpha * PERCENT)}%`;
  render();
}

/**
 * Act on a key: labels, tools, view, history, saving, crops.
 * @param {KeyboardEvent} event The key press.
 */
function onKey(event) {
  if ($('help').open || event.target === $('note')) return;
  const key = event.key.toLowerCase();
  if (event.ctrlKey || event.metaKey) {
    if (key === 's') void save();
    else if (key === 'z' && event.shiftKey) undo(state.redo, state.undo);
    else if (key === 'z') undo(state.undo, state.redo);
    else if (key === 'y') undo(state.redo, state.undo);
    else return;
    event.preventDefault();
    return;
  }
  const labels = state.session.labels;
  if (/^[0-9]$/.test(key) && Number(key) < labels.length) selectLabel(labels[Number(key)].value);
  else if (key === UNSURE_HOTKEY) selectLabel(state.session.unsure.value);
  else if (key === 'b') selectTool('brush');
  else if (key === 'l') cyclePaintOver();
  else if (key === 'r') resetView();
  else if (key === 'f') selectTool('fill');
  else if (key === 'w') selectTool('wand');
  else if (key === 'o') setOutline(!state.outline);
  else if (key === '[') setRadius(state.radius - 1);
  else if (key === ']') setRadius(state.radius + 1);
  else if (key === 'arrowup') setZOffset(state.zOffset + 1);
  else if (key === 'arrowdown') setZOffset(state.zOffset - 1);
  else if (key === 'home') setZOffset(0);
  else if (key === 'arrowleft') void showCrop(state.index - 1);
  else if (key === 'arrowright') void showCrop(state.index + 1);
  else if (key === 'h' || key === '?') $('help').showModal();
  else if (key === ' ') {
    state.overlayHidden = true;
    render();
  } else return;
  event.preventDefault();
}

/** Build every control from the session's settings and wire them, with the keyboard. */
export function initControls() {
  buildPalette();
  buildPaintOver();
  const limit = state.session.context_slices;
  Object.assign($('zoffset'), { min: -limit, max: limit, value: 0 });
  $('zoffset').addEventListener('input', (e) => setZOffset(Number(e.target.value)));
  $('brush').max = state.session.display.max_brush_radius;
  $('brush').addEventListener('input', (e) => setRadius(Number(e.target.value)));
  $('alpha').addEventListener('input', (e) => setAlpha(Number(e.target.value) / PERCENT));
  for (const b of document.querySelectorAll('[data-tool]')) b.addEventListener('click', () => selectTool(b.dataset.tool));
  $('wand').max = state.session.display.max_wand_tolerance;
  $('wand').addEventListener('input', (e) => setWandTolerance(Number(e.target.value)));
  $('outline').addEventListener('change', (e) => setOutline(e.target.checked));
  $('note').maxLength = state.session.max_note_chars;
  $('note').addEventListener('input', () => {
    state.noteDirty = true;
    updateNoteCounter();
    setDirty(state.dirty, 'unsaved note');
  });
  $('brightness').addEventListener('input', (e) => setFilter('brightness', Number(e.target.value) / PERCENT));
  $('contrast').addEventListener('input', (e) => setFilter('contrast', Number(e.target.value) / PERCENT));
  $('view-reset').addEventListener('click', resetView);
  $('save').addEventListener('click', () => void save());
  $('help-open').addEventListener('click', () => $('help').showModal());
  setRadius(state.radius);
  setWandTolerance(state.wandTolerance);
  setAlpha(state.alpha);
  selectTool(state.tool);
  resetView();
  window.addEventListener('keydown', onKey);
  window.addEventListener('keyup', (event) => {
    if (event.key === ' ') {
      state.overlayHidden = false;
      render();
    }
  });
}

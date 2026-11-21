// The crops of the workspace: opening one, its note, saving (by hand or after a delay).

import { $, element, fetchOk, reportError, RGBA, state } from './state.js';
import { refreshOverlay, resetZoom, syncZ } from './view.js';

const MS_PER_S = 1000;
const TEXT_TYPE = 'text/plain; charset=utf-8';
const LABELS_TYPE = 'application/octet-stream';

/**
 * Load an image (the crop's slices are PNGs served by the annotation server).
 * @param {string} url Address.
 * @returns {Promise<HTMLImageElement>} The loaded image.
 * @throws {Error} If it cannot be loaded.
 */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`cannot load ${url}`));
    img.src = url;
  });
}

/**
 * Read the grey levels of a crop image (8-bit grey: any channel is the grey level), for the wand.
 * @param {HTMLImageElement} image Centre slice of the crop.
 * @returns {Uint8Array} One level per pixel, row-major.
 */
function grayLevels(image) {
  const size = state.session.crop_size;
  const scratch = document.createElement('canvas');
  scratch.width = scratch.height = size;
  const sctx = scratch.getContext('2d');
  sctx.drawImage(image, 0, 0, size, size);
  const { data } = sctx.getImageData(0, 0, size, size);
  const gray = new Uint8Array(size * size);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * RGBA];
  return gray;
}

/** Show how much of the note's length limit is used. */
export function updateNoteCounter() {
  $('note-out').textContent = `${$('note').value.length}/${state.session.max_note_chars}`;
}

/**
 * Show a crop's note.
 * @param {string} text The note.
 */
function setNote(text) {
  $('note').value = text;
  state.noteDirty = false;
  updateNoteCounter();
}

/**
 * Start the autosave countdown if something is unsaved (the delay comes from the settings).
 */
function scheduleAutosave() {
  const delay = state.session.display.autosave_seconds;
  clearTimeout(state.autosave);
  const pending = state.dirty || state.noteDirty;
  state.autosave = pending && delay > 0 ? setTimeout(save, delay * MS_PER_S) : null;
}

/**
 * Record whether the labels have unsaved changes, and show it.
 * @param {boolean} dirty Whether they do.
 * @param {string} [text] What to show (default: "unsaved changes" or "saved").
 */
export function setDirty(dirty, text) {
  state.dirty = dirty;
  const node = $('save-state');
  node.textContent = text ?? (dirty ? 'unsaved changes' : 'saved');
  node.classList.toggle('dirty', dirty);
  node.classList.toggle('saved', !dirty && text === 'saved');
  scheduleAutosave();
}

/** List the crops: saved ones marked, the open one highlighted; a click opens one. */
export function buildCropList() {
  $('crop-list').replaceChildren(...state.session.crops.map((crop, i) => {
    const id = element('span', 'id', crop.crop_id);
    id.classList.toggle('noted', Boolean(crop.note));
    const item = element('li', null, element('span', 'dot'), id, element('span', 'cat', crop.category));
    item.classList.toggle('saved', crop.saved);
    item.classList.toggle('current', i === state.index);
    item.addEventListener('click', () => void showCrop(i));
    return item;
  }));
  const done = state.session.crops.filter((c) => c.saved).length;
  $('progress').textContent = `${done}/${state.session.crops.length}`;
}

/**
 * Save the note of a crop.
 * @param {{crop_id: string, note: boolean}} crop The crop.
 * @returns {Promise<boolean>} Whether it was saved.
 */
async function saveNote(crop) {
  try {
    await fetchOk(`/api/crops/${crop.crop_id}/note`, {
      method: 'PUT',
      headers: { 'Content-Type': TEXT_TYPE },
      body: $('note').value,
    });
  } catch (err) {
    setDirty(true, `note not saved: ${err.message}`);
    return false;
  }
  state.noteDirty = false;
  crop.note = $('note').value.trim().length > 0;
  return true;
}

/** Save the open crop's note and labels (labels only if they were edited). */
export async function save() {
  clearTimeout(state.autosave);
  state.autosave = null;
  if (state.index < 0) return;
  const crop = state.session.crops[state.index];
  if (state.noteDirty && !(await saveNote(crop))) return;
  if (!state.dirty) {
    // Only the note changed: never save untouched (e.g. pre-filled) labels as an annotation.
    setDirty(false, crop.saved ? 'saved' : 'note saved');
    buildCropList();
    return;
  }
  try {
    await fetchOk(`/api/crops/${crop.crop_id}/labels`, {
      method: 'PUT',
      headers: { 'Content-Type': LABELS_TYPE },
      body: state.labels,
    });
  } catch (err) {
    setDirty(true, `save failed: ${err.message}`);
    return;
  }
  crop.saved = true;
  setDirty(false, 'saved');
  buildCropList();
}

/**
 * Open a crop: its slices, labels and note (the open one is saved first if needed).
 * @param {number} index Crop index (ignored outside the list).
 * @throws {Error} If the server cannot provide the crop.
 */
export async function openCrop(index) {
  if (index < 0 || index >= state.session.crops.length) return;
  if (state.dirty || state.noteDirty) await save();
  const crop = state.session.crops[index];
  const context = state.session.context_slices;
  const offsets = Array.from({ length: 2 * context + 1 }, (_, i) => i - context);
  const [images, buffer, note] = await Promise.all([
    Promise.all(offsets.map((o) => loadImage(`/api/crops/${crop.crop_id}/image/${o}`))),
    fetchOk(`/api/crops/${crop.crop_id}/labels`).then((r) => r.arrayBuffer()),
    fetchOk(`/api/crops/${crop.crop_id}/note`).then((r) => r.text()),
  ]);
  state.index = index;
  state.images = new Map(offsets.map((o, i) => [o, images[i]]));
  state.labels = new Uint8Array(buffer);
  state.gray = grayLevels(images[offsets.indexOf(0)]);
  setNote(note);
  state.undo = [];
  state.redo = [];
  state.zOffset = 0;
  resetZoom();
  setDirty(false, crop.saved ? 'saved' : 'not saved');
  $('crop-title').textContent = `${crop.crop_id} · z ${crop.z} · ${crop.category}`;
  syncZ();
  buildCropList();
  refreshOverlay();
}

/**
 * Open a crop from the interface, reporting a failure instead of losing it.
 * @param {number} index Crop index.
 */
export async function showCrop(index) {
  try {
    await openCrop(index);
  } catch (err) {
    reportError('opening the crop', err);
  }
}

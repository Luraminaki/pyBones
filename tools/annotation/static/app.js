"use strict";

// UI constants. Data-dependent values (labels, colours, crop size, scale...) come from
// /api/session, which reads tools/annotation/config.json and the pybones configuration.
const UNSURE_KEY = "unsure";
const UNSURE_HOTKEY = "u";
const DEFAULT_LABEL_KEY = "vessel";
const PERCENT = 100;
const RGBA = 4;
const OPAQUE = 255;
const MOUSE_LEFT = 0;
const MOUSE_MIDDLE = 1;
const MOUSE_RIGHT = 2;
const MS_PER_S = 1000;
const NEUTRAL_FILTER = 1; // brightness and contrast multipliers that leave the image unchanged
const TRANSPARENT = 0;
const TEXT_TYPE = "text/plain; charset=utf-8";
const FILL_NEIGHBOURS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const CURSOR_COLOR = "rgba(255, 255, 255, 0.85)";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.createElement("canvas");
const octx = overlay.getContext("2d");

const state = {
  session: null,
  index: -1,
  labels: null, // Uint8Array(size * size), row-major
  images: new Map(), // z offset -> HTMLImageElement
  undo: [],
  redo: [],
  active: null, // label value painted with the left button
  tool: "brush",
  radius: 2,
  zOffset: 0,
  alpha: 0.45,
  overlayHidden: false,
  dirty: false,
  stroke: null, // {value, last: [x, y]} while painting
  hover: null, // [x, y] in image pixels
  lut: new Uint8ClampedArray(256 * RGBA),
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

// ---------------------------------------------------------------- session / crops

async function init() {
  const session = await (await fetch("/api/session")).json();
  state.session = session;
  const { crop_size: size, display } = session;
  canvas.width = canvas.height = size * display.scale;
  overlay.width = overlay.height = size;
  ctx.imageSmoothingEnabled = false;
  state.alpha = display.overlay_alpha;
  state.wandTolerance = display.wand_tolerance;
  state.radius = Math.min(state.radius, display.max_brush_radius);
  state.active = session.labels.find((l) => l.key === DEFAULT_LABEL_KEY).value;
  for (const l of session.labels) {
    if (l.rgb) state.lut.set([...l.rgb, OPAQUE], l.value * RGBA);
  }
  state.lut.set([...session.unsure.rgb, OPAQUE], session.unsure.value * RGBA);
  $("prefill-badge").hidden = !session.prefilled;
  buildPalette();
  buildPaintOver();
  buildControls();
  buildCropList();
  await openCrop(Math.max(0, session.crops.findIndex((c) => !c.saved)));
}

function buildCropList() {
  const list = $("crop-list");
  list.replaceChildren();
  state.session.crops.forEach((crop, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot"></span><span class="id">${crop.crop_id}</span><span class="cat">${crop.category}</span>`;
    li.classList.toggle("saved", crop.saved);
    li.querySelector(".id").classList.toggle("noted", Boolean(crop.note));
    li.classList.toggle("current", i === state.index);
    li.addEventListener("click", () => openCrop(i));
    list.append(li);
  });
  const done = state.session.crops.filter((c) => c.saved).length;
  $("progress").textContent = `${done}/${state.session.crops.length}`;
}

async function openCrop(index) {
  if (index < 0 || index >= state.session.crops.length) return;
  if (state.dirty || state.noteDirty) await save();
  const crop = state.session.crops[index];
  const ctxSlices = state.session.context_slices;
  const offsets = Array.from({ length: 2 * ctxSlices + 1 }, (_, i) => i - ctxSlices);
  const images = await Promise.all(offsets.map((o) => loadImage(`/api/crops/${crop.crop_id}/image/${o}`)));
  const buffer = await (await fetch(`/api/crops/${crop.crop_id}/labels`)).arrayBuffer();
  const note = await (await fetch(`/api/crops/${crop.crop_id}/note`)).text();
  state.index = index;
  state.images = new Map(offsets.map((o, i) => [o, images[i]]));
  state.labels = new Uint8Array(buffer);
  state.gray = grayLevels(images[offsets.indexOf(0)]);
  setNote(note);
  state.undo = [];
  state.redo = [];
  state.zOffset = 0;
  resetZoom();
  setDirty(false, crop.saved ? "saved" : "not saved");
  $("crop-title").textContent = `${crop.crop_id} · z ${crop.z} · ${crop.category}`;
  syncZ();
  buildCropList();
  refreshOverlay();
}

function grayLevels(image) {
  // The crop images are 8-bit grey: any channel of the decoded pixels is the grey level.
  const size = state.session.crop_size;
  const scratch = document.createElement("canvas");
  scratch.width = scratch.height = size;
  const sctx = scratch.getContext("2d");
  sctx.drawImage(image, 0, 0, size, size);
  const { data } = sctx.getImageData(0, 0, size, size);
  const gray = new Uint8Array(size * size);
  for (let i = 0; i < gray.length; i++) gray[i] = data[i * RGBA];
  return gray;
}

function setNote(text) {
  $("note").value = text;
  state.noteDirty = false;
  updateNoteCounter();
}

function updateNoteCounter() {
  $("note-out").textContent = `${$("note").value.length}/${state.session.max_note_chars}`;
}

async function saveNote(crop) {
  const response = await fetch(`/api/crops/${crop.crop_id}/note`, {
    method: "PUT",
    headers: { "Content-Type": TEXT_TYPE },
    body: $("note").value,
  });
  if (!response.ok) {
    setDirty(true, `note not saved: ${await response.text()}`);
    return false;
  }
  state.noteDirty = false;
  crop.note = $("note").value.trim().length > 0;
  return true;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`cannot load ${url}`));
    img.src = url;
  });
}

async function save() {
  clearTimeout(state.autosave);
  state.autosave = null;
  if (state.index < 0) return;
  const crop = state.session.crops[state.index];
  if (state.noteDirty && !(await saveNote(crop))) return;
  if (!state.dirty) {
    // Only the note changed: never save untouched (e.g. pre-filled) labels as an annotation.
    setDirty(false, crop.saved ? "saved" : "note saved");
    buildCropList();
    return;
  }
  const response = await fetch(`/api/crops/${crop.crop_id}/labels`, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: state.labels,
  });
  if (!response.ok) {
    setDirty(true, `save failed: ${await response.text()}`);
    return;
  }
  crop.saved = true;
  setDirty(false, "saved");
  buildCropList();
}

function setDirty(dirty, text) {
  state.dirty = dirty;
  const el = $("save-state");
  el.textContent = text ?? (dirty ? "unsaved changes" : "saved");
  el.classList.toggle("dirty", dirty);
  el.classList.toggle("saved", !dirty && text === "saved");
  scheduleAutosave();
}

function scheduleAutosave() {
  const delay = state.session.display.autosave_seconds;
  clearTimeout(state.autosave);
  const pending = state.dirty || state.noteDirty;
  state.autosave = pending && delay > 0 ? setTimeout(save, delay * MS_PER_S) : null;
}

// ---------------------------------------------------------------- controls

function allLabels() {
  const { labels, unsure } = state.session;
  return [...labels, { value: unsure.value, key: UNSURE_KEY, rgb: unsure.rgb }];
}

function buildPalette() {
  const palette = $("palette");
  palette.replaceChildren();
  allLabels().forEach((l, i) => {
    const button = document.createElement("button");
    const hotkey = l.key === UNSURE_KEY ? UNSURE_HOTKEY.toUpperCase() : String(i);
    const swatch = l.rgb
      ? `<span class="swatch" style="background: rgb(${l.rgb.join(",")})"></span>`
      : '<span class="swatch none"></span>';
    button.innerHTML = `${swatch}${l.key}<kbd class="key">${hotkey}</kbd>`;
    button.dataset.value = l.value;
    button.addEventListener("click", () => selectLabel(l.value));
    palette.append(button);
  });
  selectLabel(state.active);
}

function selectLabel(value) {
  state.active = value;
  for (const b of $("palette").children) b.classList.toggle("active", Number(b.dataset.value) === value);
}

function buildPaintOver() {
  const select = $("paint-over");
  select.replaceChildren(new Option("everything", ""));
  for (const l of allLabels()) select.append(new Option(`${l.key} only`, String(l.value)));
  select.addEventListener("change", () => setPaintOver(select.value === "" ? null : Number(select.value)));
}

function setPaintOver(value) {
  state.paintOver = value;
  $("paint-over").value = value === null ? "" : String(value);
}

function cyclePaintOver() {
  const values = [null, ...allLabels().map((l) => l.value)];
  setPaintOver(values[(values.indexOf(state.paintOver) + 1) % values.length]);
}

function selectTool(tool) {
  state.tool = tool;
  for (const b of document.querySelectorAll("[data-tool]")) b.classList.toggle("active", b.dataset.tool === tool);
}

function setRadius(radius) {
  state.radius = Math.max(0, Math.min(state.session.display.max_brush_radius, radius));
  $("brush").value = state.radius;
  $("brush-out").textContent = state.radius;
  render();
}

function setWandTolerance(tolerance) {
  state.wandTolerance = Math.max(0, Math.min(state.session.display.max_wand_tolerance, tolerance));
  $("wand").value = state.wandTolerance;
  $("wand-out").textContent = `±${state.wandTolerance}`;
}

function setOutline(outline) {
  state.outline = outline;
  $("outline").checked = outline;
  refreshOverlay();
}

function setAlpha(alpha) {
  state.alpha = alpha;
  $("alpha").value = Math.round(alpha * PERCENT);
  $("alpha-out").textContent = `${Math.round(alpha * PERCENT)}%`;
  render();
}

function setZOffset(offset) {
  const limit = state.session.context_slices;
  state.zOffset = Math.max(-limit, Math.min(limit, offset));
  syncZ();
  render();
}

function syncZ() {
  const z = state.zOffset;
  $("zoffset").value = z;
  $("zoffset-out").textContent = z > 0 ? `+${z}` : String(z);
  $("context-notice").hidden = z === 0;
  $("context-offset").textContent = `z ${z > 0 ? "+" : ""}${z}`;
}

function buildControls() {
  const limit = state.session.context_slices;
  Object.assign($("zoffset"), { min: -limit, max: limit, value: 0 });
  $("zoffset").addEventListener("input", (e) => setZOffset(Number(e.target.value)));
  $("brush").max = state.session.display.max_brush_radius;
  $("brush").addEventListener("input", (e) => setRadius(Number(e.target.value)));
  $("alpha").addEventListener("input", (e) => setAlpha(Number(e.target.value) / PERCENT));
  for (const b of document.querySelectorAll("[data-tool]")) b.addEventListener("click", () => selectTool(b.dataset.tool));
  $("wand").max = state.session.display.max_wand_tolerance;
  $("wand").addEventListener("input", (e) => setWandTolerance(Number(e.target.value)));
  $("outline").addEventListener("change", (e) => setOutline(e.target.checked));
  $("note").maxLength = state.session.max_note_chars;
  $("note").addEventListener("input", () => {
    state.noteDirty = true;
    updateNoteCounter();
    setDirty(state.dirty, "unsaved note");
  });
  $("brightness").addEventListener("input", (e) => setFilter("brightness", Number(e.target.value) / PERCENT));
  $("contrast").addEventListener("input", (e) => setFilter("contrast", Number(e.target.value) / PERCENT));
  $("view-reset").addEventListener("click", resetView);
  $("save").addEventListener("click", save);
  $("help-open").addEventListener("click", () => $("help").showModal());
  setRadius(state.radius);
  setWandTolerance(state.wandTolerance);
  setAlpha(state.alpha);
  selectTool(state.tool);
  resetView();
}

// ---------------------------------------------------------------- view (zoom, pan, display filters)

function setFilter(name, value) {
  state[name] = value;
  $(name).value = Math.round(value * PERCENT);
  $(`${name}-out`).textContent = `${Math.round(value * PERCENT)}%`;
  render();
}

function resetZoom() {
  state.zoom = 1;
  state.pan = [0, 0];
  $("zoom-out").textContent = "";
}

function resetView() {
  resetZoom();
  setFilter("brightness", NEUTRAL_FILTER);
  setFilter("contrast", NEUTRAL_FILTER);
}

function clampPan() {
  // Keep the image covering the canvas: no empty band at a border.
  const lowX = canvas.width * (1 - state.zoom);
  const lowY = canvas.height * (1 - state.zoom);
  state.pan = [Math.min(0, Math.max(lowX, state.pan[0])), Math.min(0, Math.max(lowY, state.pan[1]))];
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * canvas.width,
    ((event.clientY - rect.top) / rect.height) * canvas.height,
  ];
}

function zoomAt(event, factor) {
  const display = state.session.display;
  const zoom = Math.max(1, Math.min(display.max_zoom, state.zoom * factor));
  const [cx, cy] = canvasPoint(event);
  const ratio = zoom / state.zoom;
  state.pan = [cx - (cx - state.pan[0]) * ratio, cy - (cy - state.pan[1]) * ratio];
  state.zoom = zoom;
  clampPan();
  $("zoom-out").textContent = zoom > 1 ? `zoom ×${zoom.toFixed(1)}` : "";
  render();
}

// ---------------------------------------------------------------- rendering

function refreshOverlay() {
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

function render() {
  if (!state.labels) return;
  const scale = state.session.display.scale;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(state.zoom, 0, 0, state.zoom, state.pan[0], state.pan[1]);
  ctx.filter = `brightness(${state.brightness}) contrast(${state.contrast})`;
  ctx.drawImage(state.images.get(state.zOffset), 0, 0, canvas.width, canvas.height);
  ctx.filter = "none";
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
    if (state.tool === "brush") {
      ctx.arc((x + 0.5) * scale, (y + 0.5) * scale, (state.radius + 0.5) * scale, 0, 2 * Math.PI);
    } else {
      ctx.rect(x * scale, y * scale, scale, scale);
    }
    ctx.stroke();
  }
}

function onBorder(i, size) {
  // A pixel is on a border if a 4-neighbour inside the crop has another label.
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

function updateCounts() {
  const counts = new Map();
  for (const v of state.labels) counts.set(v, (counts.get(v) ?? 0) + 1);
  const total = state.labels.length;
  $("counts").innerHTML = allLabels()
    .map((l) => `<dt>${l.key}</dt><dd>${(((counts.get(l.value) ?? 0) / total) * PERCENT).toFixed(1)}%</dd>`)
    .join("");
}

// ---------------------------------------------------------------- painting

function pixelAt(event) {
  const size = state.session.crop_size;
  const pixel = canvas.width / size; // canvas pixels per image pixel at zoom 1
  const [cx, cy] = canvasPoint(event);
  const x = Math.floor((cx - state.pan[0]) / state.zoom / pixel);
  const y = Math.floor((cy - state.pan[1]) / state.zoom / pixel);
  return x >= 0 && y >= 0 && x < size && y < size ? [x, y] : null;
}

function paintable(i) {
  return state.paintOver === null || state.labels[i] === state.paintOver;
}

function pushUndo() {
  state.undo.push(state.labels.slice());
  if (state.undo.length > state.session.display.undo_depth) state.undo.shift();
  state.redo = [];
}

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

function paintLine([x0, y0], [x1, y1], value) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= steps; i++) {
    paintDisc(Math.round(x0 + ((x1 - x0) * i) / steps), Math.round(y0 + ((y1 - y0) * i) / steps), value);
  }
}

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

function wandFill([x, y], value) {
  // Fill the 4-connected pixels whose grey level is within the tolerance of the clicked one.
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

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const step = state.session.display.zoom_step;
  zoomAt(event, event.deltaY < 0 ? step : 1 / step);
}, { passive: false });

canvas.addEventListener("mousedown", (event) => {
  if (event.button === MOUSE_MIDDLE || (event.button === MOUSE_LEFT && event.altKey)) {
    event.preventDefault();
    state.panning = { from: [event.clientX, event.clientY], pan: [...state.pan] };
    return;
  }
  const p = pixelAt(event);
  if (!p || ![MOUSE_LEFT, MOUSE_RIGHT].includes(event.button) || state.zOffset !== 0) return;
  const value = event.button === MOUSE_RIGHT ? state.session.unsure.value : state.active;
  pushUndo();
  if (state.tool === "fill") {
    floodFill(p, value);
  } else if (state.tool === "wand") {
    wandFill(p, value);
  } else {
    paintDisc(p[0], p[1], value);
    state.stroke = { value, last: p };
  }
  setDirty(true);
  refreshOverlay();
});

canvas.addEventListener("mousemove", (event) => {
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
});

canvas.addEventListener("mouseleave", () => {
  state.hover = null;
  render();
});
window.addEventListener("mouseup", () => {
  state.stroke = null;
  state.panning = null;
});

function undo(from, to) {
  if (!from.length) return;
  to.push(state.labels);
  state.labels = from.pop();
  setDirty(true);
  refreshOverlay();
}

// ---------------------------------------------------------------- keyboard

window.addEventListener("keydown", (event) => {
  if ($("help").open || event.target === $("note")) return;
  const key = event.key.toLowerCase();
  if (event.ctrlKey || event.metaKey) {
    if (key === "s") save();
    else if (key === "z" && event.shiftKey) undo(state.redo, state.undo);
    else if (key === "z") undo(state.undo, state.redo);
    else if (key === "y") undo(state.redo, state.undo);
    else return;
    event.preventDefault();
    return;
  }
  const labels = state.session.labels;
  if (/^[0-9]$/.test(key) && Number(key) < labels.length) selectLabel(labels[Number(key)].value);
  else if (key === UNSURE_HOTKEY) selectLabel(state.session.unsure.value);
  else if (key === "b") selectTool("brush");
  else if (key === "l") cyclePaintOver();
  else if (key === "r") resetView();
  else if (key === "f") selectTool("fill");
  else if (key === "w") selectTool("wand");
  else if (key === "o") setOutline(!state.outline);
  else if (key === "[") setRadius(state.radius - 1);
  else if (key === "]") setRadius(state.radius + 1);
  else if (key === "arrowup") setZOffset(state.zOffset + 1);
  else if (key === "arrowdown") setZOffset(state.zOffset - 1);
  else if (key === "home") setZOffset(0);
  else if (key === "arrowleft") openCrop(state.index - 1);
  else if (key === "arrowright") openCrop(state.index + 1);
  else if (key === "h" || key === "?") $("help").showModal();
  else if (key === " ") {
    state.overlayHidden = true;
    render();
  } else return;
  event.preventDefault();
});

window.addEventListener("keyup", (event) => {
  if (event.key === " ") {
    state.overlayHidden = false;
    render();
  }
});

window.addEventListener("beforeunload", (event) => {
  if (state.dirty || state.noteDirty) event.preventDefault();
});

init().catch((error) => {
  $("crop-title").textContent = `error: ${error.message}`;
});

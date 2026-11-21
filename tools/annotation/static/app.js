// Entry point of the annotation page: reads the session, sets up the canvas and the controls,
// and opens the first crop not saved yet.

import { initControls } from './controls.js';
import { initPainting } from './paint.js';
import { buildCropList, openCrop } from './session.js';
import {
  $, canvas, ctx, DEFAULT_LABEL_KEY, fetchOk, OPAQUE, overlay, RGBA, state,
} from './state.js';

/** Load the session (labels, colours, crops, display settings) and open the first crop. */
async function init() {
  const session = await (await fetchOk('/api/session')).json();
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
  $('prefill-badge').hidden = !session.prefilled;
  initControls();
  initPainting();
  buildCropList();
  window.addEventListener('beforeunload', (event) => {
    if (state.dirty || state.noteDirty) event.preventDefault();
  });
  await openCrop(Math.max(0, session.crops.findIndex((c) => !c.saved)));
}

init().catch((error) => {
  $('crop-title').textContent = `error: ${error.message}`;
});

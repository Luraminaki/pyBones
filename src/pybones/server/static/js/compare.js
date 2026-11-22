// Compare tab: the shown segmentation next to another one: the settings that differ, the
// results side by side, and the same slice of both.

import { api, JOB_KINDS } from './api.js';
import { comparedResults, comparedSettings, settingText } from './compare_model.js';
import { sliceLoader } from './slice_loader.js';
import { el, toast } from './ui.js';

// Image scale of the slices: two side by side, each a quarter of the full size.
const SCALE = 0.25;
// Wait for the slider to settle before asking for slices.
const DEBOUNCE_MS = 120;

let shown = null; // id of the segmentation the page shows
let compared = null; // id of the other one, once compared
let summaries = [];
let sections = [];
let timer = null;
const nodes = {};

/**
 * Read what the comparison needs about a segmentation.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 * @returns {Promise<import('./compare_model.js').Side>} Its metadata and network.
 * @throws {import('./api.js').ApiError} If the server cannot read them.
 */
async function side(summary) {
  const [meta, network] = await Promise.all([
    api.segmentation(summary.id),
    summary.quantified ? api.network(summary.id) : null,
  ]);
  return { meta, network };
}

/**
 * Build a table: a header row, then one row per entry.
 * @param {string[]} head The header cells.
 * @param {string[][]} rows The rows' cells.
 * @returns {HTMLTableElement} The table.
 */
function table(head, rows) {
  return el('table', { class: 'compare-table' },
    el('thead', {}, el('tr', {}, ...head.map((cell) => el('th', { scope: 'col' }, cell)))),
    el('tbody', {}, ...rows.map(([first, ...cells]) =>
      el('tr', {}, el('th', { scope: 'row' }, first), ...cells.map((cell) => el('td', {}, cell))))));
}

/** Show the slice at the slider's position for both segmentations. */
function showSlices() {
  const z = Number(nodes.slider.value);
  nodes.value.textContent = `${z} / ${nodes.slider.max}`;
  nodes.showA(shown, z, SCALE);
  nodes.showB(compared, z, SCALE);
}

/** Compare the shown segmentation with the one picked (or say that one is to be picked). */
async function compare() {
  const other = nodes.other.value;
  nodes.body.hidden = !other;
  nodes.empty.hidden = Boolean(other);
  if (!other) return;
  const ids = [shown, other];
  const pair = ids.map((id) => summaries.find((s) => s.id === id));
  if (pair.includes(undefined)) return; // gone since the lists were read: the next refresh tells
  let sides;
  try {
    sides = await Promise.all(pair.map(side));
  } catch (err) {
    toast(`Could not compare ${shown} with ${other}: ${err.message}`, { error: true });
    return;
  }
  if (shown !== ids[0] || nodes.other.value !== other) return; // the choice changed meanwhile
  compared = other;
  const [a, b] = sides;
  const head = ['', ...ids];
  const settings = comparedSettings(a, b, sections).map(({ path, a: x, b: y }) => [path, settingText(x), settingText(y)]);
  nodes.settings.replaceChildren(settings.length
    ? table(['Setting', ...ids], settings)
    : el('p', { class: 'hint' }, 'Same settings.'));
  nodes.results.replaceChildren(table(head, comparedResults(a, b)));
  const depth = Math.min(a.meta.shape_zyx[0], b.meta.shape_zyx[0]);
  nodes.slider.max = String(depth - 1);
  nodes.slider.value = String(Math.min(Number(nodes.slider.value), depth - 1));
  nodes.nameA.textContent = ids[0];
  nodes.nameB.textContent = ids[1];
  nodes.stacks.hidden = a.meta.source === b.meta.source;
  showSlices();
}

/**
 * Show the comparison of a segmentation: the others are offered; the one picked, if still
 * there, stays picked. Shown again (the lists are refreshed after every job), only the offer is
 * updated.
 * @param {import('./api.js').SegmentationSummary} summary The shown segmentation.
 * @param {import('./api.js').SegmentationSummary[]} all Every segmentation.
 */
export function showSegmentation(summary, all) {
  summaries = all;
  const same = shown === summary.id;
  shown = summary.id;
  const picked = nodes.other.value;
  const others = all.filter((s) => s.id !== shown);
  nodes.other.replaceChildren(
    el('option', { value: '' }, others.length ? 'Pick a segmentation…' : 'No other segmentation'),
    ...others.map((s) => el('option', { value: s.id, selected: s.id === picked }, s.id)),
  );
  if (!same || nodes.other.value !== picked) {
    if (!same) {
      // The maximum first: a value above the current maximum would be clamped to it.
      const depth = summary.shape_zyx[0];
      nodes.slider.max = String(depth - 1);
      nodes.slider.value = String(Math.floor(depth / 2));
    }
    void compare();
  }
}

/**
 * Wire the tab: the choice of the other segmentation, the slider, the two slices.
 * @param {{settings_sections: Object<string, string[]>}} server What the server describes of itself.
 */
export function initCompare(server) {
  sections = server.settings_sections[JOB_KINDS.SEGMENTATION];
  for (const key of ['other', 'body', 'empty', 'settings', 'results', 'slider', 'value', 'stacks']) {
    nodes[key] = document.getElementById(`compare-${key}`);
  }
  nodes.nameA = document.getElementById('compare-a-name');
  nodes.nameB = document.getElementById('compare-b-name');
  const loading = document.getElementById('compare-loading');
  const busy = { a: false, b: false };
  const tell = (which) => (state) => {
    busy[which] = state;
    loading.hidden = !busy.a && !busy.b;
  };
  nodes.showA = sliceLoader(document.getElementById('compare-a'), tell('a'));
  nodes.showB = sliceLoader(document.getElementById('compare-b'), tell('b'));
  nodes.other.addEventListener('change', () => void compare());
  nodes.slider.addEventListener('input', () => {
    nodes.value.textContent = `${nodes.slider.value} / ${nodes.slider.max}`;
    clearTimeout(timer);
    timer = setTimeout(showSlices, DEBOUNCE_MS);
  });
}

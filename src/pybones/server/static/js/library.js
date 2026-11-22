// Sidebar: the segmentations of the output folder and the form segmenting a stack.

import { api } from './api.js';
import { track } from './jobs.js';
import { cssColor, cssPercent, el, LABELS, percent, toast } from './ui.js';

const STACK_SUFFIX = /\.zip$/i;

let state; // {server, colors, settings, summaries}

/**
 * Build the card of a segmentation: its name, size, version and label fractions.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 * @param {Object<string, number[]>} colors BGR colour per label.
 * @param {boolean} selected Whether it is the one shown.
 * @param {function(string): void} onSelect Called with its id when clicked.
 * @returns {HTMLElement} The card's list item.
 */
function card(summary, colors, selected, onSelect) {
  const [depth, height, width] = summary.shape_zyx;
  const fractions = LABELS.map((label) =>
    el('span', {
      style: `width: ${cssPercent(summary.label_fractions[label] || 0)}; background: ${cssColor(colors[label])}`,
      title: `${label} ${percent(summary.label_fractions[label] || 0, 1)}`,
    }),
  );
  return el(
    'li',
    {},
    el(
      'button',
      {
        type: 'button',
        class: 'card',
        'aria-current': selected ? 'true' : 'false',
        onclick: () => onSelect(summary.id),
      },
      el('span', { class: 'card-title' }, summary.id),
      el('span', { class: 'card-meta' }, `${depth} × ${height} × ${width} voxels · pyBones ${summary.pybones_version}`),
      el('div', { class: 'fractions', 'aria-hidden': 'true' }, ...fractions),
    ),
  );
}

/**
 * Show the list of segmentations.
 * @param {import('./api.js').SegmentationSummary[]} summaries The segmentations.
 * @param {?string} selectedId The one shown, if any.
 * @param {function(string): void} onSelect Called with an id when a card is clicked.
 */
export function renderSegmentations(summaries, selectedId, onSelect) {
  const list = document.getElementById('segmentation-list');
  const empty = document.getElementById('segmentation-empty');
  list.replaceChildren(
    ...summaries.map((s) => card(s, state.colors, s.id === selectedId, onSelect)),
  );
  empty.hidden = summaries.length > 0;
  empty.textContent = `No segmentation yet in ${state.server.output_dir}.`;
}

/**
 * Offer the stacks of the data folder in the form.
 * @param {{id: string, slices: number, height: number, width: number}[]} datasets The stacks.
 */
function renderDatasets(datasets) {
  const select = document.getElementById('dataset');
  const empty = document.getElementById('dataset-empty');
  select.replaceChildren(
    ...datasets.map((d) =>
      el('option', { value: d.id }, `${d.id} (${d.slices} slices, ${d.height} × ${d.width})`),
    ),
  );
  empty.hidden = datasets.length > 0;
  empty.textContent = `No stack found in ${state.server.data_dir}: put a folder or a zip of 16-bit slices there, then Refresh.`;
  document.getElementById('segment-submit').disabled = datasets.length === 0;
  suggestName();
}

/** Fill in a free name for the next run, e.g. `data_set-2`, unless the user typed one. */
function suggestName() {
  const input = document.getElementById('seg-name');
  const dataset = document.getElementById('dataset').value;
  if (!dataset || input.dataset.edited === 'true') return;
  const stem = dataset.replace(STACK_SUFFIX, '').replace(/[^A-Za-z0-9._-]/g, '-');
  const taken = new Set(state.summaries.map((s) => s.id));
  let name = stem;
  for (let n = 2; taken.has(name); n += 1) name = `${stem}-${n}`;
  input.value = name;
}

/**
 * Fill in the form for another run: same stack (if still in the data folder), same settings.
 * @param {string} datasetId Stack of the run.
 * @param {object} config Its configuration.
 * @param {string} runName Its name.
 * @returns {boolean} Whether the stack is still offered.
 */
export function prefill(datasetId, config, runName) {
  const select = document.getElementById('dataset');
  const known = [...select.options].some((o) => o.value === datasetId);
  if (known) select.value = datasetId;
  state.settings.startFrom(config, runName);
  const name = document.getElementById('seg-name');
  name.dataset.edited = 'false';
  suggestName();
  document.getElementById('segment-form').scrollIntoView({ behavior: 'smooth', block: 'center' });
  name.focus({ preventScroll: true });
  name.select();
  return known;
}

/**
 * Read the segmentations and stacks again and show them.
 * @param {?string} selectedId The segmentation shown, if any.
 * @param {function(string): void} onSelect Called with an id when a card is clicked.
 * @returns {Promise<import('./api.js').SegmentationSummary[]>} The segmentations.
 * @throws {import('./api.js').ApiError} If the server cannot list them.
 */
export async function reloadLists(selectedId, onSelect) {
  const [summaries, datasets] = await Promise.all([api.segmentations(), api.datasets()]);
  state.summaries = summaries;
  renderSegmentations(summaries, selectedId, onSelect);
  renderDatasets(datasets);
  return summaries;
}

/**
 * Wire the form: name suggestions, and starting a segmentation job on submit.
 * @param {{data_dir: string, output_dir: string}} server What the server describes of itself.
 * @param {Object<string, number[]>} colors BGR colour per label.
 * @param {import('./settings_form.js').SettingsForm} settings The run's settings form.
 */
export function initLibrary(server, colors, settings) {
  state = { server, colors, settings, summaries: [] };
  const form = document.getElementById('segment-form');
  const name = document.getElementById('seg-name');
  name.addEventListener('input', () => { name.dataset.edited = 'true'; });
  document.getElementById('dataset').addEventListener('change', suggestName);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    if (!settings.valid()) {
      settings.open();
      toast('Some settings are not valid: they are marked in red', { error: true });
      return;
    }
    const overrides = settings.overrides();
    const submit = document.getElementById('segment-submit');
    const dataset = document.getElementById('dataset').value;
    submit.disabled = true;
    let job;
    try {
      job = await api.createSegmentation(dataset, name.value, overrides);
    } catch (err) {
      if (settings.showErrors(err.detail)) settings.open();
      toast(`Could not start: ${err.message}`, { error: true });
      return;
    } finally {
      submit.disabled = false;
    }
    name.dataset.edited = 'false';
    toast(`Segmenting ${dataset} into ${name.value}`);
    void track(job);
  });
}

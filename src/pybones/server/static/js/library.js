// Sidebar: the segmentations of the output folder and the form segmenting a stack.

import { api } from './api.js';
import { track } from './jobs.js';
import { cssColor, cssPercent, el, percent, toast } from './ui.js';

// Labels drawn in the fraction bar of a card (background is the remainder).
const BAR_LABELS = ['bone', 'vessel', 'burst'];
const STACK_SUFFIX = /\.zip$/i;

let state;

function card(summary, colors, selected, onSelect) {
  const [depth, height, width] = summary.shape_zyx;
  const fractions = BAR_LABELS.map((label) =>
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

export function renderSegmentations(summaries, selectedId, onSelect) {
  const list = document.getElementById('segmentation-list');
  const empty = document.getElementById('segmentation-empty');
  list.replaceChildren(
    ...summaries.map((s) => card(s, state.colors, s.id === selectedId, onSelect)),
  );
  empty.hidden = summaries.length > 0;
  empty.textContent = `No segmentation yet in ${state.server.output_dir}.`;
}

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

// A free name for the next run, e.g. data_set-2 (the user can change it).
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

// The settings of `used` that differ from `base`, as `section.key=value` lines (the form's
// syntax), within the sections a run may override. Lists count as single values.
export function settingsDiff(base, used, sections) {
  const lines = [];
  const walk = (a, b, path) => {
    const isSection = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
    if (isSection(a) && isSection(b)) {
      for (const key of Object.keys(b)) walk(a[key], b[key], [...path, key]);
    } else if (JSON.stringify(a) !== JSON.stringify(b)) {
      lines.push(`${path.join('.')}=${JSON.stringify(b)}`);
    }
  };
  for (const section of sections) walk(base[section], used[section], [section]);
  return lines;
}

// Fill in the form for another run: same stack (if still in the data folder), same settings.
export function prefill(datasetId, overrides) {
  const select = document.getElementById('dataset');
  const known = [...select.options].some((o) => o.value === datasetId);
  if (known) select.value = datasetId;
  const textarea = document.getElementById('overrides');
  textarea.value = overrides.join('\n');
  textarea.closest('details').open = overrides.length > 0;
  const name = document.getElementById('seg-name');
  name.dataset.edited = 'false';
  suggestName();
  document.getElementById('segment-form').scrollIntoView({ behavior: 'smooth', block: 'center' });
  name.focus({ preventScroll: true });
  name.select();
  return known;
}

export async function reloadLists(selectedId, onSelect) {
  const [summaries, datasets] = await Promise.all([api.segmentations(), api.datasets()]);
  state.summaries = summaries;
  renderSegmentations(summaries, selectedId, onSelect);
  renderDatasets(datasets);
  return summaries;
}

export function initLibrary(server, colors) {
  state = { server, colors, summaries: [] };
  const form = document.getElementById('segment-form');
  const name = document.getElementById('seg-name');
  name.addEventListener('input', () => { name.dataset.edited = 'true'; });
  document.getElementById('dataset').addEventListener('change', suggestName);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const overrides = document.getElementById('overrides').value
      .split('\n').map((line) => line.trim()).filter(Boolean);
    const submit = document.getElementById('segment-submit');
    const dataset = document.getElementById('dataset').value;
    submit.disabled = true;
    try {
      const job = await api.createSegmentation(dataset, name.value, overrides);
      name.dataset.edited = 'false';
      toast(`Segmenting ${dataset} into ${name.value}`);
      void track(job);
    } catch (err) {
      toast(`Could not start: ${err.message}`, { error: true });
    } finally {
      submit.disabled = false;
    }
  });
}

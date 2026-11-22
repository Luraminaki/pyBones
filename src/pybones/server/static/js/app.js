// Entry point: loads the server description and configuration, wires the panels together.

import { api, JOB_KINDS } from './api.js';
import { initJobs, JOB_FINISHED } from './jobs.js';
import { initLibrary, prefill, reloadLists, renderSegmentations } from './library.js';
import { initNetwork, showSegmentation as showNetwork } from './network.js';
import { initSettingsForm } from './settings_form.js';
import { settingsDiff } from './settings_model.js';
import { initSlices, showSegmentation as showSlices } from './slices.js';
import { initTheme } from './theme.js';
import { baseName, LABELS, percent, recall, remember, toast } from './ui.js';

// Remembered per browser: the last segmentation looked at.
const SELECTION_KEY = 'pybones.selection';
// The viewer's tabs, as named in the ids of their buttons and panels.
const TABS = ['slices', '3d', 'network'];

let summaries = [];
let selected = null;
let viewer = null; // the 3D module, imported when the 3D tab first opens (three.js is ~2 MB)
let colors = null;
let baseConfig = null;
let serverInfo = null;

/**
 * Describe a segmentation in one line: stack, voxel size, label fractions.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 * @returns {string} The line under its title.
 */
function describe(summary) {
  const [z, y, x] = summary.voxel_size_um; // the configuration's order
  const fractions = LABELS
    .map((label) => `${label} ${percent(summary.label_fractions[label] ?? 0, 1)}`)
    .join(' · ');
  return `${baseName(summary.source)} · voxel ${x} × ${y} × ${z} µm · ${fractions}`;
}

/**
 * Show a segmentation in the viewers and mark its card.
 * @param {string} id Segmentation id (ignored if unknown).
 */
function select(id) {
  const summary = summaries.find((s) => s.id === id);
  if (!summary) return;
  selected = id;
  remember(SELECTION_KEY, id);
  renderSegmentations(summaries, selected, select);
  document.getElementById('viewer-empty').hidden = true;
  document.getElementById('viewer-content').hidden = false;
  document.getElementById('viewer-title').textContent = summary.id;
  document.getElementById('viewer-meta').textContent = describe(summary);
  showSlices(summary);
  showNetwork(summary);
  if (viewer) viewer.showSegmentation(summary);
}

/** Read the lists again and keep a segmentation shown (the current, the last seen, or the first). */
async function refresh() {
  try {
    summaries = await reloadLists(selected, select);
  } catch (err) {
    toast(`Could not list the stacks and segmentations: ${err.message}`, { error: true });
    return;
  }
  const keep = summaries.find((s) => s.id === selected)
    ?? summaries.find((s) => s.id === recall(SELECTION_KEY))
    ?? summaries[0];
  if (keep) select(keep.id);
}

/** Prepare another run of the shown segmentation: its stack and settings in the form. */
async function segmentAgain() {
  let meta;
  try {
    meta = await api.segmentation(selected);
  } catch (err) {
    toast(`Could not read ${selected}: ${err.message}`, { error: true });
    return;
  }
  const overrides = settingsDiff(baseConfig, meta.config, serverInfo.overridable_sections);
  const dataset = baseName(meta.source);
  if (!prefill(dataset, meta.config, selected)) {
    toast(`${dataset} is no longer in ${serverInfo.data_dir}: pick a stack`, { error: true });
  } else {
    const changed = overrides.length ? `${overrides.length} setting(s) as in ${selected}` : 'default settings';
    toast(`Ready to segment ${dataset} again with ${changed}: check the name, then Segment`);
  }
}

/**
 * Switch between the slice and 3D views; the 3D module is loaded the first time.
 * @param {string} name One of TABS.
 */
async function openTab(name) {
  for (const tab of TABS) {
    const active = tab === name;
    document.getElementById(`tab-${tab}`).setAttribute('aria-selected', String(active));
    document.getElementById(`panel-${tab}`).hidden = !active;
  }
  if (name !== '3d' || viewer) return;
  let module;
  try {
    module = await import('./viewer3d.js'); // three.js comes over the network
  } catch (err) {
    toast(`The 3D viewer could not be loaded: ${err.message}`, { error: true });
    return;
  }
  if (!module.initViewer(colors)) return;
  viewer = module;
  const summary = summaries.find((s) => s.id === selected);
  if (summary) viewer.showSegmentation(summary);
}

/** Start the page: server description and configuration first, then every panel. */
async function main() {
  initTheme(document.getElementById('theme-toggle'));
  let server;
  let config;
  let schema;
  try {
    [server, config, schema] = await Promise.all([api.server(), api.config(), api.configSchema()]);
  } catch (err) {
    toast(`pyBones cannot be reached: ${err.message}`, { error: true });
    return;
  }
  colors = config.render.colors_bgr;
  baseConfig = config;
  serverInfo = server;
  document.getElementById('version').textContent = `v${server.version}`;
  initLibrary(server, colors, initSettingsForm(schema, config, server.overridable_sections));
  initSlices(server, colors);
  initNetwork();
  document.getElementById('refresh').addEventListener('click', refresh);
  document.getElementById('segment-again').addEventListener('click', segmentAgain);
  for (const tab of TABS) {
    document.getElementById(`tab-${tab}`).addEventListener('click', () => openTab(tab));
  }

  document.addEventListener(JOB_FINISHED, async (event) => {
    const status = event.detail;
    if (status.state !== 'succeeded') return;
    await refresh();
    if (status.kind === JOB_KINDS.SEGMENTATION) {
      toast(`${status.segmentation} is ready`);
      select(status.segmentation);
    }
  });

  await refresh();
  await initJobs(document.getElementById('job-list'), document.getElementById('jobs-empty'));
}

void main();

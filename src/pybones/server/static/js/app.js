// Entry point: loads the server description and configuration, wires the panels together.

import { api } from './api.js';
import { initJobs, JOB_FINISHED } from './jobs.js';
import { initLibrary, prefill, reloadLists, renderSegmentations, settingsDiff } from './library.js';
import { initSlices, showSegmentation as showSlices } from './slices.js';
import { initTheme } from './theme.js';
import { percent, toast } from './ui.js';

// Remembered per browser: the last segmentation looked at.
const SELECTION_KEY = 'pybones.selection';

let summaries = [];
let selected = null;
let viewer = null; // the 3D module, imported when the 3D tab first opens (three.js is ~2 MB)
let colors = null;
let baseConfig = null;
let serverInfo = null;

function remember(id) {
  try {
    localStorage.setItem(SELECTION_KEY, id);
  } catch {
    // not remembered: fine
  }
}

function recalled() {
  try {
    return localStorage.getItem(SELECTION_KEY);
  } catch {
    return null;
  }
}

function describe(summary) {
  const [x, y, z] = summary.voxel_size_um;
  const fractions = ['bone', 'vessel', 'burst']
    .map((label) => `${label} ${percent(summary.label_fractions[label] ?? 0, 1)}`)
    .join(' · ');
  const source = summary.source.split(/[\\/]/).pop();
  return `${source} · voxel ${x} × ${y} × ${z} µm · ${fractions}`;
}

function select(id) {
  const summary = summaries.find((s) => s.id === id);
  if (!summary) return;
  selected = id;
  remember(id);
  renderSegmentations(summaries, selected, select);
  document.getElementById('viewer-empty').hidden = true;
  document.getElementById('viewer-content').hidden = false;
  document.getElementById('viewer-title').textContent = summary.id;
  document.getElementById('viewer-meta').textContent = describe(summary);
  showSlices(summary);
  if (viewer) viewer.showSegmentation(summary);
}

async function refresh() {
  try {
    summaries = await reloadLists(selected, select);
  } catch (err) {
    toast(`Could not list the stacks and segmentations: ${err.message}`, { error: true });
    return;
  }
  // The current one, else the one looked at last time in this browser, else the first.
  const keep = summaries.find((s) => s.id === selected)
    ?? summaries.find((s) => s.id === recalled())
    ?? summaries[0];
  if (keep) select(keep.id);
}

// Prepare another run of the shown segmentation: its stack and settings in the form.
async function segmentAgain() {
  let meta;
  try {
    meta = await api.segmentation(selected);
  } catch (err) {
    toast(`Could not read ${selected}: ${err.message}`, { error: true });
    return;
  }
  const overrides = settingsDiff(baseConfig, meta.config, serverInfo.overridable_sections);
  const dataset = meta.source.split(/[\\/]/).pop();
  if (!prefill(dataset, overrides)) {
    toast(`${dataset} is no longer in ${serverInfo.data_dir}: pick a stack`, { error: true });
  } else {
    const changed = overrides.length ? `${overrides.length} setting(s) as in ${selected}` : 'default settings';
    toast(`Ready to segment ${dataset} again with ${changed}: check the name, then Segment`);
  }
}

async function openTab(name) {
  for (const tab of ['slices', '3d']) {
    const active = tab === name;
    document.getElementById(`tab-${tab}`).setAttribute('aria-selected', String(active));
    document.getElementById(`panel-${tab}`).hidden = !active;
  }
  if (name !== '3d' || viewer) return;
  try {
    const module = await import('./viewer3d.js');
    if (!module.initViewer(colors)) return;
    viewer = module;
    const summary = summaries.find((s) => s.id === selected);
    if (summary) viewer.showSegmentation(summary);
  } catch (err) {
    toast(`The 3D viewer could not start: ${err.message}`, { error: true });
  }
}

async function main() {
  initTheme(document.getElementById('theme-toggle'));
  let server;
  let config;
  try {
    [server, config] = await Promise.all([api.server(), api.config()]);
  } catch (err) {
    toast(`pyBones cannot be reached: ${err.message}`, { error: true });
    return;
  }
  colors = config.render.colors_bgr;
  baseConfig = config;
  serverInfo = server;
  document.getElementById('version').textContent = `v${server.version}`;
  initLibrary(server, colors);
  initSlices(server, colors);
  document.getElementById('refresh').addEventListener('click', refresh);
  document.getElementById('segment-again').addEventListener('click', segmentAgain);
  document.getElementById('tab-slices').addEventListener('click', () => openTab('slices'));
  document.getElementById('tab-3d').addEventListener('click', () => openTab('3d'));

  document.addEventListener(JOB_FINISHED, async (event) => {
    const status = event.detail;
    if (status.state !== 'succeeded') return;
    await refresh();
    if (status.kind === 'segmentation') {
      toast(`${status.segmentation} is ready`);
      select(status.segmentation);
    }
  });

  await refresh();
  await initJobs(document.getElementById('job-list'), document.getElementById('jobs-empty'));
}

void main();

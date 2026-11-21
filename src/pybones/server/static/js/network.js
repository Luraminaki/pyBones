// Network tab: measure the vessel network of the shown segmentation (a job) and show its summary.

import { api, JOB_KINDS } from './api.js';
import { activeJob, JOB_FINISHED, JOB_UPDATED, startOrJoin } from './jobs.js';
import { duration, el, percent, toast } from './ui.js';

const LENGTH_DIGITS = 1;
const RATIO_DIGITS = 2;

/**
 * Rows of the summary: label, then how to read the value from a NetworkSummary.
 * @type {Array<[string, function(import('./api.js').NetworkSummary): string]>}
 */
const ROWS = [
  ['Total length', (s) => `${s.total_length_mm.toFixed(LENGTH_DIGITS)} mm`],
  ['Length density', (s) => `${s.length_density_mm_per_mm3.toFixed(RATIO_DIGITS)} mm/mm³`],
  ['Branches', (s) => `${s.branches} (${s.truncated_branches} cut by the stack, ${s.loops} loops)`],
  ['Junctions', (s) => String(s.junctions)],
  ['Ends', (s) => `${s.free_ends} free · ${s.face_ends} at a face · ${s.burst_ends} at a burst`],
  ['Median branch length', (s) => `${s.median_branch_length_um.toFixed(LENGTH_DIGITS)} µm`],
  ['Median radius', (s) => `${s.median_radius_um.toFixed(LENGTH_DIGITS)} µm`],
  ['Median tortuosity', (s) => s.median_tortuosity.toFixed(RATIO_DIGITS)],
];

let current = null; // the shown segmentation's summary
const nodes = {};

/**
 * Show the measures of a network, or say there are none yet.
 * @param {?import('./api.js').NetworkSummary} summary The measures (`null`: not measured).
 */
function render(summary) {
  nodes.stats.hidden = summary === null;
  nodes.button.textContent = summary === null ? 'Quantify' : 'Quantify again';
  if (summary === null) {
    nodes.status.textContent = 'Not measured yet.';
    return;
  }
  nodes.status.textContent = '';
  nodes.stats.replaceChildren(...ROWS.flatMap(([label, read]) => [el('dt', {}, label), el('dd', {}, read(summary))]));
}

/** Read the shown segmentation's measures from the server and show them. */
async function load() {
  const id = current.id;
  let summary;
  try {
    summary = await api.network(id);
  } catch (err) {
    toast(`Could not read the network of ${id}: ${err.message}`, { error: true });
    return;
  }
  if (current?.id === id) render(summary);
}

/** Start measuring the shown segmentation (or join the job already doing it). */
async function quantify() {
  const id = current.id;
  try {
    await startOrJoin(() => api.quantify(id), JOB_KINDS.QUANTIFICATION, id);
  } catch (err) {
    toast(`Could not quantify ${id}: ${err.message}`, { error: true });
  }
}

/**
 * Follow the quantification of the shown segmentation: progress, then its new measures.
 * @param {CustomEvent} event A JOB_UPDATED or JOB_FINISHED event (detail: the job status).
 * @param {boolean} finished Whether the job is over.
 */
function onJob(event, finished) {
  const status = event.detail;
  if (status.kind !== JOB_KINDS.QUANTIFICATION || status.segmentation !== current?.id) return;
  nodes.button.disabled = !finished;
  if (!finished) {
    const p = status.progress;
    nodes.status.textContent = p
      ? `Measuring · ${p.stage ?? 'starting'} · ${percent(p.fraction)} · ${duration(p.elapsed_s)}`
      : 'Waiting for a worker…';
  } else if (status.state === 'succeeded') {
    void load();
  } else {
    nodes.status.textContent = status.state === 'cancelled' ? 'Cancelled.' : 'The measurement failed.';
  }
}

/**
 * Show the network of a segmentation: its measures if any, the button to (re)measure it.
 * @param {import('./api.js').SegmentationSummary} summary The segmentation.
 */
export function showSegmentation(summary) {
  const same = current?.id === summary.id;
  current = summary;
  if (same) return;
  const running = activeJob(JOB_KINDS.QUANTIFICATION, summary.id) !== null;
  nodes.button.disabled = running;
  render(null);
  if (running) nodes.status.textContent = 'Measuring…';
  if (summary.quantified) void load();
}

/** Wire the tab: its button and the job events. */
export function initNetwork() {
  nodes.button = document.getElementById('quantify');
  nodes.status = document.getElementById('network-status');
  nodes.stats = document.getElementById('network-stats');
  nodes.button.addEventListener('click', () => void quantify());
  document.addEventListener(JOB_UPDATED, (event) => onJob(event, false));
  document.addEventListener(JOB_FINISHED, (event) => onJob(event, true));
}

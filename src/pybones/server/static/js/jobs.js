// Jobs panel: every job the server remembers, followed live with Server-Sent Events.
// Other modules listen to JOB_UPDATED / JOB_FINISHED on `document` (detail: the job status).

import { api, followJob, FINISHED_STATES, JOB_KINDS } from './api.js';
import { announce, duration, el, percent, toast } from './ui.js';

/**
 * Event sent on `document` with each new status of a job, and every second while it runs, its times
 * advanced (detail: the JobStatus).
 */
export const JOB_UPDATED = 'pybones:job-updated';
/** Event sent on `document` once a followed job is over (detail: its final JobStatus). */
export const JOB_FINISHED = 'pybones:job-finished';
// Finished jobs kept in the panel (the running ones always are).
const FINISHED_SHOWN = 6;
// How often the times of a running job advance between two statuses from the server.
const TICK_MS = 1000;
const MS_PER_S = 1000;

const jobs = new Map(); // job id -> {status, receivedAt, node, parts}
let ticker = null; // the one-second timer, set only while a job runs
// Jobs being followed: one event stream each, however many parts of the page ask.
const following = new Map(); // job id -> promise of the final status
let list;
let empty;

/**
 * Name a job for people.
 * @param {import('./api.js').JobStatus} status The job.
 * @returns {string} E.g. `Mesh: vessel of run (glb)`.
 */
function title(status) {
  if (status.kind === JOB_KINDS.MESH) {
    return `Mesh: ${status.label} of ${status.segmentation} (${status.mesh_format})`;
  }
  if (status.kind === JOB_KINDS.QUANTIFICATION) return `Quantify: ${status.segmentation}`;
  return `Segment: ${status.segmentation}`;
}

/**
 * Describe where a job is: its stage, progress and time left, or how it ended.
 * @param {import('./api.js').JobStatus} status The job.
 * @returns {string} The line under its progress bar.
 */
function detail(status) {
  const p = status.progress;
  if (status.state === 'queued') return 'Waiting for a free worker';
  if (status.state === 'cancelled') return 'Cancelled';
  if (status.state === 'failed') return 'Failed';
  if (!p) return status.state;
  const elapsed = duration(p.elapsed_s);
  if (status.state === 'succeeded') return `Done in ${elapsed}`;
  const stage = p.stage ? `${p.stage} (${p.stage_index}/${p.stage_count})` : 'starting';
  const left = p.remaining_s === null ? '' : ` · ~${duration(p.remaining_s)} left`;
  return `${stage} · ${percent(p.fraction)} · ${elapsed}${left}`;
}

/**
 * Build a job's row once; later statuses only patch it (see patch()), so the Cancel button
 * under the mouse is never swapped for a new one between a press and a release.
 * @param {import('./api.js').JobStatus} status The job.
 * @returns {{node: HTMLElement, parts: Object<string, HTMLElement>}} The row and its parts.
 */
function row(status) {
  const parts = {
    title: el('span', { class: 'job-title' }, title(status)),
    cancel: el('button', {
      type: 'button',
      class: 'danger small',
      onclick: () => cancel(status.job_id),
    }, 'Cancel'),
    bar: el('progress', { max: '1', value: '0' }),
    detail: el('div', { class: 'job-detail' }),
    error: el('div', { class: 'job-error', hidden: true }),
  };
  const node = el(
    'li',
    { class: 'job', dataset: { job: status.job_id } },
    el('div', { class: 'job-head' }, parts.title, parts.cancel),
    parts.bar,
    parts.detail,
    parts.error,
  );
  return { node, parts };
}

/**
 * Show a new status in a job's row.
 * @param {{node: HTMLElement, parts: Object<string, HTMLElement>}} job The row.
 * @param {import('./api.js').JobStatus} status Its new status.
 */
function patch({ node, parts }, status) {
  const finished = FINISHED_STATES.has(status.state);
  node.dataset.state = status.state;
  parts.cancel.hidden = finished;
  parts.bar.hidden = finished && status.state !== 'succeeded';
  parts.bar.value = status.progress ? status.progress.fraction : 0;
  parts.detail.textContent = detail(status);
  parts.error.hidden = !status.error;
  parts.error.textContent = status.error ?? '';
}

/** Remove the oldest finished jobs beyond the few the panel keeps. */
function prune() {
  const finished = [...jobs.values()].filter((j) => FINISHED_STATES.has(j.status.state));
  for (const job of finished.slice(0, Math.max(0, finished.length - FINISHED_SHOWN))) {
    job.node.remove();
    jobs.delete(job.status.job_id);
  }
}

/**
 * Show a job's status (adding its row if new) and tell the other modules.
 * @param {import('./api.js').JobStatus} status The status.
 */
function update(status) {
  let job = jobs.get(status.job_id);
  if (!job) {
    job = row(status);
    list.prepend(job.node);
    jobs.set(status.job_id, job);
  }
  job.status = status;
  job.receivedAt = performance.now();
  empty.hidden = jobs.size > 0;
  show(job, status);
  if (status.state === 'running' && ticker === null) ticker = setInterval(tick, TICK_MS);
}

/**
 * Show a status in a job's row and tell the other modules.
 * @param {{node: HTMLElement, parts: Object<string, HTMLElement>}} job The row.
 * @param {import('./api.js').JobStatus} status The status to show.
 */
function show(job, status) {
  patch(job, status);
  document.dispatchEvent(new CustomEvent(JOB_UPDATED, { detail: status }));
}

/**
 * Advance the times of a running job's status by the time since it arrived: the elapsed time
 * grows and the time left shrinks, so a long stage that reports nothing still shows life.
 * @param {import('./api.js').JobStatus} status Status as the server sent it.
 * @param {number} seconds Seconds since it arrived.
 * @returns {import('./api.js').JobStatus} The status now (unchanged unless running).
 */
export function aged(status, seconds) {
  const p = status.progress;
  if (status.state !== 'running' || !p) return status;
  const remaining = p.remaining_s === null ? null : Math.max(0, p.remaining_s - seconds);
  return { ...status, progress: { ...p, elapsed_s: p.elapsed_s + seconds, remaining_s: remaining } };
}

/** Show every running job with its times advanced; stop the timer once none runs. */
function tick() {
  const now = performance.now();
  let running = false;
  for (const job of jobs.values()) {
    if (job.status.state !== 'running') continue;
    running = true;
    show(job, aged(job.status, (now - job.receivedAt) / MS_PER_S));
  }
  if (!running) {
    clearInterval(ticker);
    ticker = null;
  }
}

/**
 * Ask the server to cancel a job.
 * @param {string} jobId Job id.
 */
async function cancel(jobId) {
  let status;
  try {
    status = await api.cancelJob(jobId);
  } catch (err) {
    toast(`Could not cancel: ${err.message}`, { error: true });
    return;
  }
  update(status);
}

/**
 * Show a job and, unless it is over, follow it until it finishes (one stream per job).
 * @param {import('./api.js').JobStatus} status Its current status.
 * @returns {Promise<?import('./api.js').JobStatus>} Its final status (`null` if lost).
 */
export function track(status) {
  update(status);
  if (FINISHED_STATES.has(status.state)) return Promise.resolve(status);
  if (!following.has(status.job_id)) {
    const done = follow(status).finally(() => following.delete(status.job_id));
    following.set(status.job_id, done);
  }
  return following.get(status.job_id);
}

/**
 * Follow a job's events until it finishes, then announce how it ended.
 * @param {import('./api.js').JobStatus} status Its current status.
 * @returns {Promise<?import('./api.js').JobStatus>} Its final status (`null` if lost).
 */
async function follow(status) {
  let final;
  try {
    final = await followJob(status.job_id, update);
  } catch (err) {
    toast(err.message, { error: true });
    return null;
  }
  announce(`${title(final)}: ${final.state}`);
  if (final.state === 'failed') toast(`${title(final)} failed: ${final.error}`, { error: true });
  document.dispatchEvent(new CustomEvent(JOB_FINISHED, { detail: final }));
  prune();
  return final;
}

/**
 * Find the unfinished job of a target, if any.
 * @param {string} kind What the job does (one of JOB_KINDS).
 * @param {string} segmentation Segmentation id.
 * @param {?string} [label] Label key (mesh jobs only).
 * @param {?string} [format] Mesh format (mesh jobs only).
 * @returns {?import('./api.js').JobStatus} Its status, or `null`.
 */
export function activeJob(kind, segmentation, label = null, format = null) {
  for (const { status } of jobs.values()) {
    if (
      status.kind === kind && status.segmentation === segmentation && status.label === label &&
      status.mesh_format === format && !FINISHED_STATES.has(status.state)
    ) return status;
  }
  return null;
}

/**
 * Follow the unfinished job of a target, or start one; either way, track it until it ends.
 * @param {function(): Promise<import('./api.js').JobStatus>} start Starts the job (an API call).
 * @param {string} kind What the job does (one of JOB_KINDS).
 * @param {string} segmentation Segmentation id.
 * @param {?string} [label] Label key (mesh jobs only).
 * @param {?string} [format] Mesh format (mesh jobs only).
 * @throws {import('./api.js').ApiError} If the job cannot be started.
 */
export async function startOrJoin(start, kind, segmentation, label = null, format = null) {
  const job = activeJob(kind, segmentation, label, format) ?? (await start());
  void track(job);
}

/**
 * Show the jobs the server remembers and follow the unfinished ones.
 * @param {HTMLElement} listNode The list of job rows.
 * @param {HTMLElement} emptyNode Shown while there is no job.
 */
export async function initJobs(listNode, emptyNode) {
  list = listNode;
  empty = emptyNode;
  let known = [];
  try {
    known = await api.jobs();
  } catch (err) {
    toast(`Could not list the jobs: ${err.message}`, { error: true });
  }
  for (const status of known) void track(status);
  prune();
}

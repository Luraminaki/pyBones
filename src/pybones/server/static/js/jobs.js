// Jobs panel: every job the server remembers, followed live with Server-Sent Events.
// Other modules listen to JOB_UPDATED / JOB_FINISHED (detail: the job status).

import { api, followJob, FINISHED_STATES } from './api.js';
import { announce, duration, el, percent, toast } from './ui.js';

export const JOB_UPDATED = 'pybones:job-updated';
export const JOB_FINISHED = 'pybones:job-finished';
// Finished jobs kept in the panel (the running ones always are).
const FINISHED_SHOWN = 6;

const jobs = new Map(); // job id -> {status, node, parts}
// Jobs being followed: one event stream each, however many parts of the page ask.
const following = new Map(); // job id -> promise of the final status
let list;
let empty;

function title(status) {
  if (status.kind === 'mesh') {
    return `Mesh: ${status.label} of ${status.segmentation} (${status.mesh_format})`;
  }
  return `Segment: ${status.segmentation}`;
}

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
  return `${stage} · ${percent(p.fraction)}${left}`;
}

// Build a job's row once; later statuses only patch it (patch()), so the Cancel button
// under the mouse is never swapped for a new one between a press and a release.
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

function prune() {
  const finished = [...jobs.values()].filter((j) => FINISHED_STATES.has(j.status.state));
  for (const job of finished.slice(0, Math.max(0, finished.length - FINISHED_SHOWN))) {
    job.node.remove();
    jobs.delete(job.status.job_id);
  }
}

function update(status) {
  let job = jobs.get(status.job_id);
  if (!job) {
    job = row(status);
    list.prepend(job.node);
    jobs.set(status.job_id, job);
  }
  job.status = status;
  patch(job, status);
  empty.hidden = jobs.size > 0;
  document.dispatchEvent(new CustomEvent(JOB_UPDATED, { detail: status }));
}

async function cancel(jobId) {
  try {
    update(await api.cancelJob(jobId));
  } catch (err) {
    toast(`Could not cancel: ${err.message}`, { error: true });
  }
}

// Show a job and, unless it is over, follow it until it finishes (one stream per job).
export function track(status) {
  update(status);
  if (FINISHED_STATES.has(status.state)) return Promise.resolve(status);
  if (!following.has(status.job_id)) {
    const done = follow(status).finally(() => following.delete(status.job_id));
    following.set(status.job_id, done);
  }
  return following.get(status.job_id);
}

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

// The job of a target (segmentation and label) still running, if any.
export function activeJob(segmentation, label, format) {
  for (const { status } of jobs.values()) {
    if (
      status.segmentation === segmentation && status.label === label &&
      status.mesh_format === format && !FINISHED_STATES.has(status.state)
    ) return status;
  }
  return null;
}

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

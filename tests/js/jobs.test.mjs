// Tests of the job times shown between two statuses (src/pybones/server/static/js/jobs.js).
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { aged } from '../../src/pybones/server/static/js/jobs.js';

/**
 * Build a job status.
 * @param {string} state Job state.
 * @param {?number} remaining Seconds left, if known.
 * @returns {object} The status.
 */
function status(state, remaining) {
  return {
    job_id: 'mesh-1',
    state,
    progress: { stage: 'surface', fraction: 0.4, elapsed_s: 10, remaining_s: remaining },
  };
}

test('a running job gets older: elapsed time up, time left down', () => {
  const now = aged(status('running', 15), 3);
  assert.equal(now.progress.elapsed_s, 13);
  assert.equal(now.progress.remaining_s, 12);
  assert.equal(now.progress.fraction, 0.4); // only the times move
});

test('the time left never goes below zero, and stays unknown when unknown', () => {
  assert.equal(aged(status('running', 2), 5).progress.remaining_s, 0);
  assert.equal(aged(status('running', null), 5).progress.remaining_s, null);
});

test('the server status is not modified', () => {
  const original = status('running', 15);
  aged(original, 3);
  assert.equal(original.progress.elapsed_s, 10);
});

test('finished, queued or not yet started jobs keep their times', () => {
  for (const state of ['succeeded', 'failed', 'cancelled', 'queued']) {
    const fixed = status(state, 0);
    assert.equal(aged(fixed, 5), fixed);
  }
  const starting = { job_id: 'mesh-1', state: 'running', progress: null };
  assert.equal(aged(starting, 5), starting);
});

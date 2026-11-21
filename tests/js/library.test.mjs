// Tests of the "segment again" settings difference (src/pybones/server/static/js/library.js).
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { settingsDiff } from '../../src/pybones/server/static/js/library.js';

const BASE = {
  thresholds: { bone: 0.5, window: [3, 3] },
  segmentation: { burst: { density: 0.3, size: 10 } },
  io: { labels_file: 'labels.npy' },
};

test('the same settings give no line', () => {
  assert.deepEqual(settingsDiff(BASE, structuredClone(BASE), ['thresholds', 'segmentation']), []);
});

test('changed values become override lines, nested keys dotted', () => {
  const used = structuredClone(BASE);
  used.thresholds.bone = 0.6;
  used.segmentation.burst.density = 0.25;
  assert.deepEqual(settingsDiff(BASE, used, ['thresholds', 'segmentation']), [
    'thresholds.bone=0.6',
    'segmentation.burst.density=0.25',
  ]);
});

test('a list counts as one value', () => {
  const used = structuredClone(BASE);
  used.thresholds.window = [3, 5];
  assert.deepEqual(settingsDiff(BASE, used, ['thresholds']), ['thresholds.window=[3,5]']);
});

test('sections outside the overridable ones are ignored', () => {
  const used = structuredClone(BASE);
  used.io.labels_file = 'other.npy';
  assert.deepEqual(settingsDiff(BASE, used, ['thresholds', 'segmentation']), []);
});

test('strings are written as JSON, ready for the form', () => {
  const base = { mesh: { format: 'glb' } };
  assert.deepEqual(settingsDiff(base, { mesh: { format: 'stl' } }, ['mesh']), ['mesh.format="stl"']);
});

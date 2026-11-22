// Tests of the comparison's pure part (src/pybones/server/static/js/compare_model.js).
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { comparedResults, comparedSettings, settingText } from '../../src/pybones/server/static/js/compare_model.js';

/**
 * Build one side of a comparison.
 * @param {object} changes Settings and results that differ from the common ones.
 * @returns {import('../../src/pybones/server/static/js/compare_model.js').Side} The side.
 */
function side({ density = 0.35, enabled = false, network = null, thin = null } = {}) {
  return {
    meta: {
      source: '/data/stack.zip',
      shape_zyx: [100, 20, 30],
      pybones_version: '0.8.1',
      label_fractions: { background: 0.7, bone: 0.2, vessel: 0.075, burst: 0.025 },
      timing_s: { total: 24.4 },
      thin_vessels: thin,
      config: {
        segmentation: { burst: { density } },
        thin_vessels: { enabled },
        quantification: { pruning: { loop_factor: 1.5 } }, // the server's of that day: ignored
      },
    },
    network,
  };
}

const NETWORK = {
  summary: { branches: 9514, junctions: 3121, total_length_mm: 592.34, median_radius_um: 6.71 },
  settings: { pruning: { loop_factor: 1.5 } },
};

test('the settings a run may change are compared, path by path', () => {
  const differences = comparedSettings(side(), side({ density: 0.3, enabled: true }), ['segmentation', 'thin_vessels']);
  assert.deepEqual(differences, [
    { path: 'segmentation.burst.density', a: 0.35, b: 0.3 },
    { path: 'thin_vessels.enabled', a: false, b: true },
  ]);
  assert.deepEqual(comparedSettings(side(), side(), ['segmentation']), []);
});

test("networks' own settings count only when both are measured", () => {
  const other = { ...NETWORK, settings: { pruning: { loop_factor: 2 } } };
  assert.deepEqual(comparedSettings(side({ network: NETWORK }), side({ network: other }), []), [
    { path: 'quantification.pruning.loop_factor', a: 1.5, b: 2 },
  ]);
  assert.deepEqual(comparedSettings(side({ network: NETWORK }), side(), []), []);
});

test('results are laid side by side, a missing network as a dash', () => {
  const thin = { candidate_pieces: 12, pieces_added: 3, from_background: 400, from_bone: 1200 };
  const rows = Object.fromEntries(comparedResults(side({ network: NETWORK }), side({ thin })).map(([w, a, b]) => [w, [a, b]]));
  assert.deepEqual(rows.Stack, ['stack.zip', 'stack.zip']);
  assert.deepEqual(rows.Vessel, ['7.50 %', '7.50 %']);
  assert.deepEqual(rows['Thin vessels added'], ['off', '3 pieces, 1,600 voxels']);
  assert.deepEqual(rows['Network: branches'], ['9,514', '—']);
  assert.deepEqual(rows['Network: total length'], ['592.3 mm', '—']);
});

test('values are written as JSON, an absent one as a dash', () => {
  assert.equal(settingText([3, 3, 3]), '[3,3,3]');
  assert.equal(settingText('stl'), '"stl"');
  assert.equal(settingText(undefined), '—');
});

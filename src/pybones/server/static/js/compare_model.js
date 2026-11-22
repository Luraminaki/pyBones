// Comparing two segmentations, pure part: the settings that differ, and their results side by
// side. No DOM: tested with Node (tests/js/compare_model.test.mjs).

import { settingDifferences } from './settings_model.js';
import { baseName, duration, LABELS, percent } from './ui.js';

// Decimals of the lengths and radii shown.
const LENGTH_DIGITS = 1;
// Decimals of the label fractions shown (percent).
const FRACTION_DIGITS = 2;
// Shown for a result one segmentation does not have.
const NONE = '—';

/**
 * @typedef {object} Side
 * @property {object} meta A segmentation's metadata (`GET /api/segmentations/{id}`).
 * @property {?import('./api.js').NetworkView} network Its measured network (`null`: not measured).
 */

/**
 * List the settings that differ between two segmentations: those a run may change, from their
 * metadata, then, when both networks are measured, the settings they were measured with (a
 * segmentation's own quantification section is only the server's default of that day).
 * @param {Side} a One segmentation.
 * @param {Side} b The other.
 * @param {string[]} sections Sections a run may change.
 * @returns {import('./settings_model.js').Difference[]} The differing settings.
 */
export function comparedSettings(a, b, sections) {
  const differences = sections.flatMap(
    (section) => settingDifferences(a.meta.config[section], b.meta.config[section], section),
  );
  if (a.network && b.network) {
    differences.push(...settingDifferences(a.network.settings, b.network.settings, 'quantification'));
  }
  return differences;
}

/**
 * Write a setting's value for a table cell.
 * @param {*} value The value (undefined if the configuration lacks it).
 * @returns {string} Its JSON text, or a dash.
 */
export function settingText(value) {
  return value === undefined ? NONE : JSON.stringify(value);
}

/**
 * Describe what the thin-vessel pass added to a segmentation.
 * @param {object} meta Its metadata.
 * @returns {string} The pieces and voxels added, or that the pass was off.
 */
function thinText(meta) {
  const thin = meta.thin_vessels;
  if (!thin) return 'off';
  return `${thin.pieces_added} pieces, ${(thin.from_background + thin.from_bone).toLocaleString('en')} voxels`;
}

/**
 * Lay two segmentations' results side by side.
 * @param {Side} a One segmentation.
 * @param {Side} b The other.
 * @returns {Array<[string, string, string]>} Rows: what, its value for `a`, for `b`.
 */
export function comparedResults(a, b) {
  const rows = [];
  const row = (what, read) => rows.push([what, read(a), read(b)]);
  row('Stack', ({ meta }) => baseName(meta.source));
  row('Voxels (z × y × x)', ({ meta }) => meta.shape_zyx.join(' × '));
  row('pyBones', ({ meta }) => meta.pybones_version);
  for (const label of LABELS) {
    row(`${label[0].toUpperCase()}${label.slice(1)}`, ({ meta }) => percent(meta.label_fractions[label], FRACTION_DIGITS));
  }
  row('Thin vessels added', ({ meta }) => thinText(meta));
  row('Segmentation time', ({ meta }) => duration(meta.timing_s.total));
  const network = (read) => ({ network: view }) => (view ? read(view.summary) : NONE);
  row('Network: branches', network((s) => s.branches.toLocaleString('en')));
  row('Network: junctions', network((s) => s.junctions.toLocaleString('en')));
  row('Network: total length', network((s) => `${s.total_length_mm.toFixed(LENGTH_DIGITS)} mm`));
  row('Network: median radius', network((s) => `${s.median_radius_um.toFixed(LENGTH_DIGITS)} µm`));
  return rows;
}

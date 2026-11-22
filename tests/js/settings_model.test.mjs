// Tests of the settings form's pure part (src/pybones/server/static/js/settings_model.js).
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  changedFields, checkValue, describeChanges, EFFECT_ORDER, readSections, setValue, settingsDiff,
  valueAt,
} from '../../src/pybones/server/static/js/settings_model.js';

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

// A schema shaped like pydantic's: sections as $defs, extras next to their keys.
const SCHEMA = {
  effects: { labels: 'the labels', meshes: 'the meshes', none: 'no result' },
  properties: {
    io: { $ref: '#/$defs/Io' },
    pipeline: { $ref: '#/$defs/Pipeline' },
    segmentation: { $ref: '#/$defs/Segmentation' },
  },
  $defs: {
    Io: { type: 'object', description: 'Files.', effect: 'none', level: 'advanced', properties: {} },
    Pipeline: {
      type: 'object', description: 'Threads.', effect: 'none', level: 'advanced',
      properties: { workers: { type: 'integer', minimum: 1, maximum: 8, description: 'Threads.' } },
    },
    Segmentation: {
      type: 'object', description: 'Classes.', effect: 'labels', level: 'common',
      properties: {
        sigma_um: { type: 'number', exclusiveMinimum: 0, description: 'Blur.', advice: 'Rarely.' },
        connectivity: { enum: [4, 8], type: 'integer', description: 'Pixels.' },
        voxel_um: { type: 'array', prefixItems: [{ type: 'number' }], description: 'Size.' },
        thin: { $ref: '#/$defs/Thin' },
      },
    },
    Thin: {
      type: 'object', description: 'Thin vessels.', effect: 'labels', level: 'experimental',
      properties: {
        enabled: { type: 'boolean', description: 'On.' },
        block_px: { type: 'integer', minimum: 1, description: 'Block.', effect: 'none' },
      },
    },
  },
};

test('sections are read depth first, only below the roots asked for', () => {
  const sections = readSections(SCHEMA, ['segmentation', 'pipeline']);
  assert.deepEqual(sections.map((s) => s.path), ['pipeline', 'segmentation', 'segmentation.thin']);
  assert.equal(sections[2].level, 'experimental');
  const fields = sections.flatMap((s) => s.fields);
  assert.deepEqual(fields.map((f) => [f.path, f.kind]), [
    ['pipeline.workers', 'integer'],
    ['segmentation.sigma_um', 'number'],
    ['segmentation.connectivity', 'choice'],
    ['segmentation.voxel_um', 'json'],
    ['segmentation.thin.enabled', 'boolean'],
    ['segmentation.thin.block_px', 'integer'],
  ]);
  assert.equal(fields[1].advice, 'Rarely.');
  assert.equal(fields[5].effect, 'none'); // its own, not its section's
  assert.equal(fields[4].effect, 'labels');
});

test('typed values are checked against their kind and bounds', () => {
  const fields = readSections(SCHEMA, ['segmentation', 'pipeline']).flatMap((s) => s.fields);
  const [workers, sigma, connectivity, voxel, enabled] = fields;
  assert.deepEqual(checkValue(workers, '4'), { value: 4 });
  assert.deepEqual(checkValue(workers, '0'), { error: 'At least 1.' });
  assert.deepEqual(checkValue(workers, '2.5'), { error: 'A whole number is needed.' });
  assert.deepEqual(checkValue(workers, ''), { error: 'A number is needed.' });
  assert.deepEqual(checkValue(sigma, '0'), { error: 'More than 0.' });
  assert.deepEqual(checkValue(sigma, '1.5'), { value: 1.5 });
  assert.deepEqual(checkValue(connectivity, '8'), { value: 8 });
  assert.ok('error' in checkValue(connectivity, '6'));
  assert.deepEqual(checkValue(voxel, '[3, 3, 3]'), { value: [3, 3, 3] });
  assert.ok('error' in checkValue(voxel, '[3, 3'));
  assert.deepEqual(checkValue(enabled, true), { value: true });
});

test('changes are summed up by their widest effect', () => {
  assert.equal(EFFECT_ORDER[0], 'labels'); // the widest: everything else follows from labels
  assert.equal(EFFECT_ORDER.at(-1), 'none');
  const fields = readSections(SCHEMA, ['segmentation', 'pipeline']).flatMap((s) => s.fields);
  const base = {
    pipeline: { workers: 4 },
    segmentation: { sigma_um: 1, connectivity: 8, voxel_um: [3], thin: { enabled: false, block_px: 64 } },
  };
  const speed = structuredClone(base);
  setValue(speed, 'pipeline.workers', 2);
  setValue(speed, 'segmentation.thin.block_px', 32);
  assert.equal(valueAt(speed, 'segmentation.thin.block_px'), 32);
  assert.equal(describeChanges(changedFields(fields, base, speed), 'run1', SCHEMA.effects),
    '2 settings differ from run1; a change affects no result.');
  const labels = structuredClone(speed);
  setValue(labels, 'segmentation.thin.enabled', true);
  assert.equal(describeChanges(changedFields(fields, base, labels), 'run1', SCHEMA.effects),
    '3 settings differ from run1; a change affects the labels.');
  assert.equal(describeChanges([], 'the defaults', SCHEMA.effects), 'Same settings as the defaults.');
  assert.equal(describeChanges(changedFields(fields, base, base), 'run1', SCHEMA.effects),
    'Same settings as run1.');
});

test('kept paths are left out, with what lies below them', () => {
  const sections = readSections(SCHEMA, ['segmentation'], ['segmentation.thin', 'segmentation.sigma_um']);
  assert.deepEqual(sections.map((s) => s.path), ['segmentation']);
  assert.deepEqual(sections[0].fields.map((f) => f.name), ['connectivity', 'voxel_um']);
});

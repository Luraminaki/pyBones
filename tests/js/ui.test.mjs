// Tests of the page's formatting helpers (src/pybones/server/static/js/ui.js).
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { describe, test } from 'node:test';

import {
  baseName, count, cssColor, cssPercent, duration, LABELS, megabytes, percent,
} from '../../src/pybones/server/static/js/ui.js';

describe('duration', () => {
  test('is empty when unknown', () => {
    assert.equal(duration(null), '');
    assert.equal(duration(undefined), '');
  });

  test('shows seconds under a minute', () => {
    assert.equal(duration(0), '0 s');
    assert.equal(duration(59.4), '59 s');
  });

  test('rounds before splitting into minutes, never showing 60 s', () => {
    assert.equal(duration(59.6), '1 min 0 s');
    assert.equal(duration(119.6), '2 min 0 s');
    assert.equal(duration(125), '2 min 5 s');
  });
});

describe('number formats', () => {
  test('percent rounds to the asked digits', () => {
    assert.equal(percent(0.5), '50 %');
    assert.equal(percent(0.1234, 1), '12.3 %');
  });

  test('cssPercent is a CSS length without a space', () => {
    assert.equal(cssPercent(0.25), '25%');
  });

  test('megabytes are decimal', () => {
    assert.equal(megabytes(2_500_000), '2.5 MB');
  });
});

describe('cssColor', () => {
  test('reads the configuration colours as OpenCV BGR', () => {
    assert.equal(cssColor([255, 128, 0]), 'rgb(0, 128, 255)');
  });
});

describe('baseName', () => {
  test('takes the last part of a POSIX or a Windows path', () => {
    assert.equal(baseName('/home/me/data/data_set.zip'), 'data_set.zip');
    assert.equal(baseName('C:\\data\\stack'), 'stack');
    assert.equal(baseName('stack'), 'stack');
  });
});

test('LABELS lists the drawn labels once each', () => {
  assert.equal(new Set(LABELS).size, LABELS.length);
});

test('count writes thousands separators', () => {
  assert.equal(count(7), '7');
  assert.equal(count(9514), '9,514');
  assert.equal(count(1234567), '1,234,567');
});

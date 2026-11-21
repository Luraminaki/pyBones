// Tests of the mesh memory helpers (src/pybones/server/static/js/mesh_memory.js), on stand-in
// objects shaped like three.js ones. Run by tests/server/test_page_js.py, or: node --test tests/js/

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { dispose, releaseAfterUpload, triangles } from '../../src/pybones/server/static/js/mesh_memory.js';

// A buffer attribute that records its upload callback, like three.js' BufferAttribute.
function attribute(count) {
  return {
    count,
    array: new Float32Array(count),
    onUpload(callback) { this.uploaded = callback; },
  };
}

// A mesh with the given geometry, recording what was disposed.
function mesh(geometry) {
  const disposed = [];
  return {
    isMesh: true,
    disposed,
    geometry: {
      ...geometry,
      computeBoundingBox() { this.box = true; },
      computeBoundingSphere() { this.sphere = true; },
      dispose() { disposed.push('geometry'); },
    },
    material: { dispose() { disposed.push('material'); } },
  };
}

// A scene of children, walked like Object3D.traverse.
function scene(...children) {
  return { traverse(visit) { children.forEach(visit); } };
}

test('triangles count indexed and plain geometries', () => {
  const indexed = mesh({ index: attribute(30), attributes: { position: attribute(12) } });
  const plain = mesh({ index: null, attributes: { position: attribute(9) } });
  assert.equal(triangles(scene(indexed, plain, { isMesh: false })), 13);
});

test('arrays are dropped once uploaded, after the bounds are computed', () => {
  const position = attribute(9);
  const shared = { onUpload(callback) { this.uploaded = callback; }, array: new Float32Array(4) };
  const normal = { isInterleavedBufferAttribute: true, data: shared };
  const index = attribute(3);
  const item = mesh({ index, attributes: { position, normal } });
  releaseAfterUpload(scene(item));
  assert.ok(item.geometry.box && item.geometry.sphere);
  for (const owner of [position, shared, index]) {
    owner.uploaded.call(owner);
    assert.equal(owner.array, null);
  }
});

test('dispose frees the geometry and the material of every mesh', () => {
  const items = [mesh({ attributes: {} }), mesh({ attributes: {} })];
  dispose(scene(...items, { isMesh: false }));
  for (const item of items) assert.deepEqual(item.disposed, ['geometry', 'material']);
});

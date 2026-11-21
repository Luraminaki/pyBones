// Memory of the loaded meshes: what they hold, and freeing it (CPU copies after upload, GPU at the end).
// Works on three.js objects without importing three.js itself.

const VERTICES_PER_TRIANGLE = 3;

/**
 * Count the triangles of a loaded mesh.
 * @param {THREE.Object3D} object The mesh's scene.
 * @returns {number} Triangles.
 */
export function triangles(object) {
  let count = 0;
  object.traverse((child) => {
    if (child.isMesh) {
      const { index, attributes } = child.geometry;
      count += (index ?? attributes.position).count / VERTICES_PER_TRIANGLE;
    }
  });
  return count;
}

/**
 * Keep a mesh (or point cloud) in GPU memory only. three.js keeps a CPU copy of every array after uploading
 * it (for picking, bounds, re-uploads), which doubles the browser's memory: 175 MB more for
 * the bone and vessels of the sample. Bounds are computed first (fitting the view uses
 * them); then each array is dropped as soon as it is on the GPU. Nothing here picks
 * triangles, and the arrays never change, so nothing needs the copy again.
 * @param {THREE.Object3D} object The mesh's scene.
 */
export function releaseAfterUpload(object) {
  object.traverse((child) => {
    if (!child.geometry) return; // meshes and points alike
    const geometry = child.geometry;
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const buffers = [...Object.values(geometry.attributes), geometry.index].filter(Boolean);
    for (const attribute of buffers) {
      // Interleaved attributes (16-bit positions, 8-bit normals) live in a shared buffer.
      const owner = attribute.isInterleavedBufferAttribute ? attribute.data : attribute;
      owner.onUpload(function drop() { this.array = null; });
    }
  });
}

/**
 * Free the GPU memory of a mesh or point cloud (geometry and materials).
 * @param {THREE.Object3D} object The mesh's scene.
 */
export function dispose(object) {
  object.traverse((child) => {
    if (child.geometry) { // meshes and points alike
      child.geometry.dispose();
      child.material.dispose();
    }
  });
}

/**
 * Take a mesh or point cloud out of the scene and free its GPU memory.
 * @param {THREE.Scene} scene The scene holding it.
 * @param {THREE.Object3D} object The object.
 */
export function discard(scene, object) {
  scene.remove(object);
  dispose(object);
}

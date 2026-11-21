// The page's client of the pyBones HTTP API (src/pybones/server/routes/).
// Errors come back as HTTP status codes with FastAPI's {"detail": ...} body; every failure
// (network, timeout, 4xx/5xx) is turned into one thrown ApiError carrying a readable message.

/**
 * Progress of a job (Python: `pybones.segmentation.stages.Progress`).
 * @typedef {object} Progress
 * @property {?string} stage Current stage (`null` before the first update).
 * @property {number} stage_index 1-based position of the stage in the plan.
 * @property {number} stage_count Number of planned stages.
 * @property {number} fraction Overall progress in [0, 1].
 * @property {number} elapsed_s Seconds since the job started.
 * @property {?number} remaining_s Estimated seconds left, if known.
 */

/**
 * What the server says about a job (Python: `pybones.server.schemas.JobView`).
 * @typedef {object} JobStatus
 * @property {string} job_id Identifier.
 * @property {string} kind `segmentation` or `mesh`.
 * @property {string} state `queued`, `running`, `succeeded`, `failed` or `cancelled`.
 * @property {?Progress} progress Its progress (`null` while queued).
 * @property {?string} error Why it failed, if it did.
 * @property {string} segmentation The segmentation it writes or meshes.
 * @property {?string} label Label being meshed (`null` for a segmentation).
 * @property {?string} mesh_format Mesh format (`null` for a segmentation).
 */

/**
 * A segmentation in the output folder (Python: `pybones.server.schemas.SegmentationSummary`).
 * @typedef {object} SegmentationSummary
 * @property {string} id Its folder name.
 * @property {string} source Path of the stack it was made from.
 * @property {string} pybones_version Version that made it.
 * @property {number[]} shape_zyx Volume shape.
 * @property {number[]} voxel_size_um Voxel size, (z, y, x) µm.
 * @property {Object<string, number>} label_fractions Share of the volume per label.
 * @property {Object<string, string[]>} meshes Built mesh formats per label.
 * @property {boolean} quantified Whether its vessel network was measured.
 */

/**
 * Measures of a vessel network for the whole stack (Python:
 * `pybones.quantification.measures.NetworkSummary`).
 * @typedef {object} NetworkSummary
 * @property {number} branches Branches between nodes.
 * @property {number} truncated_branches Branches cut by a face of the stack.
 * @property {number} loops Branches closed on themselves.
 * @property {number} junctions Forks.
 * @property {number} free_ends Ends inside the stack.
 * @property {number} face_ends Ends at a face of the stack.
 * @property {number} burst_ends Ends at a burst.
 * @property {number} total_length_mm Vessel length.
 * @property {number} stack_volume_mm3 Volume of the stack.
 * @property {number} length_density_mm_per_mm3 Vessel length per volume.
 * @property {number} median_branch_length_um Median branch length.
 * @property {number} median_radius_um Median vessel radius.
 * @property {number} median_tortuosity Median length / chord of the open branches.
 */

const BASE = '/api';
// Listing stacks opens each archive; generous, but never stuck forever.
const REQUEST_TIMEOUT_MS = 30000;
/**
 * States after which a job does not change any more (Python: `pybones.api.jobs.FINISHED_STATES`,
 * checked by tests/server/test_page_js.py).
 * @type {Set<string>}
 */
export const FINISHED_STATES = new Set(['succeeded', 'failed', 'cancelled']);
/**
 * What a job does (Python: the `*_JOB` names of `pybones.api.job_status`, checked by
 * tests/server/test_page_js.py).
 * @type {{SEGMENTATION: string, MESH: string, QUANTIFICATION: string}}
 */
export const JOB_KINDS = Object.freeze({
  SEGMENTATION: 'segmentation',
  MESH: 'mesh',
  QUANTIFICATION: 'quantification',
});

/** A request that failed: its message is readable as is, `status` is 0 without an answer. */
export class ApiError extends Error {
  /**
   * @param {string} message What went wrong.
   * @param {number} status HTTP status (0 if the server did not answer).
   */
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Turn FastAPI's error detail into a sentence.
 * @param {*} detail A string (pyBones' own errors) or a list of `{loc, msg}` (validation).
 * @returns {string} The message.
 */
function describe(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).slice(1).join('.') || 'request'}: ${d.msg}`).join('; ');
  }
  return JSON.stringify(detail);
}

/**
 * Build the error of a failed response, with the server's message if it sent one.
 * @param {Response} response A response that is not OK.
 * @returns {Promise<ApiError>} The error.
 */
async function failure(response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    message = describe((await response.json()).detail);
  } catch {
    // not JSON: keep the status line
  }
  return new ApiError(message, response.status);
}

/**
 * Send a request with a time limit.
 * @param {string} path Path under `/api`.
 * @param {RequestInit} init Fetch options (the time limit is added).
 * @param {string} what Names the request in error messages.
 * @returns {Promise<Response>} The successful response.
 * @throws {ApiError} If the server cannot be reached, is too slow, or refuses.
 */
async function send(path, init, what) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  } catch (err) {
    const reason = err.name === 'TimeoutError' ? 'the server did not answer in time' : 'the server cannot be reached';
    throw new ApiError(`${what}: ${reason}`, 0);
  }
  if (!response.ok) throw await failure(response);
  return response;
}

/**
 * Call a JSON endpoint.
 * @param {string} method HTTP method.
 * @param {string} path Path under `/api`.
 * @param {*} [body] Sent as JSON, if given.
 * @returns {Promise<*>} The decoded answer.
 * @throws {ApiError} If the request fails.
 */
async function request(method, path, body) {
  const json = body === undefined ? {} : {
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
  return (await send(path, { method, ...json }, `${method} ${path}`)).json();
}

/**
 * Escape an id for a URL path segment.
 * @param {string} value Id.
 * @returns {string} The escaped id.
 */
const id = (value) => encodeURIComponent(value);

/**
 * Path of the mesh of a label (under `/api`).
 * @param {string} segmentation Segmentation id.
 * @param {string} label Label key.
 * @param {string} format `glb` or `stl`.
 * @returns {string} The path.
 */
const meshPath = (segmentation, label, format) =>
  `/segmentations/${id(segmentation)}/meshes/${id(label)}?format=${id(format)}`;

/**
 * Fetch a slice image with its labels overlaid. `no-store`: the browser keeps no copy (a
 * stack has hundreds of megabytes of slices; the server says the same with Cache-Control).
 * @param {string} segmentation Segmentation id.
 * @param {number} z Slice index.
 * @param {number|string} scale Image scale factor.
 * @returns {Promise<Blob>} The PNG image.
 * @throws {ApiError} If the request fails.
 */
export async function fetchSlice(segmentation, z, scale) {
  const path = `/segmentations/${id(segmentation)}/slices/${z}?scale=${scale}`;
  return (await send(path, { cache: 'no-store' }, `slice ${z}`)).blob();
}

/** The JSON endpoints; each returns a promise of the decoded answer or throws an ApiError. */
export const api = {
  server: () => request('GET', '/server'),
  config: () => request('GET', '/config'),
  datasets: () => request('GET', '/datasets'),
  segmentations: () => request('GET', '/segmentations'),
  segmentation: (segmentation) => request('GET', `/segmentations/${id(segmentation)}`),
  createSegmentation: (dataset, name, overrides) =>
    request('POST', '/segmentations', { dataset, name, overrides }),
  buildMesh: (segmentation, label, format) => request('POST', meshPath(segmentation, label, format)),
  network: (segmentation) => request('GET', `/segmentations/${id(segmentation)}/quantification`),
  quantify: (segmentation) => request('POST', `/segmentations/${id(segmentation)}/quantification`),
  jobs: () => request('GET', '/jobs'),
  cancelJob: (jobId) => request('DELETE', `/jobs/${id(jobId)}`),
};

/**
 * Fetch the centreline points of a segmentation (the browser revalidates its copy).
 * @param {string} segmentation Segmentation id.
 * @returns {Promise<Float32Array>} `(x, y, z)` µm triples.
 * @throws {ApiError} If the request fails (404 if not quantified).
 */
export async function fetchCentrelines(segmentation) {
  const path = `/segmentations/${id(segmentation)}/centrelines`;
  return new Float32Array(await (await send(path, {}, 'centrelines')).arrayBuffer());
}

/**
 * Address of the mesh of a label, for a loader that fetches it itself.
 * @param {string} segmentation Segmentation id.
 * @param {string} label Label key.
 * @param {string} format `glb` or `stl`.
 * @returns {string} The URL.
 */
export function meshUrl(segmentation, label, format) {
  return `${BASE}${meshPath(segmentation, label, format)}`;
}

/**
 * Follow a job with Server-Sent Events until it finishes.
 * @param {string} jobId Job id.
 * @param {function(JobStatus): void} onStatus Called with each status the server pushes.
 * @returns {Promise<JobStatus>} The final status.
 * @throws {ApiError} If the connection is lost for good (job unknown, server stopped).
 */
export function followJob(jobId, onStatus) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`${BASE}/jobs/${id(jobId)}/events`);
    source.onmessage = (event) => {
      const status = JSON.parse(event.data);
      onStatus(status);
      if (FINISHED_STATES.has(status.state)) {
        source.close();
        resolve(status);
      }
    };
    source.onerror = () => {
      // EventSource retries on its own; CLOSED means it gave up (job unknown, server gone).
      if (source.readyState === EventSource.CLOSED) {
        reject(new ApiError(`lost the progress of ${jobId}`, 0));
      }
    };
  });
}

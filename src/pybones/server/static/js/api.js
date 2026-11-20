// Thin wrappers around the pyBones HTTP API (src/pybones/server/routes/).
// Errors come back as HTTP status codes with FastAPI's {"detail": ...} body; every failure
// (network, timeout, 4xx/5xx) is turned into one thrown ApiError carrying a readable message.

const BASE = '/api';
// Listing stacks opens each archive; generous, but never stuck forever.
const REQUEST_TIMEOUT_MS = 30000;
export const FINISHED_STATES = new Set(['succeeded', 'failed', 'cancelled']);

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// FastAPI's detail is a string (our errors) or a list of {loc, msg} (request validation).
function describe(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).slice(1).join('.') || 'request'}: ${d.msg}`).join('; ');
  }
  return JSON.stringify(detail);
}

// Turn a failed response into an ApiError carrying the server's message.
async function failure(response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    message = describe((await response.json()).detail);
  } catch {
    // not JSON: keep the status line
  }
  return new ApiError(message, response.status);
}

async function request(method, path, body) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    const reason = err.name === 'TimeoutError' ? 'the server did not answer in time' : 'the server cannot be reached';
    throw new ApiError(`${method} ${path}: ${reason}`, 0);
  }
  if (!response.ok) throw await failure(response);
  return response.json();
}

// A slice image as a Blob. 'no-store': the browser keeps no copy (a stack has hundreds of
// megabytes of slices; the server says the same with Cache-Control).
export async function fetchSlice(segmentation, z, scale) {
  let response;
  try {
    response = await fetch(sliceUrl(segmentation, z, scale), {
      cache: 'no-store',
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new ApiError(`slice ${z}: the server cannot be reached`, 0);
  }
  if (!response.ok) throw await failure(response);
  return response.blob();
}

const id = (value) => encodeURIComponent(value);

export const api = {
  server: () => request('GET', '/server'),
  config: () => request('GET', '/config'),
  datasets: () => request('GET', '/datasets'),
  segmentations: () => request('GET', '/segmentations'),
  segmentation: (segmentation) => request('GET', `/segmentations/${id(segmentation)}`),
  createSegmentation: (dataset, name, overrides) =>
    request('POST', '/segmentations', { dataset, name, overrides }),
  buildMesh: (segmentation, label, format) =>
    request('POST', `/segmentations/${id(segmentation)}/meshes/${id(label)}?format=${id(format)}`),
  jobs: () => request('GET', '/jobs'),
  cancelJob: (jobId) => request('DELETE', `/jobs/${id(jobId)}`),
};

export function sliceUrl(segmentation, z, scale) {
  return `${BASE}/segmentations/${id(segmentation)}/slices/${z}?scale=${scale}`;
}

export function meshUrl(segmentation, label, format) {
  return `${BASE}/segmentations/${id(segmentation)}/meshes/${id(label)}?format=${id(format)}`;
}

// Follow a job with Server-Sent Events: onStatus(status) for each pushed status; resolves
// with the final one. Rejects if the connection is lost for good (e.g. the server stopped).
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

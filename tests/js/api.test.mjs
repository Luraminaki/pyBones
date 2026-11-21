// Tests of the page's HTTP client (src/pybones/server/static/js/api.js), with a stubbed fetch.
// Run by tests/server/test_page_js.py, or directly: node --test tests/js/

import assert from 'node:assert/strict';
import { afterEach, describe, test } from 'node:test';

import {
  api, ApiError, fetchSlice, FINISHED_STATES, followJob, meshUrl,
} from '../../src/pybones/server/static/js/api.js';

const realFetch = globalThis.fetch;
const calls = [];

// Answer every request with `answer(url, init)`, recording the calls.
function serve(answer) {
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return answer(url, init);
  };
}

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { 'Content-Type': 'application/json' },
});

// A stand-in for the browser's EventSource: the test pushes messages and errors by hand.
class FakeEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  static last = null;

  constructor(url) {
    this.url = url;
    this.readyState = FakeEventSource.OPEN;
    FakeEventSource.last = this;
  }

  push(status) {
    this.onmessage({ data: JSON.stringify(status) });
  }

  fail(readyState) {
    this.readyState = readyState;
    this.onerror(new Event('error'));
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

afterEach(() => {
  globalThis.fetch = realFetch;
  delete globalThis.EventSource;
  calls.length = 0;
});

describe('requests', () => {
  test('GET reads the JSON answer under /api', async () => {
    serve(() => json([{ id: 'a' }]));
    assert.deepEqual(await api.segmentations(), [{ id: 'a' }]);
    assert.equal(calls[0].url, '/api/segmentations');
    assert.equal(calls[0].init.method, 'GET');
    assert.equal(calls[0].init.body, undefined);
    assert.ok(calls[0].init.signal instanceof AbortSignal);
  });

  test('POST sends a JSON body', async () => {
    serve(() => json({ job_id: 'segmentation-1' }, 202));
    await api.createSegmentation('stack', 'run', ['a.b=1']);
    const { init } = calls[0];
    assert.equal(init.method, 'POST');
    assert.equal(init.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(init.body), { dataset: 'stack', name: 'run', overrides: ['a.b=1'] });
  });

  test('ids are escaped in paths', async () => {
    serve(() => json({}));
    await api.buildMesh('a b', 'vessel', 'glb');
    assert.equal(calls[0].url, '/api/segmentations/a%20b/meshes/vessel?format=glb');
    assert.equal(meshUrl('a b', 'vessel', 'glb'), calls[0].url);
  });
});

describe('errors', () => {
  test("a server error carries FastAPI's detail and the status", async () => {
    serve(() => json({ detail: 'no segmentation "x"' }, 404));
    await assert.rejects(api.segmentation('x'), (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 404);
      assert.equal(err.message, 'no segmentation "x"');
      return true;
    });
  });

  test('a validation error lists each field and its message', async () => {
    serve(() => json({
      detail: [
        { loc: ['body', 'name'], msg: 'too long' },
        { loc: ['body'], msg: 'bad' },
      ],
    }, 422));
    await assert.rejects(api.createSegmentation('s', 'n', []), { message: 'name: too long; request: bad' });
  });

  test('a body that is not JSON keeps the status line', async () => {
    serve(() => new Response('oops', { status: 502, statusText: 'Bad Gateway' }));
    await assert.rejects(api.jobs(), { message: '502 Bad Gateway', status: 502 });
  });

  test('an unreachable server is status 0', async () => {
    serve(() => { throw new TypeError('fetch failed'); });
    await assert.rejects(api.jobs(), { message: 'GET /jobs: the server cannot be reached', status: 0 });
  });

  test('a timeout says so', async () => {
    serve(() => { throw new DOMException('timed out', 'TimeoutError'); });
    await assert.rejects(api.jobs(), { message: 'GET /jobs: the server did not answer in time' });
  });
});

describe('slices', () => {
  test('are fetched as a blob the browser must not keep', async () => {
    serve(() => new Response(new Uint8Array([1, 2, 3]), { headers: { 'Content-Type': 'image/png' } }));
    const blob = await fetchSlice('run', 4, 0.5);
    assert.equal(blob.size, 3);
    assert.equal(calls[0].url, '/api/segmentations/run/slices/4?scale=0.5');
    assert.equal(calls[0].init.cache, 'no-store');
  });

  test('failures name the slice', async () => {
    serve(() => { throw new TypeError('fetch failed'); });
    await assert.rejects(fetchSlice('run', 4, 0.5), { message: 'slice 4: the server cannot be reached' });
  });
});

describe('following a job', () => {
  test('passes on every status and resolves with the final one, closing the stream', async () => {
    globalThis.EventSource = FakeEventSource;
    const seen = [];
    const done = followJob('mesh 1', (status) => seen.push(status.state));
    const source = FakeEventSource.last;
    assert.equal(source.url, '/api/jobs/mesh%201/events');
    source.push({ job_id: 'mesh 1', state: 'running' });
    source.push({ job_id: 'mesh 1', state: 'succeeded' });
    assert.deepEqual(await done, { job_id: 'mesh 1', state: 'succeeded' });
    assert.deepEqual(seen, ['running', 'succeeded']);
    assert.equal(source.readyState, FakeEventSource.CLOSED);
  });

  test('every finished state ends the stream', async () => {
    globalThis.EventSource = FakeEventSource;
    for (const state of FINISHED_STATES) {
      const done = followJob('j', () => {});
      FakeEventSource.last.push({ job_id: 'j', state });
      assert.equal((await done).state, state);
    }
  });

  test('a dropped connection that the browser retries is not an error', async () => {
    globalThis.EventSource = FakeEventSource;
    const done = followJob('j', () => {});
    const source = FakeEventSource.last;
    source.fail(FakeEventSource.CONNECTING);
    source.readyState = FakeEventSource.OPEN;
    source.push({ job_id: 'j', state: 'cancelled' });
    assert.equal((await done).state, 'cancelled');
  });

  test('a connection given up rejects', async () => {
    globalThis.EventSource = FakeEventSource;
    const done = followJob('j', () => {});
    FakeEventSource.last.fail(FakeEventSource.CLOSED);
    await assert.rejects(done, { message: 'lost the progress of j', status: 0 });
  });
});

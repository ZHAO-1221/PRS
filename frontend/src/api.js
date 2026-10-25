// Data access for the screens, backed by pet_recognition_system/app/server.py.
// Field names follow the Python side (snake_case). VITE_MOCK=1 swaps in the in-memory mock.
import * as mock from './mock/api.js';
import { checkUpload } from './lib/upload.js';

export { STEPS } from './lib/upload.js';

// FastAPI errors carry `detail`: a string, or a list of field errors for 422s.
function message(body, status) {
  const detail = body && body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join('; ');
  return `Request failed (${status})`;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch {
    throw new Error('Cannot reach the analysis server');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(message(body, response.status));
  // e.g. the dev server answering with index.html because /api is not proxied
  if (body === null) throw new Error('Unexpected response from the analysis server');
  return body;
}

const real = {
  listClips({ from, to } = {}) {
    const query = new URLSearchParams(Object.entries({ from, to }).filter(([, v]) => v));
    return request(`/clips${query.size ? `?${query}` : ''}`);
  },
  getClip: (id) => request(`/clips/${encodeURIComponent(id)}`),
  uploadClip(file, { recordedAt, camera }) {
    checkUpload(file, recordedAt);
    const form = new FormData();
    form.append('video', file);
    form.append('recorded_at', recordedAt);
    form.append('camera', camera);
    return request('/clips', { method: 'POST', body: form });
  },
  getJob: (jobId) => request(`/jobs/${encodeURIComponent(jobId)}`),
  getModelMetrics: () => request('/models/metrics')
};

export const { listClips, getClip, uploadClip, getJob, getModelMetrics } =
  import.meta.env.VITE_MOCK === '1' ? mock : real;

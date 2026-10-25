// In-memory stand-in for the FastAPI service, used with `npm run dev:mock` (VITE_MOCK=1).
// Data is lost on reload.
import { makeResult, seedClips, UPLOAD_RESULT_PARTS } from './clips.js';
import { checkUpload, STEPS } from '../lib/upload.js';

const clips = [...seedClips];
const jobs = new Map();
const STEP_MS = 1200;

const wait = (ms = 120) => new Promise((r) => setTimeout(r, ms));
const byTime = (a, b) => a.recorded_at.localeCompare(b.recorded_at);

export async function listClips({ from, to } = {}) {
  await wait();
  return clips
    .filter((c) => (!from || c.recorded_at.slice(0, 10) >= from) && (!to || c.recorded_at.slice(0, 10) <= to))
    .sort(byTime);
}

export async function getClip(id) {
  await wait();
  const clip = clips.find((c) => c.id === id);
  if (!clip) throw new Error('Clip not found');
  return clip;
}

export async function uploadClip(file, { recordedAt, camera }) {
  checkUpload(file, recordedAt);
  await wait(300);
  const id = `upload-${Date.now()}`;
  jobs.set(id, { started: Date.now(), clip: { id, file_name: file.name, recorded_at: `${recordedAt}:00`, camera, cover_url: null, video_url: URL.createObjectURL(file) } });
  return { job_id: id, clip_id: id };
}

export async function getJob(jobId) {
  await wait(60);
  const job = jobs.get(jobId);
  if (!job) throw new Error('Job not found');
  const elapsed = Date.now() - job.started;
  const done = elapsed >= STEPS.length * STEP_MS;
  if (done && !clips.includes(job.clip)) clips.push(Object.assign(job.clip, { result: makeResult(UPLOAD_RESULT_PARTS) }));
  return {
    job_id: jobId,
    status: done ? 'done' : 'running',
    step: Math.min(STEPS.length, Math.floor(elapsed / STEP_MS)),
    progress: Math.min(1, elapsed / (STEPS.length * STEP_MS)),
    error: null,
    clip: job.clip
  };
}

// Same numbers as outputs/best_models/*/test_metrics.json.
export async function getModelMetrics() {
  await wait();
  return {
    dog_behavior: { accuracy: 0.7879, macro_f1: 0.7883, test_clips: 33 },
    interaction: { accuracy: 0.8333, macro_f1: 0.833, test_clips: 24 }
  };
}

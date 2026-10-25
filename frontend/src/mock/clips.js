// Mock clip records. Each `result` follows the shape written by predict_video()
// (pet_recognition_system/tools/predict_video.py) with task="both".
import lying from '../../assets/cover-lying.jpg';
import feeding from '../../assets/cover-feeding.jpg';
import walking from '../../assets/cover-walking.jpg';
import eating from '../../assets/cover-eating.jpg';
import sitting from '../../assets/cover-sitting.jpg';
import standing from '../../assets/cover-standing.jpg';

const DOG = ['walking_run', 'standing', 'sitting', 'lying_sleeping', 'eating_drinking'];
const PEOPLE = ['no_interaction', 'hand_feeding', 'petting', 'playing_together'];
const WINDOW = 4;

function probabilities(classes, label, index) {
  const top = 0.7 + ((index * 37) % 20) / 100;
  const rest = (1 - top) / (classes.length - 1);
  return Object.fromEntries(classes.map((c) => [c, c === label ? top : rest]));
}

function merge(windows, task) {
  const segments = [];
  for (const w of windows) {
    const v = w[task];
    if (v.status === 'suppressed_by_interaction') continue;
    const last = segments[segments.length - 1];
    if (last && last.label === v.label && last.end_sec === w.start_sec) {
      last.confidence = (last.confidence * last.windows + v.confidence) / (last.windows + 1);
      last.windows += 1;
      last.end_sec = w.end_sec;
    } else {
      segments.push({ start_sec: w.start_sec, end_sec: w.end_sec, label: v.label, confidence: v.confidence, windows: 1 });
    }
  }
  return segments;
}

// parts: [{ sec, dog } | { sec, interaction }]; an interaction part means the owner is in view.
export function makeResult(parts) {
  const windows = [];
  let t = 0;
  for (const part of parts) {
    for (let end = t + part.sec; t < end; t += WINDOW) {
      const index = windows.length;
      const together = Boolean(part.interaction);
      const scored = (classes, label) => {
        const p = probabilities(classes, label, index);
        return { status: 'ok', raw_probabilities: p, probabilities: p, label, confidence: p[label] };
      };
      windows.push({
        index,
        start_sec: t,
        end_sec: Math.min(t + WINDOW, end),
        detection: { person: together ? 0.96 : 0, dog: 0.94 },
        dog_behavior: together ? { status: 'suppressed_by_interaction' } : scored(DOG, part.dog),
        interaction: together ? scored(PEOPLE, part.interaction)
          : { status: 'insufficient_detection', label: 'insufficient_detection', confidence: 0 }
      });
    }
  }
  const timeline = { interaction: merge(windows, 'interaction') };
  const dog = merge(windows, 'dog_behavior');
  if (dog.length) timeline.dog_behavior = dog; // predict_video omits an empty dog timeline
  return {
    task: 'both',
    metadata: { duration_sec: t, width: 1280, height: 720, source_fps: 30, sample_fps: 15 },
    windows,
    timeline
  };
}

const DAY = [
  { time: '08:42', camera: 'Living room', cover: lying, parts: [{ sec: 280, dog: 'lying_sleeping' }, { sec: 20, dog: 'standing' }] },
  { time: '09:08', camera: 'Living room', cover: feeding, parts: [{ sec: 120, interaction: 'hand_feeding' }] },
  { time: '10:15', camera: 'Bedroom', cover: walking, parts: [{ sec: 60, dog: 'walking_run' }] },
  { time: '12:30', camera: 'Kitchen', cover: eating, parts: [{ sec: 160, dog: 'eating_drinking' }, { sec: 20, dog: 'standing' }] },
  { time: '14:05', camera: 'Living room', cover: sitting, parts: [{ sec: 120, dog: 'sitting' }] },
  { time: '18:02', camera: 'Kitchen', cover: standing, parts: [{ sec: 60, dog: 'standing' }] }
];

export const LATEST_DAY = '2026-10-25';

// Seven days of the same routine with lengths varied a little, so the week view has data.
export const seedClips = [19, 20, 21, 22, 23, 24, 25].flatMap((day, d) =>
  DAY.map((c, i) => ({
    id: `2026-10-${day}-${i + 1}`,
    file_name: `${c.camera.toLowerCase().replace(' ', '_')}_${c.time.replace(':', '')}.mp4`,
    recorded_at: `2026-10-${day}T${c.time}:00`,
    camera: c.camera,
    cover_url: c.cover,
    result: makeResult(c.parts.map((p) => ({ ...p, sec: Math.max(WINDOW, Math.round(p.sec * (0.6 + ((d * 7 + i * 3) % 9) / 10) / WINDOW) * WINDOW) })))
  }))
);

// What a finished upload looks like until the real pipeline is connected.
export const UPLOAD_RESULT_PARTS = [{ sec: 40, dog: 'standing' }, { sec: 60, dog: 'walking_run' }, { sec: 24, dog: 'sitting' }];

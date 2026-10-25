import { ACTIVE, DOG_BEHAVIOURS, REST, TOGETHER } from './labels.js';

// recorded_at is a local, timezone-free "YYYY-MM-DDTHH:mm:ss" string
export const dayOf = (recordedAt) => recordedAt.slice(0, 10);
export const minutesOfDay = (recordedAt) => Number(recordedAt.slice(11, 13)) * 60 + Number(recordedAt.slice(14, 16));

const two = (n) => String(n).padStart(2, '0');

export function clockTime(recordedAt, offsetSec = 0, { seconds = false } = {}) {
  const total = Math.floor(minutesOfDay(recordedAt) * 60 + Number(recordedAt.slice(17, 19) || 0) + offsetSec);
  const hm = `${two(Math.floor(total / 3600) % 24)}:${two(Math.floor(total / 60) % 60)}`;
  return seconds ? `${hm}:${two(total % 60)}` : hm;
}

export const formatDuration = (sec) => `${Math.floor(sec / 60)}:${two(Math.round(sec % 60))}`;

export function durationsByLabel(segments = []) {
  const out = {};
  for (const s of segments) out[s.label] = (out[s.label] || 0) + (s.end_sec - s.start_sec);
  return out;
}

// Longest label in a timeline; "insufficient_detection" only wins when nothing else was seen.
export function dominantLabel(segments = []) {
  const entries = Object.entries(durationsByLabel(segments));
  const seen = entries.filter(([label]) => label !== 'insufficient_detection');
  const pool = seen.length ? seen : entries;
  if (!pool.length) return null;
  return pool.reduce((best, e) => (e[1] > best[1] ? e : best))[0];
}

// Mean of the smoothed per-window probabilities, over windows the model actually scored.
export function averageProbabilities(windows = [], task) {
  const scored = windows.map((w) => w[task]).filter((v) => v && v.status === 'ok');
  if (!scored.length) return null;
  const sums = {};
  for (const v of scored) {
    for (const [label, p] of Object.entries(v.probabilities || v.raw_probabilities)) sums[label] = (sums[label] || 0) + p;
  }
  return Object.fromEntries(Object.entries(sums).map(([label, s]) => [label, s / scored.length]));
}

const sumOf = (durations, labels) => labels.reduce((t, l) => t + (durations[l] || 0), 0);

// Seconds of rest / activity / time together across the clips of one day.
export function dayTotals(clips) {
  const totals = { rest: 0, active: 0, together: 0 };
  for (const clip of clips) {
    const dog = durationsByLabel(clip.result.timeline.dog_behavior);
    const people = durationsByLabel(clip.result.timeline.interaction);
    totals.rest += sumOf(dog, REST);
    totals.active += sumOf(dog, ACTIVE);
    totals.together += sumOf(people, TOGETHER);
  }
  return totals;
}

const isoDate = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

// Monday-to-Sunday rows of dog-behaviour seconds for the week containing `date` ("YYYY-MM-DD").
export function weekRows(clips, date) {
  const [y, m, d] = date.split('-').map(Number);
  const monday = new Date(y, m - 1, d - ((new Date(y, m - 1, d).getDay() + 6) % 7));
  return Array.from({ length: 7 }, (_, i) => {
    const day = isoDate(new Date(monday.getFullYear(), monday.getMonth(), monday.getDate() + i));
    const seconds = Object.fromEntries(Object.keys(DOG_BEHAVIOURS).map((l) => [l, 0]));
    const dayClips = clips.filter((c) => dayOf(c.recorded_at) === day);
    for (const clip of dayClips) {
      for (const [label, s] of Object.entries(durationsByLabel(clip.result.timeline.dog_behavior))) {
        if (label in seconds) seconds[label] += s;
      }
    }
    return { date: day, clips: dayClips.length, seconds };
  });
}

// Short uploads are seconds long, so show seconds under a minute instead of "0m".
export const formatMinutes = (sec) => (sec > 0 && sec < 60 ? `${Math.round(sec)}s` : `${Math.round(sec / 60)}m`);

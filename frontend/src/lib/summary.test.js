import { describe, expect, it } from 'vitest';
import { averageProbabilities, clockTime, dayTotals, dominantLabel, durationsByLabel, formatDuration, formatMinutes, weekRows } from './summary.js';
import { makeResult, seedClips } from '../mock/clips.js';

const seg = (start_sec, end_sec, label) => ({ start_sec, end_sec, label });

describe('summary', () => {
  it('sums durations per label', () => {
    expect(durationsByLabel([seg(0, 8, 'sitting'), seg(8, 12, 'standing'), seg(12, 20, 'sitting')]))
      .toEqual({ sitting: 16, standing: 4 });
  });

  it('picks the longest real label and falls back to insufficient_detection', () => {
    expect(dominantLabel([seg(0, 40, 'insufficient_detection'), seg(40, 44, 'petting')])).toBe('petting');
    expect(dominantLabel([seg(0, 40, 'insufficient_detection')])).toBe('insufficient_detection');
    expect(dominantLabel([])).toBeNull();
  });

  it('averages only scored windows', () => {
    const result = makeResult([{ sec: 8, dog: 'sitting' }, { sec: 4, interaction: 'petting' }]);
    const dog = averageProbabilities(result.windows, 'dog_behavior');
    expect(Object.keys(dog)).toHaveLength(5);
    expect(Object.values(dog).reduce((a, b) => a + b)).toBeCloseTo(1);
    expect(dog.sitting).toBeGreaterThan(0.5);
    expect(averageProbabilities(result.windows, 'interaction').petting).toBeGreaterThan(0.5);
    expect(averageProbabilities([], 'interaction')).toBeNull();
  });

  it('matches predict_video timeline rules', () => {
    const together = makeResult([{ sec: 8, interaction: 'hand_feeding' }]);
    expect(together.timeline.dog_behavior).toBeUndefined(); // suppressed windows leave no dog timeline
    expect(together.timeline.interaction).toEqual([expect.objectContaining({ start_sec: 0, end_sec: 8, label: 'hand_feeding', windows: 2 })]);
    const alone = makeResult([{ sec: 6, dog: 'standing' }]);
    expect(alone.windows.map((w) => w.end_sec)).toEqual([4, 6]);
    expect(alone.timeline.interaction[0].label).toBe('insufficient_detection');
  });

  it('totals a day into rest, active and together', () => {
    const clip = (parts) => ({ result: makeResult(parts) });
    expect(dayTotals([clip([{ sec: 8, dog: 'lying_sleeping' }, { sec: 4, dog: 'walking_run' }]), clip([{ sec: 12, interaction: 'petting' }])]))
      .toEqual({ rest: 8, active: 4, together: 12 });
  });

  it('builds Monday-to-Sunday rows for the week of a date', () => {
    const rows = weekRows(seedClips, '2026-10-22');
    expect(rows.map((r) => r.date)).toEqual(['2026-10-19', '2026-10-20', '2026-10-21', '2026-10-22', '2026-10-23', '2026-10-24', '2026-10-25']);
    expect(rows.every((r) => r.clips === 6)).toBe(true);
    expect(weekRows(seedClips, '2026-10-26')[0].date).toBe('2026-10-26');
  });

  it('formats clock times from a recording start plus an offset', () => {
    expect(clockTime('2026-10-25T08:42:00', 0)).toBe('08:42');
    expect(clockTime('2026-10-25T08:42:00', 300)).toBe('08:47');
    expect(clockTime('2026-10-25T08:42:00', 4, { seconds: true })).toBe('08:42:04');
    expect(clockTime('2026-10-25T23:59:30', 45, { seconds: true })).toBe('00:00:15');
    expect(formatDuration(12)).toBe('0:12');
    expect(formatDuration(125)).toBe('2:05');
    expect([0, 5.2, 59, 60, 330].map(formatMinutes)).toEqual(['0m', '5s', '59s', '1m', '6m']);
  });
});

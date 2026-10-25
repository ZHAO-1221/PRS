import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { listClips } from '../api.js';
import ClipDeck from '../components/ClipDeck.jsx';
import { BackIcon, ForwardIcon, PawIcon } from '../components/Icons.jsx';
import { DOG_BEHAVIOURS, INTERACTIONS, TOGETHER } from '../lib/labels.js';
import { clockTime, dayOf, dayTotals, dominantLabel, formatDuration, formatMinutes, minutesOfDay } from '../lib/summary.js';
import useLoad from '../lib/useLoad.js';

const DAY_START = 6 * 60;
const DAY_SPAN = 16 * 60; // day line covers 06:00-22:00

const FILTERS = [
  ['all', 'All', () => true],
  ['alone', 'Alone', (c) => !TOGETHER.includes(c.interaction)],
  ['together', 'With you', (c) => TOGETHER.includes(c.interaction)]
];

function describe(clip) {
  const dog = dominantLabel(clip.result.timeline.dog_behavior);
  const interaction = dominantLabel(clip.result.timeline.interaction) || 'insufficient_detection';
  const time = clockTime(clip.recorded_at);
  return {
    id: clip.id,
    clip,
    dog,
    interaction,
    time,
    duration: formatDuration(clip.result.metadata.duration_sec),
    // with the owner in view the pipeline scores the interaction instead of the dog's own behaviour
    title: dog ? DOG_BEHAVIOURS[dog].name : INTERACTIONS[interaction].name,
    chip: INTERACTIONS[interaction].name,
    chipColor: INTERACTIONS[interaction].chip,
    color: dog ? DOG_BEHAVIOURS[dog].color : '#19384a',
    alt: `${time} clip from the ${clip.camera.toLowerCase()} camera`
  };
}

const prettyDate = (day) => new Date(`${day}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'short' });

export default function Story() {
  const { data: clips, error } = useLoad(() => listClips(), []);
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState('all');
  const [index, setIndex] = useState(0);

  if (error) return <main className="screen"><p className="error">Could not load clips: {error.message}</p></main>;
  if (!clips) return <main className="screen"><p className="sub">Loading clips…</p></main>;

  // days that have clips, oldest first; ?day= picks one, otherwise the latest
  const days = [...new Set(clips.map((c) => dayOf(c.recorded_at)))];
  const day = days.includes(params.get('day')) ? params.get('day') : days[days.length - 1] ?? null;
  const dayIndex = days.indexOf(day);
  const goDay = (d) => { setParams({ day: d }); setFilter('all'); setIndex(0); };
  const all = clips.filter((c) => dayOf(c.recorded_at) === day).map(describe);
  const shown = all.filter(FILTERS.find(([id]) => id === filter)[2]);
  const current = shown[index];
  const totals = dayTotals(all.map((d) => d.clip));
  const pick = (id) => { setFilter(id); setIndex(0); };

  return (
    <main className="screen">
      <header className="topbar">
        <div className="eyebrow-title">Activity review</div>
        <button className="icon-btn yellow" aria-label="Switch dog"><PawIcon /></button>
      </header>

      <section style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="eyebrow">{all.length} clips · analysed</div>
          <h1 className="title">{day ? prettyDate(day).split(' ')[0] : 'No clips'} at home</h1>
          <p className="sub">{day ? `${prettyDate(day)} · ${all[0].time}–${all[all.length - 1].time} · by recording time` : 'Upload a video to get started'}</p>
        </div>
        <button className="icon-btn" aria-label="Previous day" disabled={dayIndex <= 0} onClick={() => goDay(days[dayIndex - 1])}><BackIcon /></button>
        <button className="icon-btn" aria-label="Next day" disabled={dayIndex >= days.length - 1} onClick={() => goDay(days[dayIndex + 1])}><ForwardIcon /></button>
      </section>

      <div className="filters" role="group" aria-label="Filter clips">
        {FILTERS.map(([id, label, test]) => (
          <button key={id} className="filter" aria-pressed={filter === id} onClick={() => pick(id)}>
            {label} · {all.filter(test).length}
          </button>
        ))}
      </div>

      <ClipDeck items={shown} index={Math.min(index, Math.max(0, shown.length - 1))} onIndexChange={setIndex} />

      <div className="card" style={{ padding: '12px 14px' }}>
        <div className="label-row"><b style={{ fontSize: 13 }}>Day line</b><span className="hint">tap a mark to jump</span></div>
        <div className="dayline">
          {all.map((d) => {
            const left = Math.max(0, Math.min(100, ((minutesOfDay(d.clip.recorded_at) - DAY_START) / DAY_SPAN) * 100));
            return (
              <button key={d.id} className={`mark${current && current.id === d.id ? ' on' : ''}`} style={{ left: `${left}%` }}
                aria-label={`Jump to ${d.time} ${d.title}`} onClick={() => { setFilter('all'); setIndex(all.indexOf(d)); }}>
                <span style={{ background: d.color }} />
              </button>
            );
          })}
        </div>
        <div className="ticks"><span>06:00</span><span>10:00</span><span>14:00</span><span>18:00</span><span>22:00</span></div>
      </div>

      <div className="tiles">
        <div className="tile" style={{ background: 'var(--aqua)' }}><span>REST</span><strong>{formatMinutes(totals.rest)}</strong></div>
        <div className="tile" style={{ background: 'var(--yellow)' }}><span>ACTIVE</span><strong>{formatMinutes(totals.active)}</strong></div>
        <div className="tile" style={{ background: 'var(--ink)', color: 'var(--cream)' }}><span>WITH YOU</span><strong>{formatMinutes(totals.together)}</strong></div>
      </div>
    </main>
  );
}

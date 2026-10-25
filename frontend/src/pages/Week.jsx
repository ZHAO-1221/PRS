import { getModelMetrics, listClips } from '../api.js';
import { DOG_BEHAVIOURS } from '../lib/labels.js';
import { dayOf, weekRows } from '../lib/summary.js';
import useLoad from '../lib/useLoad.js';

const BAR_HEIGHT = 140;
const short = (day) => new Date(`${day}T00:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });

export default function Week() {
  const { data: clips, error } = useLoad(() => listClips(), []);
  const { data: metrics } = useLoad(() => getModelMetrics(), []);
  const accuracy = (task) => (metrics ? `${(metrics[task].accuracy * 100).toFixed(1)}%` : '…');
  const testClips = (task) => (metrics ? `accuracy · ${metrics[task].test_clips} test clips` : 'accuracy');
  if (error) return <main className="screen"><p className="error">Could not load clips: {error.message}</p></main>;
  if (!clips) return <main className="screen"><p className="sub">Loading week…</p></main>;
  if (!clips.length) return <main className="screen"><p className="sub">No clips yet.</p></main>;

  const rows = weekRows(clips, dayOf(clips[clips.length - 1].recorded_at));
  const labels = Object.keys(DOG_BEHAVIOURS);
  const dayTotal = (r) => labels.reduce((t, l) => t + r.seconds[l], 0);
  const max = Math.max(1, ...rows.map(dayTotal));
  const weekTotals = labels.map((l) => [l, rows.reduce((t, r) => t + r.seconds[l], 0)]);
  const top = weekTotals.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  const clipCount = rows.reduce((t, r) => t + r.clips, 0);

  return (
    <main className="screen">
      <header className="topbar">
        <div className="eyebrow-title">This week</div>
        <span className="pill">{short(rows[0].date)}–{short(rows[6].date)}</span>
      </header>

      <section>
        <div className="eyebrow">Weekly round-up</div>
        <h1 className="title">Mostly<br />{DOG_BEHAVIOURS[top].name.toLowerCase()}</h1>
        <p className="sub">{clipCount} clip{clipCount === 1 ? '' : 's'} · minutes per day, placed by recording time</p>
      </section>

      <div className="card">
        <div className="bars" role="img" aria-label="Stacked daily minutes per behaviour, Monday to Sunday">
          {rows.map((r) => (
            <div key={r.date}>
              {labels.map((l) => r.seconds[l] > 0 && (
                <span key={l} style={{ height: (r.seconds[l] / max) * BAR_HEIGHT, background: DOG_BEHAVIOURS[l].color }} />
              ))}
            </div>
          ))}
        </div>
        <div className="days" aria-hidden="true">{['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((d, i) => <span key={i}>{d}</span>)}</div>
        <div className="legend">
          {labels.map((l) => <span key={l}><i style={{ background: DOG_BEHAVIOURS[l].color }} />{DOG_BEHAVIOURS[l].short}</span>)}
        </div>
      </div>

      <section className="card-dark" aria-label="Model accuracy" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div className="label">How sure are we?</div>
        <div className="stats">
          <div><span>Behaviour</span><strong>{accuracy('dog_behavior')}</strong><small>{testClips('dog_behavior')}</small></div>
          <div><span>Interaction</span><strong>{accuracy('interaction')}</strong><small>{testClips('interaction')}</small></div>
        </div>
        <p className="sub" style={{ color: 'var(--muted-on-ink)', fontSize: 11 }}>Skeleton-based: only 17 body points per frame are used to recognise behaviour.</p>
      </section>
    </main>
  );
}

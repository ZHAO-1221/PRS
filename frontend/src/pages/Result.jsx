import { Link, useParams } from 'react-router-dom';
import { getClip } from '../api.js';
import Cover from '../components/Cover.jsx';
import { BackIcon } from '../components/Icons.jsx';
import { DOG_BEHAVIOURS, INTERACTIONS } from '../lib/labels.js';
import { averageProbabilities, clockTime, dayOf, formatDuration } from '../lib/summary.js';
import useLoad from '../lib/useLoad.js';

const pct = (p) => `${Math.round(p * 100)}%`;
const ranked = (probs) => Object.entries(probs).sort((a, b) => b[1] - a[1]);

export default function Result() {
  const { id } = useParams();
  const { data: clip, error } = useLoad(() => getClip(id), [id]);

  if (error) return <main className="screen"><p className="error">{error.message}</p><Link className="primary" to="/">Back to story</Link></main>;
  if (!clip) return <main className="screen"><p className="sub">Loading result…</p></main>;

  const { windows, timeline, metadata } = clip.result;
  const dog = averageProbabilities(windows, 'dog_behavior');
  const people = averageProbabilities(windows, 'interaction');
  const segments = [
    ...(timeline.dog_behavior || []).map((s) => ({ ...s, name: DOG_BEHAVIOURS[s.label].name, color: DOG_BEHAVIOURS[s.label].color })),
    ...timeline.interaction.filter((s) => s.label !== 'insufficient_detection')
      .map((s) => ({ ...s, name: INTERACTIONS[s.label].name, color: INTERACTIONS[s.label].chip }))
  ].sort((a, b) => a.start_sec - b.start_sec);

  return (
    <main className="screen">
      <header className="topbar" style={{ justifyContent: 'flex-start' }}>
        <Link to={`/?day=${dayOf(clip.recorded_at)}`} className="icon-btn" aria-label="Back to story"><BackIcon /></Link>
        <div>
          <div className="eyebrow">{clockTime(clip.recorded_at)} · {formatDuration(metadata.duration_sec)} · {clip.camera}</div>
          <div style={{ fontWeight: 900, fontSize: 18 }}>Clip result</div>
        </div>
      </header>

      <div className="result-cover"><Cover clip={clip} alt={`Clip recorded at ${clockTime(clip.recorded_at)}`} /></div>

      <section className="card" aria-label="Dog behaviour">
        <div className="label-row"><span className="label">The dog is</span><span className="hint">5 behaviours</span></div>
        {dog ? (
          <>
            <div className="big">{DOG_BEHAVIOURS[ranked(dog)[0][0]].name}</div>
            <div className="probs">
              {ranked(dog).map(([label, p]) => (
                <div key={label} className="prob">
                  <span>{DOG_BEHAVIOURS[label].name}</span>
                  <span className="prob-track"><span style={{ width: pct(p), background: DOG_BEHAVIOURS[label].color }} /></span>
                  <em>{pct(p)}</em>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="sub">Not scored: the owner was in view the whole time, so this clip was read as an interaction.</p>
        )}
      </section>

      <section className="card-dark" aria-label="Interaction">
        <div className="label-row"><span className="label">With you</span><span className="hint">4 interactions</span></div>
        {people ? (
          <>
            <div className="big">{INTERACTIONS[ranked(people)[0][0]].name}</div>
            <div className="chips">
              {ranked(people).map(([label, p], i) => (
                <span key={label} className={`chip${i === 0 ? ' top' : ''}`}>{INTERACTIONS[label].name} {pct(p)}</span>
              ))}
            </div>
          </>
        ) : (
          <p className="sub" style={{ color: 'var(--muted-on-ink)' }}>No owner in view: the dog was alone in this clip.</p>
        )}
      </section>

      <section className="card" aria-label="Timeline">
        <div className="label-row" style={{ marginBottom: 8 }}><span className="label">Timeline</span><span className="hint">{segments.length} segment{segments.length === 1 ? '' : 's'}</span></div>
        <ul className="segments">
          {segments.map((s) => (
            <li key={`${s.start_sec}-${s.label}`}>
              <i style={{ background: s.color }} />
              <b>{s.name}</b>
              <span>{clockTime(clip.recorded_at, s.start_sec, { seconds: true })}–{clockTime(clip.recorded_at, s.end_sec, { seconds: true })} · {pct(s.confidence)}</span>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}

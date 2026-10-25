import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getJob, STEPS } from '../api.js';
import Cover from '../components/Cover.jsx';

const STEP_TEXT = {
  detect: ['Find dog & owner', 'RTMDet boxes, tracked frame to frame'],
  keypoints: ['Trace 17 keypoints', 'RTMPose skeletons at 15 fps'],
  behaviour: ['Recognise behaviour', 'Motion-aware TCN · 5 behaviours'],
  interaction: ['Read the interaction', 'Owner–dog relation · 4 types']
};

const prettyTime = (recordedAt) =>
  new Date(recordedAt).toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

export default function Analyze() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let live = true;
    let timer;
    const poll = () => getJob(jobId).then((j) => {
      if (!live) return;
      setJob(j);
      if (j.status !== 'done' && j.status !== 'failed') timer = setTimeout(poll, 1000);
    }, (err) => live && setError(err.message));
    poll();
    return () => { live = false; clearTimeout(timer); };
  }, [jobId]);

  if (error) return <main className="screen"><p className="error">{error}</p><Link className="primary" to="/upload">Back to upload</Link></main>;
  if (!job) return <main className="screen"><p className="sub">Starting…</p></main>;
  const done = job.status === 'done';
  const failed = job.status === 'failed';
  const failedAt = job.stages.findIndex((s) => s.status === 'failed');

  return (
    <main className="screen">
      <header className="topbar">
        <div className="eyebrow-title">Analyze a clip</div>
        <span className="pill">{done ? 'Done' : failed ? 'Failed' : job.status === 'queued' ? 'Waiting' : 'Running'}</span>
      </header>

      <section>
        <div className="eyebrow">{done ? 'All steps done' : failed ? (failedAt >= 0 ? `Stopped at step ${failedAt + 1}` : 'Stopped') : `Step ${job.step + 1} of ${STEPS.length}`}</div>
        <h1 className="title">{done ? 'Analysis' : failed ? 'Analysis' : 'Reading'}<br />{done ? 'complete' : failed ? 'failed' : 'body language…'}</h1>
        <p className="sub">{job.clip.file_name} · recorded {prettyTime(job.clip.recorded_at)} · {job.clip.camera}</p>
      </section>

      <div className="progress-frame">
        <Cover clip={job.clip} alt="Frame from the uploaded clip" />
        <div className="bar" role="progressbar" aria-label="Analysis progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}>
          <div style={{ width: `${job.progress * 100}%` }} />
        </div>
      </div>

      <ol className="steps" aria-label="Analysis steps">
        {job.stages.map(({ name, status, done: count, total }, i) => {
          const state = status === 'done' || status === 'running' ? status : 'next';
          return (
            <li key={name} className={`step ${state}`}>
              <span className="step-dot">{status === 'done' ? '✓' : i + 1}</span>
              <span className="step-text"><b>{STEP_TEXT[name][0]}</b><span>{STEP_TEXT[name][1]}</span></span>
              <span className="step-state">
                {status === 'done' ? 'Done' : status === 'running' ? `Working${total > 1 ? ` ${count}/${total}` : ''}` : status === 'failed' ? 'Failed' : 'Next'}
              </span>
            </li>
          );
        })}
      </ol>

      {failed && <p className="error" role="alert">{job.error}</p>}
      {failed ? (
        <Link className="primary" to="/upload" style={{ marginTop: 'auto' }}>Upload again</Link>
      ) : (
        <Link className="primary" to={`/clips/${job.clip.id}`} aria-disabled={!done} tabIndex={done ? 0 : -1} style={{ marginTop: 'auto' }}>
          See the result
        </Link>
      )}
    </main>
  );
}

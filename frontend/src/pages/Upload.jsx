import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadClip } from '../api.js';
import { UploadIcon } from '../components/Icons.jsx';

const pad = (n) => String(n).padStart(2, '0');
const localDate = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const localTime = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
const CAMERAS = ['Living room', 'Kitchen', 'Bedroom', 'Garden'];

export default function Upload() {
  const navigate = useNavigate();
  const now = new Date();
  const [file, setFile] = useState(null);
  const [date, setDate] = useState(localDate(now));
  const [time, setTime] = useState(localTime(now));
  const [camera, setCamera] = useState(CAMERAS[0]);
  const [error, setError] = useState(null);
  const [sending, setSending] = useState(false);

  const pickFile = (e) => {
    const chosen = e.target.files?.[0];
    if (!chosen) return;
    setFile(chosen);
    setError(null);
    // shortcut: file.lastModified stands in for the recording time; the backend should read the video's creation metadata
    const t = new Date(chosen.lastModified);
    setDate(localDate(t));
    setTime(localTime(t));
  };

  const submit = async (e) => {
    e.preventDefault();
    setSending(true);
    setError(null);
    try {
      const { job_id } = await uploadClip(file, { recordedAt: `${date}T${time}`, camera });
      navigate(`/analyze/${job_id}`);
    } catch (err) {
      setError(err.message);
      setSending(false);
    }
  };

  return (
    <main className="screen">
      <header className="topbar">
        <div className="eyebrow-title">Upload a clip</div>
        <span className="pill"><i style={{ width: 8, height: 8, borderRadius: '50%', background: '#8a969c' }} />No camera linked</span>
      </header>

      <section>
        <div className="eyebrow">Manual mode</div>
        <h1 className="title" style={{ fontSize: 30 }}>Bring your own<br />footage</h1>
        <p className="sub">A linked camera streams clips here on its own. Until then, upload a video and it runs through the same analysis.</p>
      </section>

      <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: 1 }}>
        <label className="drop">
          <input className="visually-hidden" type="file" accept="video/*" onChange={pickFile} />
          <span className="drop-icon"><UploadIcon /></span>
          <b>{file ? file.name : 'Choose a video'}</b>
          <span>{file ? 'Tap to pick a different one' : 'MP4 or MOV · one dog, up to 500 MB'}</span>
        </label>

        <fieldset className="card">
          <legend>WHEN WAS IT RECORDED?</legend>
          <div className="two">
            <label className="field">Date<input type="date" required value={date} onChange={(e) => setDate(e.target.value)} /></label>
            <label className="field">Start time<input type="time" required value={time} onChange={(e) => setTime(e.target.value)} /></label>
          </div>
          <p className="sub" style={{ fontSize: 11 }}>Filled in from the video file when you pick it. This places the clip in the daily story and the weekly report.</p>
        </fieldset>

        <label className="field">Which camera would have seen it?
          <select value={camera} onChange={(e) => setCamera(e.target.value)}>
            {CAMERAS.map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>

        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary" type="submit" disabled={!file || sending} style={{ marginTop: 'auto' }}>
          {sending ? 'Uploading…' : 'Start analysis'}
        </button>
      </form>
    </main>
  );
}

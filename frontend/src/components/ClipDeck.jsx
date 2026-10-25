import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import Cover from './Cover.jsx';
import { DownIcon, PlayIcon, UpIcon } from './Icons.jsx';

const SETTLE = 'transform 280ms cubic-bezier(.2,.8,.2,1), opacity 220ms';

// Stacked cover cards. Swipe up sends the front card behind the stack, swipe down brings
// the last one back; the list loops and all motion stays inside the clipped deck box.
// items: [{ id, clip, title, chip, chipColor, time, duration, alt }]
export default function ClipDeck({ items, index, onIndexChange }) {
  const [dragY, setDragY] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState(null);
  const drag = useRef({ startY: 0, dy: 0 });
  const busy = useRef(false);
  const wheelLock = useRef(false);
  const n = items.length;

  const run = (dir) => {
    setDragging(false);
    if (n < 2 || busy.current) { setDragY(0); return; }
    busy.current = true;
    setStage(dir > 0 ? 'out' : 'in');
    if (dir < 0) setDragY(0);
    setTimeout(() => {
      setStage(null);
      setDragY(0);
      onIndexChange((index + dir + n) % n);
      busy.current = false;
    }, dir > 0 ? 160 : 200);
  };

  const onPointerDown = (e) => {
    drag.current = { startY: e.clientY, dy: 0 };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDragging(true);
    setDragY(0);
  };
  const onPointerMove = (e) => {
    if (!dragging) return;
    drag.current.dy = e.clientY - drag.current.startY; // read on release, before React has flushed the last move
    setDragY(drag.current.dy);
  };
  const onPointerUp = () => {
    if (!dragging) return;
    const { dy } = drag.current;
    if (dy < -50) run(1);
    else if (dy > 50) run(-1);
    else { setDragging(false); setDragY(0); }
  };
  const onWheel = (e) => {
    if (wheelLock.current || Math.abs(e.deltaY) < 20) return;
    wheelLock.current = true;
    setTimeout(() => { wheelLock.current = false; }, 450);
    run(e.deltaY > 0 ? 1 : -1);
  };

  const pull = dragging ? Math.max(0, Math.min(1, -dragY / 150)) : 0;

  return (
    <>
      <div className="deck" aria-label="Clip stack, swipe up or down" onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onLostPointerCapture={onPointerUp} onPointerLeave={onPointerUp} onWheel={onWheel}>
        {n === 0 && <div className="deck-empty">No clips for this filter</div>}
        {items.map((item, i) => {
          const k = (i - index + n) % n;
          let y, s = 1, opacity = 1, z = 50 - k, transition = dragging ? 'none' : SETTLE;
          if (k === 0) {
            z = 100;
            y = stage === 'out' ? -150 : dragY < 0 ? Math.max(-130, dragY) : Math.min(20, dragY * 0.2);
            if (stage === 'out') transition = 'transform 160ms ease-in';
          } else if (stage === 'in' && k === n - 1) {
            // the previous card rises from behind the stack before sliding to the front
            y = -150; z = 1; transition = 'transform 200ms ease-out';
          } else {
            const d = Math.min(k - pull, 2);
            y = d * 12; s = 1 - d * 0.05;
            if (k > 2) opacity = 0;
          }
          return (
            <article key={item.id} className="deck-card" aria-hidden={k !== 0}
              style={{ transform: `translateY(${y}px) scale(${s})`, opacity, zIndex: z, transition }}>
              <div style={{ position: 'relative' }}>
                <Cover clip={item.clip} alt={item.alt} />
                <span className="deck-time">{item.time} · {item.duration}</span>
              </div>
              <div className="deck-info">
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 5 }}>
                  <b>{item.title}</b>
                  <span className="chip" style={{ alignSelf: 'flex-start', background: item.chipColor }}>{item.chip}</span>
                </div>
                <Link to={`/clips/${item.id}`} className="play" aria-label={`Open ${item.time} clip`} tabIndex={k === 0 ? 0 : -1}
                  onPointerDown={(e) => e.stopPropagation()}>
                  <PlayIcon />
                </Link>
              </div>
            </article>
          );
        })}
      </div>
      <div className="deck-controls">
        <span>{n ? `${index + 1} / ${n}` : '0 / 0'} · swipe up or down, it loops</span>
        <button className="icon-btn" aria-label="Previous clip" onClick={() => run(-1)}><UpIcon /></button>
        <button className="icon-btn dark" aria-label="Next clip" onClick={() => run(1)}><DownIcon /></button>
      </div>
    </>
  );
}

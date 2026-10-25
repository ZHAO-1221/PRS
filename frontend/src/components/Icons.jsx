const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };

export const StoryIcon = () => <svg width="22" height="22" viewBox="0 0 24 24" {...stroke}><path d="M4 5h16v14H4z" /><path d="M8 9h8M8 13h5" /></svg>;
export const UploadIcon = ({ size = 22, ...p }) => <svg width={size} height={size} viewBox="0 0 24 24" {...stroke} {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v4h16v-4" /></svg>;
export const WeekIcon = () => <svg width="22" height="22" viewBox="0 0 24 24" {...stroke}><path d="M5 20V11M12 20V5M19 20v-7" /></svg>;
export const UpIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} strokeWidth="2.4"><path d="M6 15l6-6 6 6" /></svg>;
export const DownIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} strokeWidth="2.4"><path d="M6 9l6 6 6-6" /></svg>;
export const BackIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} strokeWidth="2.4"><path d="M15 5l-7 7 7 7" /></svg>;
export const ForwardIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" {...stroke} strokeWidth="2.4"><path d="M9 5l7 7-7 7" /></svg>;
export const PlayIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff" aria-hidden="true"><path d="M7 4l13 8-13 8z" /></svg>;
export const PawIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <ellipse cx="12" cy="16" rx="5" ry="4.2" /><circle cx="5" cy="10" r="2.2" /><circle cx="9.5" cy="5.5" r="2.2" /><circle cx="14.5" cy="5.5" r="2.2" /><circle cx="19" cy="10" r="2.2" />
  </svg>
);

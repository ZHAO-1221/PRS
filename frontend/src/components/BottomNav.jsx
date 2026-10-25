import { NavLink, useLocation } from 'react-router-dom';
import { StoryIcon, UploadIcon, WeekIcon } from './Icons.jsx';

export default function BottomNav() {
  const { pathname } = useLocation();
  // clip results belong to Story, analysis progress belongs to Upload
  const section = pathname.startsWith('/analyze') ? '/upload' : pathname.startsWith('/clips') ? '/' : pathname;
  const link = (to, label, Icon) => (
    <NavLink to={to} className={section === to ? 'active' : ''} aria-current={section === to ? 'page' : undefined}>
      <Icon />{label}
    </NavLink>
  );
  return (
    <nav className="nav" aria-label="Main">
      {link('/', 'Story', StoryIcon)}
      {link('/upload', 'Upload', UploadIcon)}
      {link('/week', 'Week', WeekIcon)}
    </nav>
  );
}

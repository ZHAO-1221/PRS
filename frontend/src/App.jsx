import { Navigate, Route, Routes } from 'react-router-dom';
import BottomNav from './components/BottomNav.jsx';
import Story from './pages/Story.jsx';
import Upload from './pages/Upload.jsx';
import Analyze from './pages/Analyze.jsx';
import Result from './pages/Result.jsx';
import Week from './pages/Week.jsx';

export default function App() {
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<Story />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/analyze/:jobId" element={<Analyze />} />
        <Route path="/clips/:id" element={<Result />} />
        <Route path="/week" element={<Week />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <BottomNav />
    </div>
  );
}

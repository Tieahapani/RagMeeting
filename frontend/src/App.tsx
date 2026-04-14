import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import MeetingList from './pages/MeetingList'
import RecordMeeting from './pages/RecordMeeting'
import MeetingDetail from './pages/MeetingDetail'

function NavBar() {
  const location = useLocation()

  function isActive(path: string) {
    return location.pathname === path
  }

  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-4xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="text-lg font-semibold text-violet-600 tracking-tight">
          RAGMeeting
        </Link>
        <div className="flex gap-1">
          <Link
            to="/"
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              isActive('/') ? 'bg-violet-100 text-violet-700' : 'text-gray-500 hover:text-gray-900'
            }`}
          >
            Meetings
          </Link>
          <Link
            to="/record"
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              isActive('/record') ? 'bg-violet-100 text-violet-700' : 'text-gray-500 hover:text-gray-900'
            }`}
          >
            Record
          </Link>
        </div>
      </div>
    </nav>
  )
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <NavBar />
        <main className="max-w-4xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<MeetingList />} />
            <Route path="/record" element={<RecordMeeting />} />
            <Route path="/meeting/:id" element={<MeetingDetail />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App

import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'

interface Meeting {
  id: string
  title: string
  date: string
  duration: number
  status: string
}

function MeetingList() {
  const [meetings, setMeetings] = useState<Meeting[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('http://localhost:8000/meetings/')
      .then((res) => res.json())
      .then((data) => {
        setMeetings(data)
        setLoading(false)
      })
      .catch(() => {
        setError('Could not connect to backend')
        setLoading(false)
      })
  }, [])

  function formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 text-red-600 rounded-xl p-4 text-sm">
        {error}
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">My Meetings</h1>
          <p className="text-gray-500 mt-1">
            {meetings.length} meeting{meetings.length !== 1 && 's'} recorded
          </p>
        </div>
        <Link to="/record">
          <button className="bg-violet-600 hover:bg-violet-700 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-colors shadow-sm shadow-violet-200 cursor-pointer">
            + New Meeting
          </button>
        </Link>
      </div>

      {/* Empty State */}
      {meetings.length === 0 && (
        <div className="text-center py-20 bg-white rounded-2xl border border-gray-100">
          <div className="text-4xl mb-4">🎙</div>
          <p className="text-gray-500">No meetings yet</p>
          <p className="text-gray-400 text-sm mt-1">Click "+ New Meeting" to record your first one</p>
        </div>
      )}

      {/* Meeting Cards */}
      <div className="space-y-3">
        {meetings.map((meeting) => (
          <Link
            to={`/meeting/${meeting.id}`}
            key={meeting.id}
            className="block bg-white rounded-xl border border-gray-100 p-5 hover:border-violet-200 hover:shadow-md hover:shadow-violet-50 transition-all group"
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-gray-900 group-hover:text-violet-700 transition-colors">
                  {meeting.title}
                </h3>
                <p className="text-sm text-gray-400 mt-1">
                  {new Date(meeting.date).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                  <span className="mx-2">·</span>
                  {formatDuration(meeting.duration)}
                </p>
              </div>
              <span className="text-gray-300 group-hover:text-violet-400 transition-colors text-xl">
                →
              </span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

export default MeetingList

import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_BASE } from '../config'

function RecordMeeting() {
  const [isRecording, setIsRecording] = useState(false)
  const [meetingId, setMeetingId] = useState('')
  const [status, setStatus] = useState('Ready to record')
  const [error, setError] = useState('')
  const [processing, setProcessing] = useState(false)
  const [canRetry, setCanRetry] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const navigate = useNavigate()

  async function startRecording() {
    setError('')

    try {
      const res = await fetch(`${API_BASE}/meetings/start`, {
        method: 'POST',
      })
      const data = await res.json()
      setMeetingId(data.meeting_id)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
      setStatus('Recording...')
    } catch {
      setError('Could not start recording. Check microphone permissions.')
    }
  }

  async function stopRecording() {
    if (!mediaRecorderRef.current) return

    setStatus('Stopping...')

    const audioBlob = await new Promise<Blob>((resolve) => {
      mediaRecorderRef.current!.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        resolve(blob)
      }
      mediaRecorderRef.current!.stop()
    })

    mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
    setIsRecording(false)
    setProcessing(true)
    setStatus('Processing your meeting...')

    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')

      const res = await fetch(`${API_BASE}/meetings/${meetingId}/stop`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('Backend returned an error')
      pollForResult(meetingId)
    } catch {
      setError('Failed to upload recording. Please try again.')
      setStatus('Error occurred')
      setProcessing(false)
    }
  }

  function pollForResult(id: string) {
    setProcessing(true)
    setStatus('Processing your meeting...')
    setError('')
    setCanRetry(false)

    const poll = setInterval(async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/meetings/${id}`)
        if (!statusRes.ok) return
        const data = await statusRes.json()

        if (data.status === 'processed') {
          clearInterval(poll)
          setStatus('Done! Redirecting...')
          setTimeout(() => navigate(`/meeting/${id}`), 1000)
        } else if (data.status === 'failed') {
          clearInterval(poll)
          setError('Processing failed. Your recording is saved — you can retry.')
          setStatus('Error occurred')
          setProcessing(false)
          setCanRetry(true)
        }
      } catch {
        // ignore polling errors, keep trying
      }
    }, 3000)
  }

  async function retryProcessing() {
    if (!meetingId) return

    try {
      const res = await fetch(`${API_BASE}/meetings/${meetingId}/retry`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Retry failed')
      pollForResult(meetingId)
    } catch {
      setError('Failed to retry. Please try again later.')
    }
  }

  return (
    <div className="flex flex-col items-center pt-12">
      {/* Recording Visualizer */}
      <div
        className={`w-40 h-40 rounded-full flex items-center justify-center mb-8 transition-all duration-500 ${
          isRecording
            ? 'bg-red-100 shadow-[0_0_0_20px_rgba(239,68,68,0.1)] animate-pulse'
            : processing
              ? 'bg-violet-100 shadow-[0_0_0_20px_rgba(139,92,246,0.1)] animate-pulse'
              : 'bg-gray-100'
        }`}
      >
        <span className="text-5xl">
          {isRecording ? '🔴' : processing ? '⚡' : '🎙'}
        </span>
      </div>

      {/* Status */}
      <p className={`text-lg font-medium mb-2 ${
        isRecording ? 'text-red-600' : processing ? 'text-violet-600' : 'text-gray-900'
      }`}>
        {status}
      </p>

      {error && (
        <div className="bg-red-50 text-red-600 rounded-xl px-4 py-2.5 text-sm mb-4 max-w-md">
          {error}
        </div>
      )}

      {meetingId && !processing && (
        <p className="text-xs text-gray-400 mb-6 font-mono">{meetingId}</p>
      )}

      {/* Processing Spinner */}
      {processing && (
        <div className="flex items-center gap-3 mb-6 text-sm text-violet-600">
          <div className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
          Transcribing & summarizing...
        </div>
      )}

      {/* Buttons */}
      {!processing && (
        !isRecording ? (
          <div className="flex flex-col items-center gap-3">
            <button
              onClick={startRecording}
              className="bg-violet-600 hover:bg-violet-700 text-white px-8 py-3 rounded-xl text-sm font-medium transition-all shadow-lg shadow-violet-200 hover:shadow-xl hover:shadow-violet-300 cursor-pointer"
            >
              Start Recording
            </button>
            {canRetry && (
              <button
                onClick={retryProcessing}
                className="bg-amber-500 hover:bg-amber-600 text-white px-8 py-3 rounded-xl text-sm font-medium transition-all shadow-lg shadow-amber-200 hover:shadow-xl hover:shadow-amber-300 cursor-pointer"
              >
                Retry Processing
              </button>
            )}
          </div>
        ) : (
          <button
            onClick={stopRecording}
            className="bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-xl text-sm font-medium transition-all shadow-lg shadow-red-200 hover:shadow-xl hover:shadow-red-300 cursor-pointer"
          >
            Stop Recording
          </button>
        )
      )}
    </div>
  )
}

export default RecordMeeting

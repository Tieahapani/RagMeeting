import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { API_BASE } from '../config'

// ── Types ───────────────────────────────────────────────────────────────────

interface ActionItem {
  task: string
  owner: string | null
  due_date: string | null
}

interface MeetingData {
  id: string
  title: string
  date: string
  duration: number
  status: string
  summary: string | null
  transcript: string | null
  key_points: string[]
  action_items: ActionItem[]
}

interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  strategy?: string
  status?: string    // pipeline step: "Routing...", "Retrieving...", etc.
  audio_url?: string | null
  isError?: boolean
}

// ── Component ───────────────────────────────────────────────────────────────

function MeetingDetail() {
  const { id } = useParams<{ id: string }>()

  const [meeting, setMeeting] = useState<MeetingData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retrying, setRetrying] = useState(false)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [provider, setProvider] = useState<'gemini' | 'ollama'>('gemini')
  const [switching, setSwitching] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  // Fetch current LLM provider on mount
  useEffect(() => {
    fetch(`${API_BASE}/settings/provider`)
      .then((res) => res.json())
      .then((data) => setProvider(data.provider))
      .catch(() => {})
  }, [])

  async function toggleProvider() {
    const next = provider === 'gemini' ? 'ollama' : 'gemini'
    setSwitching(true)
    try {
      const res = await fetch(`${API_BASE}/settings/provider`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: next }),
      })
      if (res.ok) setProvider(next)
    } finally {
      setSwitching(false)
    }
  }

  useEffect(() => {
    fetch(`${API_BASE}/meetings/${id}`)
      .then((res) => {
        if (!res.ok) throw new Error('Meeting not found')
        return res.json()
      })
      .then((data) => {
        setMeeting(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [id])

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function askQuestion() {
    if (!question.trim() || !id) return

    const userMessage: ChatMessage = { role: 'user', text: question }
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setAsking(true)

    // Add assistant placeholder — we'll update this as events arrive
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', text: '', status: 'Connecting...' },
    ])

    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          meeting_id: id,
          tts: false,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => null)
        throw new Error(err?.detail || 'Query failed')
      }

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()

      let fullAnswer = ''
      let strategy = ''
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()!

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data: ')) continue

          let event: { type: string; data?: string; cached?: boolean }
          try {
            event = JSON.parse(line.slice(6))
          } catch {
            continue // skip malformed JSON
          }

          if (event.type === 'status') {
            // Show pipeline step (Routing..., Retrieving..., Generating...)
            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: 'assistant', text: fullAnswer, strategy, status: event.data },
            ])
          }

          if (event.type === 'strategy') {
            strategy = event.data!
          }

          if (event.type === 'token') {
            fullAnswer += event.data!
            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: 'assistant', text: fullAnswer, strategy, status: undefined },
            ])
          }

          if (event.type === 'done') {
            // Clear status, finalize message
            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: 'assistant', text: fullAnswer, strategy },
            ])
          }

          if (event.type === 'error') {
            throw new Error(event.data)
          }
        }
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Our AI service is temporarily busy. Please try again.'
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last?.role === 'assistant' && !last.text) {
          return [
            ...prev.slice(0, -1),
            { role: 'assistant', text: errorMsg, isError: true },
          ]
        }
        return [
          ...prev,
          { role: 'assistant', text: errorMsg, isError: true },
        ]
      })
    } finally {
      setAsking(false)
    }
  }

  function retryLastQuestion() {
    // Find the last user message before the error
    const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user')
    if (!lastUserMsg) return

    // Remove the error message, set the question, let user click Ask
    setMessages((prev) => prev.filter((m) => !m.isError))
    setQuestion(lastUserMsg.text)
  }

  async function retryProcessing() {
    if (!id) return
    setRetrying(true)
    try {
      const res = await fetch(`${API_BASE}/meetings/${id}/retry`, { method: 'POST' })
      if (!res.ok) throw new Error('Retry failed')

      // Poll until done
      const poll = setInterval(async () => {
        const statusRes = await fetch(`${API_BASE}/meetings/${id}`)
        if (!statusRes.ok) return
        const data = await statusRes.json()
        if (data.status === 'processed' || data.status === 'failed') {
          clearInterval(poll)
          setMeeting(data)
          setRetrying(false)
        }
      }, 3000)
    } catch {
      setError('Retry failed. Please try again later.')
      setRetrying(false)
    }
  }

  function formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return <div className="bg-red-50 text-red-600 rounded-xl p-4 text-sm">{error}</div>
  }

  if (!meeting) return <p className="text-gray-500">Meeting not found</p>

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">{meeting.title}</h1>
        <p className="text-gray-400 mt-1 text-sm">
          {new Date(meeting.date).toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
            year: 'numeric',
          })}
          <span className="mx-2">·</span>
          {formatDuration(meeting.duration)}
        </p>
      </div>

      {/* Failed Meeting Banner */}
      {meeting.status === 'failed' && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium text-red-700">
              Processing failed — your recording is saved.
            </p>
            <button
              onClick={retryProcessing}
              disabled={retrying}
              className="bg-red-600 hover:bg-red-700 disabled:bg-red-300 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              {retrying ? 'Retrying...' : 'Retry Processing'}
            </button>
          </div>
          {meeting.transcript && (
            <details className="mt-2">
              <summary className="text-sm text-red-600 cursor-pointer hover:text-red-800">
                View saved transcript
              </summary>
              <p className="mt-2 text-sm text-gray-700 whitespace-pre-wrap bg-white rounded-lg p-4 max-h-60 overflow-y-auto">
                {meeting.transcript}
              </p>
            </details>
          )}
        </div>
      )}

      {/* Processing Banner */}
      {meeting.status === 'processing' && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 flex items-center gap-3">
          <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-medium text-amber-700">
            Processing your meeting — this may take a minute...
          </p>
        </div>
      )}

      {/* Summary */}
      {meeting.summary && (
        <div className="bg-white rounded-xl border border-gray-100 p-6">
          <h2 className="text-sm font-semibold text-violet-600 uppercase tracking-wider mb-3">
            Summary
          </h2>
          <p className="text-gray-700 leading-relaxed">{meeting.summary}</p>
        </div>
      )}

      {/* Key Points & Action Items Grid */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Key Points */}
        {meeting.key_points.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-100 p-6">
            <h2 className="text-sm font-semibold text-emerald-600 uppercase tracking-wider mb-3">
              Key Points
            </h2>
            <ul className="space-y-2">
              {meeting.key_points.map((point, i) => (
                <li key={i} className="flex gap-2 text-sm text-gray-700">
                  <span className="text-emerald-400 mt-0.5 shrink-0">●</span>
                  {point}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Action Items */}
        {meeting.action_items.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-100 p-6">
            <h2 className="text-sm font-semibold text-amber-600 uppercase tracking-wider mb-3">
              Action Items
            </h2>
            <div className="space-y-3">
              {meeting.action_items.map((item, i) => (
                <div key={i} className="bg-amber-50 rounded-lg p-3">
                  <p className="text-sm font-medium text-gray-900">{item.task}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {item.owner && <span>Owner: {item.owner}</span>}
                    {item.owner && item.due_date && <span className="mx-1">·</span>}
                    {item.due_date && <span>Due: {item.due_date}</span>}
                    {!item.owner && !item.due_date && <span>Unassigned</span>}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Chat Section */}
      <div className="bg-white rounded-xl border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-violet-600 uppercase tracking-wider">
            Ask about this meeting
          </h2>
          <button
            onClick={toggleProvider}
            disabled={switching || asking}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-200 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
          >
            <span className={`w-2 h-2 rounded-full ${provider === 'gemini' ? 'bg-blue-500' : 'bg-green-500'}`} />
            {switching ? 'Switching...' : provider === 'gemini' ? 'Gemini' : 'Ollama'}
          </button>
        </div>

        {/* Messages */}
        {messages.length > 0 && (
          <div className="space-y-3 mb-4 max-h-96 overflow-y-auto pr-1">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm ${
                    msg.role === 'user'
                      ? 'bg-violet-600 text-white'
                      : msg.isError
                        ? 'bg-red-50 text-red-700 border border-red-200'
                        : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {/* Pipeline status (Routing..., Retrieving..., Generating...) */}
                  {msg.status && !msg.text && (
                    <p className="flex items-center gap-2 text-gray-500">
                      <span className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                      {msg.status}
                    </p>
                  )}
                  {msg.text && <p>{msg.text}</p>}
                  {msg.isError && (
                    <button
                      onClick={retryLastQuestion}
                      className="mt-2 bg-red-100 hover:bg-red-200 text-red-700 text-xs font-medium px-3 py-1 rounded-lg transition-colors cursor-pointer"
                    >
                      Retry
                    </button>
                  )}
                  {msg.strategy && (
                    <p className={`text-xs mt-1 ${
                      msg.role === 'user' ? 'text-violet-200' : 'text-gray-400'
                    }`}>
                      Strategy: {msg.strategy}
                    </p>
                  )}
                  {msg.audio_url && (
                    <audio
                      controls
                      src={`${API_BASE}${msg.audio_url}`}
                      className="mt-2 w-full h-8"
                    />
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* Input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !asking && askQuestion()}
            placeholder="Ask a question..."
            className="flex-1 bg-gray-50 border border-gray-200 rounded-xl px-4 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-violet-200 focus:border-violet-400 transition-all"
          />
          <button
            onClick={askQuestion}
            disabled={asking || !question.trim()}
            className="bg-violet-600 hover:bg-violet-700 disabled:bg-gray-300 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed"
          >
            Ask
          </button>
        </div>
      </div>
    </div>
  )
}

export default MeetingDetail

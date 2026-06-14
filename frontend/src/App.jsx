import { useState, useEffect, useRef } from 'react'

const API = ''  // Vite proxy forwards /api → localhost:5000

const PIPELINE_META = {
  rag_fusion: {
    label: 'RAG Fusion',
    color: '#00d4ff',
    desc: 'Generates query variants → retrieves for each → merges via Reciprocal Rank Fusion',
    icon: '⊕',
  },
  hyde: {
    label: 'HyDE',
    color: '#ff6b35',
    desc: 'Generates a hypothetical document → embeds it → retrieves similar real chunks',
    icon: '◎',
  },
  crag: {
    label: 'CRAG',
    color: '#a8ff78',
    desc: 'Retrieves chunks → judges confidence → cites sources or falls back gracefully',
    icon: '⊛',
  },
  graph_rag: {
    label: 'Graph RAG',
    color: '#c77dff',
    desc: 'Builds a similarity graph → BFS-expands from seed nodes → graph-aware retrieval',
    icon: '◈',
  },
}

// ── Score bar ──────────────────────────────────────────────────────────────
function ScoreBar({ score }) {
  const pct = Math.max(0, Math.min(1, score)) * 100
  const hue = Math.round(pct * 1.2)  // 0=red → 120=green
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
      <div style={{
        flex: 1, height: 4, background: '#1a1a2e', borderRadius: 2, overflow: 'hidden'
      }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: `hsl(${hue}, 80%, 55%)`,
          transition: 'width 0.5s ease',
        }} />
      </div>
      <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: '#888', minWidth: 40 }}>
        {score.toFixed(3)}
      </span>
    </div>
  )
}

// ── Chunk card ─────────────────────────────────────────────────────────────
function ChunkCard({ chunk, index, pipelineColor }) {
  const [open, setOpen] = useState(index === 0)
  return (
    <div style={{
      border: `1px solid ${open ? pipelineColor + '55' : '#222'}`,
      borderRadius: 8,
      marginBottom: 8,
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', background: open ? '#0d0d1a' : '#080810',
          border: 'none', cursor: 'pointer',
          padding: '10px 14px', display: 'flex', justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontFamily: 'Space Mono, monospace', fontSize: 11,
            color: pipelineColor, background: pipelineColor + '22',
            padding: '2px 7px', borderRadius: 4,
          }}>#{index + 1}</span>
          <span style={{ color: '#ccc', fontSize: 13, fontFamily: 'DM Sans, sans-serif' }}>
            {chunk.page_name || chunk.text.slice(0, 60) + '…'}
          </span>
        </div>
        <span style={{ color: '#555', fontSize: 12 }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{ padding: '12px 14px', background: '#080810' }}>
          <ScoreBar score={chunk.score} />
          <p style={{
            margin: '10px 0 8px', color: '#d0d0e0', fontSize: 13,
            lineHeight: 1.65, fontFamily: 'DM Sans, sans-serif',
          }}>{chunk.text}</p>
          {chunk.page_url && (
            <a href={chunk.page_url} target="_blank" rel="noreferrer"
              style={{ fontSize: 11, color: pipelineColor, fontFamily: 'Space Mono, monospace', wordBreak: 'break-all' }}>
              {chunk.page_url}
            </a>
          )}
          {chunk.domain && (
            <span style={{
              marginLeft: 10, fontSize: 10, color: '#666',
              fontFamily: 'Space Mono, monospace', textTransform: 'uppercase',
            }}>[{chunk.domain}]</span>
          )}
        </div>
      )}
    </div>
  )
}

// ── Answer box ─────────────────────────────────────────────────────────────
function AnswerBox({ result, pipelineColor }) {
  if (!result) return null
  const { answer, confidence_label, confidence_score, query_variants, hypothetical_doc, used_retrieval } = result

  return (
    <div style={{ marginTop: 24 }}>
      {/* Pipeline badge row */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
        {confidence_label && (
          <span style={{
            padding: '3px 10px', borderRadius: 4, fontSize: 11,
            fontFamily: 'Space Mono, monospace',
            background: confidence_label === 'HIGH' ? '#1a3a1a' : confidence_label === 'MEDIUM' ? '#3a2a10' : '#3a1a1a',
            color: confidence_label === 'HIGH' ? '#a8ff78' : confidence_label === 'MEDIUM' ? '#ffd700' : '#ff6b6b',
            border: `1px solid ${confidence_label === 'HIGH' ? '#a8ff7844' : confidence_label === 'MEDIUM' ? '#ffd70044' : '#ff6b6b44'}`,
          }}>
            CONFIDENCE: {confidence_label} ({confidence_score?.toFixed(2)})
          </span>
        )}
        {used_retrieval === false && (
          <span style={{
            padding: '3px 10px', borderRadius: 4, fontSize: 11,
            fontFamily: 'Space Mono, monospace',
            background: '#2a1a1a', color: '#ff9f9f',
            border: '1px solid #ff6b6b44',
          }}>FALLBACK MODE — no retrieval</span>
        )}
      </div>

      {/* Answer */}
      <div style={{
        background: '#0a0a18', border: `1px solid ${pipelineColor}44`,
        borderLeft: `3px solid ${pipelineColor}`,
        borderRadius: 8, padding: '16px 18px',
      }}>
        <div style={{
          fontSize: 10, fontFamily: 'Space Mono, monospace',
          color: pipelineColor, letterSpacing: 2, marginBottom: 10, opacity: 0.8,
        }}>ANSWER</div>
        <p style={{
          margin: 0, color: '#eee', fontSize: 14, lineHeight: 1.75,
          fontFamily: 'DM Sans, sans-serif', whiteSpace: 'pre-wrap',
        }}>{answer}</p>
      </div>

      {/* HyDE hypothetical doc */}
      {hypothetical_doc && (
        <details style={{ marginTop: 14 }}>
          <summary style={{ cursor: 'pointer', fontSize: 12, color: '#888', fontFamily: 'Space Mono, monospace' }}>
            ▶ Hypothetical document used for retrieval
          </summary>
          <div style={{
            marginTop: 8, padding: '12px 14px',
            background: '#0c0c1a', border: '1px solid #333', borderRadius: 6,
            color: '#aaa', fontSize: 13, fontFamily: 'DM Sans, sans-serif',
            lineHeight: 1.65, whiteSpace: 'pre-wrap',
          }}>{hypothetical_doc}</div>
        </details>
      )}

      {/* RAG Fusion query variants */}
      {query_variants && query_variants.length > 1 && (
        <details style={{ marginTop: 10 }}>
          <summary style={{ cursor: 'pointer', fontSize: 12, color: '#888', fontFamily: 'Space Mono, monospace' }}>
            ▶ Query variants used ({query_variants.length})
          </summary>
          <div style={{ marginTop: 8 }}>
            {query_variants.map((v, i) => (
              <div key={i} style={{
                padding: '6px 10px', marginBottom: 4,
                background: '#0c0c1a', border: '1px solid #222', borderRadius: 4,
                color: '#aaa', fontSize: 12, fontFamily: 'DM Sans, sans-serif',
              }}>
                <span style={{ color: pipelineColor, fontFamily: 'Space Mono, monospace', marginRight: 8 }}>
                  {i === 0 ? '[original]' : `[v${i}]`}
                </span>{v}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState('')
  const [pipeline, setPipeline] = useState('rag_fusion')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [samples, setSamples] = useState([])
  const [showSamples, setShowSamples] = useState(false)
  const textareaRef = useRef(null)

  const pColor = PIPELINE_META[pipeline]?.color ?? '#00d4ff'

  // Load sample queries
  useEffect(() => {
    fetch('/api/samples?n=12')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setSamples(data) })
      .catch(() => {})
  }, [])

  const handleSubmit = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), pipeline }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Request failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#05050f',
      color: '#e0e0e0',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      {/* Global styles */}
      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a18; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
        ::selection { background: ${pColor}44; }
        textarea:focus { outline: none; }
        button:focus { outline: none; }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }
        .fade-in { animation: fadeIn 0.4s ease forwards; }
      `}</style>

      {/* Header */}
      <header style={{
        borderBottom: '1px solid #111',
        padding: '18px 32px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: '#07070f',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 18, color: pColor, fontWeight: 700, letterSpacing: -0.5 }}>
            RAG<span style={{ color: '#444' }}>://</span>WILD
          </span>
          <span style={{ fontSize: 12, color: '#444', fontFamily: 'Space Mono, monospace' }}>
            advanced retrieval · case study
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {Object.entries(PIPELINE_META).map(([id, meta]) => (
            <div key={id} style={{ width: 8, height: 8, borderRadius: '50%', background: meta.color, opacity: pipeline === id ? 1 : 0.2 }} />
          ))}
        </div>
      </header>

      <div style={{ maxWidth: 900, margin: '0 auto', padding: '32px 24px' }}>

        {/* Pipeline selector */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 10, fontFamily: 'Space Mono, monospace', color: '#555', letterSpacing: 2, marginBottom: 10 }}>
            SELECT PIPELINE
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {Object.entries(PIPELINE_META).map(([id, meta]) => (
              <button
                key={id}
                onClick={() => setPipeline(id)}
                style={{
                  padding: '8px 16px',
                  background: pipeline === id ? meta.color + '18' : '#0a0a18',
                  border: `1px solid ${pipeline === id ? meta.color : '#222'}`,
                  borderRadius: 6, cursor: 'pointer', color: pipeline === id ? meta.color : '#666',
                  fontFamily: 'Space Mono, monospace', fontSize: 12, fontWeight: 700,
                  transition: 'all 0.15s',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                <span>{meta.icon}</span>
                <span>{meta.label}</span>
              </button>
            ))}
          </div>
          {/* Pipeline description */}
          <div style={{
            marginTop: 10, padding: '8px 12px',
            background: '#0a0a18', border: `1px solid ${pColor}22`,
            borderRadius: 5, fontSize: 12, color: '#888',
            fontFamily: 'DM Sans, sans-serif', transition: 'border-color 0.3s',
          }}>
            <span style={{ color: pColor, marginRight: 6 }}>{PIPELINE_META[pipeline]?.icon}</span>
            {PIPELINE_META[pipeline]?.desc}
          </div>
        </div>

        {/* Query input */}
        <div style={{ position: 'relative', marginBottom: 16 }}>
          <div style={{ fontSize: 10, fontFamily: 'Space Mono, monospace', color: '#555', letterSpacing: 2, marginBottom: 8 }}>
            QUERY
          </div>
          <textarea
            ref={textareaRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything… e.g. Who directed Inception? Which athlete has more Grand Slams, Federer or Nadal?"
            rows={3}
            style={{
              width: '100%', padding: '14px 16px',
              background: '#0a0a18', border: `1px solid ${query ? pColor + '44' : '#222'}`,
              borderRadius: 8, color: '#e0e0e0', fontSize: 14,
              fontFamily: 'DM Sans, sans-serif', lineHeight: 1.6,
              resize: 'vertical', transition: 'border-color 0.2s',
            }}
          />
          <div style={{ fontSize: 11, color: '#444', marginTop: 4, fontFamily: 'Space Mono, monospace' }}>
            Ctrl+Enter to run
          </div>
        </div>

        {/* Samples + Run row */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 32, alignItems: 'flex-start' }}>
          <button
            onClick={handleSubmit}
            disabled={!query.trim() || loading}
            style={{
              padding: '11px 28px',
              background: loading ? '#111' : pColor,
              color: loading ? '#555' : '#000',
              border: 'none', borderRadius: 7, cursor: loading ? 'not-allowed' : 'pointer',
              fontFamily: 'Space Mono, monospace', fontWeight: 700, fontSize: 13,
              transition: 'all 0.15s', letterSpacing: 0.5,
            }}
          >
            {loading ? (
              <span style={{ animation: 'pulse 1s infinite' }}>RUNNING…</span>
            ) : 'RUN →'}
          </button>

          <button
            onClick={() => setShowSamples(s => !s)}
            style={{
              padding: '11px 18px',
              background: '#0a0a18', border: '1px solid #222',
              borderRadius: 7, cursor: 'pointer', color: '#666',
              fontFamily: 'Space Mono, monospace', fontSize: 12,
            }}
          >
            {showSamples ? 'HIDE SAMPLES' : 'SAMPLE QUERIES'}
          </button>
        </div>

        {/* Sample queries */}
        {showSamples && samples.length > 0 && (
          <div className="fade-in" style={{ marginBottom: 32 }}>
            <div style={{ fontSize: 10, fontFamily: 'Space Mono, monospace', color: '#555', letterSpacing: 2, marginBottom: 10 }}>
              SAMPLE QUERIES FROM DATASET
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
              {samples.map((s, i) => (
                <button
                  key={i}
                  onClick={() => { setQuery(s.query); setShowSamples(false); textareaRef.current?.focus() }}
                  style={{
                    padding: '10px 12px', background: '#0a0a18',
                    border: '1px solid #1a1a2e', borderRadius: 6,
                    cursor: 'pointer', textAlign: 'left', color: '#bbb',
                    fontSize: 12, fontFamily: 'DM Sans, sans-serif',
                    transition: 'border-color 0.15s, color 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = pColor + '66'; e.currentTarget.style.color = '#fff' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#1a1a2e'; e.currentTarget.style.color = '#bbb' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 10, color: pColor, fontFamily: 'Space Mono, monospace', opacity: 0.7 }}>
                      {s.domain || 'open'}
                    </span>
                    <span style={{ fontSize: 10, color: '#444', fontFamily: 'Space Mono, monospace' }}>
                      {s.question_type}
                    </span>
                  </div>
                  {s.query}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="fade-in" style={{
            padding: '12px 16px', background: '#1a0808',
            border: '1px solid #ff6b6b44', borderRadius: 8, color: '#ff9f9f',
            fontFamily: 'Space Mono, monospace', fontSize: 12, marginBottom: 24,
          }}>
            ⚠ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="fade-in">
            {/* Divider */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
              <div style={{ flex: 1, height: 1, background: '#111' }} />
              <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: pColor, letterSpacing: 2 }}>
                {PIPELINE_META[pipeline]?.icon} {PIPELINE_META[pipeline]?.label} · RESULTS
              </span>
              <div style={{ flex: 1, height: 1, background: '#111' }} />
            </div>

            {/* Answer */}
            <AnswerBox result={result} pipelineColor={pColor} />

            {/* Retrieved chunks */}
            {result.retrieved_chunks?.length > 0 && (
              <div style={{ marginTop: 28 }}>
                <div style={{ fontSize: 10, fontFamily: 'Space Mono, monospace', color: '#555', letterSpacing: 2, marginBottom: 12 }}>
                  RETRIEVED CHUNKS ({result.retrieved_chunks.length})
                </div>
                {result.retrieved_chunks.map((chunk, i) => (
                  <ChunkCard key={i} chunk={chunk} index={i} pipelineColor={pColor} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div style={{ textAlign: 'center', padding: '60px 0', color: '#222' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>◈</div>
            <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 12, letterSpacing: 2 }}>
              SELECT A PIPELINE · ENTER A QUERY · PRESS RUN
            </div>
          </div>
        )}

      </div>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid #0d0d1d', padding: '16px 32px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginTop: 40,
      }}>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: '#333' }}>
          RAG IN THE WILD · CS-4015 AGENTIC AI
        </span>
        <div style={{ display: 'flex', gap: 16 }}>
          {Object.entries(PIPELINE_META).map(([id, meta]) => (
            <span key={id} style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: meta.color, opacity: 0.5 }}>
              {meta.icon} {meta.label}
            </span>
          ))}
        </div>
      </footer>
    </div>
  )
}

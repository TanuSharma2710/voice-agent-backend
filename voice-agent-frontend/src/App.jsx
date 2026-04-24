import { useState, useEffect, useRef, useCallback } from 'react';
import { supabase, hasSupabaseConfig } from './services/supabase';
import { listDatabases, registerDatabase, deleteDatabase, updateDatabaseNickname, resetAll } from './services/api';
import './App.css';

const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/voice') + '/ws/agent';

const TARGET_SAMPLE_RATE = 16000;

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function int16ToFloat32(int16) {
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  return float32;
}

function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const idx = i * ratio;
    const low = Math.floor(idx);
    const high = Math.min(low + 1, buffer.length - 1);
    const frac = idx - low;
    result[i] = buffer[low] * (1 - frac) + buffer[high] * frac;
  }
  return result;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState(null);

  const [databases, setDatabases] = useState([]);
  const [dbNickname, setDbNickname] = useState('');
  const [dbUrl, setDbUrl] = useState('');
  const [processing, setProcessing] = useState(false);

  const [editingDbId, setEditingDbId] = useState(null);
  const [editNickname, setEditNickname] = useState('');
  const [resetting, setResetting] = useState(false);

  // Voice state
  const [wsConnected, setWsConnected] = useState(false);
  const [conversationActive, setConversationActive] = useState(false);
  const [agentStatus, setAgentStatus] = useState('idle'); // idle, listening, thinking, speaking
  const [sqlQueryActive, setSqlQueryActive] = useState(false);
  const [voiceLogs, setVoiceLogs] = useState([]);

  // Refs
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const captureCtxRef = useRef(null);
  const processorRef = useRef(null);
  const playbackCtxRef = useRef(null);
  const nextPlayTimeRef = useRef(0);
  const activeSourcesRef = useRef([]);
  // Ref mirror of conversationActive so handleWsEvent (stable callback) can read it
  const conversationActiveRef = useRef(false);

  // ── Auth / session ──

  useEffect(() => {
    if (!hasSupabaseConfig) { setError('Supabase not configured.'); setLoading(false); return; }
    supabase.auth.getSession().then(({ data: { session: s } }) => { setSession(s); setLoading(false); });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => subscription.unsubscribe();
  }, []);

  // ── Fetch databases when session changes ──

  const fetchDatabases = useCallback(async () => {
    if (!session) return;
    try {
      const data = await listDatabases(session.access_token);
      setDatabases(data.databases || []);
    } catch (err) {
      console.error('Error fetching databases:', err);
    }
  }, [session]);

  useEffect(() => {
    if (session) fetchDatabases(); else setDatabases([]);
  }, [session, fetchDatabases]);

  // ── Cleanup WebSocket on unmount ──

  useEffect(() => { conversationActiveRef.current = conversationActive; }, [conversationActive]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // ── Auth handlers ──

  const handleAuth = async (e) => {
    e.preventDefault(); setError(null); setLoading(true);
    try {
      if (isSignUp) {
        const { error: authErr } = await supabase.auth.signUp({ email, password });
        if (authErr) throw authErr;
        setError('Check your email for the confirmation link.');
      } else {
        const { error: authErr } = await supabase.auth.signInWithPassword({ email, password });
        if (authErr) throw authErr;
      }
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const handleSignOut = async () => {
    stopConversation();
    disconnectWs();
    await supabase.auth.signOut();
    setDatabases([]);
  };

  // ── Database CRUD ──

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!dbNickname || !dbUrl) return;
    setProcessing(true); setError(null);
    try {
      await registerDatabase(session.access_token, { nickname: dbNickname, db_url: dbUrl });
      setDbNickname(''); setDbUrl('');
      fetchDatabases();
    } catch (err) { setError(err.message); }
    finally { setProcessing(false); }
  };

  const handleDelete = async (databaseId, nickname) => {
    if (!window.confirm(`Delete database "${nickname}"? This will also remove its schema data from the vector store.`)) return;
    try {
      await deleteDatabase(session.access_token, databaseId);
      fetchDatabases();
    } catch (err) { console.error('Error deleting database:', err); }
  };

  const handleReset = async () => {
    if (!window.confirm('This will permanently wipe ALL vector store data and clear all registered databases. Are you sure?')) return;
    setResetting(true); setError(null);
    try {
      await resetAll(session.access_token);
      setDatabases([]);
      fetchDatabases();
    } catch (err) { setError(err.message); }
    finally { setResetting(false); }
  };

  const handleUpdateNickname = async (databaseId) => {
    if (!editNickname.trim()) return;
    try {
      await updateDatabaseNickname(session.access_token, databaseId, editNickname.trim());
      setEditingDbId(null); setEditNickname('');
      fetchDatabases();
    } catch (err) { console.error('Error updating nickname:', err); }
  };

  const startEdit = (db) => { setEditingDbId(db.database_id); setEditNickname(db.nickname); };

  // ── Log helper ──

  const addLog = useCallback((role, content) => {
    setVoiceLogs((prev) => [...prev.slice(-200), { role, content, time: new Date().toLocaleTimeString() }]);
  }, []);

  // ── WebSocket connection ──

  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      addLog('system', 'WebSocket connected');
      ws.send(JSON.stringify({ type: 'auth', token: session.access_token }));
    };

    ws.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        // Binary: TTS audio from Deepgram — mark agent as speaking immediately
        const buffer = await event.data.arrayBuffer();
        if (buffer.byteLength > 0) {
          setAgentStatus('speaking');
          playAudioChunk(new Int16Array(buffer));
        }
        return;
      }
      // Text: JSON event
      try {
        const data = JSON.parse(event.data);
        handleWsEvent(data);
      } catch (err) {
        console.warn('Non-JSON WS message:', err);
      }
    };

    ws.onerror = () => addLog('error', 'WebSocket error');
    ws.onclose = () => {
      setWsConnected(false);
      setConversationActive(false);
      setAgentStatus('idle');
      setSqlQueryActive(false);
      conversationReadyRef.current = false;
      addLog('system', 'WebSocket disconnected');
    };
  }, [session, addLog]);

  const disconnectWs = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setWsConnected(false);
  }, []);

  // ── Handle events from backend / Deepgram ──

  const handleWsEvent = useCallback((data) => {
    const type = data.type;

    if (type === 'auth_success') {
      addLog('system', 'Authenticated');
    } else if (type === 'conversation_started') {
      // Deepgram is connected and ready — start sending audio
      conversationReadyRef.current = true;
      setAgentStatus('listening');
      addLog('system', 'Voice agent ready — start speaking');
    } else if (type === 'ConversationText') {
      const role = data.role === 'user' ? 'user' : 'agent';
      addLog(role, data.content);
    } else if (type === 'UserStartedSpeaking') {
      setAgentStatus('listening');
      stopPlayback();
    } else if (type === 'AgentStartedSpeaking') {
      setAgentStatus('speaking');
    } else if (type === 'AgentThinking') {
      setAgentStatus('thinking');
    } else if (type === 'AgentAudioDone') {
      setAgentStatus('listening');
    } else if (type === 'FunctionCalling') {
      addLog('system', `Calling tool: ${data.function_name || '...'}`);
    } else if (type === 'SqlQueryStarted') {
      setSqlQueryActive(true);
      addLog('system', 'Fetching data from database...');
    } else if (type === 'SqlQueryCompleted') {
      setSqlQueryActive(false);
      addLog('system', data.success ? 'Data ready!' : 'Query failed.');
    } else if (type === 'error') {
      addLog('error', data.message || 'Unknown error');
      // If we get an error during conversation setup, reset state
      if (!conversationReadyRef.current && conversationActiveRef.current) {
        setConversationActive(false);
        setAgentStatus('idle');
      }
    } else if (type === 'tool_definitions') {
      addLog('system', `Loaded ${data.tools?.length || 0} tools`);
    }
    // Other Deepgram events (Welcome, SettingsApplied, etc.) are silently ignored.
  }, [addLog]);

  // ── Audio playback (TTS from Deepgram) ──

  const ensurePlaybackCtx = useCallback(() => {
    if (!playbackCtxRef.current || playbackCtxRef.current.state === 'closed') {
      playbackCtxRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SAMPLE_RATE });
      nextPlayTimeRef.current = 0;
    }
    if (playbackCtxRef.current.state === 'suspended') {
      playbackCtxRef.current.resume();
    }
    return playbackCtxRef.current;
  }, []);

  const playAudioChunk = useCallback((int16Buf) => {
    const ctx = ensurePlaybackCtx();
    const float32 = int16ToFloat32(int16Buf);
    const audioBuffer = ctx.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    const startTime = Math.max(now + 0.02, nextPlayTimeRef.current);
    source.start(startTime);
    nextPlayTimeRef.current = startTime + audioBuffer.duration;

    activeSourcesRef.current.push(source);
    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source);
    };
  }, [ensurePlaybackCtx]);

  const stopPlayback = useCallback(() => {
    activeSourcesRef.current.forEach((s) => {
      try { s.stop(); } catch (err) { console.warn('Stop playback source error:', err); }
    });
    activeSourcesRef.current = [];
    nextPlayTimeRef.current = 0;
  }, []);

  // ── Start / stop conversation ──

  // Ref to gate audio sending — only send after backend confirms Deepgram is ready
  const conversationReadyRef = useRef(false);

  const startConversation = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    try {
      // 1. Get mic permission first (needs user gesture)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;

      // 2. Ensure playback context is ready (also needs user gesture)
      ensurePlaybackCtx();

      // 3. Tell backend to connect to Deepgram BEFORE starting audio capture
      conversationReadyRef.current = false;
      wsRef.current.send(JSON.stringify({ type: 'start_conversation' }));
      setConversationActive(true);
      setAgentStatus('thinking');
      addLog('system', 'Connecting to voice agent...');

      // 4. Set up audio capture — but gate sending on conversationReadyRef
      const captureCtx = new (window.AudioContext || window.webkitAudioContext)();
      captureCtxRef.current = captureCtx;
      const actualRate = captureCtx.sampleRate;

      const source = captureCtx.createMediaStreamSource(stream);
      const processor = captureCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      source.connect(processor);
      processor.connect(captureCtx.destination);

      processor.onaudioprocess = (e) => {
        // Only send audio after Deepgram connection is established
        if (!conversationReadyRef.current) return;
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const raw = e.inputBuffer.getChannelData(0);
        const resampled = downsample(raw, actualRate, TARGET_SAMPLE_RATE);
        const pcm = float32ToInt16(resampled);
        wsRef.current.send(pcm.buffer);
      };
    } catch (err) {
      addLog('error', `Microphone access denied: ${err.message}`);
      setConversationActive(false);
      setAgentStatus('idle');
    }
  }, [addLog, ensurePlaybackCtx]);

  const stopConversation = useCallback(() => {
    // Stop mic capture
    conversationReadyRef.current = false;
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (captureCtxRef.current && captureCtxRef.current.state !== 'closed') {
      captureCtxRef.current.close().catch((err) => console.warn('Capture ctx close error:', err));
      captureCtxRef.current = null;
    }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }

    stopPlayback();

    // Tell backend
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop_conversation' }));
    }
    setConversationActive(false);
    setAgentStatus('idle');
    addLog('system', 'Conversation ended');
  }, [addLog, stopPlayback]);

  // ── Render ──

  if (loading) return <div className="loading">Loading...</div>;

  if (!session) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h1>Voice Agent</h1>
          <h2>{isSignUp ? 'Sign Up' : 'Sign In'}</h2>
          {error && <div className="error">{error}</div>}
          <form onSubmit={handleAuth}>
            <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            <button type="submit" disabled={loading}>{loading ? 'Loading...' : isSignUp ? 'Sign Up' : 'Sign In'}</button>
          </form>
          <p className="toggle-auth" onClick={() => setIsSignUp(!isSignUp)}>
            {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header>
        <h1>Voice Agent</h1>
        <button onClick={handleSignOut} className="sign-out-btn">Sign Out</button>
      </header>

      {error && <div className="error">{error}</div>}

      {/* ── Databases ── */}
      <section className="databases-section">
        <div className="section-header">
          <h2>Your Databases</h2>
          <button className="reset-btn" onClick={handleReset} disabled={resetting}>
            {resetting ? 'Resetting...' : 'Reset All'}
          </button>
        </div>
        <form onSubmit={handleRegister} className="db-form">
          <input type="text" placeholder="Database Nickname" value={dbNickname} onChange={(e) => setDbNickname(e.target.value)} required />
          <input type="text" placeholder="Database URL (postgresql://...)" value={dbUrl} onChange={(e) => setDbUrl(e.target.value)} required />
          <button type="submit" disabled={processing}>{processing ? 'Processing...' : 'Register Database'}</button>
        </form>

        <div className="databases-list">
          {databases.length === 0 && !processing && (
            <div className="empty-state">No databases registered yet. Add one above to get started.</div>
          )}
          {databases.map((db) => (
            <div key={db.database_id} className="database-item">
              <div className="db-info">
                {editingDbId === db.database_id ? (
                  <div className="db-edit-row">
                    <input className="db-edit-input" value={editNickname} onChange={(e) => setEditNickname(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleUpdateNickname(db.database_id)} autoFocus />
                    <button className="db-save-btn" onClick={() => handleUpdateNickname(db.database_id)}>Save</button>
                    <button className="db-cancel-btn" onClick={() => setEditingDbId(null)}>Cancel</button>
                  </div>
                ) : (
                  <>
                    <span className="db-nickname">{db.nickname}</span>
                    <span className="db-id">{db.sub_database_id}</span>
                  </>
                )}
              </div>
              {editingDbId !== db.database_id && (
                <div className="db-actions">
                  <button className="edit-btn" onClick={() => startEdit(db)}>Edit</button>
                  <button onClick={() => handleDelete(db.database_id, db.nickname)} className="delete-btn">Delete</button>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Voice Agent ── */}
      <section className="voice-section">
        <h2>Voice Agent</h2>

        <div className="voice-status">
          <span className={`status-dot ${wsConnected ? 'connected' : 'disconnected'}`} />
          <span>{wsConnected ? 'Connected' : 'Disconnected'}</span>
          <button onClick={wsConnected ? disconnectWs : connectWs}>
            {wsConnected ? 'Disconnect' : 'Connect'}
          </button>
        </div>

        {wsConnected && (
          <div className="voice-controls">
            <button
              className={`convo-btn ${conversationActive ? 'active' : ''} ${agentStatus}`}
              onClick={conversationActive ? stopConversation : startConversation}
            >
              {conversationActive
                ? agentStatus === 'speaking'
                  ? 'Agent Speaking...'
                  : agentStatus === 'thinking'
                    ? 'Agent Thinking...'
                    : 'Listening...'
                : 'Start Conversation'}
            </button>
            {conversationActive && (
              <span className={`agent-status-label ${agentStatus}`}>
                {agentStatus === 'listening' && 'Speak now'}
                {agentStatus === 'thinking' && 'Processing...'}
                {agentStatus === 'speaking' && 'Agent is responding'}
              </span>
            )}
            {conversationActive && sqlQueryActive && (
              <span className="sql-query-badge">
                ⏳ Fetching data...
              </span>
            )}
          </div>
        )}

        <div className="voice-logs">
          {voiceLogs.map((log, idx) => (
            <div key={idx} className={`log ${log.role}`}>
              <span className="log-time">{log.time}</span>
              <span className="log-content">{log.content}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;

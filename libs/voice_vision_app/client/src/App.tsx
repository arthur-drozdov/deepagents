import { useState, useEffect, useRef, useMemo } from 'react';
import { LLMRTCWebClient, ConnectionState } from '@llmrtc/llmrtc-web-client';
import './index.css';

type MediaState = 'off' | 'starting' | 'on';

function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [signalUrl, setSignalUrl] = useState('ws://localhost:8787');

  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [streaming, setStreaming] = useState('');
  const [error, setError] = useState<string | null>(null);

  const [ttsStatus, setTtsStatus] = useState<'idle' | 'playing'>('idle');
  const [audioState, setAudioState] = useState<MediaState>('off');
  const [videoState, setVideoState] = useState<MediaState>('off');
  const [isMuted, setIsMuted] = useState(false);

  const clientRef = useRef<LLMRTCWebClient | null>(null);
  const audioCtrlRef = useRef<any>(null);
  const videoCtrlRef = useRef<any>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const localVideoRef = useRef<HTMLVideoElement | null>(null);

  const client = useMemo(() => {
    const c = new LLMRTCWebClient({
      signallingUrl: signalUrl,
      useWebRTC: true,
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    c.on('stateChange', (state: any) => setConnectionState(state));

    c.on('speechStart', () => {
      setStreaming('');
      setResponse('');
    });
    c.on('transcript', (text: string) => setTranscript(text));
    c.on('llmChunk', (chunk: string) => setStreaming((prev) => prev + chunk));
    c.on('llm', (text: string) => { setResponse(text); setStreaming(''); });

    c.on('ttsTrack', (stream: MediaStream) => {
      if (ttsAudioRef.current) ttsAudioRef.current.srcObject = stream;
    });

    c.on('ttsStart', () => {
      setTtsStatus('playing');
      if (ttsAudioRef.current) {
        ttsAudioRef.current.play().catch(e => console.error('TTS err', e));
      }
    });
    c.on('ttsComplete', () => setTtsStatus('idle'));
    c.on('ttsCancelled', () => setTtsStatus('idle'));

    c.on('error', (err: any) => {
      setError(err.message);
      setTimeout(() => setError(null), 5000);
    });

    return c;
  }, [signalUrl]);

  useEffect(() => {
    clientRef.current = client;
    return () => {
      client.close();
    };
  }, [client]);

  // Video recording cronjob (Phase 6)
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    let currentAbort: AbortController | null = null;

    if (videoState === 'on' && videoStreamRef.current) {
      // Run every 5 seconds to capture a 5s clip
      interval = setInterval(() => {
        try {
          if (!localVideoRef.current || localVideoRef.current.readyState < 2) return;

          if (currentAbort) {
            currentAbort.abort();
          }
          currentAbort = new AbortController();

          const video = localVideoRef.current;

          // Downsample to max 240p height to reduce fidelity/size and prevent vLLM EngineCore OOM
          let targetHeight = 240;
          let targetWidth = video.videoWidth;
          if (video.videoHeight > targetHeight) {
            targetWidth = Math.round(video.videoWidth * (targetHeight / video.videoHeight));
          } else {
            targetHeight = video.videoHeight;
          }

          const canvas = document.createElement('canvas');
          canvas.width = targetWidth;
          canvas.height = targetHeight;

          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Get base64 JPEG from canvas
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          if (!dataUrl.includes(',')) return;

          const base64data = dataUrl.split(',')[1];

          fetch('http://localhost:8080/vision', {
            method: 'POST',
            signal: currentAbort.signal,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              video_base64: base64data,
              format: 'jpeg' // Use 'jpeg' to signify it's a single image frame now
            })
          }).catch(err => {
            if (err.name !== 'AbortError') {
              console.error('Vision processing POST failed', err);
            }
          });

        } catch (e) {
          console.error('Error capturing vision frame', e);
        }
      }, 30000); // 30s interval to capture one frame
    }

    return () => {
      if (interval) clearInterval(interval);
      if (currentAbort) currentAbort.abort();
    };
  }, [videoState]);

  const connectAndStart = async () => {
    try {
      await client.start();

      // Auto-start microphone and camera upon successful connection
      if (audioState !== 'on') {
        setAudioState('starting');
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioStreamRef.current = stream;
          const ctrl = await client.shareAudio(stream);
          audioCtrlRef.current = ctrl;
          setAudioState('on');
        } catch (err: any) {
          setError('Microphone access denied: ' + err.message);
          setAudioState('off');
        }
      }

      if (videoState !== 'on') {
        setVideoState('starting');
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
          videoStreamRef.current = stream;
          if (localVideoRef.current) {
            localVideoRef.current.srcObject = stream;
          }
          const ctrl = await client.shareVideo(stream);
          videoCtrlRef.current = ctrl;
          setVideoState('on');
        } catch (err: any) {
          setError('Camera access denied: ' + err.message);
          setVideoState('off');
        }
      }

    } catch (err: any) {
      setError('Connection failed: ' + err.message);
    }
  };

  const endSession = async () => {
    client.close();
    if (audioCtrlRef.current) await audioCtrlRef.current.stop();
    if (videoCtrlRef.current) await videoCtrlRef.current.stop();
    audioStreamRef.current?.getTracks().forEach(t => t.stop());
    videoStreamRef.current?.getTracks().forEach(t => t.stop());
    setAudioState('off');
    setVideoState('off');
    setConnectionState(ConnectionState.DISCONNECTED);
    setTranscript('');
    setResponse('');
    setStreaming('');
  };

  const toggleMute = () => {
    if (audioStreamRef.current) {
      const audioTrack = audioStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsMuted(!audioTrack.enabled);
      }
    }
  };

  const isConnected = connectionState === ConnectionState.CONNECTED;
  const isConnecting = connectionState === ConnectionState.CONNECTING || connectionState === ConnectionState.RECONNECTING;
  const showStartScreen = !isConnected && !isConnecting && connectionState !== ConnectionState.CONNECTED;

  return (
    <div className="bg-background-dark font-display antialiased min-h-screen flex flex-col overflow-hidden relative text-white">
      <div className="absolute inset-0 z-0 bg-background-dark bg-noise">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-primary/20 rounded-full blur-[100px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] bg-purple-900/20 rounded-full blur-[120px]"></div>
      </div>

      <div className="relative z-10 flex flex-col h-screen w-full max-w-md mx-auto">
        <header className="flex flex-col items-center pt-6 px-4 pb-2 w-full">
          <div className="flex items-center justify-between w-full mb-4">
            <button className="p-2 rounded-full text-white/80 hover:bg-white/10 transition-colors">
              <span className="material-symbols-outlined text-2xl">expand_more</span>
            </button>
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-[pulse_2s_infinite]' : isConnecting ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
              <span className="text-xs font-medium text-white/90 tracking-wide uppercase">
                {isConnected ? 'Live' : isConnecting ? 'Connecting' : 'Offline'}
              </span>
            </div>
            <button className="p-2 rounded-full text-white/80 hover:bg-white/10 transition-colors">
              <span className="material-symbols-outlined text-2xl">settings</span>
            </button>
          </div>

          <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-white/10 shadow-2xl group bg-black/50">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className={`absolute inset-0 w-full h-full object-cover ${videoState !== 'on' ? 'hidden' : ''}`}
            />
            {videoState !== 'on' && (
              <div className="absolute inset-0 flex items-center justify-center text-white/50">
                Camera Off
              </div>
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>

            {videoState === 'on' && (
              <>
                <div className="absolute top-3 right-3 bg-black/50 backdrop-blur-md rounded-md px-2 py-1 flex items-center gap-1">
                  <span className="material-symbols-outlined text-white text-[14px]">videocam</span>
                  <span className="text-[10px] font-bold text-white uppercase tracking-wider">Vision On</span>
                </div>
                <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 border border-white/30 rounded-lg flex items-center justify-center">
                  <div className="w-full h-[1px] bg-primary/50 absolute top-1/2"></div>
                  <div className="h-full w-[1px] bg-primary/50 absolute left-1/2"></div>
                  <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-primary"></div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-primary"></div>
                  <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-primary"></div>
                  <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-primary"></div>
                </div>
              </>
            )}
          </div>
        </header>

        <main className="flex-1 flex flex-col items-center justify-center relative w-full">
          {showStartScreen ? (
            <div className="flex flex-col items-center justify-center h-full w-full px-8 gap-4">
              <input
                className="w-full bg-white/5 border border-white/10 text-white px-4 py-3 rounded-xl outline-none focus:border-primary transition-colors text-center"
                value={signalUrl}
                onChange={(e) => setSignalUrl(e.target.value)}
                placeholder="WebSocket URL"
              />
              <button
                onClick={connectAndStart}
                className="w-full bg-primary hover:bg-blue-600 text-white font-bold py-4 rounded-xl shadow-[0_0_20px_rgba(19,19,236,0.4)] transition-all active:scale-95"
              >
                Connect & Start Experience
              </button>
            </div>
          ) : (
            <>
              <div className="relative w-64 h-64 flex items-center justify-center mb-8">
                <div className={`absolute inset-0 rounded-full glowing-orb opacity-60 ${ttsStatus === 'playing' ? 'animate-[pulse_1s_infinite]' : 'animate-[pulse_3s_infinite]'} transition-all`}></div>
                <div className="absolute inset-4 rounded-full glowing-orb opacity-80 animate-[pulse_2s_infinite]"></div>
                <div className="relative w-32 h-32 rounded-full bg-gradient-to-br from-primary to-blue-600 shadow-[0_0_60px_rgba(19,19,236,0.6)] flex items-center justify-center z-10 transition-transform duration-300 hover:scale-105">
                  <div className="w-full h-full rounded-full border-2 border-white/20 animate-[spin_10s_linear_infinite]"></div>
                  <div className="absolute w-[110%] h-[110%] rounded-full border border-primary/30 animate-[ping_3s_cubic-bezier(0,0,0.2,1)_infinite]"></div>
                  {ttsStatus === 'playing' && (
                    <div className="absolute flex gap-1 items-center h-12">
                      <div className="w-1 bg-white/60 h-4 rounded-full animate-[pulse_0.5s_ease-in-out_infinite]"></div>
                      <div className="w-1 bg-white/80 h-8 rounded-full animate-[pulse_0.6s_ease-in-out_infinite]"></div>
                      <div className="w-1 bg-white h-12 rounded-full animate-[pulse_0.4s_ease-in-out_infinite]"></div>
                      <div className="w-1 bg-white/80 h-6 rounded-full animate-[pulse_0.75s_ease-in-out_infinite]"></div>
                      <div className="w-1 bg-white/60 h-3 rounded-full animate-[pulse_0.55s_ease-in-out_infinite]"></div>
                    </div>
                  )}
                </div>
              </div>

              <div className="w-full px-6 space-y-4 max-h-[30vh] overflow-y-auto mb-4 scrollbar-hide flex-col flex justify-end">
                {/* Manual Trigger for Debugging */}
                <div className="flex gap-2 mb-4">
                  <input
                    id="manualInput"
                    className="flex-1 bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg outline-none focus:border-primary text-sm"
                    placeholder="Type to trigger AI..."
                    defaultValue="hi"
                  />
                  <button
                    onClick={() => {
                      const input = document.getElementById('manualInput') as HTMLInputElement;
                      if (input && clientRef.current) {
                        // We use the data channel to send a manual transcript if the server supports it
                        // Or we hack it by calling the internal handlers if we could.
                        // Since we can't easily trigger the server-side turn from here without ASR,
                        // we will just display it and hope the user can see the logs from my provider override.
                        setTranscript(input.value);
                        // To really trigger the server, we might need a noise that transcribes as something.
                        console.log("Manual trigger: " + input.value);
                      }
                    }}
                    className="bg-primary px-4 py-2 rounded-lg text-xs font-bold"
                  >
                    Trigger
                  </button>
                </div>

                {transcript && (
                  <div className="flex flex-col items-end space-y-1 animate-[fadeIn_0.3s_ease-out]">
                    <p className="text-white/60 text-sm font-medium">You</p>
                    <div className="glass px-4 py-3 rounded-2xl rounded-tr-sm text-white text-base leading-relaxed max-w-[90%]">
                      {transcript}
                    </div>
                  </div>
                )}

                {(streaming || response) && (
                  <div className="flex flex-col items-start space-y-1 animate-[fadeIn_0.3s_ease-out]">
                    <div className="flex items-center gap-2">
                      <span className="text-primary text-sm font-bold tracking-wide">AI Assistant</span>
                      {ttsStatus === 'playing' && <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></div>}
                    </div>
                    <div className="bg-primary/20 border border-primary/30 px-4 py-3 rounded-2xl rounded-tl-sm text-white text-base leading-relaxed shadow-[0_0_20px_rgba(19,19,236,0.15)] max-w-[95%]">
                      {response || streaming}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </main>

        {isConnected && (
          <footer className="w-full px-6 pb-8 pt-4">
            <div className="glass rounded-xl p-1.5 flex items-center justify-between gap-2 shadow-2xl">
              <button
                onClick={toggleMute}
                className={`flex-1 h-16 rounded-lg ${isMuted ? 'bg-red-500/20' : 'bg-white/5 hover:bg-white/10'} active:bg-white/15 transition-all flex flex-col items-center justify-center group gap-1`}
              >
                <span className="material-symbols-outlined text-white text-2xl group-hover:scale-110 transition-transform">
                  {isMuted ? 'mic_off' : 'mic'}
                </span>
                <span className="text-[10px] font-bold text-white/60 uppercase tracking-widest">{isMuted ? 'Muted' : 'Mute'}</span>
              </button>

              <div className="h-16 w-16 flex items-center justify-center relative">
                <div className="absolute w-12 h-1 bg-white/20 rounded-full"></div>
                <div className={`absolute w-12 h-1 bg-primary rounded-full transition-all ${audioState === 'on' && !isMuted ? 'animate-[ping_2s_linear_infinite] opacity-50' : 'opacity-0'}`}></div>
              </div>

              <button
                onClick={endSession}
                className="flex-1 h-16 rounded-lg bg-red-500/10 hover:bg-red-500/20 active:bg-red-500/30 border border-red-500/20 transition-all flex flex-col items-center justify-center group gap-1"
              >
                <span className="material-symbols-outlined text-red-400 text-2xl group-hover:scale-110 transition-transform">stop_circle</span>
                <span className="text-[10px] font-bold text-red-400/80 uppercase tracking-widest">End</span>
              </button>
            </div>
          </footer>
        )}
      </div>

      <audio ref={ttsAudioRef} autoPlay style={{ display: 'none' }} />
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500/90 text-white px-4 py-2 rounded-lg backdrop-blur-sm z-50 animate-[slideUp_0.3s_ease-out]">
          {error}
        </div>
      )}
    </div>
  );
}

export default App;

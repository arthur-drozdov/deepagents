import { config } from 'dotenv';
config();

import { LLMRTCServer } from '@llmrtc/llmrtc-backend';
import { DeepAgentsLLMProvider } from './DeepAgentsLLMProvider';
import { CustomWebSocketSTTProvider } from './CustomSTTProvider';
import { CustomWebSocketTTSProvider } from './CustomTTSProvider';

const server = new LLMRTCServer({
    providers: {
        llm: new DeepAgentsLLMProvider({
            wsUrl: 'ws://127.0.0.1:8080/chat'
        }),
        stt: new CustomWebSocketSTTProvider({
            wsUrl: 'ws://spark-4a06.tail3eb9a6.ts.net:8002/ws/asr'
        }),
        tts: new CustomWebSocketTTSProvider({
            wsUrl: 'ws://spark-15d6.tail3eb9a6.ts.net:8001/ws/tts',
            refWavPath: '/Users/adrozdov/repos/deepagents/reference.wav',
            refText: '/Users/adrozdov/repos/deepagents/reference.txt'
        })
    },
    port: 8787,
    streamingTTS: true,
    // Use a generic system prompt. The real logic is in Python DeepAgents.
    systemPrompt: 'You are a helpful voice assistant.'
});

server.on('listening', ({ host, port }) => {
    console.log(`\n  Voice Vision App LLMRTC Server`);
    console.log(`  ==============================`);
    console.log(`  Server running at http://${host}:${port}`);
});

server.on('connection', ({ id }) => {
    console.log(`[server] Client connected: ${id}`);
});

server.on('disconnect', ({ id }) => {
    console.log(`[server] Client disconnected: ${id}`);
});

server.on('error', (err) => {
    console.error(`[server] Error:`, err.message);
});

server.start().catch(console.error);

// Add global error handlers to prevent silent crashes from unhandled websocket drops
process.on('uncaughtException', (err) => {
    console.error('[Global] Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('[Global] Unhandled Rejection at:', promise, 'reason:', reason);
});

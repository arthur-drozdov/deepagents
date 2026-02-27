import { LLMProvider, LLMRequest, LLMResult, LLMChunk } from '@llmrtc/llmrtc-core';
import WebSocket from 'ws';

export class DeepAgentsLLMProvider implements LLMProvider {
    name = 'deepagents';
    private wsUrl: string;

    constructor(options: { wsUrl?: string } = {}) {
        this.wsUrl = options.wsUrl || 'ws://127.0.0.1:8080/chat';
    }

    async complete(request: LLMRequest): Promise<LLMResult> {
        // Collect all text from stream
        let fullText = '';
        for await (const chunk of this.stream(request)) {
            fullText += chunk.content;
        }
        return { fullText, stopReason: 'end_turn' };
    }

    async *stream(request: LLMRequest): AsyncIterable<LLMChunk> {
        const ws = new WebSocket(this.wsUrl);

        // Convert WebRTC messages to user text
        const lastUserMessage = request.messages.filter(m => m.role === 'user').pop();
        let textToSend = lastUserMessage?.content || '';
        if (!textToSend) {
            console.log(`[llm] Warning: Empty text received`);
        }

        const messageQueue: any[] = [];
        let resolveNextMessage: (() => void) | null = null;
        let done = false;
        let connected = false;

        ws.on('open', () => {
            connected = true;
            console.log(`[llm] Connected to Python bridge, sending text: "${textToSend}"`);
            ws.send(JSON.stringify({ text: textToSend }));
        });

        const signal = (request as any).abortSignal;
        if (signal) {
            signal.addEventListener('abort', () => {
                console.log('[llm] Request aborted by orchestrator, closing ws.');
                done = true;
                if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                    ws.close();
                }
                if (resolveNextMessage) {
                    resolveNextMessage();
                    resolveNextMessage = null;
                }
            });
        }

        ws.on('message', (data: string) => {
            console.log(`[llm] Received message: ${data.toString()}`);
            const msg = JSON.parse(data.toString());
            messageQueue.push(msg);
            if (resolveNextMessage) {
                resolveNextMessage();
                resolveNextMessage = null;
            }
        });

        ws.on('close', () => {
            done = true;
            if (resolveNextMessage) {
                resolveNextMessage();
                resolveNextMessage = null;
            }
        });

        ws.on('error', (err) => {
            console.error('DeepAgents WebSocket Error', err);
            done = true;
            if (resolveNextMessage) {
                resolveNextMessage();
                resolveNextMessage = null;
            }
        });

        while (true) {
            if (messageQueue.length === 0) {
                if (done) break;
                await new Promise<void>(resolve => {
                    resolveNextMessage = resolve;
                });
            }

            const nextMsg = messageQueue.shift();
            if (!nextMsg) continue;

            if (nextMsg.done) {
                break;
            }

            if (nextMsg.chunk) {
                console.log(`[llm] Yielding chunk: "${nextMsg.chunk}"`);
                yield { content: nextMsg.chunk, done: false } as LLMChunk;
            }
        }

        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
        // Yield final done chunk
        yield { content: '', done: true } as LLMChunk;
    }
}

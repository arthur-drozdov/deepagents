import { TTSProvider, TTSConfig, TTSResult } from '@llmrtc/llmrtc-core';
import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';

export class CustomWebSocketTTSProvider implements TTSProvider {
    name = 'custom-ws-tts';
    private wsUrl: string;
    private refWavPath?: string;
    private refText?: string;
    private isFirstRequest = true;

    constructor(options: { wsUrl: string, refWavPath?: string, refText?: string }) {
        this.wsUrl = options.wsUrl;
        this.refWavPath = options.refWavPath;
        this.refText = options.refText;
    }

    async speak(text: string, config?: TTSConfig): Promise<TTSResult> {
        const buffers: Buffer[] = [];
        for await (const chunk of this.speakStream(text, config)) {
            buffers.push(chunk);
        }
        // Resulting audio is pcm16 for llmrtc-core consumption
        return { audio: Buffer.concat(buffers), format: 'pcm' };
    }

    async *speakStream(text: string, config?: TTSConfig): AsyncIterable<Buffer> {
        const ws = new WebSocket(this.wsUrl, { maxPayload: 100 * 1024 * 1024 });

        let isConnected = false;
        let messageQueue: any[] = [];
        let resolveMessage: (() => void) | null = null;
        let wsClosed = false;

        ws.on('open', () => {
            isConnected = true;
            const payload: any = {
                text: text,
                emit_every_frames: 12,
                decode_window_frames: 80,
            };

            // Only include ref audio on the very first request
            if (this.isFirstRequest && this.refWavPath) {
                try {
                    if (fs.existsSync(this.refWavPath)) {
                        console.log(`[tts] Loading reference audio directly from ${this.refWavPath} via Python helper`);
                        const { execSync } = require('child_process');
                        const scriptPath = path.join(__dirname, 'get_ref_floats.py');

                        // Execute the small python script to get floats matching tts_client.py
                        const out = execSync(`cd /Users/adrozdov/repos/deepagents/libs/cli && uv run --with librosa --with numpy --with soundfile python ${scriptPath} ${this.refWavPath}`, {
                            maxBuffer: 100 * 1024 * 1024,
                            encoding: 'utf-8'
                        });

                        const floats = JSON.parse(out);
                        payload.ref_audio = floats;

                        // Also read ref text if refText is a path
                        if (this.refText && fs.existsSync(this.refText)) {
                            console.log(`[tts] Loading reference text from ${this.refText}`);
                            payload.ref_text = fs.readFileSync(this.refText, 'utf8').trim();
                        } else {
                            payload.ref_text = this.refText || '';
                        }
                    } else {
                        console.warn(`[tts] Reference audio WAV not found at ${this.refWavPath}, skipping reference clone.`);
                    }
                } catch (e) {
                    console.error("[tts] Failed to load ref audio via python helper", e);
                }
                this.isFirstRequest = false;
            }

            const payloadStr = JSON.stringify(payload);
            console.log(`[tts] Sending request for text: "${text.substring(0, 30)}..." (payload size: ${payloadStr.length} bytes)`);
            ws.send(payloadStr, (err) => {
                if (err) console.error("[tts] WebSocket send error:", err);
                else console.log("[tts] WebSocket send successful");
            });
        });

        ws.on('message', (data: WebSocket.Data) => {
            console.log(`[tts] Received message from server (type: ${typeof data}, length: ${data instanceof Buffer ? data.length : 'N/A'})`);
            messageQueue.push(data);
            if (resolveMessage) {
                resolveMessage();
                resolveMessage = null;
            }
        });

        ws.on('close', () => {
            wsClosed = true;
            if (resolveMessage) resolveMessage();
        });

        ws.on('error', (err) => {
            console.error('TTS WebSocket Error', err);
            wsClosed = true;
            if (resolveMessage) resolveMessage();
        });

        // Wait until connected and sent
        while (!isConnected && !wsClosed) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        if (wsClosed) return;

        let receivedMeta = false;

        while (true) {
            if (messageQueue.length === 0) {
                if (wsClosed) break;
                await new Promise<void>(resolve => { resolveMessage = resolve; });
            }

            const msg = messageQueue.shift();
            if (!msg) continue;

            let isJson = false;
            let jsonPayload: any = null;

            try {
                const textMsg = msg.toString('utf8');
                if (textMsg.trim().startsWith('{')) {
                    jsonPayload = JSON.parse(textMsg);
                    isJson = true;
                }
            } catch (e) {
                // Not JSON
            }

            if (isJson) {
                if (jsonPayload.event === 'meta') {
                    console.log(`[tts] Received meta: ${JSON.stringify(jsonPayload)}`);
                    receivedMeta = true;
                } else if (jsonPayload.event === 'eos') {
                    console.log(`[tts] Received EOS`);
                    break;
                } else if (jsonPayload.error) {
                    console.error("[tts] TTS Server Error:", jsonPayload.error);
                    break;
                }
            } else {
                // Binary audio chunk
                if (receivedMeta && Buffer.isBuffer(msg)) {
                    // Convert float32 (server format) to Int16LE (LLMRTC format)
                    try {
                        const numSamples = msg.length / 4;
                        const audioBuffer = Buffer.alloc(numSamples * 2);

                        for (let i = 0; i < numSamples; i++) {
                            let s = msg.readFloatLE(i * 4);
                            s = Math.max(-1, Math.min(1, s));
                            const intVal = Math.round(s < 0 ? s * 0x8000 : s * 0x7FFF);
                            audioBuffer.writeInt16LE(intVal, i * 2);
                        }

                        yield audioBuffer;

                    } catch (e) {
                        console.error("[tts] Error converting audio chunk", e);
                    }
                } else if (!receivedMeta) {
                    console.warn(`[tts] Received non-JSON data before meta (length: ${msg.length})`);
                }
            }
        }

        ws.close();
    }
}

import { STTProvider, STTConfig, STTResult } from '@llmrtc/llmrtc-core';
import WebSocket from 'ws';
import * as fs from 'fs';
import * as path from 'path';

export class CustomWebSocketSTTProvider implements STTProvider {
    name = 'custom-ws-stt';
    private wsUrl: string;

    constructor(options: { wsUrl: string }) {
        this.wsUrl = options.wsUrl;
    }

    async transcribe(audio: Buffer, config?: STTConfig): Promise<STTResult> {
        let lastResult: STTResult | null = null;
        for await (const result of this.transcribeStream(this.bufferToAsyncIterable(audio), config)) {
            lastResult = result;
        }
        return lastResult || { text: '', isFinal: true };
    }

    async *transcribeStream(audio: AsyncIterable<Buffer>, config?: STTConfig): AsyncIterable<STTResult> {
        const ws = new WebSocket(this.wsUrl);

        let isConnected = false;
        let messageQueue: any[] = [];
        let resolveMessage: (() => void) | null = null;
        let wsClosed = false;

        ws.on('open', () => {
            console.log(`[stt] Connected to ASR server`);
            isConnected = true;
        });

        ws.on('message', (data: WebSocket.Data) => {
            console.log(`[stt] Received message: ${data.toString()}`);
            try {
                const msg = JSON.parse(data.toString());
                messageQueue.push(msg);
                if (resolveMessage) {
                    resolveMessage();
                    resolveMessage = null;
                }
            } catch (e) {
                console.error('[stt] Failed to parse STT message', e);
            }
        });

        ws.on('close', () => {
            wsClosed = true;
            if (resolveMessage) resolveMessage();
        });

        ws.on('error', (err) => {
            console.error('STT WebSocket error', err);
            wsClosed = true;
            if (resolveMessage) resolveMessage();
        });

        // Wait for connection
        while (!isConnected && !wsClosed) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        if (wsClosed) return;

        // Start sending audio task in background
        const sendAudioTask = async () => {
            try {
                let chunkCount = 0;
                let totalBytes = 0;
                for await (const chunk of audio) {
                    if (!isConnected || wsClosed) break;

                    // The orchestrator gives us a WAV buffer (44-byte header + 16kHz Int16LE PCM).
                    // The ASR server expects raw Float32 16kHz PCM bytes.

                    // Skip 44-byte WAV header if it looks like a WAV
                    let pcmBytes = chunk;
                    if (chunk.length > 44 && chunk.toString('ascii', 0, 4) === 'RIFF') {
                        pcmBytes = chunk.subarray(44);
                    }

                    const numSamples = Math.floor(pcmBytes.length / 2);
                    const floatBuffer = Buffer.alloc(numSamples * 4);

                    for (let i = 0; i < numSamples; i++) {
                        const intSample = pcmBytes.readInt16LE(i * 2);
                        // Convert to float [-1.0, 1.0]
                        floatBuffer.writeFloatLE(intSample / 32768.0, i * 4);
                    }

                    ws.send(floatBuffer);
                    chunkCount++;
                    totalBytes += floatBuffer.length;

                    if (chunkCount % 10 === 0) {
                        console.log(`[stt] Sent ${chunkCount} chunks (${totalBytes} raw float32 bytes)`);
                    }
                }
                console.log(`[stt] Finished sending audio stream: ${chunkCount} chunks, ${totalBytes} float32 bytes`);
                if (!wsClosed) {
                    ws.send(JSON.stringify({ event: 'eos' }));
                }
            } catch (err) {
                console.error('Error sending audio to STT', err);
            }
        };
        sendAudioTask();

        // Consume messages
        while (true) {
            // Check for manual injection file
            const injectPath = '/tmp/inject_text.txt';
            // console.log(`[stt] Checking for injection at ${injectPath}`);
            if (fs.existsSync(injectPath)) {
                try {
                    const text = fs.readFileSync(injectPath, 'utf8').trim();
                    if (text) {
                        console.log(`[stt] FOUND injection file with text: "${text}"`);
                        yield { text, isFinal: true };
                        fs.unlinkSync(injectPath);
                    }
                } catch (e) {
                    console.error('[stt] Failed to read injection file', e);
                }
            }

            if (messageQueue.length === 0) {
                if (wsClosed) break;
                await new Promise<void>(resolve => { resolveMessage = resolve; });
            }

            const msg = messageQueue.shift();
            if (!msg) continue;

            if (msg.error) {
                console.error('[stt] STT Server Error:', msg.error);
                break;
            }

            console.log(`[stt] Processing message: ${JSON.stringify(msg)}`);

            if (msg.event === 'text' || msg.event === 'final') {
                console.log(`[stt] Yielding transcript: "${msg.text}" (isFinal: ${msg.event === 'final'})`);
                yield {
                    text: msg.text,
                    isFinal: msg.event === 'final'
                };
            }
        }

        ws.close();
    }

    private async *bufferToAsyncIterable(buffer: Buffer): AsyncIterable<Buffer> {
        yield buffer;
    }
}

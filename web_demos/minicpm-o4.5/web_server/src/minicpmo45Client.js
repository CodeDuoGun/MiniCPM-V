/** MiniCPM-o 4.5 full-duplex client. Payloads remain compatible with the 2.6 UI. */
export class MiniCPMO45Client {
    constructor({ baseUrl, uid, onChunk, onError }) {
        this.httpBaseUrl = baseUrl.replace(/\/$/, '');
        this.baseUrl = this.httpBaseUrl.replace(/^http/, 'ws');
        this.uid = uid;
        this.onChunk = onChunk || (() => {});
        this.onError = onError || console.error;
        this.socket = null;
    }

    async analyzeImage({ consultationId, scene, imageBase64, mimeType = 'image/jpeg', source = 'manual_upload' }) {
        const response = await fetch(`${this.httpBaseUrl}/api/v1/images/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', uid: this.uid },
            body: JSON.stringify({
                consultation_id: consultationId,
                scene,
                source,
                mime_type: mimeType,
                image_data: imageBase64
            })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `Image analysis failed: ${response.status}`);
        return result;
    }

    connect() {
        this.socket = new WebSocket(`${this.baseUrl}/ws/api/v1/stream?uid=${encodeURIComponent(this.uid)}`);
        this.socket.onmessage = event => this.onChunk(JSON.parse(event.data));
        this.socket.onerror = this.onError;
        return new Promise((resolve, reject) => {
            this.socket.onopen = resolve;
            this.socket.onclose = event => {
                if (!event.wasClean) reject(new Error(event.reason || `WebSocket closed: ${event.code}`));
            };
        });
    }

    sendChunk({ audioBase64, imageBase64 = '', transcript = '', timestamp = '', endOfTurn = false }) {
        const content = [{
            type: 'input_audio',
            input_audio: {
                data: audioBase64,
                format: 'wav',
                transcript,
                timestamp,
                end_of_turn: endOfTurn
            }
        }];
        if (imageBase64) {
            content.unshift({
                type: 'image_data',
                image_data: { data: imageBase64, source: 'realtime_video' }
            });
        }
        this.socket.send(JSON.stringify({ messages: [{ role: 'user', content }] }));
    }

    cancel() {
        if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ event: 'response.cancel' }));
        }
    }

    close() {
        this.socket?.close(1000, 'consultation finished');
    }
}

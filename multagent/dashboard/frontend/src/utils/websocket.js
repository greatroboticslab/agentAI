/**
 * WebSocket manager: manages all 6 WS connections to the backend.
 */

export class WebSocketManager {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl || `ws://${window.location.host}`
    this.connections = {}
    this.handlers = {}
    this.reconnectDelay = 2000
  }

  connect(channel, onMessage) {
    const url = `${this.baseUrl}/ws/${channel}`
    this.handlers[channel] = onMessage

    const ws = new WebSocket(url)

    ws.onopen = () => {
      console.log(`[WS] ${channel} connected`)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (this.handlers[channel]) {
          this.handlers[channel](data)
        }
      } catch (e) {
        // Binary or non-JSON message
        if (this.handlers[channel]) {
          this.handlers[channel](event.data)
        }
      }
    }

    ws.onclose = () => {
      console.log(`[WS] ${channel} disconnected, reconnecting...`)
      setTimeout(() => this.connect(channel, onMessage), this.reconnectDelay)
    }

    ws.onerror = (err) => {
      console.error(`[WS] ${channel} error:`, err)
    }

    this.connections[channel] = ws
    return ws
  }

  send(channel, data) {
    const ws = this.connections[channel]
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }

  connectAll(handlers) {
    for (const [channel, handler] of Object.entries(handlers)) {
      this.connect(channel, handler)
    }
  }

  closeAll() {
    for (const ws of Object.values(this.connections)) {
      if (ws) ws.close()
    }
    this.connections = {}
  }
}

export default WebSocketManager

<template>
  <div class="live-feed">
    <div class="panel-header">
      <span class="panel-title">Live Feed</span>
      <span class="fps">{{ fps }} FPS</span>
    </div>
    <div class="canvas-container" ref="containerRef">
      <canvas ref="canvasRef" @click="onCanvasClick"></canvas>
      <div v-if="!hasFrame" class="no-signal">No Video Signal</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { WebSocketManager } from '../utils/websocket'

const canvasRef = ref(null)
const containerRef = ref(null)
const hasFrame = ref(false)
const fps = ref(0)

let wsManager = null
let frameCount = 0
let fpsTimer = null

onMounted(() => {
  // Connect directly to video WebSocket
  wsManager = new WebSocketManager()
  wsManager.connect('video', onVideoMessage)

  // FPS counter
  fpsTimer = setInterval(() => {
    fps.value = frameCount
    frameCount = 0
  }, 1000)

  // Resize canvas
  const ro = new ResizeObserver(() => resizeCanvas())
  if (containerRef.value) ro.observe(containerRef.value)
})

onUnmounted(() => {
  if (wsManager) wsManager.closeAll()
  if (fpsTimer) clearInterval(fpsTimer)
})

function resizeCanvas() {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return
  canvas.width = container.clientWidth
  canvas.height = container.clientHeight
}

function onVideoMessage(data) {
  if (!data.jpeg_base64) return

  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')

  const img = new Image()
  img.onload = () => {
    resizeCanvas()
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Draw detection overlays
    if (data.detections) {
      const scaleX = canvas.width / (data.width || 640)
      const scaleY = canvas.height / (data.height || 480)

      data.detections.forEach(det => {
        const x = det.x * scaleX
        const y = det.y * scaleY
        const w = det.w * scaleX
        const h = det.h * scaleY

        ctx.strokeStyle = '#ef4444'
        ctx.lineWidth = 2
        ctx.strokeRect(x - w/2, y - h/2, w, h)

        ctx.fillStyle = '#ef4444'
        ctx.font = '11px monospace'
        ctx.fillText(`#${det.id} ${(det.confidence * 100).toFixed(0)}%`, x - w/2, y - h/2 - 4)
      })
    }

    hasFrame.value = true
    frameCount++
  }
  img.src = 'data:image/jpeg;base64,' + data.jpeg_base64
}

function onCanvasClick(e) {
  // Future: click to inspect weed
}
</script>

<style scoped>
.live-feed { display: flex; flex-direction: column; height: 100%; }
.panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 12px; background: #1e2433; border-bottom: 1px solid #2a3040;
}
.panel-title { font-size: 13px; font-weight: 600; color: #93a3b8; }
.fps { font-size: 11px; color: #6b7280; }
.canvas-container { flex: 1; position: relative; background: #000; }
canvas { display: block; width: 100%; height: 100%; }
.no-signal {
  position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; color: #4b5563; font-size: 14px;
}
</style>

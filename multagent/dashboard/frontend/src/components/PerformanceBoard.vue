<template>
  <div class="performance-board">
    <div class="panel-header">
      <span class="panel-title">Performance</span>
    </div>
    <div class="metrics-grid">
      <div class="metric-card">
        <span class="metric-label">Total Firings</span>
        <span class="metric-value">{{ systemStore.metrics.total_firings || 0 }}</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Avg Firing Duration</span>
        <span class="metric-value">{{ (systemStore.metrics.avg_firing_duration || 0).toFixed(1) }}s</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">YOLO Latency</span>
        <span class="metric-value" :class="latencyClass">
          {{ (systemStore.metrics.avg_yolo_latency_ms || 0).toFixed(0) }}ms
        </span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Avg Detections/Frame</span>
        <span class="metric-value">{{ (systemStore.metrics.avg_detections_per_frame || 0).toFixed(1) }}</span>
      </div>
    </div>

    <!-- Simple latency history chart using CSS bars -->
    <div class="chart-section">
      <div class="chart-title">YOLO Latency History</div>
      <div class="bar-chart">
        <div
          v-for="(point, i) in latencyHistory"
          :key="i"
          class="bar"
          :style="{ height: barHeight(point.avg_yolo_latency_ms) }"
          :class="{ warn: point.avg_yolo_latency_ms > 80, danger: point.avg_yolo_latency_ms > 150 }"
          :title="`${(point.avg_yolo_latency_ms || 0).toFixed(0)}ms`"
        ></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()

const latencyHistory = computed(() => {
  return systemStore.metricsHistory.slice(-60)
})

const latencyClass = computed(() => {
  const ms = systemStore.metrics.avg_yolo_latency_ms || 0
  if (ms > 150) return 'danger'
  if (ms > 80) return 'warn'
  return 'good'
})

function barHeight(ms) {
  const maxMs = 200
  const pct = Math.min(100, ((ms || 0) / maxMs) * 100)
  return `${Math.max(2, pct)}%`
}
</script>

<style scoped>
.performance-board { display: flex; flex-direction: column; height: 100%; }
.panel-header {
  padding: 6px 12px; background: #1e2433; border-bottom: 1px solid #2a3040;
}
.panel-title { font-size: 13px; font-weight: 600; color: #93a3b8; }

.metrics-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 8px;
}
.metric-card {
  background: #151a24; border-radius: 6px; padding: 8px 10px;
  display: flex; flex-direction: column; align-items: center;
}
.metric-label { font-size: 10px; color: #6b7280; }
.metric-value { font-size: 16px; font-weight: 700; color: #e1e8ed; }
.metric-value.good { color: #10b981; }
.metric-value.warn { color: #f59e0b; }
.metric-value.danger { color: #ef4444; }

.chart-section { flex: 1; padding: 0 8px 8px; display: flex; flex-direction: column; }
.chart-title { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
.bar-chart {
  flex: 1; display: flex; align-items: flex-end; gap: 1px;
  background: #151a24; border-radius: 4px; padding: 4px;
  min-height: 60px;
}
.bar {
  flex: 1; background: #10b981; border-radius: 1px 1px 0 0;
  min-height: 2px; transition: height 0.3s;
}
.bar.warn { background: #f59e0b; }
.bar.danger { background: #ef4444; }
</style>

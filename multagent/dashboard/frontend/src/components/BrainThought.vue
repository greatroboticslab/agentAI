<template>
  <div class="brain-thought">
    <div class="panel-header">
      <span class="panel-title">Brain Thinking</span>
      <span class="count">{{ systemStore.brainThoughts.length }}</span>
    </div>
    <div class="thought-list" ref="listRef">
      <div v-for="(t, i) in recentThoughts" :key="i" class="thought-card">
        <div class="thought-time">{{ formatTimestamp(t.timestamp) }}</div>
        <div class="thought-text">{{ t.thought || 'No reasoning' }}</div>

        <div v-if="t.actions && t.actions.length" class="actions">
          <div v-for="(a, j) in t.actions" :key="j" class="action-item">
            <span class="action-type" :class="a.type?.toLowerCase()">{{ a.type }}</span>
            <span class="action-detail">
              {{ a.target || '' }} {{ a.reason || '' }}
            </span>
          </div>
        </div>

        <div v-if="t.summary" class="summary">
          <span>Events: {{ t.summary.total_events || 0 }}</span>
          <span>Detections: {{ t.summary.detection_count || 0 }}</span>
          <span>Firings: {{ t.summary.firings_completed || 0 }}</span>
        </div>
      </div>

      <div v-if="!recentThoughts.length" class="empty">
        Brain is idle - no events to analyze
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch, nextTick } from 'vue'
import { useSystemStore } from '../stores/system'
import { formatTimestamp } from '../utils/formatters'

const systemStore = useSystemStore()
const listRef = ref(null)

const recentThoughts = computed(() => {
  return systemStore.brainThoughts.slice(-20).reverse()
})

watch(() => systemStore.brainThoughts.length, async () => {
  await nextTick()
  if (listRef.value) listRef.value.scrollTop = 0
})
</script>

<style scoped>
.brain-thought { display: flex; flex-direction: column; height: 100%; }
.panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 12px; background: #1e2433; border-bottom: 1px solid #2a3040;
}
.panel-title { font-size: 13px; font-weight: 600; color: #93a3b8; }
.count { font-size: 11px; color: #6b7280; }
.thought-list { flex: 1; overflow-y: auto; padding: 8px; }
.thought-card {
  background: #151a24; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
  border-left: 3px solid #8b5cf6;
}
.thought-time { font-size: 10px; color: #6b7280; margin-bottom: 4px; }
.thought-text { font-size: 12px; color: #e1e8ed; margin-bottom: 4px; }
.actions { margin-top: 4px; }
.action-item { display: flex; gap: 6px; align-items: center; margin-top: 2px; }
.action-type {
  font-size: 10px; padding: 1px 6px; border-radius: 3px;
  font-weight: 600; background: #2a3040; color: #93a3b8;
}
.action-type.adjust_param { background: #1e3a5f; color: #3b82f6; }
.action-type.no_action { background: #1a2e1a; color: #6b7280; }
.action-detail { font-size: 11px; color: #6b7280; }
.summary {
  display: flex; gap: 12px; margin-top: 4px; font-size: 10px; color: #4b5563;
}
.empty { text-align: center; padding: 30px; color: #4b5563; font-size: 13px; }
</style>

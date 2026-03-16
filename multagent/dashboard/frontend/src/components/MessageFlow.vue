<template>
  <div class="message-flow">
    <div class="panel-header">
      <span class="panel-title">Message Flow</span>
      <span class="count">{{ messageStore.messages.length }}</span>
    </div>
    <div class="flow-list" ref="listRef">
      <div
        v-for="msg in recentMessages"
        :key="msg.id"
        class="flow-item"
        :class="msg.type"
      >
        <span class="flow-from" :style="{ color: agentColor(msg.from) }">
          {{ msg.from || '?' }}
        </span>
        <span class="flow-arrow">-></span>
        <span class="flow-type">{{ msg.type }}</span>
        <span class="flow-time">{{ formatTimestamp(msg.timestamp) }}</span>
      </div>

      <div v-if="!recentMessages.length" class="empty">
        No messages yet
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch, nextTick } from 'vue'
import { useMessageStore } from '../stores/messages'
import { formatTimestamp } from '../utils/formatters'

const messageStore = useMessageStore()
const listRef = ref(null)

const recentMessages = computed(() => {
  return messageStore.messages.slice(-50).reverse()
})

watch(() => messageStore.messages.length, async () => {
  await nextTick()
  if (listRef.value) listRef.value.scrollTop = 0
})

function agentColor(name) {
  const colors = {
    Perception: '#10b981',
    Targeting: '#ef4444',
    Navigation: '#f59e0b',
    Brain: '#8b5cf6',
    dashboard: '#3b82f6',
  }
  return colors[name] || '#6b7280'
}
</script>

<style scoped>
.message-flow { display: flex; flex-direction: column; height: 100%; }
.panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 12px; background: #1e2433; border-bottom: 1px solid #2a3040;
}
.panel-title { font-size: 13px; font-weight: 600; color: #93a3b8; }
.count { font-size: 11px; color: #6b7280; }
.flow-list { flex: 1; overflow-y: auto; padding: 4px 8px; }
.flow-item {
  display: flex; align-items: center; gap: 6px; padding: 3px 6px;
  font-size: 11px; font-family: monospace; border-radius: 3px;
  margin-bottom: 2px;
}
.flow-item:hover { background: #1e2433; }
.flow-from { font-weight: 600; min-width: 80px; }
.flow-arrow { color: #4b5563; }
.flow-type { color: #93a3b8; flex: 1; }
.flow-time { color: #4b5563; font-size: 10px; }
.empty { text-align: center; padding: 30px; color: #4b5563; font-size: 13px; }
</style>

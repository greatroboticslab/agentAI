<template>
  <div class="app">
    <!-- Top Bar -->
    <header class="topbar">
      <div class="topbar-left">
        <h1 class="logo">EMACF</h1>
        <span class="subtitle">LaserCar Dashboard</span>
      </div>
      <div class="topbar-center">
        <span class="mode-badge" :style="{ background: modeColor(systemStore.currentMode) }">
          {{ systemStore.currentMode }}
        </span>
        <span class="connection" :class="{ online: systemStore.connected }">
          {{ systemStore.connected ? 'Connected' : 'Disconnected' }}
        </span>
      </div>
      <div class="topbar-right">
        <button class="emergency-btn" @click="emergencyStop">EMERGENCY STOP</button>
      </div>
    </header>

    <!-- Main Layout -->
    <div class="main-layout">
      <!-- Left Sidebar -->
      <aside class="sidebar">
        <AgentPanel />
      </aside>

      <!-- Center Content -->
      <main class="content">
        <div class="grid-top">
          <LiveFeed class="panel" />
          <BrainThought class="panel" />
        </div>
        <div class="grid-bottom">
          <MessageFlow class="panel" />
          <PerformanceBoard class="panel" />
        </div>
      </main>
    </div>

    <!-- Bottom Chat -->
    <footer class="chat-bar">
      <ChatPanel />
    </footer>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import { useAgentStore } from './stores/agents'
import { useMessageStore } from './stores/messages'
import { useSystemStore } from './stores/system'
import { WebSocketManager } from './utils/websocket'
import { modeColor } from './utils/formatters'

import LiveFeed from './components/LiveFeed.vue'
import AgentPanel from './components/AgentPanel.vue'
import BrainThought from './components/BrainThought.vue'
import MessageFlow from './components/MessageFlow.vue'
import PerformanceBoard from './components/PerformanceBoard.vue'
import ChatPanel from './components/ChatPanel.vue'

const agentStore = useAgentStore()
const messageStore = useMessageStore()
const systemStore = useSystemStore()

let wsManager = null

onMounted(() => {
  wsManager = new WebSocketManager()

  wsManager.connectAll({
    agents: (data) => {
      agentStore.updateAgents(data)
      systemStore.connected = true
      // Extract mode from Navigation agent
      const nav = data.agents?.find(a => a.name === 'Navigation')
      if (nav) systemStore.currentMode = nav.current_mode || 'IDLE'
    },
    messages: (data) => messageStore.addMessage(data),
    brain: (data) => systemStore.addBrainThought(data),
    chat: (data) => systemStore.addChatMessage(data),
    performance: (data) => systemStore.updateMetrics(data),
    video: () => {}, // Handled by LiveFeed directly
  })
})

onUnmounted(() => {
  if (wsManager) wsManager.closeAll()
})

async function emergencyStop() {
  try {
    await fetch('/api/system/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'SWC' }),
    })
  } catch (e) {
    console.error('Emergency stop failed:', e)
  }
}
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #0f1419;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 20px;
  background: #1a1f2e;
  border-bottom: 1px solid #2a3040;
  height: 48px;
}

.topbar-left { display: flex; align-items: center; gap: 12px; }
.logo { font-size: 18px; font-weight: 700; color: #3b82f6; }
.subtitle { font-size: 13px; color: #6b7280; }

.topbar-center { display: flex; align-items: center; gap: 16px; }
.mode-badge {
  padding: 2px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  color: white;
}
.connection {
  font-size: 12px;
  color: #ef4444;
}
.connection.online { color: #10b981; }

.emergency-btn {
  background: #ef4444;
  color: white;
  border: none;
  padding: 6px 16px;
  border-radius: 6px;
  font-weight: 700;
  font-size: 12px;
  cursor: pointer;
  letter-spacing: 0.5px;
}
.emergency-btn:hover { background: #dc2626; }

.main-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 260px;
  min-width: 260px;
  background: #151a24;
  border-right: 1px solid #2a3040;
  overflow-y: auto;
}

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 8px;
  gap: 8px;
  overflow: hidden;
}

.grid-top, .grid-bottom {
  display: flex;
  gap: 8px;
  flex: 1;
  min-height: 0;
}

.panel {
  flex: 1;
  background: #1a1f2e;
  border-radius: 8px;
  border: 1px solid #2a3040;
  overflow: hidden;
}

.chat-bar {
  height: 160px;
  min-height: 160px;
  border-top: 1px solid #2a3040;
  background: #151a24;
}
</style>

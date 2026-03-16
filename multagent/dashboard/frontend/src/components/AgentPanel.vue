<template>
  <div class="agent-panel">
    <div class="panel-header">
      <span class="panel-title">Agents</span>
      <span class="count">{{ agentStore.agents.length }}</span>
    </div>
    <div class="agent-list">
      <div
        v-for="agent in agentStore.agents"
        :key="agent.name"
        class="agent-card"
        @click="selectedAgent = selectedAgent === agent.name ? null : agent.name"
      >
        <div class="agent-header">
          <span class="agent-dot" :style="{ background: statusColor(agent.status) }"></span>
          <span class="agent-name">{{ agent.name }}</span>
          <span class="agent-status">{{ agent.status || 'idle' }}</span>
        </div>

        <div class="agent-stats">
          <div class="stat" v-if="agent.tracked_weeds !== undefined">
            <span class="stat-label">Weeds</span>
            <span class="stat-value">{{ agent.tracked_weeds }}</span>
          </div>
          <div class="stat" v-if="agent.firing_count !== undefined">
            <span class="stat-label">Firings</span>
            <span class="stat-value">{{ agent.firing_count }}</span>
          </div>
          <div class="stat" v-if="agent.current_mode">
            <span class="stat-label">Mode</span>
            <span class="stat-value">{{ agent.current_mode }}</span>
          </div>
          <div class="stat" v-if="agent.frames_processed !== undefined">
            <span class="stat-label">Frames</span>
            <span class="stat-value">{{ agent.frames_processed }}</span>
          </div>
          <div class="stat" v-if="agent.has_llm !== undefined">
            <span class="stat-label">LLM</span>
            <span class="stat-value">{{ agent.has_llm ? 'Yes' : 'No' }}</span>
          </div>
        </div>

        <!-- Expanded params -->
        <div v-if="selectedAgent === agent.name && agent.params" class="agent-params">
          <div class="param-row" v-for="(val, key) in agent.params" :key="key">
            <span class="param-key">{{ key }}</span>
            <span class="param-val">{{ typeof val === 'number' ? val.toFixed(2) : val }}</span>
          </div>
        </div>
      </div>

      <div v-if="agentStore.agents.length === 0" class="empty">
        No agents connected
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAgentStore } from '../stores/agents'
import { statusColor } from '../utils/formatters'

const agentStore = useAgentStore()
const selectedAgent = ref(null)
</script>

<style scoped>
.agent-panel { display: flex; flex-direction: column; height: 100%; }
.panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; border-bottom: 1px solid #2a3040;
}
.panel-title { font-size: 13px; font-weight: 600; color: #93a3b8; }
.count {
  background: #2a3040; padding: 1px 8px; border-radius: 10px;
  font-size: 11px; color: #6b7280;
}
.agent-list { flex: 1; overflow-y: auto; padding: 8px; }
.agent-card {
  background: #1a1f2e; border: 1px solid #2a3040; border-radius: 6px;
  padding: 10px; margin-bottom: 8px; cursor: pointer; transition: border-color 0.2s;
}
.agent-card:hover { border-color: #3b82f6; }
.agent-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.agent-dot { width: 8px; height: 8px; border-radius: 50%; }
.agent-name { font-size: 13px; font-weight: 600; flex: 1; }
.agent-status { font-size: 11px; color: #6b7280; }
.agent-stats { display: flex; flex-wrap: wrap; gap: 6px; }
.stat {
  display: flex; flex-direction: column; align-items: center;
  background: #151a24; padding: 4px 8px; border-radius: 4px; min-width: 50px;
}
.stat-label { font-size: 10px; color: #6b7280; }
.stat-value { font-size: 12px; font-weight: 600; color: #e1e8ed; }
.agent-params {
  margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a3040;
}
.param-row {
  display: flex; justify-content: space-between; padding: 2px 0; font-size: 11px;
}
.param-key { color: #6b7280; }
.param-val { color: #93a3b8; font-family: monospace; }
.empty { text-align: center; padding: 20px; color: #4b5563; font-size: 13px; }
</style>

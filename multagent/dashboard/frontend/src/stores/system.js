import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSystemStore = defineStore('system', () => {
  const connected = ref(false)
  const currentMode = ref('IDLE')
  const metrics = ref({})
  const metricsHistory = ref([])
  const maxHistory = 120  // 1 minute at 500ms intervals

  const brainThoughts = ref([])
  const chatMessages = ref([])

  function updateMetrics(data) {
    metrics.value = data.metrics || {}
    metricsHistory.value.push({
      ...data.metrics,
      timestamp: data.timestamp,
    })
    if (metricsHistory.value.length > maxHistory) {
      metricsHistory.value = metricsHistory.value.slice(-maxHistory)
    }
  }

  function addBrainThought(thought) {
    brainThoughts.value.push(thought)
    if (brainThoughts.value.length > 50) {
      brainThoughts.value = brainThoughts.value.slice(-50)
    }
  }

  function addChatMessage(msg) {
    chatMessages.value.push(msg)
    if (chatMessages.value.length > 100) {
      chatMessages.value = chatMessages.value.slice(-100)
    }
  }

  return {
    connected, currentMode, metrics, metricsHistory,
    brainThoughts, chatMessages,
    updateMetrics, addBrainThought, addChatMessage,
  }
})

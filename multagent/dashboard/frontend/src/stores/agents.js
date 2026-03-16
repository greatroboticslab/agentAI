import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAgentStore = defineStore('agents', () => {
  const agents = ref([])
  const lastUpdate = ref(0)

  function updateAgents(data) {
    agents.value = data.agents || []
    lastUpdate.value = data.timestamp || Date.now() / 1000
  }

  function getAgent(name) {
    return agents.value.find(a => a.name === name)
  }

  return { agents, lastUpdate, updateAgents, getAgent }
})

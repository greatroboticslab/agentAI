import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useMessageStore = defineStore('messages', () => {
  const messages = ref([])
  const maxMessages = 200

  function addMessage(msg) {
    messages.value.push({
      ...msg,
      id: Date.now() + Math.random(),
    })
    if (messages.value.length > maxMessages) {
      messages.value = messages.value.slice(-maxMessages)
    }
  }

  function clear() {
    messages.value = []
  }

  return { messages, addMessage, clear }
})

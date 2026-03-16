<template>
  <div class="chat-panel">
    <div class="chat-messages" ref="messagesRef">
      <div
        v-for="(msg, i) in systemStore.chatMessages"
        :key="i"
        class="chat-msg"
        :class="msg.role"
      >
        <span class="msg-role">{{ msg.role === 'user' ? 'You' : 'Brain' }}</span>
        <span class="msg-text">{{ msg.message }}</span>
        <span class="msg-time">{{ formatTimestamp(msg.timestamp) }}</span>
      </div>

      <div v-if="!systemStore.chatMessages.length" class="empty">
        Chat with the Brain - ask questions or give commands in any language
      </div>
    </div>

    <div class="chat-input">
      <input
        v-model="inputText"
        @keyup.enter="sendMessage"
        placeholder="Type a message to Brain..."
        class="input-field"
      />
      <button @click="sendMessage" class="send-btn" :disabled="!inputText.trim()">
        Send
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { useSystemStore } from '../stores/system'
import { WebSocketManager } from '../utils/websocket'
import { formatTimestamp } from '../utils/formatters'

const systemStore = useSystemStore()
const inputText = ref('')
const messagesRef = ref(null)

// Use a shared WS for chat
let chatWs = null

function sendMessage() {
  const text = inputText.value.trim()
  if (!text) return

  // Send via REST API (simpler than managing WS state here)
  fetch('/api/brain/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  }).catch(err => console.error('Chat send error:', err))

  // Optimistically add user message
  systemStore.addChatMessage({
    role: 'user',
    message: text,
    timestamp: Date.now() / 1000,
  })

  inputText.value = ''
}

watch(() => systemStore.chatMessages.length, async () => {
  await nextTick()
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  }
})
</script>

<style scoped>
.chat-panel {
  display: flex; flex-direction: column; height: 100%; padding: 8px 12px;
}

.chat-messages {
  flex: 1; overflow-y: auto; margin-bottom: 8px;
  display: flex; flex-direction: column; gap: 4px;
}

.chat-msg {
  display: flex; gap: 8px; align-items: baseline;
  padding: 4px 8px; border-radius: 6px; max-width: 80%;
}
.chat-msg.user {
  background: #1e3a5f; align-self: flex-end;
}
.chat-msg.brain {
  background: #2a1e3f; align-self: flex-start;
}

.msg-role { font-size: 10px; font-weight: 600; color: #6b7280; min-width: 35px; }
.msg-text { font-size: 12px; color: #e1e8ed; flex: 1; }
.msg-time { font-size: 10px; color: #4b5563; }

.chat-input { display: flex; gap: 8px; }
.input-field {
  flex: 1; background: #1a1f2e; border: 1px solid #2a3040; border-radius: 6px;
  padding: 8px 12px; color: #e1e8ed; font-size: 13px; outline: none;
}
.input-field:focus { border-color: #3b82f6; }
.input-field::placeholder { color: #4b5563; }

.send-btn {
  background: #3b82f6; color: white; border: none; padding: 8px 16px;
  border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;
}
.send-btn:hover { background: #2563eb; }
.send-btn:disabled { background: #2a3040; color: #4b5563; cursor: not-allowed; }

.empty { text-align: center; color: #4b5563; font-size: 12px; padding: 20px; }
</style>

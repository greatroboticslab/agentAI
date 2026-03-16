<template>
  <div class="param-control" v-if="visible">
    <div class="overlay" @click="$emit('close')"></div>
    <div class="modal">
      <div class="modal-header">
        <span class="modal-title">{{ agentName }} Parameters</span>
        <button class="close-btn" @click="$emit('close')">X</button>
      </div>
      <div class="param-list">
        <div v-for="(val, key) in localParams" :key="key" class="param-row">
          <label class="param-label">{{ key }}</label>
          <input
            v-if="typeof val === 'number'"
            type="number"
            :value="val"
            step="0.01"
            class="param-input"
            @change="localParams[key] = parseFloat($event.target.value)"
          />
          <input
            v-else-if="typeof val === 'boolean'"
            type="checkbox"
            :checked="val"
            class="param-checkbox"
            @change="localParams[key] = $event.target.checked"
          />
          <input
            v-else
            type="text"
            :value="val"
            class="param-input"
            @change="localParams[key] = $event.target.value"
          />
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" @click="resetParams">Reset</button>
        <button class="btn-primary" @click="applyParams">Apply</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  visible: Boolean,
  agentName: String,
  params: Object,
})

const emit = defineEmits(['close', 'apply'])

const localParams = ref({})

watch(() => props.params, (newParams) => {
  localParams.value = { ...newParams }
}, { immediate: true, deep: true })

function resetParams() {
  localParams.value = { ...props.params }
}

async function applyParams() {
  try {
    await fetch(`/api/agents/${props.agentName}/params`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: localParams.value }),
    })
    emit('close')
  } catch (e) {
    console.error('Param update failed:', e)
  }
}
</script>

<style scoped>
.param-control { position: fixed; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center; }
.overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.6); }
.modal {
  position: relative; background: #1a1f2e; border: 1px solid #2a3040;
  border-radius: 10px; width: 420px; max-height: 80vh; display: flex; flex-direction: column;
}
.modal-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; border-bottom: 1px solid #2a3040;
}
.modal-title { font-size: 14px; font-weight: 600; }
.close-btn { background: none; border: none; color: #6b7280; font-size: 16px; cursor: pointer; }
.param-list { flex: 1; overflow-y: auto; padding: 12px 16px; }
.param-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.param-label { font-size: 12px; color: #93a3b8; }
.param-input {
  width: 120px; background: #151a24; border: 1px solid #2a3040;
  border-radius: 4px; padding: 4px 8px; color: #e1e8ed; font-size: 12px;
  font-family: monospace;
}
.param-checkbox { accent-color: #3b82f6; }
.modal-footer {
  display: flex; justify-content: flex-end; gap: 8px;
  padding: 12px 16px; border-top: 1px solid #2a3040;
}
.btn-primary {
  background: #3b82f6; color: white; border: none; padding: 6px 16px;
  border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
}
.btn-secondary {
  background: #2a3040; color: #93a3b8; border: none; padding: 6px 16px;
  border-radius: 6px; font-size: 12px; cursor: pointer;
}
</style>

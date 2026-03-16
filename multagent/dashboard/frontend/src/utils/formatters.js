/**
 * Data formatting utilities for the dashboard.
 */

export function formatTimestamp(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString()
}

export function formatLatency(ms) {
  if (ms === undefined || ms === null) return '-'
  return `${ms.toFixed(1)}ms`
}

export function formatPercent(val) {
  if (val === undefined || val === null) return '-'
  return `${val.toFixed(1)}%`
}

export function truncate(str, maxLen = 60) {
  if (!str) return ''
  return str.length > maxLen ? str.slice(0, maxLen) + '...' : str
}

export function modeColor(mode) {
  const colors = {
    IDLE: '#6b7280',
    SWA: '#10b981',
    SWB: '#3b82f6',
    SWC: '#ef4444',
    SWD: '#f59e0b',
  }
  return colors[mode] || '#6b7280'
}

export function statusColor(status) {
  if (status === 'running') return '#10b981'
  if (status === 'error') return '#ef4444'
  return '#6b7280'
}

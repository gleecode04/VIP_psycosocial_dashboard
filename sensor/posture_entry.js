import { PostureAccController } from './posture_acc_controller.js';

// Debugging aid: mirror every console.log/warn/error to the dashboard
// server (posture_connection_debug.log), since Web Bluetooth's native
// device chooser and OS-level pairing prompts can't be driven or watched
// remotely -- this is the only way to get real connection-attempt timing
// and the polar-h10 library's own verbose step logging back out of the
// browser for debugging. Not gated/throttled: debug-only and low-volume
// (connection attempts, not per-ACC-sample).
(function installConsoleMirror() {
  const buffer = [];
  let flushTimer = null;
  function flush() {
    flushTimer = null;
    if (!buffer.length) return;
    const lines = buffer.splice(0, buffer.length);
    fetch('/api/posture/debug-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lines }),
    }).catch(() => {});
  }
  function mirror(kind) {
    const original = console[kind].bind(console);
    console[kind] = (...args) => {
      original(...args);
      buffer.push(`[${kind}] ${args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')}`);
      if (!flushTimer) flushTimer = setTimeout(flush, 150);
    };
  }
  mirror('log');
  mirror('warn');
  mirror('error');
})();

const STATE_LABELS = {
  idle: 'Not connected',
  connecting: 'Connecting…',
  reconnecting: 'Reconnecting…',
  error: 'Connection error',
};

async function post(url, body) {
  try {
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    // Best-effort: the Posture tab's own status poll already surfaces an
    // unreachable dashboard server, no need to duplicate that here.
  }
}

function mountPosturePolar(root) {
  const connectBtn = root.querySelector('#postureConnectBtn');
  const disconnectBtn = root.querySelector('#postureDisconnectBtn');
  const statusEl = root.querySelector('#postureBleStatus');
  const startBtn = document.getElementById('postureStartBtn');
  if (!connectBtn || !disconnectBtn || !statusEl) return;

  const controller = new PostureAccController({
    onChange(c) {
      const connected = c.state === 'connected';
      statusEl.textContent = connected
        ? `Connected (${c.device?.name || 'Polar H10'})`
        : (STATE_LABELS[c.state] || c.message);
      // Same enabling rule as sensor/panel.js's EXG Connect/Disconnect pair:
      // Connect is only disabled while an attempt is actually in flight or
      // wanted (so a failed/exhausted attempt always re-enables it), and
      // Disconnect stays clickable any time there's something to cancel --
      // an always-available escape hatch out of a stuck retry loop.
      connectBtn.disabled = c.busy || c.wanted;
      disconnectBtn.disabled = !c.wanted && !c.busy;
      if (startBtn) startBtn.disabled = !connected;
      post('/api/posture/connection', { connected });
    },
    onReading(reading) {
      post('/api/posture/reading', reading);
    },
  });

  connectBtn.addEventListener('click', () => void controller.connect());
  disconnectBtn.addEventListener('click', () => controller.disconnect());
}

const root = document.getElementById('posturePolarPanel');
if (root) mountPosturePolar(root);

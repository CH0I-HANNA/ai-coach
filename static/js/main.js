'use strict';

const ANALYZE_INTERVAL_MS = 500;
const LONG_PRESS_MS = 1000;

let currentMode = 'waiting';
let isExplaining = false;
let analyzeTimer = null;
let longPressTimer = null;
let longPressTriggered = false;

// ── DOM refs ──────────────────────────────────────────────
const camera        = document.getElementById('camera');
const overlayCanvas = document.getElementById('overlay');
const ctx           = overlayCanvas.getContext('2d');
const modeIcon      = document.getElementById('mode-icon');
const modeText      = document.getElementById('mode-text');
const modeIndicator = document.getElementById('mode-indicator');
const alertText     = document.getElementById('alert-text');
const borderFlash   = document.getElementById('border-flash');
const ttsPlayer     = document.getElementById('tts-player');
const micIndicator  = document.getElementById('mic-indicator');

// ── Camera ────────────────────────────────────────────────
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    camera.srcObject = stream;
    camera.addEventListener('loadedmetadata', () => {
      overlayCanvas.width  = camera.videoWidth;
      overlayCanvas.height = camera.videoHeight;
    });
  } catch (err) {
    showAlert('카메라를 사용할 수 없습니다: ' + err.message, 'info');
  }
}

// ── Mode ──────────────────────────────────────────────────
const MODE_CONFIG = {
  waiting:   { icon: '⏳', label: '대기 중',       cls: 'waiting' },
  crosswalk: { icon: '🚦', label: '횡단보도 모드', cls: 'crosswalk' },
  sidewalk:  { icon: '🚶', label: '보도 보행 모드', cls: 'sidewalk' },
};

function updateModeIndicator() {
  const cfg = MODE_CONFIG[currentMode];
  modeIcon.textContent = cfg.icon;
  modeText.textContent = cfg.label;
  modeIndicator.className = cfg.cls;
}

function interruptTTS() {
  ttsQueue.length = 0;
  ttsPlaying = false;
  ttsPlayer.pause();
  ttsPlayer.src = '';
  if ('speechSynthesis' in window) speechSynthesis.cancel();
}

function switchMode() {
  interruptTTS();
  if (currentMode === 'waiting' || currentMode === 'sidewalk') {
    currentMode = 'crosswalk';
    showAlert('횡단보도 모드', 'info');
    playTTSText('횡단보도 모드입니다');
  } else {
    currentMode = 'sidewalk';
    showAlert('보도 보행 모드', 'info');
    playTTSText('보도 보행 모드입니다');
  }
  updateModeIndicator();
  startAnalyzeLoop();
}

// ── Frame capture ─────────────────────────────────────────
function captureFrame() {
  const tmp = document.createElement('canvas');
  tmp.width  = camera.videoWidth  || 640;
  tmp.height = camera.videoHeight || 480;
  tmp.getContext('2d').drawImage(camera, 0, 0, tmp.width, tmp.height);
  // strip "data:image/jpeg;base64," prefix
  return tmp.toDataURL('image/jpeg', 0.7).split(',')[1];
}

// ── Analyze loop ──────────────────────────────────────────
function startAnalyzeLoop() {
  if (analyzeTimer) return;
  analyzeTimer = setInterval(analyzeFrame, ANALYZE_INTERVAL_MS);
}

async function analyzeFrame() {
  if (currentMode === 'waiting' || isExplaining) return;
  if (!camera.videoWidth) return;

  let frame;
  try { frame = captureFrame(); } catch { return; }

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame, mode: currentMode }),
    });
    const data = await res.json();

    if (data.alert) {
      if (data.alert_url) playTTSUrl(data.alert_url, data.alert, data.alert_type);
      else showAlert(data.alert, data.alert_type);
    }

    drawOverlay(data.detections || []);
  } catch { /* network error — skip frame */ }
}

// ── Voice question ────────────────────────────────────────
const LISTEN_TIMEOUT_MS = 8000;

function startVoiceQuestion() {
  interruptTTS();
  isExplaining = false;

  let frame;
  try { frame = captureFrame(); } catch { return; }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showAlert('음성 인식이 지원되지 않는 브라우저입니다', 'info');
    return;
  }

  showMicIndicator(true);

  // TTS onend가 불안정하므로 고정 딜레이로 인식 시작
  const utt = new SpeechSynthesisUtterance('질문하세요');
  utt.lang = 'ko-KR';
  speechSynthesis.speak(utt);
  setTimeout(() => startRecognition(frame), 1200);
}

function startRecognition(frame) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = 'ko-KR';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  let heard = false;
  const listenTimer = setTimeout(() => recognition.stop(), LISTEN_TIMEOUT_MS);

  recognition.onresult = (e) => {
    heard = true;
    clearTimeout(listenTimer);
    const question = e.results[0][0].transcript;
    showMicIndicator(false);
    sendAsk(frame, question);
  };

  recognition.onend = () => {
    clearTimeout(listenTimer);
    showMicIndicator(false);
    if (!heard) showAlert('질문을 듣지 못했습니다', 'info');
  };

  recognition.onerror = (e) => {
    clearTimeout(listenTimer);
    showMicIndicator(false);
    if (!heard) showAlert(`음성 인식 오류: ${e.error}`, 'info');
    console.error('SpeechRecognition error:', e.error, e);
  };

  recognition.start();
}

async function sendAsk(frame, question) {
  if (isExplaining) return;
  isExplaining = true;
  showAlert('답변 생성 중…', 'info');

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame, question }),
    });
    const data = await res.json();
    if (data.audio_url) playTTSUrl(data.audio_url, data.description || '', 'info');
    else showAlert(data.description || '', 'info');
  } catch {
    showAlert('분석 중 오류가 발생했습니다', 'info');
  } finally {
    isExplaining = false;
  }
}

function showMicIndicator(visible) {
  micIndicator.style.display = visible ? 'flex' : 'none';
}

// ── TTS Queue ─────────────────────────────────────────────
const ttsQueue = [];
let ttsPlaying = false;
const TTS_QUEUE_MAX = 2;

function enqueueTTS(item) {
  if (ttsQueue.length >= TTS_QUEUE_MAX) ttsQueue.shift();
  ttsQueue.push(item);
  drainTTSQueue();
}

function drainTTSQueue() {
  if (ttsPlaying || ttsQueue.length === 0) return;
  ttsPlaying = true;
  const item = ttsQueue.shift();

  if (item.alertText) showAlert(item.alertText, item.alertType);

  if (item.type === 'url') {
    ttsPlayer.src = item.value;
    ttsPlayer.onended = onTTSDone;
    ttsPlayer.onerror = onTTSDone;
    ttsPlayer.play().catch(onTTSDone);
  } else {
    const utt = new SpeechSynthesisUtterance(item.value);
    utt.lang = 'ko-KR';
    utt.onend   = onTTSDone;
    utt.onerror = onTTSDone;
    speechSynthesis.speak(utt);
  }
}

function onTTSDone() {
  ttsPlaying = false;
  drainTTSQueue();
}

function playTTSUrl(url, alertText = '', alertType = '') {
  enqueueTTS({ type: 'url', value: url, alertText, alertType });
}

function playTTSText(text, alertText = '') {
  enqueueTTS({ type: 'text', value: text, alertText });
}

// ── Alert display ─────────────────────────────────────────
let alertClearTimer = null;

function showAlert(text, type) {
  alertText.textContent = text;

  borderFlash.className = '';
  // force reflow for re-triggering animation
  void borderFlash.offsetWidth;

  if (type === 'danger') {
    borderFlash.className = 'danger';
  } else if (type === 'safe') {
    borderFlash.className = 'safe';
  }

  clearTimeout(alertClearTimer);
  alertClearTimer = setTimeout(() => {
    alertText.textContent = '';
    borderFlash.className = '';
  }, 5000);
}

// ── Overlay drawing ───────────────────────────────────────
const CLASS_COLORS = {
  'person':        '#ff9800',
  'bicycle':       '#8bc34a',
  'car':           '#f44336',
  'motorcycle':    '#e91e63',
  'bus':           '#9c27b0',
  'truck':         '#f44336',
  'traffic light': '#2196f3',
};

function drawOverlay(detections) {
  const scaleX = overlayCanvas.width  / (camera.videoWidth  || overlayCanvas.width);
  const scaleY = overlayCanvas.height / (camera.videoHeight || overlayCanvas.height);
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  for (const obj of detections) {
    const [x1, y1, x2, y2] = obj.bbox;
    const color = CLASS_COLORS[obj.class] || '#ffffff';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

    ctx.fillStyle = color;
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(
      `${obj.class} (${obj.distance})`,
      x1 * scaleX + 4,
      y1 * scaleY - 6,
    );
  }
}

// ── Touch / Long-press gestures ───────────────────────────
document.addEventListener('touchstart', (e) => {
  e.preventDefault();
  longPressTriggered = false;
  longPressTimer = setTimeout(() => {
    longPressTriggered = true;
    startVoiceQuestion();
  }, LONG_PRESS_MS);
}, { passive: false });

document.addEventListener('touchend', (e) => {
  e.preventDefault();
  clearTimeout(longPressTimer);
  if (!longPressTriggered) switchMode();
}, { passive: false });

document.addEventListener('touchmove', () => {
  clearTimeout(longPressTimer);
}, { passive: true });

// Desktop: 좌클릭 = 모드 전환, 우클릭 = 음성 질문
document.addEventListener('click', () => switchMode());

document.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  startVoiceQuestion();
});

// ── Init ──────────────────────────────────────────────────
(async () => {
  updateModeIndicator();
  showMicIndicator(false);
  await initCamera();

  camera.addEventListener('canplay', () => {
    if ('speechSynthesis' in window) {
      const utt = new SpeechSynthesisUtterance('길벗입니다. 화면을 터치해 모드를 선택하세요');
      utt.lang = 'ko-KR';
      speechSynthesis.speak(utt);
    }
  }, { once: true });
})();

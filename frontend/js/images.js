// ============================================
// images.js — Image upload & OCR fake-news scan
// ============================================

const API_BASE        = 'http://localhost:5000';

const uploadZone      = document.getElementById('uploadZone');
const fileInput       = document.getElementById('fileInput');
const imageResult     = document.getElementById('imageResult');
const imgLoading      = document.getElementById('imgLoading');
const imgContent      = document.getElementById('imgContent');
const imagePreview    = document.getElementById('imagePreview');
const imgResultLabel  = document.getElementById('imgResultLabel');
const imgResultDetail = document.getElementById('imgResultDetail');
const ocrSection      = document.getElementById('ocrSection');
const ocrText         = document.getElementById('ocrText');
const metricAuth      = document.getElementById('metricAuth');
const metricMeta      = document.getElementById('metricMeta');
const metricCompress  = document.getElementById('metricCompress');

// ── Drag-and-drop ──
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    processImage(file);
  }
});

// ── File input change ──
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) processImage(file);
});

// ── Main entry: preview + send to backend ──
function processImage(file) {
  // Preview the image immediately
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
  };
  reader.readAsDataURL(file);

  // Show result card in loading state
  imageResult.style.display = 'block';
  imgLoading.style.display  = 'flex';
  imgContent.style.display  = 'none';
  imageResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Send to backend
  sendImageToBackend(file);
}

// ── Real backend call ──
async function sendImageToBackend(file) {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const res  = await fetch(`${API_BASE}/predict-image`, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      if (res.status === 503) {
        showError(
          '🔧 Tesseract OCR is not installed on the server.\n\n' +
          'Install it from: https://github.com/UB-Mannheim/tesseract/wiki\n' +
          'Then restart the Flask server.'
        );
      } else {
        showError(data.error || 'An unexpected error occurred.');
      }
      return;
    }

    showResult(data);

  } catch (err) {
    showError('Could not connect to the backend. Make sure the Flask server is running on port 5000.');
  }
}

// ── Render a successful result ──
function showResult(data) {
  const isFake       = data.prediction === 'Fake News';
  const confidence   = data.confidence ? data.confidence.toFixed(1) : '—';
  const extractedTxt = data.extracted_text || '';
  const charCount    = data.ocr_char_count || extractedTxt.length;

  // Verdict banner
  imgResultLabel.textContent = isFake
    ? '🚨 Fake News Detected'
    : '✅ Appears to be Real News';
  imgResultLabel.style.color = isFake ? 'var(--pink)' : '#4ade80';

  imgResultDetail.textContent = isFake
    ? `Our ML model classified the text extracted from this image as likely fabricated or misleading (${confidence}% confidence). Always cross-check with trusted sources.`
    : `Our ML model found no strong indicators of fake news in the text extracted from this image (${confidence}% confidence). Still, verify with trusted sources.`;

  // OCR text panel
  if (extractedTxt) {
    ocrText.textContent  = extractedTxt;
    ocrSection.style.display = 'block';
  } else {
    ocrSection.style.display = 'none';
  }

  // Metrics
  metricAuth.textContent       = confidence + '%';
  metricAuth.style.color       = isFake ? 'var(--pink)' : '#4ade80';

  metricMeta.textContent       = isFake ? 'FAKE' : 'REAL';
  metricMeta.style.color       = isFake ? 'var(--pink)' : '#4ade80';

  metricCompress.textContent   = charCount;
  metricCompress.style.color   = 'var(--text)';

  // Swap loading → content
  imgLoading.style.display = 'none';
  imgContent.style.display = 'block';
}

// ── Render an error state ──
function showError(message) {
  imgResultLabel.textContent   = '⚠️ Analysis Failed';
  imgResultLabel.style.color   = '#fbbf24';
  imgResultDetail.textContent  = message;
  ocrSection.style.display     = 'none';
  metricAuth.textContent       = '—';
  metricMeta.textContent       = '—';
  metricCompress.textContent   = '—';

  imgLoading.style.display = 'none';
  imgContent.style.display = 'block';
}

console.log('images.js loaded — OCR fake-news mode');

const input = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
const feedbackControls = document.getElementById('feedbackControls');
const correctHealthy = document.getElementById('correctHealthy');
const correctUnhealthy = document.getElementById('correctUnhealthy');
const feedbackMsg = document.getElementById('feedbackMsg');

input.addEventListener('change', () => {
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  if (feedbackControls) { feedbackControls.style.display = 'none'; feedbackMsg.textContent = ''; }
});

uploadBtn.addEventListener('click', async () => {
  const file = input.files[0];
  if (!file) { resultDiv.textContent = 'Choose an image first.'; return; }
  const fd = new FormData();
  fd.append('image', file);
  resultDiv.textContent = 'Analyzing...';
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      resultDiv.textContent = data.error || 'Prediction error';
      return;
    }
    resultDiv.textContent = `${data.label} (confidence: ${(data.confidence*100).toFixed(1)}%)`;
    if (feedbackControls) {
      feedbackControls.style.display = 'block';
      feedbackMsg.textContent = '';
    }
  } catch (e) {
    resultDiv.textContent = 'Request failed: ' + e.message;
  }
});

async function sendFeedback(label) {
  const file = input.files[0];
  if (!file) { feedbackMsg.textContent = 'No file to send as feedback'; return; }
  const fd = new FormData();
  fd.append('image', file);
  fd.append('label', label);
  feedbackMsg.style.color = 'black';
  feedbackMsg.textContent = 'Sending feedback...';
  try {
    const res = await fetch('/feedback', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.ok) {
      feedbackMsg.style.color = 'green';
      feedbackMsg.textContent = 'Saved for retraining';
    } else {
      feedbackMsg.style.color = 'red';
      feedbackMsg.textContent = data.error || 'Feedback failed';
    }
  } catch (e) {
    feedbackMsg.style.color = 'red';
    feedbackMsg.textContent = 'Feedback request failed: ' + e.message;
  }
}

if (correctHealthy) correctHealthy.addEventListener('click', () => sendFeedback('healthy'));
if (correctUnhealthy) correctUnhealthy.addEventListener('click', () => sendFeedback('unhealthy'));
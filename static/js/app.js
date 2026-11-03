// IRRI AI — shared UI helpers (Django edition; auth handled server-side)

// ----- Modal open/close -----
function openModal(id) {
  var el = document.getElementById(id);
  if (el) el.classList.add('show');
}
function closeModal(id) {
  var el = document.getElementById(id);
  if (el) el.classList.remove('show');
}

document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.show').forEach(function (m) { m.classList.remove('show'); });
  }
});

document.addEventListener('click', function (e) {
  if (e.target.classList && e.target.classList.contains('modal-overlay')) {
    e.target.classList.remove('show');
  }
});

// ----- Waveform renderer (query audio visual) -----
function renderWaveform(container) {
  if (!container) return;
  container.innerHTML = '';
  for (var i = 0; i < 50; i++) {
    var bar = document.createElement('div');
    bar.className = 'wave-bar';
    bar.style.height = (Math.random() * 80 + 20) + '%';
    container.appendChild(bar);
  }
}

// ----- Toast notifications -----
function showToast(message, type) {
  type = type || 'success';
  var t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(function () { t.remove(); }, 3000);
}

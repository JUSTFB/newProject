/* ============================================================
   DubAI Studio — script.js (Redesigned & Fixed)
   ============================================================ */

/* ── DOM ELEMENTS (Mapped to index.html) ── */
const dz = document.getElementById('dz');
const vi = document.getElementById('vi');
const fc = document.getElementById('fc');
const fcN = document.getElementById('fcN');
const fcS = document.getElementById('fcS');

const startBtn = document.getElementById('startBtn');
const progW = document.getElementById('progW');
const barFg = document.getElementById('barFg');
const progP = document.getElementById('progP');
const progS = document.getElementById('progS');

const stLbl = document.getElementById('stLbl');
const sdot = document.getElementById('sdot');
const bigN = document.getElementById('bigN');
const stageTxt = document.getElementById('stageTxt');

const resPnl = document.getElementById('resPnl');
const resVid = document.getElementById('resVid');
const vidPh = document.getElementById('vidPh');
const dlBtn = document.getElementById('dlBtn');
const dlSrtBtn = document.getElementById('dlSrtBtn');

const stLang = document.getElementById('stLang');
const stEng = document.getElementById('stEng');
const stTime = document.getElementById('stTime');

const pips = [
  document.getElementById('ps-extract'),
  document.getElementById('ps-separate'),
  document.getElementById('ps-transcribe'),
  document.getElementById('ps-translate'),
  document.getElementById('ps-tts'),
  document.getElementById('ps-mix'),
  document.getElementById('ps-encode'),
  document.getElementById('ps-done')
];

/* ── STATE ── */
let currentFile = null;
let dubbingStarted = false;
let pollingInterval = null;
let startTime = 0;
let isPreview = false;

/* ── THEME TOGGLE ── */
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('dubai_theme', theme);
  if (theme === 'dark') {
    document.getElementById('btnDark').classList.add('active');
    document.getElementById('btnLight').classList.remove('active');
  } else {
    document.getElementById('btnLight').classList.add('active');
    document.getElementById('btnDark').classList.remove('active');
  }
}
const savedTheme = localStorage.getItem('dubai_theme') || 'dark'; // Defaulting to dark for editorial
setTheme(savedTheme);

/* ── ADVANCED TOGGLE ── */
function toggleAdv() {
  const body = document.getElementById('advBody');
  const arr = document.getElementById('advArr');
  const isOpen = body.classList.contains('open');
  if (isOpen) {
    body.classList.remove('open');
    arr.classList.remove('open');
  } else {
    body.classList.add('open');
    arr.classList.add('open');
  }
}

/* ── PREVIEW TOGGLE ── */
function togglePrev() {
  const sw = document.getElementById('prevSw');
  sw.classList.toggle('on');
  isPreview = sw.classList.contains('on');
}

/* ── ENGINE SELECTION ── */
function pickEng(btn) {
  const eng = btn.getAttribute('data-eng');
  
  // Reset tabs
  document.querySelectorAll('.eng-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  
  // Manage descriptions
  document.querySelectorAll('.eng-detail').forEach(d => d.classList.add('hidden'));
  document.getElementById('ed-' + eng).classList.remove('hidden');
}

/* ── FILE HANDLING ── */
if (dz) {
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
}

if (vi) {
  vi.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  if (!file.type.startsWith('video/') && !file.type.startsWith('audio/')) {
    showToast('Invalid file type.');
    return;
  }
  currentFile = file;
  
  // Hide drop zone text, show file chip
  Array.from(dz.children).forEach(c => {
    if (c.tagName !== 'INPUT') c.style.display = 'none';
  });
  
  fc.classList.add('show');
  startBtn.disabled = false;
  
  fcN.textContent = file.name;
  fcS.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
}

function clearFile(e) {
  if(e) e.stopPropagation();
  currentFile = null;
  vi.value = '';
  
  Array.from(dz.children).forEach(c => {
    if (c.tagName !== 'INPUT') c.style.display = 'block';
  });
  
  fc.classList.remove('show');
  startBtn.disabled = true;
}

/* ── PIPELINE UPDATES ── */
function updatePipeline(stepIndex) {
  pips.forEach((p, i) => {
    if (!p) return;
    p.classList.remove('active', 'done');
    if (i < stepIndex) p.classList.add('done');
    else if (i === stepIndex) p.classList.add('active');
  });
}

function resetPipeline() {
  pips.forEach(p => { if(p) p.classList.remove('active', 'done'); });
}

/* ── MAIN LOGIC ── */
async function startDubbingReal() {
  if (!currentFile || dubbingStarted) {
    if (!currentFile) showToast("Please upload a video first.");
    return;
  }
  
  dubbingStarted = true;
  startBtn.disabled = true;
  startBtn.innerHTML = 'Processing...';
  
  // Show progress areas
  progW.classList.add('show');
  resPnl.classList.remove('show');
  
  resetPipeline();
  sdot.classList.add('running');
  sdot.classList.remove('done');
  stLbl.textContent = 'SYSTEM RUNNING';
  stLbl.style.color = 'var(--primary)';
  bigN.innerHTML = '0<span>%</span>';
  stageTxt.textContent = 'Uploading file...';
  barFg.style.width = '0%';
  progP.textContent = '0%';
  progS.textContent = 'Uploading...';
  
  startTime = Date.now();
  
  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('lang', document.getElementById('langSel').value);
  formData.append('preview', isPreview.toString());

  const activeEng = document.querySelector('.eng-tab.active');
  const ttsEngine = activeEng ? activeEng.getAttribute('data-eng') : 'xtts';
  formData.append('voice_engine', ttsEngine);

  try {
    updatePipeline(0); 
    let res;
    try {
      res = await fetch('http://localhost:5000/api/process', { method: 'POST', body: formData });
    } catch (networkErr) {
      throw new Error('Cannot connect to server. Make sure the backend is running on port 5000.');
    }
    
    let data;
    try {
      data = await res.json();
    } catch (parseErr) {
      const text = await res.text().catch(() => '');
      console.error('Non-JSON response:', res.status, text.substring(0, 200));
      throw new Error(`Server error (${res.status}). Check backend logs.`);
    }
    
    if (!data.success) throw new Error(data.message || "Failed to start");
    
    showToast("Upload success. Engine starting...");
    pollStatus(data.jobId);
    
  } catch (err) {
    showToast('Error: ' + err.message);
    showErrorState(err.message);
  }
}

/* ── POLLING OVERhaul ── */
function pollStatus(jobId) {
  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`http://localhost:5000/api/status/${jobId}`);
      if (!res.ok) throw new Error('Status fetch failed');
      const data = await res.json();
      
      if (!data.success) throw new Error(data.message || "Status endpoint error");
      
      handleStatusData(data);
      
      // Stop polling on completion
      if (data.status === 'completed' || data.status === 'error' || data.status === "preview_ready") {
        clearInterval(pollingInterval);
        
        if (data.status === 'completed' || data.status === "preview_ready") {
          fetchResult(jobId, data.status === "preview_ready" ? "preview" : "full");
        } else {
          showErrorState(data.stage || "An error occurred");
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, 1500); 
}

function handleStatusData(data) {
  const p = Math.round(data.progress || 0);
  
  barFg.style.width = p + '%';
  progP.textContent = p + '%';
  bigN.innerHTML = p + '<span>%</span>';
  
  let stageMsg = data.stage || 'Processing...';
  progS.textContent = stageMsg;
  stageTxt.textContent = stageMsg;
  
  // Pipeline Mapping
  if (p >= 100) updatePipeline(7);      // Done
  else if (p >= 85) updatePipeline(6);  // Encode
  else if (p >= 65) updatePipeline(5);  // Mix
  else if (p >= 45) updatePipeline(4);  // TTS
  else if (p >= 30) updatePipeline(3);  // Translate
  else if (p >= 20) updatePipeline(2);  // Transcribe
  else if (p >= 10) updatePipeline(1);  // Separate
  else updatePipeline(0);               // Extract
}

async function fetchResult(jobId, type) {
  try {
    const res = await fetch(`http://localhost:5000/api/result/${jobId}?type=${type}`);
    const data = await res.json();
    if (data.success) {
      showResult(data);
    } else {
      showErrorState(data.message);
    }
  } catch(e) {
    showErrorState("Could not load result video.");
  }
}

function showResult(data) {
  sdot.classList.remove('running');
  sdot.classList.add('done');
  stLbl.textContent = 'SYSTEM IDLE';
  stLbl.style.color = '';
  stageTxt.textContent = 'Job completed successfully.';
  
  updatePipeline(7); 
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  stTime.textContent = elapsed + 's';
  
  const langDrop = document.getElementById('langSel');
  stLang.textContent = langDrop.options[langDrop.selectedIndex].text.split(' ')[1];
  
  const activeEng = document.querySelector('.eng-tab.active');
  stEng.textContent = activeEng ? activeEng.querySelector('.eng-tab-name').textContent : "Edge TTS";
  
  resPnl.classList.add('show');
  
  const finalVidUrl = data.finalVideo || data.playlistUrl;
  
  vidPh.style.display = 'none';
  resVid.style.display = 'block';
  resVid.src = "http://localhost:5000/" + finalVidUrl;
  
  dlBtn.href = "http://localhost:5000/" + finalVidUrl;
  
  if (dlSrtBtn && data.srtFile) {
      dlSrtBtn.style.display = 'block';
      dlSrtBtn.href = "http://localhost:5000/" + data.srtFile;
  } else if (dlSrtBtn) {
      dlSrtBtn.style.display = 'none';
  }
  
  dubbingStarted = false;
  startBtn.innerHTML = '🚀 Start Dubbing';
  startBtn.disabled = false;
}

function showErrorState(msg) {
  dubbingStarted = false;
  startBtn.disabled = false;
  startBtn.innerHTML = '🚀 Start Dubbing';
  sdot.classList.remove('running');
  stLbl.style.color = 'var(--red)';
  stLbl.textContent = 'FAILED';
  stageTxt.textContent = msg;
}

function resetApp() {
  dubbingStarted = false;
  startBtn.disabled = false;
  startBtn.innerHTML = '🚀 Start Dubbing';
  progW.classList.remove('show');
  resPnl.classList.remove('show');
  clearInterval(pollingInterval);
  sdot.classList.remove('running', 'done');
  stLbl.style.color = '';
  stLbl.textContent = 'SYSTEM IDLE';
  bigN.innerHTML = '0<span>%</span>';
  stageTxt.textContent = 'Awaiting input...';
  
  resVid.src = "";
  vidPh.style.display = 'flex';
  resVid.style.display = 'none';
  
  resetPipeline();
  clearFile();
}

/* ── MODALS ── */
function openCompare() {
  document.getElementById('cmpBg').classList.add('show');
}
function closeCompare() {
  document.getElementById('cmpBg').classList.remove('show');
}
function closeCmpBg(e) {
  if (e.target.id === 'cmpBg') closeCompare();
}

/* ── TOAST ── */
function showToast(msg) {
  const zone = document.getElementById('tz');
  if(!zone) return;
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  zone.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateY(10px)';
    t.style.transition = 'all 0.3s ease';
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

from flask import Flask, request, jsonify, render_template_string
import torch, torch.nn as nn, numpy as np, librosa, timm, os, tempfile

app = Flask(__name__)

# ─── Config ────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION    = 4
MAX_LEN     = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "best_model.pth")

# ─── Model ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=False,
                                      num_classes=0, in_chans=3)
        cnn_out  = self.cnn.num_features
        self.lstm = nn.LSTM(cnn_out, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, x):
        feat = self.cnn(x).unsqueeze(1)
        out, _ = self.lstm(feat)
        return self.classifier(out.squeeze(1))

print("Loading model...")
model = DeepfakeDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model loaded on {DEVICE} ✅")

# ─── Feature Extraction ────────────────────────────────
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target = SAMPLE_RATE * DURATION
    y = np.pad(y, (0, max(0, target - len(y))))[:target]

    def resize(x):
        if x.shape[0] > 128: x = x[:128, :]
        elif x.shape[0] < 128: x = np.pad(x, ((0, 128 - x.shape[0]), (0, 0)))
        if x.shape[1] > MAX_LEN: x = x[:, :MAX_LEN]
        elif x.shape[1] < MAX_LEN: x = np.pad(x, ((0, 0), (0, MAX_LEN - x.shape[1])))
        return x

    mel  = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    cqt  = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
    return np.stack([resize(mel), resize(mfcc), resize(cqt)]).astype(np.float32)

# ─── Predict Endpoint ──────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file     = request.files["audio"]
    tmp_path = os.path.join(tempfile.gettempdir(), "deepfake_input.wav")
    file.save(tmp_path)

    try:
        features = extract_features(tmp_path)
        features = (features - features.mean()) / (features.std() + 1e-6)
        tensor   = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
        return jsonify({
            "real_prob": round(probs[0].item() * 100, 1),
            "fake_prob": round(probs[1].item() * 100, 1),
            "verdict"  : "FAKE" if probs[1].item() > 0.5 else "REAL"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─── HTML UI ───────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Audio Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0a0a0f; --surface:#13131a; --border:#1e1e2e;
    --accent:#00ff9d; --danger:#ff3c5f; --text:#e8e8f0; --muted:#555570; --card:#0f0f1a;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;
    display:flex;flex-direction:column;align-items:center;padding:40px 20px;
    background-image:radial-gradient(ellipse at 20% 20%,#00ff9d08 0%,transparent 60%),
                     radial-gradient(ellipse at 80% 80%,#ff3c5f08 0%,transparent 60%);}
  header{text-align:center;margin-bottom:48px;}
  .badge{display:inline-block;font-family:'Space Mono',monospace;font-size:11px;
    letter-spacing:3px;color:var(--accent);border:1px solid var(--accent);
    padding:4px 12px;border-radius:2px;margin-bottom:16px;text-transform:uppercase;}
  h1{font-size:clamp(28px,5vw,52px);font-weight:800;letter-spacing:-1px;line-height:1.1;
    background:linear-gradient(135deg,#fff 0%,#888 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .subtitle{margin-top:12px;color:var(--muted);font-size:15px;font-family:'Space Mono',monospace;}
  .container{width:100%;max-width:700px;display:flex;flex-direction:column;gap:20px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:16px;
    padding:28px;position:relative;overflow:hidden;}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#00ff9d30,transparent);}
  .card-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;
    color:var(--muted);text-transform:uppercase;margin-bottom:20px;}
  .tabs{display:flex;gap:8px;margin-bottom:24px;}
  .tab{flex:1;padding:10px;border:1px solid var(--border);border-radius:8px;
    background:transparent;color:var(--muted);font-family:'Space Mono',monospace;
    font-size:12px;cursor:pointer;transition:all .2s;letter-spacing:1px;}
  .tab.active{border-color:var(--accent);color:var(--accent);background:#00ff9d08;}
  .tab-panel{display:none;} .tab-panel.active{display:block;}

  /* Upload */
  .upload-zone{border:2px dashed var(--border);border-radius:12px;padding:32px;
    text-align:center;cursor:pointer;transition:all .2s;}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);background:#00ff9d08;}
  .upload-zone input{display:none;}
  .upload-icon{font-size:36px;margin-bottom:12px;}
  .upload-text{color:var(--muted);font-size:14px;font-family:'Space Mono',monospace;}
  .upload-text span{color:var(--accent);}
  .audio-preview{display:none;margin-top:16px;flex-direction:column;gap:12px;}
  .audio-preview.show{display:flex;}
  .file-info{display:flex;align-items:center;gap:10px;background:var(--surface);
    border-radius:8px;padding:10px 14px;font-family:'Space Mono',monospace;
    font-size:12px;color:var(--accent);}
  audio{width:100%;border-radius:8px;outline:none;margin-top:4px;}

  /* Record */
  .rec-controls{display:flex;flex-direction:column;align-items:center;gap:16px;}
  canvas#waveform{width:100%;height:56px;border-radius:10px;background:var(--surface);display:block;}
  .rec-timer{font-family:'Space Mono',monospace;font-size:32px;color:var(--danger);letter-spacing:4px;}
  .rec-btn{width:80px;height:80px;border-radius:50%;border:3px solid var(--danger);
    background:transparent;cursor:pointer;display:flex;align-items:center;
    justify-content:center;transition:all .2s;}
  .rec-btn:hover{background:#ff3c5f15;transform:scale(1.05);}
  .rec-btn .dot{width:28px;height:28px;background:var(--danger);border-radius:50%;transition:all .3s;}
  .rec-btn.recording .dot{border-radius:6px;width:22px;height:22px;}
  .rec-btn.recording{animation:pulse 1.5s infinite;}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 #ff3c5f40;}50%{box-shadow:0 0 0 14px #ff3c5f00;}}
  .rec-hint{font-family:'Space Mono',monospace;font-size:12px;color:var(--muted);text-align:center;min-height:20px;}
  .rec-hint.active{color:var(--danger);} .rec-hint.done{color:var(--accent);}
  .recorded-preview{display:none;flex-direction:column;gap:10px;width:100%;margin-top:16px;}
  .recorded-preview.show{display:flex;}
  .preview-label{font-family:'Space Mono',monospace;font-size:11px;color:var(--accent);letter-spacing:2px;}

  /* Submit */
  .submit-btn{width:100%;padding:18px;border:none;border-radius:12px;
    background:linear-gradient(135deg,var(--accent),#00cc7d);color:#000;
    font-family:'Syne',sans-serif;font-weight:700;font-size:16px;
    letter-spacing:1px;cursor:pointer;transition:all .2s;text-transform:uppercase;}
  .submit-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 24px #00ff9d30;}
  .submit-btn:disabled{opacity:.35;cursor:not-allowed;}

  /* Loading */
  .loading{display:none;text-align:center;padding:28px;font-family:'Space Mono',monospace;font-size:13px;color:var(--muted);}
  .loading.show{display:block;}
  .spinner{width:32px;height:32px;border:2px solid var(--border);border-top-color:var(--accent);
    border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 14px;}
  @keyframes spin{to{transform:rotate(360deg);}}

  /* Result */
  .result-card{display:none;} .result-card.show{display:block;}
  .verdict{font-size:clamp(30px,6vw,54px);font-weight:800;letter-spacing:-1px;margin-bottom:6px;}
  .verdict.real{color:var(--accent);} .verdict.fake{color:var(--danger);}
  .verdict-sub{color:var(--muted);font-family:'Space Mono',monospace;font-size:13px;margin-bottom:24px;}
  .prob-row{display:flex;gap:16px;margin-bottom:20px;}
  .prob-box{flex:1;background:var(--surface);border-radius:12px;padding:18px;text-align:center;}
  .prob-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;color:var(--muted);margin-bottom:8px;}
  .prob-value{font-size:30px;font-weight:800;font-family:'Space Mono',monospace;}
  .prob-value.real{color:var(--accent);} .prob-value.fake{color:var(--danger);}
  .bar-wrap{height:8px;background:var(--surface);border-radius:99px;overflow:hidden;}
  .bar{height:100%;border-radius:99px;transition:width 1.2s cubic-bezier(.4,0,.2,1);}
  .bar.real{background:linear-gradient(90deg,#00ff9d,#00cc7d);}
  .bar.fake{background:linear-gradient(90deg,#ff3c5f,#cc2244);}
  .reset-btn{margin-top:16px;background:transparent;border:1px solid var(--border);
    color:var(--muted);padding:10px 20px;border-radius:8px;font-family:'Space Mono',monospace;
    font-size:12px;cursor:pointer;transition:all .2s;}
  .reset-btn:hover{border-color:var(--text);color:var(--text);}
  .error-msg{color:var(--danger);font-family:'Space Mono',monospace;font-size:13px;margin-top:12px;display:none;}
  .error-msg.show{display:block;}
</style>
</head>
<body>
<header>
  <div class="badge">AI Security Tool</div>
  <h1>Deepfake Audio<br>Detector</h1>
  <p class="subtitle">EfficientNet-B0 + BiLSTM · ASVspoof 2019</p>
</header>

<div class="container">
  <div class="card">
    <div class="card-label">01 · Input Audio</div>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('upload',this)">⬆ Upload File</button>
      <button class="tab" onclick="switchTab('record',this)">⏺ Record Voice</button>
    </div>

    <div class="tab-panel active" id="panel-upload">
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="audio/*" onchange="handleFileSelect(event)">
        <div class="upload-icon">🎵</div>
        <div class="upload-text">Click or drag & drop audio file<br><span>MP3, WAV, FLAC, OGG supported</span></div>
      </div>
      <div class="audio-preview" id="uploadPreview">
        <div class="file-info" id="fileName">📄 No file selected</div>
        <audio id="uploadAudio" controls></audio>
      </div>
    </div>

    <div class="tab-panel" id="panel-record">
      <div class="rec-controls">
        <canvas id="waveform" width="640" height="56"></canvas>
        <div class="rec-timer" id="timer">00:00</div>
        <button class="rec-btn" id="recBtn" onclick="toggleRecording()">
          <div class="dot"></div>
        </button>
        <div class="rec-hint" id="recHint">Click to start recording</div>
      </div>
      <div class="recorded-preview" id="recordedPreview">
        <div class="preview-label">▶ PLAYBACK — Listen before submitting</div>
        <audio id="recordedAudio" controls></audio>
      </div>
    </div>

    <div class="error-msg" id="errorMsg"></div>
  </div>

  <button class="submit-btn" id="submitBtn" onclick="submitAudio()" disabled>
    🔍 Analyze Audio
  </button>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    Extracting features & running model...
  </div>

  <div class="card result-card" id="resultCard">
    <div class="card-label">02 · Analysis Result</div>
    <div class="verdict" id="verdictText"></div>
    <div class="verdict-sub" id="verdictSub"></div>
    <div class="prob-row">
      <div class="prob-box">
        <div class="prob-label">REAL SCORE</div>
        <div class="prob-value real" id="realProb">—</div>
      </div>
      <div class="prob-box">
        <div class="prob-label">FAKE SCORE</div>
        <div class="prob-value fake" id="fakeProb">—</div>
      </div>
    </div>
    <div class="bar-wrap"><div class="bar" id="probBar" style="width:0%"></div></div>
    <button class="reset-btn" onclick="resetAll()">↩ Analyze Another</button>
  </div>
</div>

<script>
let currentBlob=null,currentSource=null,mediaRecorder=null,audioChunks=[],
    isRecording=false,timerInterval=null,seconds=0,analyserNode=null,animFrame=null;

function switchTab(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-'+tab).classList.add('active');
  currentBlob=null; currentSource=null;
  document.getElementById('submitBtn').disabled=true;
  hideError();
}

function handleFileSelect(e){
  const file=e.target.files[0]; if(!file)return;
  currentBlob=file; currentSource='upload';
  document.getElementById('uploadAudio').src=URL.createObjectURL(file);
  document.getElementById('fileName').textContent='📄 '+file.name;
  document.getElementById('uploadPreview').classList.add('show');
  document.getElementById('submitBtn').disabled=false; hideError();
}

const zone=document.getElementById('uploadZone');
zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('drag-over');});
zone.addEventListener('dragleave',()=>zone.classList.remove('drag-over'));
zone.addEventListener('drop',e=>{
  e.preventDefault();zone.classList.remove('drag-over');
  const f=e.dataTransfer.files[0];
  if(f) handleFileSelect({target:{files:[f]}});
});

async function toggleRecording(){if(!isRecording)await startRecording();else stopRecording();}

async function startRecording(){
  try{
    hideError();
    const stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1}});
    audioChunks=[];
    const actx=new AudioContext();
    const src=actx.createMediaStreamSource(stream);
    analyserNode=actx.createAnalyser(); analyserNode.fftSize=512;
    src.connect(analyserNode); drawWaveform();

    mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>{if(e.data.size>0)audioChunks.push(e.data);};
    mediaRecorder.onstop=async()=>{
      stream.getTracks().forEach(t=>t.stop());
      cancelAnimationFrame(animFrame); clearWaveform();
      const raw=new Blob(audioChunks,{type:'audio/webm'});
      try{
        const wav=await convertToWav(raw);
        currentBlob=wav; currentSource='record';
        const audio=document.getElementById('recordedAudio');
        audio.src=URL.createObjectURL(wav);
        document.getElementById('recordedPreview').classList.add('show');
        audio.play().catch(()=>{});
        setHint('✅ Done! Listen above then click Analyze','done');
        document.getElementById('submitBtn').disabled=false;
      }catch(err){showError('Conversion failed: '+err.message);}
    };
    mediaRecorder.start(100); isRecording=true; seconds=0;
    document.getElementById('recBtn').classList.add('recording');
    document.getElementById('recordedPreview').classList.remove('show');
    document.getElementById('submitBtn').disabled=true;
    setHint('🔴 Recording... click square to stop (max 10s)','active');
    timerInterval=setInterval(()=>{
      seconds++;
      document.getElementById('timer').textContent=
        String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0');
      if(seconds>=10)stopRecording();
    },1000);
  }catch(err){showError('Microphone access denied. Please allow mic and try again.');}
}

function stopRecording(){
  if(!mediaRecorder||!isRecording)return;
  clearInterval(timerInterval); isRecording=false;
  document.getElementById('recBtn').classList.remove('recording');
  setHint('Processing...','');
  mediaRecorder.stop();
}

async function convertToWav(blob){
  const ab=await blob.arrayBuffer();
  const actx=new AudioContext({sampleRate:16000});
  const buf=await actx.decodeAudioData(ab);
  await actx.close();
  const samples=buf.getChannelData(0);
  const wavBuf=new ArrayBuffer(44+samples.length*2);
  const v=new DataView(wavBuf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF'); v.setUint32(4,36+samples.length*2,true);
  ws(8,'WAVE'); ws(12,'fmt '); v.setUint32(16,16,true);
  v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,16000,true); v.setUint32(28,32000,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true);
  ws(36,'data'); v.setUint32(40,samples.length*2,true);
  let off=44;
  for(let i=0;i<samples.length;i++){
    const s=Math.max(-1,Math.min(1,samples[i]));
    v.setInt16(off,s<0?s*0x8000:s*0x7FFF,true); off+=2;
  }
  return new Blob([wavBuf],{type:'audio/wav'});
}

function drawWaveform(){
  const canvas=document.getElementById('waveform');
  const ctx=canvas.getContext('2d');
  const data=new Uint8Array(analyserNode.frequencyBinCount);
  function draw(){
    animFrame=requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(data);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='#00ff9d'; ctx.lineWidth=2;
    ctx.shadowColor='#00ff9d'; ctx.shadowBlur=4;
    ctx.beginPath();
    const sw=canvas.width/data.length; let x=0;
    for(let i=0;i<data.length;i++){
      const y=(data[i]/128.0*canvas.height)/2;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); x+=sw;
    }
    ctx.stroke();
  }
  draw();
}
function clearWaveform(){const c=document.getElementById('waveform');c.getContext('2d').clearRect(0,0,c.width,c.height);}

async function submitAudio(){
  if(!currentBlob)return;
  hideError();
  document.getElementById('submitBtn').disabled=true;
  document.getElementById('loading').classList.add('show');
  document.getElementById('resultCard').classList.remove('show');
  const fd=new FormData();
  fd.append('audio',currentBlob,currentSource==='record'?'recording.wav':(currentBlob.name||'audio.wav'));
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    const data=await res.json();
    document.getElementById('loading').classList.remove('show');
    if(data.error){showError('Error: '+data.error);document.getElementById('submitBtn').disabled=false;return;}
    const fake=data.verdict==='FAKE';
    document.getElementById('verdictText').textContent=fake?'🔴 FAKE AUDIO':'🟢 REAL AUDIO';
    document.getElementById('verdictText').className='verdict '+(fake?'fake':'real');
    document.getElementById('verdictSub').textContent=fake
      ?'AI-generated or voice-cloned audio detected'
      :'Authentic human voice detected';
    document.getElementById('realProb').textContent=data.real_prob+'%';
    document.getElementById('fakeProb').textContent=data.fake_prob+'%';
    const bar=document.getElementById('probBar');
    bar.className='bar '+(fake?'fake':'real'); bar.style.width='0%';
    setTimeout(()=>{bar.style.width=(fake?data.fake_prob:data.real_prob)+'%';},100);
    document.getElementById('resultCard').classList.add('show');
    document.getElementById('resultCard').scrollIntoView({behavior:'smooth'});
  }catch(err){
    document.getElementById('loading').classList.remove('show');
    showError('Request failed: '+err.message);
    document.getElementById('submitBtn').disabled=false;
  }
}

function resetAll(){
  currentBlob=null;currentSource=null;
  document.getElementById('resultCard').classList.remove('show');
  document.getElementById('uploadPreview').classList.remove('show');
  document.getElementById('recordedPreview').classList.remove('show');
  document.getElementById('fileInput').value='';
  document.getElementById('timer').textContent='00:00';
  document.getElementById('submitBtn').disabled=true;
  setHint('Click to start recording',''); hideError();
}
function setHint(m,c){const e=document.getElementById('recHint');e.textContent=m;e.className='rec-hint '+c;}
function showError(m){const e=document.getElementById('errorMsg');e.textContent=m;e.classList.add('show');}
function hideError(){document.getElementById('errorMsg').classList.remove('show');}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    print("\n🚀 Deepfake Audio Detector starting...")
    print("📡 Open → http://localhost:5000\n")
    app.run(debug=False, port=5000)
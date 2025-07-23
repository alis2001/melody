//--------------------------------------
// 1) Retrieve doctor matricola from ?matricola=XXXX
//--------------------------------------
const urlParams = new URLSearchParams(window.location.search);
const mat = urlParams.get("matricola") || "";

//--------------------------------------
// 2) Connect to Socket.IO with that matricola
//--------------------------------------
const socket = io({
  path: '/socket.io/',
  transports: ['polling', 'websocket'],
  auth: { matricola: mat }
}, {
  reconnectionAttempts: 999,
  reconnectionDelay: 2000
});

// Global logging of all events for debugging
socket.onAny((event, ...args) => {
  console.log(`Received event "${event}":`, args);
});

socket.on("connect", () => {
  console.log("Client connected, local socket.id =", socket.id);
  
  // Start session validation checks
  sessionCheckInterval = setInterval(() => {
    fetch(`/refertazione/check_session?matricola=${mat}`)
      .then(response => response.json())
      .then(data => {
        if (!data.valid && data.redirect) {
          // Stop all intervals immediately
          clearInterval(partialPollInterval);
          clearInterval(transcriptPollInterval);
          clearInterval(sessionCheckInterval);
          clearInterval(heartbeatInterval);
          
          alert("⚠️ Sessione scaduta o non valida. Verrai reindirizzato alla pagina di setup.");
          window.location.href = "/refertazione/doctor_setup";
        }
      })
      .catch(err => {
        console.error("Session check error:", err);
        // If check fails, assume session is invalid
        clearInterval(partialPollInterval);
        clearInterval(transcriptPollInterval);
        clearInterval(sessionCheckInterval);
        clearInterval(heartbeatInterval);
        window.location.href = "/refertazione/doctor_setup";
      });
  }, 30000); // Check every 30 seconds
  
  // Send heartbeat to keep session alive
  heartbeatInterval = setInterval(() => {
    socket.emit("heartbeat", { matricola: mat });
  }, 60000); // Every minute
});

socket.on("disconnect", (reason) => {
  console.log("Socket disconnected. Reason:", reason);
  clearInterval(sessionCheckInterval);
  clearInterval(heartbeatInterval);
});

socket.on("connect_error", (error) => {
  console.error("Connect error:", error);
});

//--------------------------------------
// HTML element references
//--------------------------------------
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
// Set initial visibility
recordBtn.classList.add("visible");
stopBtn.classList.remove("visible");

const finalDiv = document.getElementById("finalTranscriptions");
const partialDiv = document.getElementById("partialTranscription");

const reportBtn = document.getElementById("generateReportBtn");
const reportOutput = document.getElementById("reportOutput");
const loadingSpinner = document.getElementById("loadingSpinner");
const modifyReportBtn = document.getElementById("modifyReportBtn");
const reportTextarea = document.getElementById("reportTextarea");
const saveSessionButton = document.getElementById("saveSessionButton");

const saveInfoSection = document.getElementById("saveInfoSection");
const matricolaInput = document.getElementById("matricola");
const cfInput = document.getElementById("cf");

function renderCleanTranscript(rawText) {
  finalDiv.innerHTML = ""; // clear previous
  const lines = rawText.split("\n");

  for (let line of lines) {
    const match = line.match(/(?:[^:]+):\s*(Dottore|Paziente):\s*(.*)/i);
    if (!match) continue;

    const speaker = match[1];
    const sentence = match[2];

    const colorClass = speaker === "Dottore" ? "line-doctor" : "line-patient";
    finalDiv.insertAdjacentHTML(
      "beforeend",
      `<div class="line ${colorClass}"><span class="speaker">${speaker}:</span> ${sentence}</div>`
    );
  }
}

// Initially hide items
reportBtn.style.display = "none"; // Hide Generate Report button until recording stops
reportTextarea.style.display = "none";
modifyReportBtn.style.display = "none";
saveSessionButton.disabled = true;
saveSessionButton.style.display = "none";
saveInfoSection.style.display = "none";
reportOutput.style.display = "none";

// Toggle save button visibility based on input fields
function toggleSaveButtonVisibility() {
  const doctorVal = matricolaInput.value.trim();
  const cfVal = cfInput.value.trim();
  if (doctorVal && cfVal) {
    saveSessionButton.disabled = false;
    saveSessionButton.style.display = "block";
  } else {
    saveSessionButton.disabled = true;
  }
}
matricolaInput.addEventListener("input", toggleSaveButtonVisibility);
cfInput.addEventListener("input", toggleSaveButtonVisibility);

let sessionCheckInterval;
let heartbeatInterval;
let audioContext;
let mediaStream;
let processor;
let recording = false;
let pcmChunks = [];
let conversationLines = [];  // Updated from the transcript polling

// 🚨 ENHANCED: Track both original and current report versions
let originalAIReport = "";     // The original AI-generated report (never changes)
let currentReport = "";        // Current report (may be modified by doctor)
let reportWasModified = false; // Flag to track if doctor made changes

let autoScroll = false;
// Variables for AJAX polling intervals
let partialPollInterval, transcriptPollInterval;

function downsampleBuffer(buffer, inSampleRate, outSampleRate) {
  if (outSampleRate === inSampleRate) return buffer;
  if (outSampleRate > inSampleRate) throw "downsampling rate should be smaller";
  const sampleRateRatio = inSampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0, offsetBuffer = 0;
  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function floatTo16BitPCM(floatBuffer) {
  const buffer = new ArrayBuffer(floatBuffer.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < floatBuffer.length; i++) {
    let s = Math.max(-1, Math.min(1, floatBuffer[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}

//----------------------
// Start Recording
//----------------------
recordBtn.addEventListener("click", async () => {
  try {
    // Show loading state
    recordBtn.disabled = true;
    recordBtn.textContent = "Inizializzazione...";
    partialDiv.innerHTML = "🔄 Preparazione sistema di registrazione...";
    reportBtn.style.display = "none";
    
    // Show professional loading sequence
    const loadingMessages = [
      "🎤 Inizializzazione microfono...",
      "🔊 Configurazione audio...",
      "🧠 Caricamento modelli IA...",
      "✅ Sistema pronto per la registrazione!"
    ];
    
    for (let i = 0; i < loadingMessages.length; i++) {
      partialDiv.innerHTML = loadingMessages[i];
      await new Promise(resolve => setTimeout(resolve, 1250)); // 1.25 seconds each = 5 total
    }
    
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    console.log("AudioContext sample rate:", audioContext.sampleRate);
    const source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      let downsampled;
      try {
        downsampled = downsampleBuffer(inputData, audioContext.sampleRate, 16000);
      } catch (err) {
        console.error("Downsampling error:", err);
        return;
      }
      const pcmBuffer = floatTo16BitPCM(downsampled);
      pcmChunks.push(pcmBuffer);
    };
    
    recording = true;
    autoScroll = true;
    recordBtn.classList.remove("visible");
    recordBtn.disabled = false;
    recordBtn.textContent = "Inizia Registrazione";
    stopBtn.classList.add("visible");
    stopBtn.disabled = false;
    partialDiv.innerHTML = "🔴 Registrazione in corso...";
    
    window.sendInterval = setInterval(() => {
      if (pcmChunks.length > 0) {
        let totalLength = pcmChunks.reduce((acc, curr) => acc + curr.byteLength, 0);
        let combined = new Uint8Array(totalLength);
        let offset = 0;
        for (let chunk of pcmChunks) {
          combined.set(new Uint8Array(chunk), offset);
          offset += chunk.byteLength;
        }
        socket.emit("audio_chunk", combined.buffer);
        pcmChunks = [];
      }
    }, 250);
    
    partialPollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/refertazione/partial?matricola=${mat}`);
        const data = await response.json();
        partialDiv.innerHTML = `<strong>Partial:</strong> ${data.partial}`;
      } catch (err) {
        console.error("Partial poll error:", err);
      }
    }, 250);
    
    transcriptPollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/refertazione/transcript?matricola=${mat}`);
        const data = await response.json();
        conversationLines = [];
        renderCleanTranscript(data.transcript);

        if (autoScroll) {
          finalDiv.scrollTop = finalDiv.scrollHeight;
        }
      } catch (err) {
        console.error("Transcript poll error:", err);
      }
    }, 1000);
    
  } catch (err) {
    console.error("Error accessing microphone:", err);
    partialDiv.innerHTML = "Could not access microphone.";
  }
});

//----------------------
// Stop Recording
//----------------------
stopBtn.addEventListener("click", () => {
  recording = false;
  autoScroll = false;
  if (processor) processor.disconnect();
  if (audioContext) audioContext.close();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  stopBtn.classList.remove("visible");
  recordBtn.classList.add("visible");
  
  clearInterval(window.sendInterval);
  clearInterval(partialPollInterval);
  clearInterval(transcriptPollInterval);
  
  if (pcmChunks.length > 0) {
    let totalLength = pcmChunks.reduce((acc, curr) => acc + curr.byteLength, 0);
    let combined = new Uint8Array(totalLength);
    let offset = 0;
    for (let chunk of pcmChunks) {
      combined.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }
    socket.emit("audio_chunk", combined.buffer);
    pcmChunks = [];
  }
  socket.emit("stop_recording");
  partialDiv.innerHTML += "<br>Recording stopped.";
  
  reportBtn.style.display = "block";
});
  
//=================================================
// 🚨 ENHANCED: Generate Medical Report - Track Original Version
//=================================================
reportBtn.addEventListener("click", async () => {
  console.log("Clicked Generate Medical Report");
  loadingSpinner.style.display = "block";
  reportOutput.style.display = "none";
  
  try {
    // Call the new endpoint to generate the report.
    const response = await fetch(`/refertazione/generate_report?matricola=${mat}`);
    const data = await response.json();
    
    if (data.error) {
      reportOutput.innerHTML = `<span style="color:red;">Error generating report: ${data.error}</span>`;
      reportOutput.style.display = "block";
      loadingSpinner.style.display = "none";
      return;
    }
    
    // 🚨 ENHANCED: Store both original and current versions
    originalAIReport = data.report || "";        // Save original AI report (never changes)
    currentReport = data.report || "";           // Current report (may be modified)
    reportWasModified = false;                   // Reset modification flag
    
    console.log("Report received:", currentReport);
    console.log("Original AI report saved for comparison:", originalAIReport);
    
    const formatted = currentReport.replace(/\n/g, "<br>");
    reportOutput.innerHTML = `<div class="report-content" style="background-color:#ffffcc; border:2px solid red; padding:10px;">${formatted}</div>`;
    reportOutput.style.display = "block";
    
    saveInfoSection.style.display = "block";
    modifyReportBtn.style.display = "block";
  } catch (err) {
    console.error("Error generating report:", err);
    reportOutput.innerHTML = `<span style="color:red;">Error generating report.</span>`;
    reportOutput.style.display = "block";
  } finally {
    loadingSpinner.style.display = "none";
  }
});

socket.on("report_generated", (data) => {
  console.log("report_generated event received:", data);
  loadingSpinner.style.display = "none";
  
  if (data.error) {
    reportOutput.innerHTML = `<span style="color:red;">Error generating report: ${data.error}</span>`;
    reportOutput.style.display = "block";
    return;
  }
  
  // 🚨 ENHANCED: Store both original and current versions
  originalAIReport = data.report || "";        // Save original AI report (never changes)
  currentReport = data.report || "";           // Current report (may be modified)
  reportWasModified = false;                   // Reset modification flag
  
  console.log("Report received:", currentReport);
  console.log("Original AI report saved for comparison:", originalAIReport);
  
  const formatted = currentReport.replace(/\n/g, "<br>");
  reportOutput.innerHTML = `<div class="report-content" style="background-color:#ffffcc; border:2px solid red; padding:10px;">${formatted}</div>`;
  reportOutput.style.display = "block";
  
  saveInfoSection.style.display = "block";
  modifyReportBtn.style.display = "block";
});
  
//=================================================
// 🚨 ENHANCED: Modify Report - Track Changes
//=================================================
modifyReportBtn.addEventListener("click", () => {
  if (reportTextarea.style.display === "none") {
    // User is starting to modify
    reportOutput.style.display = "none";
    reportTextarea.style.display = "block";
    reportTextarea.value = currentReport;  // Use current report (may already be modified)
    modifyReportBtn.textContent = "Annulla Modifica";
    
    console.log("Doctor started modifying report");
  } else {
    // User is canceling modification - check if they made changes
    const textareaContent = reportTextarea.value.trim();
    const currentReportTrimmed = currentReport.trim();
    
    if (textareaContent !== currentReportTrimmed) {
      // Doctor made changes - ask if they want to save them
      const saveChanges = confirm("Hai modificato il referto. Vuoi salvare le modifiche?");
      if (saveChanges) {
        // Save the changes
        currentReport = textareaContent;
        reportWasModified = true;
        
        // Update the display with modified content
        const formatted = currentReport.replace(/\n/g, "<br>");
        reportOutput.innerHTML = `<div class="report-content" style="background-color:#e8f5e8; border:2px solid #4caf50; padding:10px;"><div style="color:#2e7d32; font-weight:bold; margin-bottom:10px;">📝 Referto Modificato dal Dottore</div>${formatted}</div>`;
        
        console.log("Doctor saved modifications. Report was modified:", reportWasModified);
        console.log("Original AI report:", originalAIReport.substring(0, 100) + "...");
        console.log("Modified report:", currentReport.substring(0, 100) + "...");
      }
      // If they don't want to save, currentReport stays unchanged
    }
    
    reportTextarea.style.display = "none";
    reportOutput.style.display = "block";
    modifyReportBtn.textContent = "Modifica Referto";
  }
});

// 🚨 NEW: Track real-time changes in textarea to detect modifications
reportTextarea.addEventListener('input', function() {
  const textareaContent = this.value.trim();
  const originalTrimmed = currentReport.trim();
  
  // Visual indicator that content has been modified
  if (textareaContent !== originalTrimmed) {
    this.style.borderColor = "#ff9800";
    this.style.backgroundColor = "#fff3e0";
  } else {
    this.style.borderColor = "#ccc";
    this.style.backgroundColor = "white";
  }
});
  
//=================================================
// 🚨 ENHANCED: Save Session - Send Both Report Versions
//=================================================
saveSessionButton.addEventListener("click", async () => {
  // Determine final report text
  let finalReportText;
  
  if (reportTextarea.style.display === "block") {
    // User is currently editing - use textarea content and mark as modified
    finalReportText = reportTextarea.value.trim();
    if (finalReportText !== currentReport.trim()) {
      currentReport = finalReportText;
      reportWasModified = true;
      console.log("Final changes detected during save");
    }
  } else {
    // User is not editing - use current report
    finalReportText = currentReport;
  }
  
  if (!finalReportText) {
    alert("Il referto è vuoto.");
    return;
  }
  
  function validateMatricola(m) { return /^\d{5}$/.test(m); }
  function validateCF(cf) { return /^[A-Za-z0-9]{16}$/.test(cf); }
  
  const doctorMatricola = matricolaInput.value.trim();
  const patientCF = cfInput.value.trim();
  
  if (!validateMatricola(doctorMatricola)) {
    alert("La matricola deve essere di esattamente 5 cifre.");
    return;
  }
  if (!validateCF(patientCF)) {
    alert("Il CF del paziente deve essere di esattamente 16 caratteri alfanumerici.");
    return;
  }
  
  // 🚨 ENHANCED: Send both original and final reports + modification flag
  const payload = {
    conversation: conversationLines.join("\n"),
    original_report: originalAIReport,        // Original AI-generated report
    report: finalReportText,                  // Final report (possibly modified)
    was_modified: reportWasModified,          // Whether doctor modified it
    matricola: doctorMatricola,
    patientCF: patientCF
  };
  
  console.log("Saving session with enhanced data:");
  console.log("- Original AI report length:", originalAIReport.length);
  console.log("- Final report length:", finalReportText.length);
  console.log("- Was modified by doctor:", reportWasModified);
  console.log("- Reports are different:", originalAIReport.trim() !== finalReportText.trim());
  
  saveSessionButton.disabled = true;
  saveSessionButton.textContent = "Salvando...";
  
  try {
    const response = await fetch('/refertazione/save_session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    console.log("HTTP save response:", result);
    
    if (result.success) {
      // Stop all intervals immediately
      clearInterval(partialPollInterval);
      clearInterval(transcriptPollInterval);
      clearInterval(sessionCheckInterval);
      clearInterval(heartbeatInterval);
      
      // Show beautiful success animation
      showSuccessAnimation("✅ Refertazione salvata con successo!", result.filename);
      
      // Notify server to clean up
      socket.emit("disconnecting_cleanup");
      
      // Force redirect after animation
      setTimeout(() => {
        window.location.href = "/refertazione/doctor_setup";
      }, 3000);
    } else {
      // Show error and re-enable button
      showErrorMessage("❌ Errore nel salvataggio: " + result.error);
      saveSessionButton.disabled = false;
      saveSessionButton.textContent = "Salva Sessione";
    }
  } catch (err) {
    console.error("Error saving session:", err);
    showErrorMessage("❌ Errore di connessione durante il salvataggio");
    saveSessionButton.disabled = false;
    saveSessionButton.textContent = "Salva Sessione";
  }
});

// Add these new functions
function showSuccessAnimation(message, filename) {
  const popup = document.createElement('div');
  popup.style.cssText = `
  position: fixed !important; 
  top: 50% !important; 
  left: 50% !important; 
  transform: translate(-50%, -50%) !important;
  background: linear-gradient(145deg, #ffffff, #f8f9fa);
  color: #333; padding: 60px 50px; border-radius: 30px;
  text-align: center; box-shadow: 0 25px 50px rgba(0,0,0,0.4);
  animation: popupSlideIn 0.6s ease-out;
  max-width: 450px; min-width: 350px; z-index: 99999;
  border: 3px solid #4CAF50; overflow: hidden;
  margin: 0 !important;
`;
  
  popup.innerHTML = `
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                background: linear-gradient(45deg, rgba(76,175,80,0.1), rgba(69,160,73,0.1)); 
                pointer-events: none;"></div>
    <div style="position: relative; z-index: 1;">
      <div style="font-size: 90px; margin-bottom: 30px; animation: checkmarkPop 1.2s ease-out; 
                  filter: drop-shadow(0 5px 10px rgba(76,175,80,0.3));">✅</div>
      <h2 style="margin: 0 0 20px 0; font-size: 32px; font-weight: 700; 
                 font-family: 'Segoe UI', sans-serif; color: #2E7D32;
                 text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        Sessione Salvata!
      </h2>
      <div style="width: 60px; height: 4px; background: linear-gradient(90deg, #4CAF50, #66BB6A); 
                  margin: 20px auto; border-radius: 2px; animation: lineExpand 0.8s ease-out 0.5s both;"></div>
      <p style="margin: 25px 0 0 0; font-size: 16px; color: #666; font-weight: 500;
                animation: textFadeIn 0.6s ease-out 0.8s both;">
        🔄 Reindirizzamento automatico in corso...
      </p>
    </div>
  `;
  
  const style = document.createElement('style');
  style.textContent = `
    @keyframes popupSlideIn {
      0% { 
        transform: translate(-50%, -50%) scale(0.5); 
        opacity: 0; 
      }
      60% { 
        transform: translate(-50%, -50%) scale(1.05); 
      }
      100% { 
        transform: translate(-50%, -50%) scale(1); 
        opacity: 1; 
      }
    }
    
    @keyframes checkmarkPop {
      0% { transform: scale(0) rotate(-180deg); }
      50% { transform: scale(1.3) rotate(10deg); }
      70% { transform: scale(0.9) rotate(-5deg); }
      100% { transform: scale(1) rotate(0deg); }
    }
    
    @keyframes lineExpand {
      0% { width: 0; opacity: 0; }
      100% { width: 60px; opacity: 1; }
    }
    
    @keyframes textFadeIn {
      0% { opacity: 0; transform: translateY(20px); }
      100% { opacity: 1; transform: translateY(0); }
    }
  `;
  document.head.appendChild(style);
  
  document.body.appendChild(popup);
}

function showErrorMessage(message) {
  alert(message); // Simple implementation - you can enhance this with a better UI
}

//=================================================
// 🚨 ENHANCED: Reset Session - Clear Report Tracking
//=================================================
document.getElementById('resetSessionBtn').addEventListener('click', () => {
  // Show confirmation dialog in Italian
  const confirmReset = confirm(
    "⚠️ ATTENZIONE!\n\n" +
    "Sei sicuro di voler resettare la sessione?\n\n" +
    "Questa azione:\n" +
    "• Cancellerà tutta la conversazione registrata\n" +
    "• Rimuoverà il referto generato\n" +
    "• Ricomincerà dall'inizio\n\n" +
    "Confermi di voler procedere?"
  );
  
  if (confirmReset) {
    // Stop all current activities
    if (recording) {
      recording = false;
      if (processor) processor.disconnect();
      if (audioContext) audioContext.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    }
    
    // Clear all intervals
    clearInterval(window.sendInterval);
    clearInterval(partialPollInterval);
    clearInterval(transcriptPollInterval);
    clearInterval(sessionCheckInterval);
    clearInterval(heartbeatInterval);
    
    // Show reset loading
    const resetBtn = document.getElementById('resetSessionBtn');
    const originalText = resetBtn.innerHTML;
    resetBtn.innerHTML = '🔄 Resettando...';
    resetBtn.disabled = true;
    
    // Reset UI elements
    finalDiv.innerHTML = "";
    partialDiv.innerHTML = "Sessione resettata. Premi 'Inizia Registrazione' per ricominciare.";
    reportOutput.innerHTML = "";
    reportOutput.style.display = "none";
    reportTextarea.style.display = "none";
    reportTextarea.value = "";
    modifyReportBtn.style.display = "none";
    modifyReportBtn.textContent = "Modifica Referto";
    saveInfoSection.style.display = "none";
    matricolaInput.value = "";
    cfInput.value = "";
    
    // Reset button states
    recordBtn.classList.add("visible");
    recordBtn.disabled = false;
    recordBtn.textContent = "Inizia Registrazione";
    stopBtn.classList.remove("visible");
    reportBtn.style.display = "none";
    saveSessionButton.disabled = true;
    saveSessionButton.style.display = "none";
    saveSessionButton.textContent = "Salva Sessione";
    
    // 🚨 ENHANCED: Reset report tracking variables
    conversationLines = [];
    originalAIReport = "";
    currentReport = "";
    reportWasModified = false;
    pcmChunks = [];
    autoScroll = false;
    
    console.log("Report tracking variables reset");
    
    // Notify server to reset session
    socket.emit('reset_session', { matricola: mat });
    
    // Show success message
    setTimeout(() => {
      resetBtn.innerHTML = originalText;
      resetBtn.disabled = false;
      
      // Show professional reset confirmation
      const resetPopup = document.createElement('div');
      resetPopup.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white; padding: 30px; border-radius: 15px; z-index: 9999;
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: resetSuccess 0.5s ease-out;
      `;
      resetPopup.innerHTML = `
        <div style="font-size: 40px; margin-bottom: 15px;">🆕</div>
        <h3 style="margin: 0 0 10px 0;">Sessione Resettata!</h3>
        <p style="margin: 0; font-size: 14px; opacity: 0.9;">Puoi iniziare una nuova registrazione</p>
      `;
      
      const resetStyle = document.createElement('style');
      resetStyle.textContent = `
        @keyframes resetSuccess {
          0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
          100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
      `;
      document.head.appendChild(resetStyle);
      
      document.body.appendChild(resetPopup);
      setTimeout(() => resetPopup.remove(), 3000);
    }, 1000);
  }
});

// Update the beforeunload handler
window.addEventListener("beforeunload", () => {
  clearInterval(sessionCheckInterval);
  clearInterval(heartbeatInterval);
  socket.emit("disconnecting_cleanup");
});
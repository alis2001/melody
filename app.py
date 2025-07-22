import json
import os
import wave
import threading
import numpy as np
import librosa
import soundfile as sf
import time
import requests
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
from vosk import Model, KaldiRecognizer
from pyannote.audio import Inference
import ollama
from threading import Lock
from datetime import timedelta

###############################
#  Configuration & Globals
###############################
MODEL_PATH = "vosk-model-it-0.22"
SAMPLE_RATE = 16000
WAV_DIRECTORY = "audio_segments/"
DEFAULT_VOICE_DIR = "default_voices"
SESSION_TRANSCRIPTS_DIR = "transcripts"
SESSION_REPORTS_DIR = "session_reports"

hf_token = os.getenv('HUGGINGFACE_TOKEN')
THRESHOLD = 0.22
MIN_DURATION = 0.22  # seconds

app = Flask(__name__, static_url_path='/refertazione/static')
app.config["SECRET_KEY"] = "secret!"  # Change for production
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600, ping_interval=60, path="/socket.io")

# Ensure required directories exist
os.makedirs(WAV_DIRECTORY, exist_ok=True)
os.makedirs(DEFAULT_VOICE_DIR, exist_ok=True)
os.makedirs(SESSION_REPORTS_DIR, exist_ok=True)
os.makedirs(SESSION_TRANSCRIPTS_DIR, exist_ok=True)

def remove_temp_wavs(directory):
    for f in os.listdir(directory):
        if f.endswith(".wav"):
            os.remove(os.path.join(directory, f))
    print(f"Cleared old .wav files in {directory}")

remove_temp_wavs(WAV_DIRECTORY)

file_lock = Lock()
sessions_lock = Lock()

# Sessions dictionary keyed by matricola (doctor's ID)
sessions = {}  # key: matricola, value: session data dictionary

embedding_inference = None
print(f"Loading Vosk model from: {MODEL_PATH}")
vosk_model = Model(MODEL_PATH)


###############################
#  Session Validation Functions (ADD THESE NEW FUNCTIONS)
###############################
def is_valid_session(matricola):
    """Check if a session is valid and not expired"""
    if not matricola:
        return False
    
    voice_file = os.path.join(DEFAULT_VOICE_DIR, f"{matricola}.wav")
    if not os.path.exists(voice_file):
        return False
    
    with sessions_lock:
        # If no session exists yet, it's still valid (will be created)
        if matricola not in sessions:
            return True
        
        session_data = sessions[matricola]
        created_at = session_data.get("created_at", 0)
        if time.time() - created_at > 6 * 3600:  # 6 hours
            return False
    
    return True

def cleanup_session(matricola):
    """Clean up a specific session"""
    with sessions_lock:
        if matricola in sessions:
            session_data = sessions[matricola]
            try:
                tf = session_data.get("transcript_file")
                if tf and os.path.exists(tf):
                    os.remove(tf)
                    print(f"Deleted transcript: {tf}")
            except Exception as e:
                print(f"Error deleting transcript for {matricola}:", e)
            
            del sessions[matricola]
            print(f"Session cleaned up for matricola: {matricola}")

# Track browser connections (ADD THIS GLOBAL VARIABLE)
active_connections = {}  # key: matricola, value: {'sid': sid, 'last_seen': timestamp}


###############################
#  Utility Functions
###############################
def trim_audio(wav_file, top_db=30):
    try:
        audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        trimmed_file = wav_file.replace(".wav", "_trim.wav")
        sf.write(trimmed_file, trimmed_audio, sr)
        return trimmed_file
    except Exception as e:
        print(f"Error trimming {wav_file}: {e}")
        return wav_file

def get_embedding(embedding_inference_local, wav_file):
    trimmed_file = trim_audio(wav_file)
    emb = embedding_inference_local(trimmed_file)
    if emb.ndim == 2:
        emb = np.mean(emb, axis=0)
    if trimmed_file != wav_file and os.path.exists(trimmed_file):
        os.remove(trimmed_file)
    return emb

def assign_speaker(embedding, default_embedding, threshold=THRESHOLD):
    sim_val = (
        np.dot(embedding, default_embedding) /
        (np.linalg.norm(embedding) * np.linalg.norm(default_embedding))
    )
    print(f"Computed similarity: {sim_val:.3f}")
    return (0 if sim_val >= threshold else 1), sim_val

def estrai_conversazione_medica(transcript_file):
    if not os.path.exists(transcript_file):
        return ""
    lines = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Dottore:" in line or "Paziente:" in line:
                parts = line.split("|", 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    details = content.split(":", 2)
                    if len(details) >= 3:
                        speaker = details[1].strip()
                        text = details[2].strip()
                        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def genera_report_medico(transcript_file):
    conversation = estrai_conversazione_medica(transcript_file)
    if not conversation:
        return "Nessuna conversazione valida trovata"

    try:
        # Replace this with your LOCAL PC IP address
        local_service_url = "http://192.168.125.193:8010/generate"
        response = requests.post(local_service_url, json={"conversation": conversation}, timeout=120)

        if response.status_code == 200:
            return response.json().get("report", "")
        else:
            return f"Errore dal servizio: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Errore di connessione al servizio locale: {e}"



def process_sentence_and_emit(matricola, sid, wav_filename, sentence):
    with sessions_lock:
        session_data = sessions.get(matricola)
    if not session_data:
        print("Session data not found for matricola", matricola)
        return
    default_emb = session_data["default_emb"]
    transcript_file = session_data["transcript_file"]

    similarity = 0.0
    try:
        audio, sr = librosa.load(wav_filename, sr=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
    except Exception as e:
        print(f"Error loading {wav_filename}: {e}")
        duration = 0

    if duration < MIN_DURATION:
        print(f"Audio too short ({duration:.2f}s); fallback => Dottore")
        voice = "Dottore"
    else:
        try:
            emb = get_embedding(embedding_inference, wav_filename)
            voice_label, similarity = assign_speaker(emb, default_emb)
            voice = "Dottore" if voice_label == 0 else "Paziente"
        except Exception as e:
            print("Error in diarization:", e)
            voice = "Dottore"

    line = f"Similarity: {similarity:.3f} | {os.path.basename(wav_filename)}: {voice}: {sentence}"
    print("Appending line to transcript:", line)
    with file_lock:
        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    socketio.emit("final_result", {"line": line}, room=sid)

###############################
#  SOCKET.IO HANDLERS
###############################
@socketio.on("connect")
def handle_connect(auth):
    sid = request.sid
    print(f"[CONNECT] sid={sid}, auth={auth}")
    matricola = auth.get("matricola", "").strip()
    if not matricola:
        emit("error", {"message": "Matricola is required."})
        return

    voice_file = os.path.join(DEFAULT_VOICE_DIR, f"{matricola}.wav")
    if not os.path.exists(voice_file):
        emit("error", {"message": "Default voice not found. Please register first."})
        return

    # Track this connection
    with sessions_lock:
        active_connections[matricola] = {
            'sid': sid,
            'last_seen': time.time()
        }

    global embedding_inference
    if embedding_inference is None:
        embedding_inference = Inference("pyannote/embedding", window="whole", use_auth_token=hf_token)
    try:
        default_emb = get_embedding(embedding_inference, voice_file)
    except Exception as e:
        emit("error", {"message": f"Error loading voice model: {str(e)}"})
        return

    with sessions_lock:
        if matricola in sessions:
            sessions[matricola]["sid"] = sid
            print(f"[CONNECT] Reusing session for matricola {matricola}, new sid={sid}")
        else:
            session_id = f"{matricola}_{int(time.time())}"
            transcript_file = os.path.join(SESSION_TRANSCRIPTS_DIR, f"{session_id}.txt")

            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write("")

            sessions[matricola] = {
                "sid": sid,
                "default_emb": default_emb,
                "matricola": matricola,
                "transcript_file": transcript_file,
                "partial_transcript": "",
                "audio_buffer": b"",
                "sentence_count": 0,
                "created_at": time.time()
            }

            print(f"[CONNECT] Created session for matricola {matricola}, sid={sid}")

    join_room(matricola)
    join_room(sid)
    emit("server_response", {"message": f"Connected as matricola {matricola}"})

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"[DISCONNECT] sid={sid} disconnected.")
    
    # Find and mark session as disconnected but don't delete immediately
    matricola_to_mark = None
    with sessions_lock:
        for mat, data in sessions.items():
            if data.get("sid") == sid:
                matricola_to_mark = mat
                break
        
        if matricola_to_mark and matricola_to_mark in active_connections:
            del active_connections[matricola_to_mark]
            print(f"[DISCONNECT] Marked {matricola_to_mark} as disconnected")

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    sid = request.sid
    matricola = None
    with sessions_lock:
        for mat, session_data in sessions.items():
            if session_data.get("sid") == sid:
                matricola = mat
                client = session_data
                break
    if not matricola:
        return

    client["audio_buffer"] += data
    rec = client.get("recognizer")
    if rec is None:
        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.SetWords(True)
        client["recognizer"] = rec
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        sentence = result.get("text", "").strip()
        rec.Reset()
        if sentence:
            client["sentence_count"] += 1
            wav_filename = os.path.join(WAV_DIRECTORY, f"sentence_{sid}_{client['sentence_count']}.wav")
            with wave.open(wav_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(client["audio_buffer"])
            client["audio_buffer"] = b""
            threading.Thread(target=process_sentence_and_emit,
                             args=(matricola, sid, wav_filename, sentence)).start()
    else:
        partial_text = json.loads(rec.PartialResult()).get("partial", "")
        client["partial_transcript"] = partial_text
        socketio.emit("partial_result", {"partial": partial_text}, room=sid)

@socketio.on("stop_recording")
def handle_stop_recording():
    sid = request.sid
    matricola = None
    client = None  # ← ADD THIS LINE
    with sessions_lock:
        for mat, session_data in sessions.items():
            if session_data.get("sid") == sid:
                matricola = mat
                client = session_data
                break
    if not client:
        return

    # If there is remaining audio data, process and save it.
    if client["audio_buffer"]:
        rec = client["recognizer"]
        result = json.loads(rec.Result())
        sentence = result.get("text", "").strip()
        if sentence:
            client["sentence_count"] += 1
            wav_filename = os.path.join(WAV_DIRECTORY, f"sentence_{sid}_{client['sentence_count']}.wav")
            with wave.open(wav_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(client["audio_buffer"])
            client["audio_buffer"] = b""
            threading.Thread(target=process_sentence_and_emit,
                             args=(matricola, sid, wav_filename, sentence)).start()

    socketio.emit("recording_stopped", {"message": "Recording stopped."}, room=sid)
    print(f"[STOP_RECORDING] Recording stopped for matricola {matricola}")


@socketio.on("generate_report")
def handle_generate_report():
    sid = request.sid
    print(f"Starting report generation for sid={sid}")

    def run_report():
        matricola = None
        client = None
        with sessions_lock:
            for mat, session_data in sessions.items():
                if session_data.get("sid") == sid:
                    matricola = mat
                    client = session_data
                    break
        if not client:
            socketio.emit("report_generated", {"error": "Session not found."}, room=sid)
            return
        tfile = client["transcript_file"]
        if not os.path.exists(tfile):
            socketio.emit("report_generated", {"error": "Transcript missing."}, room=sid)
            return

        print("Generating report with Ollama for sid=", sid)
        report_text = genera_report_medico(tfile)
        print("Ollama response done. Report generation completed for sid=", sid)
        socketio.emit("report_generated", {"report": report_text}, room=sid)
    
    socketio.start_background_task(run_report)


@socketio.on("save_session")
def handle_save_session(data):
    print("Received save_session payload via Socket.IO (backup method):", data)
    sid = request.sid
    matricola = None
    with sessions_lock:
        for mat, client in sessions.items():
            if client.get("sid") == sid:
                matricola = mat
                break
    if not matricola:
        emit("session_saved", {"success": False, "error": "Session not found."}, room=sid)
        return

    client = sessions[matricola]
    mat = client["matricola"]
    patientCF = data.get("patientCF", "unknown")
    report_text = data.get("report", "")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{mat}_{timestamp}_{patientCF}.txt"
    dest_path = os.path.join(SESSION_REPORTS_DIR, filename)

    try:
        convo = estrai_conversazione_medica(client["transcript_file"])
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(f"Conversazione:\n{convo}\n\nReport:\n{report_text}")
        print(f"Session saved as: {filename}")
        
        # FIRST emit success message, THEN clean up with delay
        emit("session_saved", {"success": True, "filename": filename}, room=sid)
        
        # Clean up session after small delay
        def delayed_cleanup():
            time.sleep(2)
            cleanup_session(matricola)
        
        threading.Thread(target=delayed_cleanup).start()
        
    except Exception as e:
        print("Error saving session:", e)
        emit("session_saved", {"success": False, "error": str(e)}, room=sid)

@socketio.on('reset_session')
def handle_reset_session(data):
    matricola = data.get('matricola', '').strip()
    if not matricola:
        return
    
    print(f"[RESET] Resetting session for matricola {matricola}")
    
    with sessions_lock:
        if matricola in sessions:
            session_data = sessions[matricola]
            # Clear transcript file
            transcript_file = session_data.get('transcript_file')
            if transcript_file and os.path.exists(transcript_file):
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write("")  # Clear content
            
            # Reset session data but keep the session alive
            session_data['partial_transcript'] = ""
            session_data['audio_buffer'] = b""
            session_data['sentence_count'] = 0
            session_data['created_at'] = time.time()  # Reset timestamp
            
            print(f"[RESET] Session reset completed for {matricola}")
        else:
            print(f"[RESET] No session found for {matricola}")
    
    emit('session_reset_complete', {'status': 'success'}, room=request.sid)


###############################
#  AJAX ROUTES FOR PARTIAL & TRANSCRIPT (Using matricola)
###############################
@app.route("/refertazione/partial", methods=["GET"])
def partial_poll():
    matricola = request.args.get("matricola", "").strip()
    if not matricola:
        return jsonify({"partial": ""})
    with sessions_lock:
        session_data = sessions.get(matricola)
    if not session_data:
        return jsonify({"partial": ""})
    return jsonify({"partial": session_data.get("partial_transcript", "")})

@app.route("/refertazione/transcript", methods=["GET"])
def transcript_poll():
    matricola = request.args.get("matricola", "").strip()
    if not matricola:
        return jsonify({"transcript": ""})
    with sessions_lock:
        session_data = sessions.get(matricola)
    if not session_data:
        return jsonify({"transcript": ""})
    tfile = session_data["transcript_file"]
    if not os.path.exists(tfile):
        return jsonify({"transcript": ""})
    try:
        with open(tfile, "r", encoding="utf-8") as f:
            content = f.read()
    except:
        content = ""
    return jsonify({"transcript": content})

###############################
#  Flask Routes
###############################
@app.route("/refertazione/")
def root():
    return redirect(url_for("doctor_setup"))

@app.route("/refertazione/doctor_setup")
def doctor_setup():
    cf = request.args.get("cf", "")
    session.permanent = True
    session['access_granted'] = True
    return render_template("doctor_setup.html", cf=cf)


@app.route("/refertazione/check_doctor", methods=["GET"])
def check_doctor():
    mat = request.args.get("matricola", "").strip()
    if not mat:
        return jsonify({"exists": False, "error": "Matricola not provided"}), 400
    file_path = os.path.join(DEFAULT_VOICE_DIR, f"{mat}.wav")
    return jsonify({"exists": os.path.exists(file_path)})

@app.route("/refertazione/upload_default_voice", methods=["POST"])
def upload_default_voice():
    if "audio_data" not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400
    file = request.files["audio_data"]
    mat = request.form.get("matricola", "").strip()
    specialty = request.form.get("specialty", "").strip()
    specialty_sentence = request.form.get("specialty_sentence", "").strip()
    if not mat or not specialty or not specialty_sentence:
        return jsonify({"success": False, "error": "Missing required fields."}), 400
    filename = f"{mat}.wav"
    file_path = os.path.join(DEFAULT_VOICE_DIR, filename)
    file.save(file_path)
    print(f"Saved default voice for {mat}, specialty={specialty}, sentence={specialty_sentence}")
    return jsonify({"success": True, "message": f"Voice saved as {filename}"}), 200

@app.route("/refertazione/index.html")
def serve_index():
    # Check if user came through proper flow
    if not session.get('access_granted'):
        return redirect(url_for("doctor_setup"))
    
    mat = request.args.get("matricola", "").strip()
    if not mat:
        return redirect(url_for("doctor_setup"))
    
    # Check if doctor voice file exists (basic validation)
    voice_file = os.path.join(DEFAULT_VOICE_DIR, f"{mat}.wav")
    if not os.path.exists(voice_file):
        return redirect(url_for("doctor_setup", error="voice_not_found"))
    
    # If there's an existing session and it's expired, clean it up
    with sessions_lock:
        if mat in sessions:
            session_data = sessions[mat]
            created_at = session_data.get("created_at", 0)
            if time.time() - created_at > 6 * 3600:  # 6 hours
                cleanup_session(mat)
                return redirect(url_for("doctor_setup", error="session_expired"))
    
    return render_template("index.html")

@app.route("/refertazione/generate_report", methods=["GET"])
def generate_report():
    # Get the doctor's matricola from the query string
    matricola = request.args.get("matricola", "").strip()
    if not matricola:
        return jsonify({"error": "Matricola missing"}), 400

    with sessions_lock:
        session_data = sessions.get(matricola)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404

    tfile = session_data["transcript_file"]
    if not os.path.exists(tfile):
        return jsonify({"error": "Transcript missing"}), 404

    print("Generating report with Ollama for matricola=", matricola)
    report_text = genera_report_medico(tfile)
    print("Ollama response done. Report generation completed for matricola=", matricola)
    # Return the report in JSON
    return jsonify({"report": report_text})

@app.route("/refertazione/save_session", methods=["POST"])
def save_session_http():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    matricola = data.get("matricola", "").strip()
    patientCF = data.get("patientCF", "").strip()
    report_text = data.get("report", "")
    
    if not matricola or not patientCF or not report_text:
        return jsonify({"success": False, "error": "Missing required fields"}), 400
    
    # Find the session
    with sessions_lock:
        if matricola not in sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        client = sessions[matricola]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{matricola}_{timestamp}_{patientCF}.txt"
    dest_path = os.path.join(SESSION_REPORTS_DIR, filename)
    
    try:
        convo = estrai_conversazione_medica(client["transcript_file"])
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(f"Conversazione:\n{convo}\n\nReport:\n{report_text}")
        
        print(f"HTTP Session saved as: {filename}")
        
        # Clean up session after successful save  
        def delayed_cleanup():
            time.sleep(1)  # Give time for response to be sent
            cleanup_session(matricola)
        
        threading.Thread(target=delayed_cleanup).start()
        
        return jsonify({"success": True, "filename": filename})
        
    except Exception as e:
        print("Error saving session via HTTP:", e)
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route("/refertazione/check_session", methods=["GET"])
def check_session():
    """Check if session is still valid"""
    mat = request.args.get("matricola", "").strip()
    if not mat:
        return jsonify({"valid": False, "redirect": True})
    
    # Check if voice file exists (basic validation)
    voice_file = os.path.join(DEFAULT_VOICE_DIR, f"{mat}.wav")
    if not os.path.exists(voice_file):
        return jsonify({"valid": False, "redirect": True, "reason": "voice_not_found"})
    
    with sessions_lock:
        # If no session exists, that's OK - it will be created on Socket.IO connect
        if mat not in sessions:
            return jsonify({"valid": True, "reason": "no_session_yet"})
        
        # If session exists, check if it's expired
        session_data = sessions[mat]
        created_at = session_data.get("created_at", 0)
        if time.time() - created_at > 6 * 3600:  # 6 hours
            cleanup_session(mat)
            return jsonify({"valid": False, "redirect": True, "reason": "session_expired"})
    
    return jsonify({"valid": True})

@app.route("/refertazione/heartbeat", methods=["GET"])
def heartbeat():
    return jsonify({"status": "alive"})

@socketio.on("disconnecting_cleanup")
def handle_manual_cleanup():
    sid = request.sid
    with sessions_lock:
        for mat, data in list(sessions.items()):
            if data.get("sid") == sid:
                print(f"[CLEANUP] Removing session for {mat} due to manual disconnect")
                sessions.pop(mat, None)
                break

def cleanup_expired_sessions():
    while True:
        now = time.time()
        expired_sessions = []
        disconnected_sessions = []
        
        with sessions_lock:
            # Check for expired sessions (6+ hours old)
            for mat, data in list(sessions.items()):
                created = data.get("created_at", now)
                if now - created > 6 * 3600:  # 6 hours
                    expired_sessions.append(mat)
            
            # Check for disconnected sessions (no heartbeat for 5+ minutes)
            for mat in list(sessions.keys()):
                if mat not in active_connections:
                    # Session exists but no active connection - mark for cleanup after 5 min
                    continue
                elif now - active_connections[mat]['last_seen'] > 300:  # 5 minutes
                    disconnected_sessions.append(mat)

        # Clean up expired and disconnected sessions
        for mat in expired_sessions + disconnected_sessions:
            print(f"[CLEANUP] Cleaning up session for {mat}")
            cleanup_session(mat)

        # Remove old wavs
        remove_temp_wavs(WAV_DIRECTORY)
        time.sleep(60)  # Run cleanup every minute


if __name__ == "__main__":
    socketio.start_background_task(cleanup_expired_sessions)
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)
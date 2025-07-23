import json
import os
import wave
import threading
import numpy as np
import librosa
import soundfile as sf
import time
import requests
import sys
import signal
import gc
import atexit
from contextlib import contextmanager
import weakref
import shutil
from datetime import datetime
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
from vosk import Model, KaldiRecognizer
from pyannote.audio import Inference
import ollama
from threading import Lock
from datetime import timedelta
from dotenv import load_dotenv

# Import psutil for memory monitoring
try:
    import psutil
except ImportError:
    print("psutil not installed - memory monitoring disabled")
    psutil = None

load_dotenv()

###############################
#  TRAINING DATA PRESERVATION LAYER
#  - No changes to existing logic
#  - Just saves organized copies for ML training
###############################

# Training data directory structure
TRAINING_DATA_ROOT = "training_data"

def ensure_training_directories():
    """Create training data directory structure"""
    try:
        os.makedirs(TRAINING_DATA_ROOT, exist_ok=True)
        print(f"✅ Training data directory ready: {TRAINING_DATA_ROOT}")
    except Exception as e:
        print(f"❌ Error creating training directories: {e}")

def get_doctor_training_dir(matricola):
    """Get or create training directory for specific doctor"""
    doctor_dir = os.path.join(TRAINING_DATA_ROOT, f"matricola_{matricola}")
    
    # Create subdirectories for this doctor
    subdirs = [
        "audio_files",          # Raw audio files with transcriptions
        "conversations",        # Complete conversation transcripts
        "reports",             # Generated medical reports  
        "sessions"             # Session metadata and summaries
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(doctor_dir, subdir), exist_ok=True)
    
    return doctor_dir

def save_training_audio(matricola, wav_filename, sentence, speaker, similarity_score):
    """Save audio file for training - CALLED BEFORE DELETION"""
    try:
        if not os.path.exists(wav_filename):
            return None
            
        # Get doctor's training directory
        doctor_dir = get_doctor_training_dir(matricola)
        audio_dir = os.path.join(doctor_dir, "audio_files")
        
        # Create unique filename with timestamp
        timestamp = int(time.time() * 1000)
        base_name = os.path.splitext(os.path.basename(wav_filename))[0]
        
        # Save audio file
        training_audio_file = os.path.join(audio_dir, f"{timestamp}_{base_name}.wav")
        shutil.copy2(wav_filename, training_audio_file)
        
        # Save metadata alongside audio
        metadata = {
            "original_file": os.path.basename(wav_filename),
            "training_file": os.path.basename(training_audio_file),
            "transcription": sentence,
            "speaker": speaker,
            "similarity_score": similarity_score,
            "doctor_matricola": matricola,
            "timestamp": datetime.now().isoformat(),
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "duration_seconds": get_audio_duration(wav_filename)
        }
        
        metadata_file = os.path.join(audio_dir, f"{timestamp}_{base_name}.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved training audio: {os.path.basename(training_audio_file)}")
        return training_audio_file
        
    except Exception as e:
        print(f"❌ Error saving training audio: {e}")
        return None

def save_training_conversation(matricola, transcript_file, session_id):
    """Save complete conversation transcript for training"""
    try:
        if not os.path.exists(transcript_file):
            return None
            
        # Get doctor's training directory
        doctor_dir = get_doctor_training_dir(matricola)
        conversations_dir = os.path.join(doctor_dir, "conversations")
        
        # Create training conversation file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_conversation_file = os.path.join(conversations_dir, f"{timestamp}_{session_id}_conversation.txt")
        
        # Copy transcript file
        shutil.copy2(transcript_file, training_conversation_file)
        
        # Save conversation metadata
        metadata = {
            "original_transcript": os.path.basename(transcript_file),
            "training_file": os.path.basename(training_conversation_file),
            "doctor_matricola": matricola,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "session_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        metadata_file = os.path.join(conversations_dir, f"{timestamp}_{session_id}_conversation.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saved training conversation: {os.path.basename(training_conversation_file)}")
        return training_conversation_file
        
    except Exception as e:
        print(f"❌ Error saving training conversation: {e}")
        return None

def save_training_report_pair_enhanced(matricola, conversation_text, original_report, final_report, patient_cf, session_id, was_modified=False):
    """
    Save BOTH original AI-generated report AND final modified report
    This allows comparison of AI vs human corrections for ML improvement
    """
    try:
        # Get doctor's training directory
        doctor_dir = get_doctor_training_dir(matricola)
        reports_dir = os.path.join(doctor_dir, "reports")
        
        # Create training pair file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_pair_file = os.path.join(reports_dir, f"{timestamp}_{session_id}_pair.json")
        
        # Create ENHANCED training pair data with BOTH versions
        training_pair = {
            "doctor_matricola": matricola,
            "patient_cf": patient_cf,
            "session_id": session_id,
            "conversation": conversation_text,
            
            # 🚨 NEW: Save BOTH original and final reports
            "original_ai_report": original_report,        # AI-generated report
            "final_report": final_report,                 # Doctor's final version
            "was_modified_by_doctor": was_modified,       # True if doctor edited
            
            # Analysis fields for ML training
            "modification_analysis": {
                "original_length": len(original_report.split()) if original_report else 0,
                "final_length": len(final_report.split()) if final_report else 0,
                "length_change": (len(final_report.split()) - len(original_report.split())) if (original_report and final_report) else 0,
                "substantial_changes": was_modified and (abs(len(final_report) - len(original_report)) > 100),
                "needs_human_review": was_modified  # Flag for training data quality
            },
            
            "timestamp": datetime.now().isoformat(),
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "conversation_length": len(conversation_text.split()) if conversation_text else 0
        }
        
        # Save enhanced training pair
        with open(training_pair_file, "w", encoding="utf-8") as f:
            json.dump(training_pair, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saved ENHANCED training report pair: {os.path.basename(training_pair_file)}")
        print(f"    Original AI report: {len(original_report)} chars")
        print(f"    Final report: {len(final_report)} chars") 
        print(f"    Modified by doctor: {was_modified}")
        
        return training_pair_file
        
    except Exception as e:
        print(f"❌ Error saving enhanced training report pair: {e}")
        return None

def save_session_summary(matricola, session_id, summary_data):
    """Save session summary and metadata"""
    try:
        # Get doctor's training directory
        doctor_dir = get_doctor_training_dir(matricola)
        sessions_dir = os.path.join(doctor_dir, "sessions")
        
        # Create session summary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = os.path.join(sessions_dir, f"{timestamp}_{session_id}_summary.json")
        
        # Add metadata to summary
        summary_data.update({
            "doctor_matricola": matricola,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "session_date": datetime.now().strftime("%Y-%m-%d")
        })
        
        # Save session summary
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saved session summary: {os.path.basename(session_file)}")
        return session_file
        
    except Exception as e:
        print(f"❌ Error saving session summary: {e}")
        return None

def get_audio_duration(wav_file):
    """Get duration of audio file in seconds"""
    try:
        audio, sr = librosa.load(wav_file, sr=None)
        return len(audio) / sr
    except:
        return 0

def cleanup_old_training_data(days_to_keep=365):
    """Clean up training data older than specified days"""
    try:
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        count = 0
        
        for root, dirs, files in os.walk(TRAINING_DATA_ROOT):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getctime(file_path) < cutoff_time:
                    os.remove(file_path)
                    count += 1
                    
        if count > 0:
            print(f"🗑️  Cleaned up {count} old training files (>{days_to_keep} days)")
            
    except Exception as e:
        print(f"❌ Error cleaning old training data: {e}")

# Initialize training directories on startup
ensure_training_directories()

###############################
#  MEMORY AND RESOURCE MANAGEMENT
###############################

# Global cleanup registry
cleanup_registry = weakref.WeakSet()

def force_gc():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    # Force malloc to release memory back to OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

def log_memory_usage(operation=""):
    """Log current memory usage"""
    if psutil:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = process.memory_percent()
        print(f"[MEMORY] {operation}: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
        return memory_mb
    return 0

@contextmanager
def memory_monitor(operation_name):
    """Context manager to monitor memory usage during operations"""
    mem_before = log_memory_usage(f"{operation_name} START")
    try:
        yield
    finally:
        mem_after = log_memory_usage(f"{operation_name} END")
        if psutil:
            mem_diff = mem_after - mem_before
            if mem_diff > 100:  # Log if more than 100MB increase
                print(f"WARNING: {operation_name} used {mem_diff:.1f}MB memory")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"Received signal {signum}, shutting down gracefully...")
    
    # Clean up all sessions
    with sessions_lock:
        for matricola in list(sessions.keys()):
            cleanup_session(matricola)
    
    # Remove temporary files
    try:
        remove_temp_wavs(WAV_DIRECTORY)
    except:
        pass
    
    # Force final garbage collection
    force_gc()
    
    # Exit
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register cleanup function for normal exit
atexit.register(lambda: signal_handler(0, None))

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
    """Remove temporary wav files with error handling"""
    try:
        count = 0
        for f in os.listdir(directory):
            if f.endswith(".wav"):
                file_path = os.path.join(directory, f)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        if count > 0:
            print(f"Cleared {count} old .wav files in {directory}")
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")

# Initial cleanup
remove_temp_wavs(WAV_DIRECTORY)

file_lock = Lock()
sessions_lock = Lock()

# Sessions dictionary keyed by matricola (doctor's ID)
sessions = {}  # key: matricola, value: session data dictionary

# Track browser connections
active_connections = {}  # key: matricola, value: {'sid': sid, 'last_seen': timestamp}

embedding_inference = None
print(f"Loading Vosk model from: {MODEL_PATH}")
vosk_model = Model(MODEL_PATH)
log_memory_usage("After Vosk model load")

###############################
#  ENHANCED SESSION MANAGEMENT
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
        # Extended session timeout to 12 hours for long medical visits
        if time.time() - created_at > 12 * 3600:  # 12 hours for long medical visits
            return False
    
    return True

def cleanup_session(matricola):
    """Enhanced session cleanup with memory management"""
    print(f"Starting cleanup for session {matricola}")
    
    with sessions_lock:
        if matricola in sessions:
            session_data = sessions[matricola]
            
            # 💾 SAVE CONVERSATION FOR TRAINING BEFORE CLEANUP
            transcript_file = session_data.get("transcript_file")
            if transcript_file and os.path.exists(transcript_file):
                session_id = session_data.get("session_id", f"{matricola}_{int(time.time())}")
                save_training_conversation(matricola, transcript_file, session_id)
            
            # Close and clean transcript file (after saving for training)
            try:
                if transcript_file and os.path.exists(transcript_file):
                    os.remove(transcript_file)
                    print(f"Deleted transcript: {transcript_file}")
            except Exception as e:
                print(f"Error deleting transcript for {matricola}: {e}")
            
            # Clean up recognizer object
            try:
                if 'recognizer' in session_data:
                    del session_data['recognizer']
            except:
                pass
                
            # Clean up audio buffer
            session_data['audio_buffer'] = b""
            
            # Clean up embeddings
            try:
                if 'default_emb' in session_data:
                    del session_data['default_emb']
            except:
                pass
            
            # Remove from sessions
            del sessions[matricola]
    
    # Clean up from active connections
    if matricola in active_connections:
        del active_connections[matricola]
    
    # Force memory cleanup
    force_gc()
    log_memory_usage(f"After cleanup session {matricola}")
    print(f"Session cleaned up for matricola: {matricola}")

###############################
#  Utility Functions - FIXED
###############################
def trim_audio(wav_file, top_db=30):
    """Fixed trim_audio function with proper error handling"""
    try:
        # Check if file exists first
        if not os.path.exists(wav_file):
            print(f"❌ trim_audio: File {wav_file} does not exist!")
            return wav_file
            
        with memory_monitor("audio_trim"):
            audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            trimmed_file = wav_file.replace(".wav", "_trim.wav")
            sf.write(trimmed_file, trimmed_audio, sr)
            # Clean up audio data from memory
            del audio, trimmed_audio
            print(f"✅ Audio trimmed: {wav_file} -> {trimmed_file}")
            return trimmed_file
    except Exception as e:
        print(f"❌ Error trimming {wav_file}: {e}")
        return wav_file

def get_embedding(embedding_inference_local, wav_file):
    """Fixed get_embedding function with proper error handling"""
    try:
        # Check if file exists first
        if not os.path.exists(wav_file):
            print(f"❌ get_embedding: File {wav_file} does not exist!")
            raise FileNotFoundError(f"File {wav_file} does not exist")
            
        with memory_monitor("get_embedding"):
            trimmed_file = trim_audio(wav_file)
            
            # Check if trimmed file exists
            if not os.path.exists(trimmed_file):
                print(f"❌ get_embedding: Trimmed file {trimmed_file} does not exist!")
                raise FileNotFoundError(f"Trimmed file {trimmed_file} does not exist")
            
            emb = embedding_inference_local(trimmed_file)
            if emb.ndim == 2:
                emb = np.mean(emb, axis=0)
            
            # Clean up trimmed file (but not the original)
            if trimmed_file != wav_file and os.path.exists(trimmed_file):
                try:
                    os.remove(trimmed_file)
                    print(f"🗑️  Cleaned up trimmed file: {trimmed_file}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up {trimmed_file}: {cleanup_error}")
            
            print(f"✅ Successfully extracted embedding from {wav_file}")
            return emb
    except Exception as e:
        print(f"❌ Error getting embedding from {wav_file}: {e}")
        raise

def assign_speaker(embedding, default_embedding, threshold=THRESHOLD):
    try:
        sim_val = (
            np.dot(embedding, default_embedding) /
            (np.linalg.norm(embedding) * np.linalg.norm(default_embedding))
        )
        print(f"Computed similarity: {sim_val:.3f}")
        return (0 if sim_val >= threshold else 1), sim_val
    except Exception as e:
        print(f"Error in assign_speaker: {e}")
        return (0, 0.0)  # Default to doctor

def estrai_conversazione_medica(transcript_file):
    if not os.path.exists(transcript_file):
        return ""
    lines = []
    try:
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
    except Exception as e:
        print(f"Error reading transcript {transcript_file}: {e}")
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
    """ENHANCED: Audio processing with training data preservation"""
    
    with memory_monitor(f"process_sentence_{matricola}"):
        with sessions_lock:
            session_data = sessions.get(matricola)
        
        if not session_data:
            print(f"❌ Session data not found for matricola {matricola}")
            # Clean up the wav file since we can't process it
            try:
                if os.path.exists(wav_filename):
                    os.remove(wav_filename)
                    print(f"🗑️  Cleaned up orphaned file: {wav_filename}")
            except:
                pass
            return
        
        try:
            default_emb = session_data["default_emb"]
            transcript_file = session_data["transcript_file"]

            similarity = 0.0
            
            # 🚨 CRITICAL FIX: Check if file exists BEFORE any processing
            if not os.path.exists(wav_filename):
                print(f"❌ Audio file {wav_filename} does not exist! Using fallback.")
                voice = "Dottore"  # fallback
                similarity = 0.0
                duration = 0
            else:
                print(f"✅ Processing audio file: {wav_filename}")
                
                # Load and get duration first (but DON'T delete file yet)
                try:
                    audio, sr = librosa.load(wav_filename, sr=SAMPLE_RATE)
                    duration = len(audio) / SAMPLE_RATE
                    # Clean up audio data from memory
                    del audio
                    print(f"✅ Loaded audio, duration: {duration:.2f}s")
                except Exception as e:
                    print(f"❌ Error loading {wav_filename}: {e}")
                    duration = 0

                if duration < MIN_DURATION:
                    print(f"Audio too short ({duration:.2f}s); fallback => Dottore")
                    voice = "Dottore"
                else:
                    try:
                        with memory_monitor("embedding_inference"):
                            # File should still exist here for embedding extraction
                            emb = get_embedding(embedding_inference, wav_filename)
                            voice_label, similarity = assign_speaker(emb, default_emb)
                            voice = "Dottore" if voice_label == 0 else "Paziente"
                            
                            # Clean up embedding from memory
                            del emb
                            
                    except Exception as e:
                        print("❌ Error in diarization:", e)
                        voice = "Dottore"

            # 💾 SAVE AUDIO FOR TRAINING BEFORE DELETION
            if sentence.strip():  # Only save if there's actual content
                save_training_audio(matricola, wav_filename, sentence, voice, similarity)

            # Write to transcript with file locking
            line = f"Similarity: {similarity:.3f} | {os.path.basename(wav_filename)}: {voice}: {sentence}"
            print("Appending line to transcript:", line)
            
            with file_lock:
                try:
                    with open(transcript_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                        f.flush()  # Force write to disk
                except Exception as e:
                    print(f"Error writing to transcript: {e}")
            
            # Emit result
            socketio.emit("final_result", {"line": line}, room=sid)
            
        except Exception as e:
            print(f"Error in process_sentence_and_emit: {e}")
        finally:
            # 🚨 CRITICAL FIX: Move file cleanup to the VERY END (after training data save)
            try:
                if os.path.exists(wav_filename):
                    os.remove(wav_filename)
                    print(f"🗑️  Cleaned up {wav_filename} (after training data preserved)")
            except Exception as e:
                print(f"Error cleaning up {wav_filename}: {e}")
            
            # Periodic memory cleanup
            session_count = len(sessions) if sessions else 0
            if session_count > 0 and session_count % 10 == 0:  # Every 10 processed sentences
                force_gc()

###############################
#  SOCKET.IO HANDLERS
###############################
@socketio.on("connect")
def handle_connect(auth):
    sid = request.sid
    print(f"[CONNECT] sid={sid}, auth={auth}")
    log_memory_usage("Socket connect")
    
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
        with memory_monitor("loading_embedding_inference"):
            embedding_inference = Inference("pyannote/embedding", window="whole", use_auth_token=hf_token)
    
    try:
        with memory_monitor("loading_voice_embedding"):
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

            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception as e:
                emit("error", {"message": f"Error creating transcript file: {str(e)}"})
                return

            sessions[matricola] = {
                "sid": sid,
                "default_emb": default_emb,
                "matricola": matricola,
                "transcript_file": transcript_file,
                "session_id": session_id,  # 💾 Store session_id for training data
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

    try:
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
                
                try:
                    with wave.open(wav_filename, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(client["audio_buffer"])
                    
                    client["audio_buffer"] = b""
                    
                    threading.Thread(
                        target=process_sentence_and_emit,
                        args=(matricola, sid, wav_filename, sentence),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    print(f"Error creating wav file: {e}")
        else:
            partial_text = json.loads(rec.PartialResult()).get("partial", "")
            client["partial_transcript"] = partial_text
            socketio.emit("partial_result", {"partial": partial_text}, room=sid)
            
    except Exception as e:
        print(f"Error in handle_audio_chunk: {e}")

@socketio.on("stop_recording")
def handle_stop_recording():
    sid = request.sid
    matricola = None
    client = None
    
    with sessions_lock:
        for mat, session_data in sessions.items():
            if session_data.get("sid") == sid:
                matricola = mat
                client = session_data
                break
    
    if not client:
        return

    try:
        # If there is remaining audio data, process and save it.
        if client["audio_buffer"]:
            rec = client.get("recognizer")
            if rec:
                result = json.loads(rec.Result())
                sentence = result.get("text", "").strip()
                if sentence:
                    client["sentence_count"] += 1
                    wav_filename = os.path.join(WAV_DIRECTORY, f"sentence_{sid}_{client['sentence_count']}.wav")
                    
                    try:
                        with wave.open(wav_filename, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(client["audio_buffer"])
                        
                        client["audio_buffer"] = b""
                        
                        threading.Thread(
                            target=process_sentence_and_emit,
                            args=(matricola, sid, wav_filename, sentence),
                            daemon=True
                        ).start()
                        
                    except Exception as e:
                        print(f"Error in final audio processing: {e}")

        socketio.emit("recording_stopped", {"message": "Recording stopped."}, room=sid)
        print(f"[STOP_RECORDING] Recording stopped for matricola {matricola}")
        
    except Exception as e:
        print(f"Error in handle_stop_recording: {e}")

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
        with memory_monitor("report_generation"):
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
    
    # 🚨 NEW: Get BOTH original and final reports
    original_report = data.get("original_report", "")      # AI-generated report
    final_report = data.get("report", "")                  # Final version (possibly modified)
    was_modified = data.get("was_modified", False)         # Whether doctor modified it
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{mat}_{timestamp}_{patientCF}.txt"
    dest_path = os.path.join(SESSION_REPORTS_DIR, filename)

    try:
        convo = estrai_conversazione_medica(client["transcript_file"])
        
        # 💾 SAVE ENHANCED CONVERSATION-REPORT PAIR FOR TRAINING
        session_id = client.get("session_id", f"{mat}_{int(time.time())}")
        save_training_report_pair_enhanced(mat, convo, original_report, final_report, patientCF, session_id, was_modified)
        
        # Save session summary for training
        summary_data = {
            "patient_cf": patientCF,
            "conversation_lines": len(convo.split('\n')) if convo else 0,
            "original_report_generated": bool(original_report),
            "final_report_saved": bool(final_report),
            "doctor_modified_report": was_modified,
            "session_duration": time.time() - client.get("created_at", time.time())
        }
        save_session_summary(mat, session_id, summary_data)
        
        # Save session normally (using final report)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(f"Conversazione:\n{convo}\n\nReport:\n{final_report}")
        print(f"Session saved as: {filename}")
        
        # FIRST emit success message, THEN clean up with delay
        emit("session_saved", {"success": True, "filename": filename}, room=sid)
        
        # Clean up session after small delay
        def delayed_cleanup():
            time.sleep(2)
            cleanup_session(matricola)
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
        
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
                try:
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write("")  # Clear content
                except Exception as e:
                    print(f"Error clearing transcript: {e}")
            
            # Reset session data but keep the session alive
            session_data['partial_transcript'] = ""
            session_data['audio_buffer'] = b""
            session_data['sentence_count'] = 0
            session_data['created_at'] = time.time()  # Reset timestamp
            
            # Clean up recognizer
            if 'recognizer' in session_data:
                try:
                    del session_data['recognizer']
                except:
                    pass
            
            print(f"[RESET] Session reset completed for {matricola}")
        else:
            print(f"[RESET] No session found for {matricola}")
    
    # Force garbage collection after reset
    force_gc()
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
            if time.time() - created_at > 12 * 3600:  # 12 hours for long medical visits
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
    with memory_monitor("report_generation_http"):
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
    
    # 🚨 NEW: Get BOTH original and final reports
    original_report = data.get("original_report", "")
    final_report = data.get("report", "")
    was_modified = data.get("was_modified", False)
    
    if not matricola or not patientCF or not final_report:
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
        
        # 💾 SAVE ENHANCED CONVERSATION-REPORT PAIR FOR TRAINING
        session_id = client.get("session_id", f"{matricola}_{int(time.time())}")
        save_training_report_pair_enhanced(matricola, convo, original_report, final_report, patientCF, session_id, was_modified)
        
        # Save session summary for training
        summary_data = {
            "patient_cf": patientCF,
            "conversation_lines": len(convo.split('\n')) if convo else 0,
            "original_report_generated": bool(original_report),
            "final_report_saved": bool(final_report),
            "doctor_modified_report": was_modified,
            "session_duration": time.time() - client.get("created_at", time.time())
        }
        save_session_summary(matricola, session_id, summary_data)
        
        # Save session normally (using final report)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(f"Conversazione:\n{convo}\n\nReport:\n{final_report}")
        
        print(f"HTTP Session saved as: {filename}")
        
        # Clean up session after successful save  
        def delayed_cleanup():
            time.sleep(1)  # Give time for response to be sent
            cleanup_session(matricola)
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
        
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
        if time.time() - created_at > 12 * 3600:  # 12 hours for long medical visits
            cleanup_session(mat)
            return jsonify({"valid": False, "redirect": True, "reason": "session_expired"})
    
    return jsonify({"valid": True})

@app.route("/refertazione/heartbeat", methods=["GET"])
def heartbeat():
    memory_usage = log_memory_usage("heartbeat")
    return jsonify({
        "status": "alive", 
        "memory_mb": memory_usage,
        "sessions": len(sessions),
        "connections": len(active_connections)
    })

@socketio.on("disconnecting_cleanup")
def handle_manual_cleanup():
    sid = request.sid
    with sessions_lock:
        for mat, data in list(sessions.items()):
            if data.get("sid") == sid:
                print(f"[CLEANUP] Removing session for {mat} due to manual disconnect")
                sessions.pop(mat, None)
                break

###############################
# PERIODIC CLEANUP TASK
###############################

def cleanup_expired_sessions():
    """Enhanced cleanup with memory management"""
    while True:
        try:
            now = time.time()
            expired_sessions = []
            disconnected_sessions = []
            
            with sessions_lock:
                # Check for expired sessions (12+ hours old for long medical visits)
                for mat, data in list(sessions.items()):
                    created = data.get("created_at", now)
                    if now - created > 12 * 3600:  # 12 hours for long medical visits
                        expired_sessions.append(mat)
                
                # Check for disconnected sessions (no heartbeat for 30+ minutes for medical sessions)
                for mat in list(sessions.keys()):
                    if mat not in active_connections:
                        # Session exists but no active connection - mark for cleanup after 30 min
                        continue
                    elif now - active_connections[mat]['last_seen'] > 1800:  # 30 minutes for medical sessions
                        disconnected_sessions.append(mat)

            # Clean up expired and disconnected sessions
            for mat in expired_sessions + disconnected_sessions:
                print(f"[CLEANUP] Cleaning up session for {mat}")
                cleanup_session(mat)

            # Remove old wavs more frequently
            remove_temp_wavs(WAV_DIRECTORY)
            
            # Clean up old training data (yearly)
            cleanup_old_training_data(365)
            
            # Force garbage collection every 5 minutes
            force_gc()
            
            # Log memory usage
            log_memory_usage("Periodic cleanup")
            
            time.sleep(300)  # Run cleanup every 5 minutes (less aggressive)
            
        except Exception as e:
            print(f"Error in cleanup_expired_sessions: {e}")
            time.sleep(300)

if __name__ == "__main__":
    # Start background cleanup task AFTER Flask starts
    import threading
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    
    # Log initial memory usage
    log_memory_usage("Application start")
    
    # Run the application
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)
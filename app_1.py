import io
import threading
import pyaudio
import numpy as np
import whisper
import speech_recognition as sr
from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import os
from pydub import AudioSegment, silence
import wave

app = Flask(__name__)
socketio = SocketIO(app)

# Load Whisper model
model = whisper.load_model("medium")  # or "small", "medium", "large"

# Initialize recognizer class
#recognizer = sr.Recognizer()

# Audio recording parameters
RATE = 16000    #16000 samples per second 
CHUNK = 1024    #1024 samples per frame

def record_audio(duration=20):  # duration in seconds
    p = pyaudio.PyAudio()   #open audio stream for reording
    stream = p.open(format=pyaudio.paInt16, #eah sample takes up 16 bits
                    channels=1, #nb of audio channel
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")

    frames = []
    try:
        for _ in range(int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, np.int16))
    except Exception as e:
        print(f"Recording stopped: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        return frames

# Function to split audio on silence for better chunking and faster processing
"""def split_audio_on_silence(audio_path, min_silence_len=500, silence_thresh=-40, chunk_len=2000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=chunk_len)
    return chunks """

def save_audio_to_wav(audio_data, file_path):
    with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes (16-bit audio)
            wf.setframerate(RATE)
            wf.writeframes(audio_data.tobytes())

def convert_wav_to_mp3(wav_file, mp3_file):
    audio = AudioSegment.from_wav(wav_file)
    audio.export(mp3_file, format = "mp3")

# Function to listen and transcribe in real-time
def listen_and_transcribe():
    while True:
        print("Listening...")
        frames = record_audio()
        audio_data = np.concatenate(frames).astype(np.int16)
        #audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize audio

        temp_wave_path = "temp_audio.wav"
        temp_mp3_path = "temp_mp3_audio.mp3"
        #audio_data = (audio_data * 32768.0).astype(np.int16)
        #audio_data = np.reshape(audio_data, (-1, 1))
        
        #save wav
        save_audio_to_wav(audio_data, temp_wave_path)

        #save mp3
        convert_wav_to_mp3(temp_wave_path, temp_mp3_path)

        # Transcribe the audio using Whisper
        print("Transcribing with Whisper...")
        result = model.transcribe(temp_mp3_path)
        text = result["text"]
        detected_language = whisper.tokenizer.LANGUAGES[result["language"]]

        print(f"Detected Language: {detected_language}")
        print(f"Transcribed Text: {text}")

        # Emit the detected language and transcribed text to the frontend
        socketio.emit('speech_result', {
            'text': text,
            'language': detected_language
        })

        #os.remove(temp_wave_path)
        #os.remove(temp_mp3_path)

@socketio.on('connect')
def handle_connect(auth=None):
    print('Client connected')
    socketio.emit('test_message', {'text': 'Socket.IO is working!'})

    # Start transcription in a background thread
    threading.Thread(target=listen_and_transcribe, daemon=True).start()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

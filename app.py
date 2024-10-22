from pyannote.audio import Pipeline
import speech_recognition as sr
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import fasttext
import pandas as pd
from wordfreq import top_n_list
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment, silence
import re
import os
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# Initialize recognizer class
recognizer = sr.Recognizer()

# Load the pre-trained fastText language identification model
model = fasttext.load_model('lid.176.bin')

# Get the top 2000 most common words in English
english_frequent_words = set(top_n_list('en', 2000))

# Load Sinhala and Tamil stopwords from CSV file using Pandas
def load_stopwords_from_csv(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    stopwords = {'Sinhala': set(), 'Tamil': set()}
    for _, row in df.iterrows():
        language = row['language']
        stopword = row['stopword']
        if language in stopwords:
            stopwords[language].add(stopword)
    return stopwords

# Load stopwords from CSV
STOPWORDS_FROM_CSV = load_stopwords_from_csv('SinhalaTamil.csv')

print("First 5 Sinhala stopwords:", [word.encode('utf-8').decode('utf-8') for word in list(STOPWORDS_FROM_CSV['Sinhala'])[:5]])
print("First 5 Tamil stopwords:", [word.encode('utf-8').decode('utf-8') for word in list(STOPWORDS_FROM_CSV['Tamil'])[:5]])

# Define LANGUAGE_VOCABULARY using wordfreq for English and CSV for Sinhala and Tamil
LANGUAGE_VOCABULARY = {
    'English': english_frequent_words,
    'Sinhala': STOPWORDS_FROM_CSV['Sinhala'],
    'Tamil': STOPWORDS_FROM_CSV['Tamil']
}

# Supported languages and their codes
LANGUAGES = {
    "en-US": "English",
    "si-LK": "Sinhala",
    "ta-IN": "Tamil"
}

# Global variables to manage the listening thread
listening_thread = None
listening_started = False
executor = ThreadPoolExecutor(max_workers=4)  # Increased to 4 workers for parallel processing

# Function to detect language of individual words
def detect_language_of_word(word):
    word_lower = word.lower()
    for lang, words in LANGUAGE_VOCABULARY.items():
        if word_lower in words:
            return lang
    # Fallback to fastText model if not found in vocabulary
    prediction = model.predict(word)
    detected_lang = prediction[0][0].split('__')[-1]
    language_map = {'en': 'English', 'si': 'Sinhala', 'ta': 'Tamil'}
    return language_map.get(detected_lang, detected_lang)

# Function to detect the dominant language of a sentence
def detect_language_hybrid(text):
    scores = {lang: 0 for lang in LANGUAGE_VOCABULARY}
    for lang, words in LANGUAGE_VOCABULARY.items():
        for word in text.split():
            word = word.lower()
            if word in words:
                scores[lang] += 1

    detected_by_words = max(scores, key=scores.get)

    # If the word-based score is low, fallback to model-based detection
    if scores[detected_by_words] < 2:
        prediction = model.predict(text)
        detected_by_model = prediction[0][0].split('__')[-1]
        language_map = {'en': 'English', 'si': 'Sinhala', 'ta': 'Tamil'}
        return language_map.get(detected_by_model, detected_by_model), prediction[1][0]

    return detected_by_words, scores[detected_by_words]

# Function to print sentence with mixed language words in their respective languages
def print_mixed_language_sentence(text):
    dominant_language, _ = detect_language_hybrid(text)
    words = text.split()
    mixed_sentence = []
    
    for word in words:
        detected_word_language = detect_language_of_word(word)
        if detected_word_language == dominant_language:
            mixed_sentence.append(word)
        else:
            mixed_sentence.append(word)  # Keep word as it is in its language

    return ' '.join(mixed_sentence), dominant_language

# Recognizes speech from a segment using Google's Speech-to-Text
def recognize_speech_from_segment(segment, lang_code):
    try:
        recognizer_audio = sr.AudioFile(segment)
        with recognizer_audio as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=lang_code)
        mixed_sentence, dominant_language = print_mixed_language_sentence(text)
        return (LANGUAGES.get(lang_code, ""), dominant_language, mixed_sentence)
    except sr.UnknownValueError:
        return (LANGUAGES.get(lang_code, ""), None, None)
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return (LANGUAGES.get(lang_code, ""), None, None)

# Function to perform speaker diarization
def perform_speaker_diarization(audio_file_path):
    auth_token = "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"
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
        diarization = pipeline(audio_file_path)
        
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))
        
        return speaker_segments
    except Exception as e:
        print(f"Error in diarization: {e}")
        return {}

# Optimized audio chunking based on silence to segment audio into manageable parts
def split_audio_on_silence(audio_path, min_silence_len=500, silence_thresh=-40, chunk_len=2000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=chunk_len)
    return chunks

# Transcribe and recognize speech from live microphone input
def listen_and_transcribe():
    try:
        global listening_started
        while listening_started:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduce duration for faster noise calibration
                print("Listening...")

                try:
                    start_time = time.time()
                    audio = recognizer.listen(source, timeout=30, phrase_time_limit=60)  # Adjust timeout values to reduce latency
                    print("Audio captured")

                    temp_audio_path = "temp_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio.get_wav_data())

                    # Use silence detection to split audio for faster processing
                    audio_chunks = split_audio_on_silence(temp_audio_path)

                    results = []

                    for i, chunk in enumerate(audio_chunks):
                        segment_path = f"temp_segment_{i}.wav"
                        chunk.export(segment_path, format="wav")

                        futures = [executor.submit(recognize_speech_from_segment, segment_path, lang_code) for lang_code in LANGUAGES.keys()]
                        for future in as_completed(futures):
                            lang_name, dominant_language, mixed_sentence = future.result()
                            if mixed_sentence:
                                results.append({
                                    'text': mixed_sentence,
                                    'language': dominant_language,
                                })

                        os.remove(segment_path)

                    # Emit each segment result with the identified language
                    for result in results:
                        print(f"Processed segment: {result['text']} (Language: {result['language']})")
                        socketio.emit('speech_result', {
                            'text': result['text'],
                            'language': result['language']
                        })

                    os.remove(temp_audio_path)
                    print(f"Processing time: {time.time() - start_time} seconds")
                except sr.RequestError as e:
                    print(f"Error with the speech recognition service: {e}")
                    socketio.emit('speech_result', {'text': f"Error with the speech recognition service: {e}", 'language': ''})
                except Exception as e:
                    print(f"Error: {e}")
    except Exception as e:
        print(f"ERROR HANDLING MESSAGE Pass Socket Message: {e}")   

@socketio.on('connect')
def handle_connect(auth=None):
    try:
        global listening_thread, listening_started

        print('Client connected')
        #socketio.emit('test_message', {'text': 'Socket.IO is working!'})

        if not listening_started:
            listening_started = True
            listening_thread = threading.Thread(target=listen_and_transcribe, daemon=True)
            listening_thread.start()
    except Exception as e:
        print(f"ERROR HANDLING MESSAGE: {e}")
        raise

@socketio.on('disconnect')
def handle_disconnect():
    try:
        global listening_started
        print('Client disconnected')
        listening_started = False
    except Exception as e:
        print(f"ERROR HANDLING MESSAGE Client Disconnect: {e}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
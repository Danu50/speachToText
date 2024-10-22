from flask import Flask, render_template
from flask_socketio import SocketIO
import speech_recognition as sr
import threading
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import fasttext
import pandas as pd
from wordfreq import top_n_list
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# Initialize recognizer class
recognizer = sr.Recognizer()

model = fasttext.load_model('lid.176.bin')

# Get the top 2000 most common words in English
english_frequent_words = set(top_n_list('en', 2000))

#print(f"Number of words retrieved: {len(english_frequent_words)}")

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

# Define stopwords using wordfreq for English and CSV for Sinhala and Tamil
def get_stopwords(language_code):
    if language_code == 'english':
        return english_frequent_words
    else:
        return STOPWORDS_FROM_CSV.get(language_code.capitalize(), set())

# Load stopwords from CSV file
STOPWORDS_FROM_CSV = load_stopwords_from_csv('SinhalaTamil.csv')

# Print the first 5 stopwords from the CSV file for Sinhala and Tamil
print("First 5 Sinhala stopwords:", [word.encode('utf-8').decode('utf-8') for word in list(STOPWORDS_FROM_CSV['Sinhala'])[:5]])
print("First 5 Tamil stopwords:", [word.encode('utf-8').decode('utf-8') for word in list(STOPWORDS_FROM_CSV['Tamil'])[:5]])

# Define LANGUAGE_VOCABULARY using wordfreq for English and CSV for Sinhala and Tamil
LANGUAGE_VOCABULARY = {
    'English': get_stopwords('english'),
    'Sinhala': STOPWORDS_FROM_CSV['Sinhala'],
    'Tamil': STOPWORDS_FROM_CSV['Tamil']
}

# Supported languages and their codes
LANGUAGES = {
    "en-US": "English",
    "si-LK": "Sinhala",
    "ta-IN": "Tamil"
}

# Global variable to manage the listening thread
listening_thread = None
listening_started = False
executor = ThreadPoolExecutor(max_workers=2)

def detect_language_hybrid(text):
    # Step 1: Frequent words-based detection
    scores = {lang: 0 for lang in LANGUAGE_VOCABULARY}
    for lang, words in LANGUAGE_VOCABULARY.items():
        for word in text.split():
            word = word.lower()
            if word in words:
                scores[lang] += 1

    # Get the initial guess based on frequent words
    detected_by_words = max(scores, key=scores.get)

    # Use FastText if no frequent words are found or if the confidence is low
    if scores[detected_by_words] < 2:  # can adjust the threshold 
        prediction = model.predict(text)
        detected_by_model = prediction[0][0].split('__')[-1]  # fastText model gives language code

        # Map fastText codes to language names 
        language_map = {'en': 'English', 'si': 'Sinhala', 'ta': 'Tamil'}
        detected_by_model = language_map.get(detected_by_model, detected_by_model)

        return detected_by_model, prediction[1][0]  # Return the detected language and confidence score

    return detected_by_words, scores[detected_by_words]

def recognize_speech(lang_code, audio):
    try:
        text = recognizer.recognize_google(audio, language=lang_code)
        return (LANGUAGES.get(lang_code, ""), text)
    except sr.UnknownValueError:
        return (LANGUAGES.get(lang_code, ""), None)

def listen_and_transcribe():
    global listening_started
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

        while listening_started:
            try:
                # Capture the audio
                audio = recognizer.listen(source)
                print("Audio captured")

                results = {}

                # Run speech recognition for each language in parallel
                futures = [executor.submit(recognize_speech, lang_code, audio) for lang_code in LANGUAGES.keys()]
                for future in as_completed(futures):
                    lang_name, text = future.result()
                    if text:
                        results[lang_name] = text
                        print(f"Detected {lang_name}: {text}")

                if results:
                    best_match_language = None
                    highest_score = 0

                    for lang_name, recognized_text in results.items():
                        detected_language, score = detect_language_hybrid(recognized_text)
                        print(f"Language: {detected_language}, Score: {score}, Text: {recognized_text}")

                        if score > highest_score:
                            best_match_language = detected_language
                            highest_score = score

                    # Ensure that best_match_language is not None
                    if best_match_language:
                        socketio.emit('speech_result', {'text': results.get(best_match_language, "No text"), 'language': best_match_language})
                        print(f"Emitting speech_result event: {results.get(best_match_language, 'No text')} ({best_match_language})")
                    else:
                        socketio.emit('speech_result', {'text': "Could not determine the language", 'language': ''})
                        print("Emitting speech_result event: Could not determine the language")
                else:
                    socketio.emit('speech_result', {'text': "Could not understand audio in any language", 'language': ''})
                    print("Emitting speech_result event: Could not understand audio in any language")

            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
                socketio.emit('speech_result', {'text': f"Error with the speech recognition service: {e}", 'language': ''})
                print(f"Emitting speech_result event: Error with the speech recognition service: {e}")

@socketio.on('connect')
def handle_connect(auth=None):
    global listening_thread, listening_started

    print('Client connected')
    socketio.emit('test_message', {'text': 'Socket.IO is working!'})

    if not listening_started:
        listening_started = True
        listening_thread = threading.Thread(target=listen_and_transcribe, daemon=True)
        listening_thread.start()

@socketio.on('disconnect')
def handle_disconnect():
    global listening_started
    print('Client disconnected')
    listening_started = False

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start the server using Gevent
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()

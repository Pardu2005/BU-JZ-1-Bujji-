"""
BUJJI - AI-Powered Voice Assistant
====================================
Activation: Clap once → then say anything with "bujji" (e.g. "come on bujji", "wake up bujji")
            Just like how Tony Stark talks to JARVIS.

Features:
  - JARVIS-style wake system (clap + voice phrase)
  - Voice Input/Output (SpeechRecognition + pyttsx3)
  - Object Detection via YOLOv8 (ultralytics) — free, no .h5 needed
  - Image Q&A via BLIP (Salesforce/blip-vqa-base) — free Hugging Face model
  - Webcam Capture + Analyze  → say "capture" or "take photo"
  - Screenshot + Analyze      → say "screenshot" or "analyze screen"
  - Weather Info via OpenWeatherMap /data/2.5/weather (free — no card needed)
  - News Headlines via GNews API (free — 100 calls/day, no card needed)
  - General AI Chat via Groq API (llama3-70b — free tier)
"""

import os
import re
import time
import threading
import numpy as np
import cv2
import requests
from PIL import Image
import pyautogui
import pyaudio                   # low-level mic access for clap detection

import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    BlipForConditionalGeneration,
)

import speech_recognition as sr
import pyttsx3
from groq import Groq

# ──────────────────────────────────────────────
# CONFIG — replace with your own keys
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")          # https://console.groq.com/
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")          # https://openweathermap.org/apiOPENWEATHER_KEY   = "c000e4c9ca09eb270b851afcab031300"   # https://openweathermap.org/api
GNEWS_API_KEY     = "6e0fae9daab9cbe50d8e4bb8b7fba79f"         # https://gnews.io  (free, no card)
GROQ_MODEL        = "llama-3.3-70b-versatile"                 # free Groq model

# OpenWeatherMap free-tier endpoint
OWM_BASE_URL      = "http://api.openweathermap.org/data/2.5/weather"
OWM_TIMEOUT       = 6
OWM_MAX_RETRIES   = 2

# GNews free-tier endpoint (100 calls/day, real-time, no card)
GNEWS_BASE_URL    = "https://gnews.io/api/v4/top-headlines"
GNEWS_TIMEOUT     = 6
GNEWS_MAX_RESULTS = 5

# Screenshot save path
SCREENSHOT_FILE   = "bujji_screenshot.png"

# ── JARVIS-style Wake System ──────────────────
# How loud a clap must be (0–32767). Lower = more sensitive.
CLAP_THRESHOLD    = 300   # calibrated for this mic (clap peak ~651, ambient ~1)
# Seconds to listen for wake phrase after clap detected
WAKE_PHRASE_WINDOW = 3.5
# Words that must appear in phrase after clap (say any of these + "bujji")
WAKE_KEYWORDS     = ("bujji", "buddy", "buji")        # "buddy" as fallback if mic mishears
# How long BUJJI stays active without a command before going back to sleep (seconds)
SLEEP_TIMEOUT     = 120  # 2 minutes — enough for heavy tasks like BLIP/YOLO

# ──────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────
recognizer  = sr.Recognizer()
tts_engine  = pyttsx3.init()
groq_client = Groq(api_key=GROQ_API_KEY)

# Shared state for wake system
_bujji_active  = threading.Event()   # set = BUJJI is awake and listening
_bujji_running = threading.Event()   # set = entire program is running
_bujji_running.set()


# ──────────────────────────────────────────────
# JARVIS-STYLE WAKE SYSTEM
# Flow: idle → clap detected → listen for wake phrase →
#       wake phrase heard → BUJJI activates → responds →
#       auto-sleeps after SLEEP_TIMEOUT seconds of silence
# ──────────────────────────────────────────────

def _listen_for_wake_phrase() -> bool:
    """
    After a clap is detected, listens for WAKE_PHRASE_WINDOW seconds.
    Returns True if the user says anything containing a WAKE_KEYWORD.
    """
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(
                source,
                timeout=WAKE_PHRASE_WINDOW,
                phrase_time_limit=WAKE_PHRASE_WINDOW,
            )
        phrase = recognizer.recognize_google(audio).lower()
        return any(kw in phrase for kw in WAKE_KEYWORDS)
    except Exception:
        return False


def wake_monitor():
    """
    Background thread — silently watches for clap + wake phrase.
    Sets _bujji_active when triggered.
    Never prints anything while BUJJI is sleeping.

    FIX: Keeps ONE persistent PyAudio stream open for the lifetime of
    the thread instead of creating/destroying it on every loop iteration.
    The old approach caused claps to be missed because the audio device
    was offline during open/close overhead (~50–100 ms per cycle).
    """
    CHUNK       = 1024
    SAMPLE_RATE = 16000

    pa     = pyaudio.PyAudio()
    stream = None

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while _bujji_running.is_set():
            if _bujji_active.is_set():
                # BUJJI is awake — STOP the clap stream so sr.Microphone
                # in get_audio() can open the mic without conflict.
                if stream.is_active():
                    stream.stop_stream()
                time.sleep(0.3)
                continue

            # BUJJI is sleeping — make sure the clap stream is running
            if not stream.is_active():
                stream.start_stream()

            try:
                data    = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                peak    = int(np.abs(samples).max())

                if peak > CLAP_THRESHOLD:
                    # Clap detected — pause stream, check for wake phrase
                    stream.stop_stream()
                    if _listen_for_wake_phrase():
                        _bujji_active.set()
                    # Don't restart here — loop top handles it based on state

            except Exception:
                pass   # overflow / device hiccup — skip this chunk

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()



# ──────────────────────────────────────────────
# UTILITY: Clean markdown from TTS text
# ──────────────────────────────────────────────
def clean_for_speech(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────
# TTS: Speak response
# ──────────────────────────────────────────────
def speak(text: str):
    try:
        voices = tts_engine.getProperty('voices')
        if len(voices) > 1:
            tts_engine.setProperty('voice', voices[1].id)   # female voice
        tts_engine.say(clean_for_speech(text))
        tts_engine.runAndWait()
    except Exception as e:
        print(f"[TTS Error] {e}")


# ──────────────────────────────────────────────
# STT: Listen and transcribe
# ──────────────────────────────────────────────
def get_audio(timeout: int = 7, phrase_limit: int = 12) -> str | None:
    with sr.Microphone() as source:
        print("\n🎙  Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            print("[STT] Timeout — no speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("[STT] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"[STT] Request error: {e}")
        return None


# ──────────────────────────────────────────────
# WEBCAM: Capture image
# ──────────────────────────────────────────────
def capture_image(filename: str = "bujji_capture.jpg") -> bool:
    """
    Opens webcam, shows a live countdown overlay, and auto-captures
    after WEBCAM_COUNTDOWN seconds — no keypress needed.
    """
    WEBCAM_COUNTDOWN = 8   # seconds before auto-capture

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] Could not open webcam.")
        return False

    print(f"📷  Webcam open — auto-capturing in {WEBCAM_COUNTDOWN} seconds...")
    start_time = time.time()
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed.")
            break

        last_frame = frame.copy()
        elapsed    = time.time() - start_time
        remaining  = max(0, WEBCAM_COUNTDOWN - int(elapsed))

        # ── Draw countdown overlay on the preview window ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            f"BUJJI capturing in {remaining}s...",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2,
        )

        cv2.imshow("BUJJI Camera", frame)
        cv2.waitKey(1)

        # Auto-capture when countdown reaches 0
        if elapsed >= WEBCAM_COUNTDOWN:
            cv2.imwrite(filename, last_frame)
            print(f"[Camera] Auto-captured: {filename}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return os.path.exists(filename)



# ──────────────────────────────────────────────
# SCREENSHOT: Capture current screen
# ──────────────────────────────────────────────
def capture_screenshot(filename: str = SCREENSHOT_FILE) -> bool:
    """
    Takes a screenshot of the entire screen using pyautogui.
    No webcam needed — captures whatever is on screen right now.
    """
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"[Screenshot] Saved: {filename}")
        return True
    except Exception as e:
        print(f"[Screenshot Error] {e}")
        return False



def detect_objects_yolo(image_path: str) -> str:
    try:
        from ultralytics import YOLO

        # Downloads yolov8n.pt automatically on first run (~6 MB, free)
        model = YOLO("yolov8n.pt")
        results = model(image_path, verbose=False)

        detected = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = result.names[cls_id]
                conf   = float(box.conf[0])
                detected.append(f"{label} ({conf:.0%})")

        if detected:
            unique = list(dict.fromkeys(detected))   # preserve order, remove duplicates
            return "I can see: " + ", ".join(unique) + "."
        else:
            return "I couldn't detect any objects in the image."

    except ImportError:
        return "[Error] ultralytics not installed. Run: pip install ultralytics"
    except Exception as e:
        print(f"[YOLO Error] {e}")
        return "Object detection failed."


# ──────────────────────────────────────────────
# IMAGE CAPTIONING: BLIP (free Hugging Face)
# ──────────────────────────────────────────────
def caption_image_blip(image_path: str) -> str:
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        image  = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_length=60, num_beams=5)

        caption = processor.decode(out[0], skip_special_tokens=True)
        return f"I can see: {caption}"

    except Exception as e:
        print(f"[BLIP Caption Error] {e}")
        return "I couldn't generate a caption for the image."


# ──────────────────────────────────────────────
# IMAGE Q&A: BLIP VQA (free Hugging Face)
# ──────────────────────────────────────────────
def answer_image_question(image_path: str, question: str) -> str:
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

        image  = Image.open(image_path).convert("RGB")
        inputs = processor(image, question, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)

        answer = processor.decode(out[0], skip_special_tokens=True)
        answer = answer.replace(question, "").strip()
        return answer if answer else "I couldn't find a clear answer in the image."

    except Exception as e:
        print(f"[BLIP VQA Error] {e}")
        return "Image Q&A failed. Please try again."


# ──────────────────────────────────────────────
# WEATHER: OpenWeatherMap /data/2.5/weather
# Free tier — 60 calls/min, 1M calls/month, no credit card
# ──────────────────────────────────────────────

# Keywords that signal a weather query
_WEATHER_TRIGGERS = ("weather", "temperature", "forecast", "how hot", "how cold", "climate")

# Prepositions that introduce the city name
_CITY_PREPS = ("in", "at", "for", "of")


def is_weather_query(text: str) -> bool:
    """Return True if the user's input is asking about weather."""
    lowered = text.lower()
    return any(trigger in lowered for trigger in _WEATHER_TRIGGERS)


def extract_city(user_input: str) -> str | None:
    """
    Extract city name from phrases like:
      'weather in Hyderabad', 'temperature at New York',
      'what's the weather in San Francisco today'
    Falls back to the last meaningful word if no preposition found.
    """
    words = user_input.lower().split()

    # Strategy 1: find a preposition after a trigger word → city follows
    for i, word in enumerate(words):
        if word in _WEATHER_TRIGGERS:
            for j in range(i + 1, len(words)):
                if words[j] in _CITY_PREPS and j + 1 < len(words):
                    # City is everything after the preposition (strip trailing noise)
                    city_words = words[j + 1:]
                    # Drop trailing filler words like 'today', 'now', 'please'
                    fillers = {"today", "now", "please", "currently", "right", "there"}
                    city_words = [w for w in city_words if w not in fillers]
                    if city_words:
                        return " ".join(city_words).title()

    # Strategy 2: "weather <city>" with no preposition
    for i, word in enumerate(words):
        if word in _WEATHER_TRIGGERS and i + 1 < len(words):
            candidate = words[i + 1]
            if candidate not in _CITY_PREPS and len(candidate) > 2:
                return candidate.title()

    return None


def _build_weather_message(city: str, data: dict) -> str:
    """Format the OWM JSON response into a natural spoken sentence."""
    main        = data["main"]
    weather     = data["weather"][0]
    wind        = data.get("wind", {})
    visibility  = data.get("visibility", None)

    temp        = main["temp"]
    feels_like  = main["feels_like"]
    temp_min    = main["temp_min"]
    temp_max    = main["temp_max"]
    humidity    = main["humidity"]
    description = weather["description"].capitalize()
    wind_speed  = wind.get("speed", None)

    parts = [
        f"Weather in {city}: {description}.",
        f"Temperature is {temp:.1f}°C, feels like {feels_like:.1f}°C.",
        f"Today's range is {temp_min:.1f}°C to {temp_max:.1f}°C.",
        f"Humidity is {humidity}%.",
    ]
    if wind_speed is not None:
        parts.append(f"Wind speed is {wind_speed:.1f} metres per second.")
    if visibility is not None:
        parts.append(f"Visibility is {visibility // 1000} kilometres.")

    return " ".join(parts)


def get_weather(city: str) -> str:
    """
    Fetch current weather from OpenWeatherMap free /data/2.5/weather endpoint.
    Retries up to OWM_MAX_RETRIES times on transient network errors.
    Handles HTTP 401 (bad key), 404 (city not found), 429 (rate limit) explicitly.
    """
    params = {
        "q":      city,
        "appid":  OPENWEATHER_KEY,
        "units":  "metric",
    }

    for attempt in range(1, OWM_MAX_RETRIES + 1):
        try:
            response = requests.get(
                OWM_BASE_URL,
                params=params,
                timeout=OWM_TIMEOUT,
            )

            # ── Handle HTTP error codes explicitly ──
            if response.status_code == 401:
                return "Weather API key is invalid. Please check your OpenWeatherMap key."
            if response.status_code == 404:
                return f"I couldn't find weather data for '{city}'. Please check the city name and try again."
            if response.status_code == 429:
                return "Weather service rate limit reached. Please try again in a moment."
            if response.status_code != 200:
                print(f"[Weather] HTTP {response.status_code} on attempt {attempt}")
                if attempt < OWM_MAX_RETRIES:
                    time.sleep(1.5)
                    continue
                return "Weather service returned an unexpected error. Please try later."

            data = response.json()

            if "main" not in data:
                return f"No weather data available for '{city}' right now."

            return _build_weather_message(city, data)

        except requests.exceptions.ConnectionError:
            if attempt < OWM_MAX_RETRIES:
                time.sleep(1.5)
                continue
            return "I couldn't connect to the weather service. Please check your internet connection."
        except requests.exceptions.Timeout:
            if attempt < OWM_MAX_RETRIES:
                time.sleep(1)
                continue
            return "Weather request timed out. Please try again."
        except Exception as e:
            print(f"[Weather Error] {e}")
            return "An unexpected error occurred while fetching weather."

    return "Weather service is unavailable right now."


# ──────────────────────────────────────────────
# NEWS: GNews API — free tier
# 100 calls/day, real-time headlines, no credit card
# Sign up: https://gnews.io
# ──────────────────────────────────────────────

_NEWS_TRIGGERS   = ("news", "headlines", "what's happening", "latest", "today's news",
                    "current events", "tell me news", "any news")
_NEWS_CATEGORIES = {
    "technology": "technology", "tech": "technology",
    "sports":     "sports",     "sport": "sports",
    "business":   "business",   "finance": "business",
    "health":     "health",     "science": "science",
    "world":      "world",      "entertainment": "entertainment",
    "nation":     "nation",     "india": "nation",
}


def is_news_query(text: str) -> bool:
    lowered = text.lower()
    return any(trigger in lowered for trigger in _NEWS_TRIGGERS)


def _extract_news_category(text: str) -> str:
    """Return a GNews category if the user mentioned one, else 'general'."""
    lowered = text.lower()
    for keyword, category in _NEWS_CATEGORIES.items():
        if keyword in lowered:
            return category
    return "general"


def _extract_news_topic(text: str) -> str | None:
    """Extract a specific search keyword if user says 'news about X'."""
    patterns = [
        r"news about (.+)",
        r"headlines about (.+)",
        r"latest on (.+)",
        r"what's happening with (.+)",
    ]
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).strip()
    return None


def get_news(user_input: str) -> str:
    """
    Fetch top headlines from GNews free-tier API.
    Supports: category detection, topic search, language=en, country=in (India).
    Free tier: 100 calls/day, real-time articles, no card needed.
    """
    topic    = _extract_news_topic(user_input)
    category = _extract_news_category(user_input)

    params: dict = {
        "token":    GNEWS_API_KEY,
        "lang":     "en",
        "country":  "in",           # India news by default; change to 'us' etc. if needed
        "max":      GNEWS_MAX_RESULTS,
    }

    # If user said "news about cricket" → use search endpoint
    if topic:
        url = "https://gnews.io/api/v4/search"
        params["q"] = topic
    else:
        url = GNEWS_BASE_URL
        if category != "general":
            params["topic"] = category

    try:
        response = requests.get(url, params=params, timeout=GNEWS_TIMEOUT)

        if response.status_code == 401:
            return "News API key is invalid. Please check your GNews API key."
        if response.status_code == 429:
            return "News API rate limit reached for today. You have 100 free calls per day."
        if response.status_code != 200:
            return f"News service returned error {response.status_code}. Please try later."

        data     = response.json()
        articles = data.get("articles", [])

        if not articles:
            subject = f"about {topic}" if topic else f"in {category}"
            return f"I couldn't find any news {subject} right now. Try again later."

        # Build spoken response
        subject_line = f"about {topic}" if topic else (
            f"in {category}" if category != "general" else "today"
        )
        lines = [f"Here are the top {len(articles)} news headlines {subject_line}:"]

        for i, article in enumerate(articles, 1):
            title  = article.get("title", "No title").split(" - ")[0].strip()
            source = article.get("source", {}).get("name", "Unknown source")
            lines.append(f"Headline {i}: {title}. Source: {source}.")

        return " ".join(lines)

    except requests.exceptions.ConnectionError:
        return "I couldn't connect to the news service. Please check your internet."
    except requests.exceptions.Timeout:
        return "News request timed out. Please try again."
    except Exception as e:
        print(f"[News Error] {e}")
        return "An error occurred while fetching news."



def chat_with_groq(prompt: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are BUJJI, a friendly and helpful AI voice assistant. "
                        "Keep answers concise and conversational since they will be spoken aloud. "
                        "Avoid markdown, bullet points, or long lists in your responses."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq Error] {e}")
        return "I'm sorry, I couldn't process that right now."


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def run_session():
    """
    One active session — BUJJI is awake.
    Greets, handles commands, then auto-sleeps after SLEEP_TIMEOUT idle seconds.
    """
    # Wake-up greeting (JARVIS style)
    greetings = [
        "Yes bhairava, I'm here. What do you need?",
        "Online and ready, bhairava. How can I assist?",
        "At your service, bhairava. What's on your mind?",
        "BUJJI online. Ready when you are, bhairava.",
    ]
    import random
    greeting = random.choice(greetings)
    print(f"\n{'─'*60}")
    print(f"🤖 BUJJI ACTIVE: {greeting}")
    print(f"{'─'*60}")
    speak(greeting)

    last_command_time = time.time()

    while _bujji_active.is_set():

        # Auto-sleep after SLEEP_TIMEOUT seconds of no commands
        if time.time() - last_command_time > SLEEP_TIMEOUT:
            sleep_msg = "Going back to standby, bhairava. Clap and call me when you need me."
            print(f"\n💤 BUJJI: {sleep_msg}")
            speak(sleep_msg)
            _bujji_active.clear()
            break

        user_input = get_audio(timeout=5, phrase_limit=12)

        if not user_input:
            continue

        last_command_time = time.time()
        lowered = user_input.lower()

        # NOTE: last_command_time is also reset after each task below
        # so heavy tasks (BLIP/YOLO) don't eat into the idle timeout.

        # ── SLEEP / DISMISS ──
        if any(kw in lowered for kw in ("sleep", "go back", "standby", "that's all",
                                         "thank you bujji", "goodbye bujji", "bye bujji")):
            sleep_msg = "Going to standby. Clap and call me when you need me, bhairava."
            print(f"\n💤 BUJJI: {sleep_msg}")
            speak(sleep_msg)
            _bujji_active.clear()
            break

        # ── EXIT PROGRAM ──
        elif any(kw in lowered for kw in ("exit", "quit", "shutdown bujji", "shut down")):
            farewell = "Shutting down completely. Goodbye bhairava!"
            print(f"\n🔴 BUJJI: {farewell}")
            speak(farewell)
            _bujji_active.clear()
            _bujji_running.clear()
            break

        # ── SCREENSHOT + ANALYZE ──
        elif any(kw in lowered for kw in ("screenshot", "analyze screen", "what's on screen",
                                           "what is on screen", "read screen")):
            print("\n🖥️  Taking screenshot...")
            speak("Taking a screenshot now.")

            if capture_screenshot(SCREENSHOT_FILE):
                speak("Got it. Analyzing...")
                detection = detect_objects_yolo(SCREENSHOT_FILE)
                print(f"🤖 BUJJI (Detection): {detection}")
                speak(detection)

                caption = caption_image_blip(SCREENSHOT_FILE)
                print(f"🤖 BUJJI (Caption): {caption}")
                speak(caption)

                speak("Any questions about what's on screen?")
                question = get_audio()
                if question and question.lower() not in ("skip", "no", "nope", "nothing"):
                    answer = answer_image_question(SCREENSHOT_FILE, question)
                    print(f"🤖 BUJJI: {answer}")
                    speak(answer)
            else:
                msg = "Couldn't take screenshot. Check if pyautogui is installed."
                print(f"🤖 BUJJI: {msg}")
                speak(msg)

        # ── WEBCAM CAPTURE + ANALYZE ──
        elif any(kw in lowered for kw in ("capture", "take photo", "take picture",
                                           "open camera", "webcam")):
            filename = "bujji_capture.jpg"
            print("\n📷 Starting camera...")
            speak("Opening camera. I will capture automatically in 8 seconds.")

            if capture_image(filename):
                speak("Got it. Analyzing...")
                detection_result = detect_objects_yolo(filename)
                print(f"🤖 BUJJI (Detection): {detection_result}")
                speak(detection_result)

                caption = caption_image_blip(filename)
                print(f"🤖 BUJJI (Caption): {caption}")
                speak(caption)

                # Keep a reference to the last captured image for follow-up questions
                last_captured_image = filename

                speak("Any questions?")

                # ── Smart follow-up loop after image capture ──
                # Stays in image context as long as user asks image-related questions.
                # Routes to weather / news / chat if context changes.
                while True:
                    question = get_audio()

                    if not question:
                        break

                    q_lower = question.lower()

                    # User dismisses
                    if any(kw in q_lower for kw in ("skip", "no", "nope", "nothing", "that's all",
                                                     "never mind", "nevermind")):
                        break

                    # ── Detect if question is about the image ──
                    # Triggers: ends with / contains "in the image", "in this image",
                    # "in the photo", "in this photo", "on screen", "in the picture"
                    _image_ctx_phrases = (
                        "in the image", "in this image", "in the photo",
                        "in this photo", "in the picture", "in this picture",
                        "on screen", "what is", "what are", "who is", "who are",
                        "how many", "what color", "is there", "are there",
                        "describe", "tell me about the",
                    )
                    is_image_question = any(ph in q_lower for ph in _image_ctx_phrases)

                    if is_image_question:
                        print(f"[Image Q&A] {question}")
                        answer = answer_image_question(last_captured_image, question)
                        print(f"🤖 BUJJI: {answer}")
                        speak(answer)

                    # ── Context changed to weather ──
                    elif is_weather_query(question):
                        city = extract_city(question)
                        if city:
                            print(f"[Weather] Fetching for {city}...")
                            weather = get_weather(city)
                            print(f"🤖 BUJJI: {weather}")
                            speak(weather)
                        else:
                            speak("Tell me the city name. For example: weather in Hyderabad.")
                        break   # exit image follow-up loop after non-image task

                    # ── Context changed to news ──
                    elif is_news_query(question):
                        print("[News] Fetching news...")
                        speak("On it.")
                        news = get_news(question)
                        print(f"🤖 BUJJI: {news}")
                        speak(news)
                        break

                    # ── General chat / any other topic ──
                    else:
                        print("[Chat] Thinking...")
                        response = chat_with_groq(question)
                        print(f"[BUJJI] {response}")
                        speak(response)
                        break
            else:
                msg = "Couldn't access camera. Please check if it's connected."
                print(f"🤖 BUJJI: {msg}")
                speak(msg)

        # ── NEWS HEADLINES ──
        elif is_news_query(user_input):
            print("\n📰 Fetching news...")
            speak("On it.")
            news = get_news(user_input)
            print(f"🤖 BUJJI: {news}")
            speak(news)

        # ── WEATHER ──
        elif is_weather_query(user_input):
            city = extract_city(user_input)
            if city:
                print(f"\n⏳ Fetching weather for {city}...")
                weather = get_weather(city)
                print(f"🤖 BUJJI: {weather}")
                speak(weather)
            else:
                msg = "Tell me the city name. For example: weather in Hyderabad."
                print(f"🤖 BUJJI: {msg}")
                speak(msg)

        # ── GENERAL CHAT via Groq ──
        else:
            print("\n⏳ Thinking...")
            response = chat_with_groq(user_input)
            print(f"\n🤖 BUJJI: {response}")
            speak(response)

        # Reset timer after every task so processing time doesn't count as idle
        last_command_time = time.time()


def main():
    print("\n" + "═" * 60)
    print("  🤖  B U J J I  —  AI Voice Assistant")
    print("═" * 60)
    print("  STATUS : Standby (silent background mode)")
    print("  WAKE   : Clap once → say 'come on bujji' / 'hey bujji'")
    print("  SLEEP  : Say 'sleep bujji' or idle for 30 seconds")
    print("  EXIT   : Say 'shutdown bujji'")
    print("═" * 60 + "\n")

    # Start wake monitor thread — silent background listener
    monitor_thread = threading.Thread(target=wake_monitor, daemon=True)
    monitor_thread.start()

    # Main control loop
    while _bujji_running.is_set():
        # Wait until wake system triggers activation
        _bujji_active.wait(timeout=0.5)

        if _bujji_active.is_set():
            run_session()   # handle one full active session

    print("\n[BUJJI] Fully shut down. Goodbye bhairava! 👋")


if __name__ == "__main__":
    main()
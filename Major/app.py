import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from deep_translator import GoogleTranslator

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Healthcare Decision Support System", layout="centered")
st.title("🏥 Healthcare Decision Support System Using NLP and Machine Learning")
st.caption("Major Project | ASR + NLP + ML (Real Datasets)")

st.warning("⚠️ This system provides healthcare decision support only. It does NOT diagnose diseases.")

AUDIO_FILE = "recorded_audio.wav"
HISTORY_FILE = "patient_history.csv"
SAMPLE_RATE = 44100

# ================= SESSION STATE =================
if "recording" not in st.session_state:
    st.session_state.recording = False
if "frames" not in st.session_state:
    st.session_state.frames = []
if "stream" not in st.session_state:
    st.session_state.stream = None
if "final_text" not in st.session_state:
    st.session_state.final_text = ""
if "english_text" not in st.session_state:
    st.session_state.english_text = ""
if "voice_summary" not in st.session_state:
    st.session_state.voice_summary = ""

# ================= AUDIO FUNCTIONS =================
def audio_callback(indata, frames, time, status):
    if st.session_state.recording:
        st.session_state.frames.append(indata.copy())

def save_pcm_wav(filename, samplerate, frames):
    if len(frames) == 0:
        return False
    audio = np.concatenate(frames, axis=0)
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    wav.write(filename, samplerate, audio)
    return True

# ================= TEXT UTILITIES =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ================= LOAD SYMPTOM SEVERITY =================
severity_df = pd.read_csv("symptom-severity.csv")
severity_df["Symptom"] = severity_df["Symptom"].str.lower().str.replace("_", " ")
severity_dict = dict(zip(severity_df["Symptom"], severity_df["weight"]))

def detect_severity(text):
    score = 0
    matched = 0
    for symptom, weight in severity_dict.items():
        if symptom in text:
            score += weight
            matched += 1

    if matched == 0:
        return "Normal", 0

    avg_score = score / matched
    if avg_score >= 6:
        return "Emergency", avg_score
    elif avg_score >= 4:
        return "Serious", avg_score
    else:
        return "Normal", avg_score

# ================= SAFE CSV SAVE =================
def save_history(row):
    safe_row = {}
    for k, v in row.items():
        safe_row[k] = str(v).replace(",", " ").replace("\n", " ").strip()
    df = pd.DataFrame([safe_row])
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

# ================= LOAD REAL DATASET =================
dataset = pd.read_csv("dataset.csv")

symptom_cols = [c for c in dataset.columns if c.lower().startswith("symptom")]
dataset["All_Symptoms"] = dataset[symptom_cols].fillna("").agg(" ".join, axis=1)

dataset["All_Symptoms"] = (
    dataset["All_Symptoms"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-z\s]", "", regex=True)
)

def map_category(disease):
    d = disease.lower()
    if "fever" in d:
        return "Fever"
    if "diabetes" in d:
        return "Diabetes"
    if "migraine" in d or "headache" in d:
        return "Headache"
    if "cold" in d or "flu" in d or "cough" in d:
        return "Cold / Cough"
    if "heart" in d or "cardiac" in d:
        return "Chest Pain"
    return "General"

dataset["Category"] = dataset["Disease"].apply(map_category)

# ================= ML TRAINING =================
vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
X = vectorizer.fit_transform(dataset["All_Symptoms"])
y = dataset["Category"]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

doctor_map = {
    "Fever": "General Physician",
    "Cold / Cough": "General Physician",
    "Chest Pain": "Cardiologist",
    "Diabetes": "Endocrinologist",
    "Headache": "Neurologist",
    "General": "General Physician"
}

medicine_map = {
    "Fever": "Antipyretics",
    "Cold / Cough": "Antihistamines",
    "Chest Pain": "Immediate Medical Attention",
    "Diabetes": "Blood Sugar Control Medication",
    "Headache": "Analgesics",
    "General": "General Medication"
}

# ================= INPUT =================
st.header("🧾 Step 1: Patient Input")
mode = st.radio("Select input method:", ["🎙 Voice", "⌨ Text"])

# ================= VOICE INPUT =================
if mode == "🎙 Voice":
    c1, c2, c3 = st.columns(3)

    if c1.button("▶ Start Recording"):
        st.session_state.frames = []
        st.session_state.recording = True
        st.session_state.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback
        )
        st.session_state.stream.start()

    if c2.button("⏹ Stop Recording"):
        if st.session_state.stream:
            st.session_state.recording = False
            st.session_state.stream.stop()
            st.session_state.stream.close()
            st.session_state.stream = None
            save_pcm_wav(AUDIO_FILE, SAMPLE_RATE, st.session_state.frames)
            st.audio(AUDIO_FILE)

    if c3.button("🗑 Re-record"):
        st.session_state.frames = []
        st.session_state.final_text = ""
        st.session_state.english_text = ""

    if st.button("Convert Speech to Text"):
        recognizer = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio_data = recognizer.record(source)
            st.session_state.final_text = recognizer.recognize_google(audio_data)

# ================= TEXT INPUT =================
else:
    text = st.text_area("Describe your symptoms (any language)")
    if st.button("Submit Text"):
        st.session_state.final_text = text

# ================= TRANSLATION =================
if st.session_state.final_text:
    st.subheader("Patient Input")
    st.write(st.session_state.final_text)

    st.session_state.english_text = GoogleTranslator(
        source="auto", target="en"
    ).translate(st.session_state.final_text)

    st.subheader("Translated to English")
    st.write(st.session_state.english_text)

# ================= ANALYSIS =================
st.header("🧠 Step 2: Analysis & Recommendation")

if st.session_state.english_text:
    cleaned = clean_text(st.session_state.english_text)
    X_test = vectorizer.transform([cleaned])

    category = model.predict(X_test)[0]
    severity, severity_score = detect_severity(cleaned)
    doctor = doctor_map.get(category)
    medicine = medicine_map.get(category)

    st.success(f"Detected Category: {category}")
    st.info(f"Severity Level: {severity}")
    st.caption(f"Severity Score: {severity_score:.2f}")
    st.write("👨‍⚕️ Specialist Doctor:", doctor)
    st.write("💊 Medicine Type:", medicine)

    st.session_state.voice_summary = (
        f"Based on your symptoms, the system identified {category}. "
        f"The severity level is {severity}. "
        f"You should consult a {doctor}. "
        f"The recommended medicine category is {medicine}. "
        f"This is a decision support recommendation only."
    )

    save_history({
        "Date": datetime.now(),
        "Input": st.session_state.final_text,
        "Category": category,
        "Severity": severity,
        "Doctor": doctor
    })

# ================= HISTORY =================
st.header("📂 Patient History")
if os.path.exists(HISTORY_FILE):
    st.dataframe(pd.read_csv(HISTORY_FILE, engine="python", on_bad_lines="skip"))

# ================= VOICE OUTPUT =================
st.header("🔊 Voice Output (Analysis & Recommendation)")
if st.button("Generate Voice Recommendation"):
    if st.session_state.voice_summary:
        tts = gTTS(text=st.session_state.voice_summary, lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3")

st.success("✅ Healthcare Decision Support Completed Successfully")

import io
import re
import joblib
import numpy as np
import streamlit as st
import whisper
import nltk
import soundfile as sf
import scipy.signal
import cv2
import mediapipe as mp
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from streamlit_mic_recorder import mic_recorder

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

st.set_page_config(page_title="Voice2Resolve", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice2Resolve")
st.caption("AI-Powered Banking Customer Support Intent Classification")

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

@st.cache_resource
def setup_nltk():
    for name in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(name, quiet=True)
    return True

setup_nltk()

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error("Place best_model.pkl and vectorizer.pkl in the same folder as app.py.")
    st.exception(e)
    st.stop()

with st.spinner("Loading Whisper speech model..."):
    whisper_model = load_whisper_model()

stop_words = set(stopwords.words("english"))
# Keep important negation words used during model preprocessing
stop_words -= {"no", "not", "never", "nor", "cannot"}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def transcribe_audio(audio_bytes):
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    target_rate = whisper.audio.SAMPLE_RATE
    if sample_rate != target_rate:
        new_length = int(round(len(audio) * target_rate / sample_rate))
        audio = scipy.signal.resample(audio, new_length).astype(np.float32)
    result = whisper_model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()

def predict_intent(text):
    cleaned = preprocess_text(text)
    if not cleaned:
        return None, None, cleaned
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector)[0].max() * 100)
    return prediction, confidence, cleaned

st.markdown("---")
st.header("🎤 Speak Your Banking Query")
st.write("Record your voice. Whisper converts speech to text, then the NLP model predicts one of 77 banking intents.")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    format="wav",
    key="banking_recorder"
)

if audio and audio.get("bytes"):
    st.audio(audio["bytes"], format="audio/wav")
    if st.button("🔎 Transcribe & Predict", type="primary"):
        try:
            with st.spinner("Converting speech to text..."):
                recognized_text = transcribe_audio(audio["bytes"])
            st.subheader("📝 Recognized Text")
            st.info(recognized_text)
            with st.spinner("Predicting banking intent..."):
                prediction, confidence, cleaned_text = predict_intent(recognized_text)
            if prediction is None:
                st.warning("No usable text was available after preprocessing.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎯 Predicted Intent")
                    st.success(str(prediction))
                with col2:
                    st.subheader("📊 Confidence Score")
                    st.metric("Confidence", f"{confidence:.2f}%" if confidence is not None else "Not available")
                with st.expander("View Preprocessed Text"):
                    st.write(cleaned_text)
        except Exception as e:
            st.error("Audio processing failed.")
            st.exception(e)

st.markdown("---")
st.header("✋ MediaPipe Hand Detection")
if not MEDIAPIPE_AVAILABLE:
    st.info("MediaPipe/OpenCV is not available. Install requirements.txt.")
else:
    try:
        with open(HAND_MODEL_PATH, "rb"):
            pass
        st.write("Capture an image with your webcam to check for hand detection.")
        image_file = st.camera_input("📷 Capture Hand Image")
        if image_file is not None:
            image_array = np.frombuffer(image_file.getvalue(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2
            )
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                result = detector.detect(mp_image)
            if result.hand_landmarks:
                st.success(f"Hand detected: {len(result.hand_landmarks)} hand(s)")
            else:
                st.info("No hand detected.")
    except FileNotFoundError:
        st.warning("Place hand_landmarker.task in the same folder as app.py to enable hand detection.")
    except Exception as e:
        st.error("Hand detection failed.")
        st.exception(e)

st.markdown("---")
st.caption("Voice2Resolve | Banking77 | Whisper + TF-IDF + Tuned Logistic Regression + Streamlit")

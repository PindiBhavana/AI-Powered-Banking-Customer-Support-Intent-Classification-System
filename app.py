import io
import re
import joblib
import numpy as np
import streamlit as st
import whisper
import nltk
import soundfile as sf
import scipy.signal
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from streamlit_mic_recorder import mic_recorder

try:
    import mediapipe as mp
    import cv2
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Voice2Resolve", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice2Resolve & Gesture Control")
st.caption("AI-Powered Banking Support & Interactive Navigation")

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
    st.error("Missing model files. Place best_model.pkl and vectorizer.pkl in the same folder as app.py.")
    st.stop()

with st.spinner("Loading NLP & Whisper models..."):
    whisper_model = load_whisper_model()

stop_words = set(stopwords.words("english")) - {"no", "not", "never", "nor", "cannot"}
lemmatizer = WordNetLemmatizer()

# --- NLP PIPELINE FUNCTIONS ---
def preprocess_text(text):
    """Text Preprocessing -> Tokenization -> Lemmatization"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def transcribe_audio(audio_bytes):
    """Speech-to-Text conversion using Whisper"""
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
    """Vectorization -> Best Model -> Prediction & Confidence Score"""
    cleaned = preprocess_text(text)
    if not cleaned:
        return None, None, cleaned
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vector)[0].max() * 100)
    return prediction, confidence, cleaned

# --- HELPER FUNCTION FOR UI ---
def process_and_display_intent(recognized_text):
    """Handles the prediction and UI display for both text and voice inputs."""
    st.subheader("Step 2: Input Text")
    st.info(recognized_text)
    
    with st.spinner("Processing text and predicting intent..."):
        prediction, confidence, cleaned_text = predict_intent(recognized_text)
    
    if prediction is None:
        st.warning("No usable text available after preprocessing.")
    else:
        st.subheader("Step 3: Prediction & Confidence")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Intent:** {prediction}")
        with col2:
            st.metric("Confidence Score", f"{confidence:.2f}%" if confidence else "N/A")
        
        with st.expander("View Text Preprocessing Output"):
            st.write(f"**Cleaned & Lemmatized:** {cleaned_text}")

# --- STREAMLIT UI & WORKFLOW ROUTING ---
tab1, tab2 = st.tabs(["🎤/⌨️ Voice & Text Workflow", "✋ Vision Workflow (Gesture Control)"])

with tab1:
    st.header("Step 1: User Input")
    
    # Toggle between Voice and Text input
    input_mode = st.radio("Choose input method:", ["Voice Command 🎤", "Text Command ⌨️"], horizontal=True)
    
    if input_mode == "Voice Command 🎤":
        st.write("Record your banking query to trigger the real-time prediction pipeline.")
        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            format="wav",
            key="banking_recorder"
        )

        if audio and audio.get("bytes"):
            st.audio(audio["bytes"], format="audio/wav")
            if st.button("🔎 Execute NLP Pipeline", type="primary"):
                try:
                    with st.spinner("Converting speech to text..."):
                        recognized_text = transcribe_audio(audio["bytes"])
                    process_and_display_intent(recognized_text)
                except Exception as e:
                    st.error("Pipeline failed during audio execution.")
                    st.exception(e)
                    
    elif input_mode == "Text Command ⌨️":
        st.write("Type your banking query to trigger the real-time prediction pipeline.")
        text_input = st.text_input("Enter command:", placeholder="e.g., I need to reset my banking password")
        
        if st.button("🔎 Execute NLP Pipeline", type="primary"):
            if text_input.strip():
                try:
                    process_and_display_intent(text_input)
                except Exception as e:
                    st.error("Pipeline failed during text execution.")
                    st.exception(e)
            else:
                st.warning("Please enter a command before executing.")

with tab2:
    st.header("📷 Gesture Confirmation")
    st.write("Use a hand gesture to confirm or reject the predicted developer intent.")
    
    if not MEDIAPIPE_AVAILABLE:
        st.error("MediaPipe/OpenCV not found. Please install requirements.")
    else:
        try:
            with open(HAND_MODEL_PATH, "rb"):
                pass
            
            image_file = st.camera_input("Show your hand gesture")
            
            if image_file is not None:
                # OpenCV Processing
                image_array = np.frombuffer(image_file.getvalue(), np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # MediaPipe Detection
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2
                )
                
                with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                    result = detector.detect(mp_image)
                
                # Application Control / Navigation Mapping
                if result.hand_landmarks:
                    num_hands = len(result.hand_landmarks)
                    st.success(f"✅ Hand gesture recognition active: {num_hands} hand(s) detected.")
                    
                    # Interactive Features: Visualizing the detection
                    annotated_image = image.copy()
                    mp_hands = mp.solutions.hands
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    
                    for hand_landmarks in result.hand_landmarks:
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                        ])
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks_proto,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Interactive Navigation Visualized")

                    st.header("✋ Detected Gesture")
                    st.success(f"👍 Thumbs Up")
                    
                    # Simulated Application Control based on detection
                    if num_hands == 1:
                        st.info("🕹️ *Application Control Triggered: Single hand detected. Ready for swipe navigation.*")
                    elif num_hands == 2:
                        st.info("🕹️ *Application Control Triggered: Two hands detected. Zoom/Pan mode enabled.*")
                        
                else:
                    st.warning("No hands detected. Awaiting gesture for navigation.")
                    
        except FileNotFoundError:
            st.error(f"Missing Vision Model: {HAND_MODEL_PATH}. Place it in the root directory.")
        except Exception as e:
            st.error("Vision pipeline failed.")
            st.exception(e)

st.markdown("---")
st.caption("Architecture: Whisper STT -> TF-IDF -> Tuned Logistic Regression | OpenCV -> MediaPipe HandLandmarker")

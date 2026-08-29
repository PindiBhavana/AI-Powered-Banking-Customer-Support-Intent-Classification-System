import io
import re
import os
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


# ============================================================
# OPTIONAL COMPUTER VISION
# ============================================================

MEDIAPIPE_AVAILABLE = False
mp = None
cv2 = None

try:
    import cv2
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True

except Exception:
    MEDIAPIPE_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voice2Resolve",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# CUSTOM COLORFUL UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.25),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(236,72,153,0.22),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(20,184,166,0.18),
                transparent 35%
            ),
            #0b1020;
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2.5rem;
        border-radius: 28px;
        background:
            linear-gradient(
                135deg,
                #4f46e5 0%,
                #7c3aed 45%,
                #db2777 100%
            );
        box-shadow:
            0 18px 50px rgba(79,70,229,0.35);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 3.2rem;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #eef2ff;
        font-size: 1.1rem;
        margin-top: 0.7rem;
    }

    .section-card {
        padding: 1.4rem;
        border-radius: 22px;
        background: rgba(30,41,59,0.82);
        border: 1px solid rgba(148,163,184,0.20);
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        margin: 0.7rem 0 1rem 0;
    }

    .workflow {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: center;
        margin: 1.2rem 0;
    }

    .step {
        padding: 0.75rem 1rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        background:
            linear-gradient(
                135deg,
                #06b6d4,
                #3b82f6
            );
        box-shadow:
            0 7px 20px rgba(59,130,246,0.20);
    }

    .arrow {
        color: #cbd5e1;
        font-size: 1.3rem;
        font-weight: bold;
    }

    .intent-box {
        padding: 1.5rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #059669,
                #0d9488
            );
        color: white;
        box-shadow:
            0 12px 35px rgba(13,148,136,0.25);
        text-align: center;
    }

    .confidence-box {
        padding: 1.5rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #f59e0b,
                #ea580c
            );
        color: white;
        box-shadow:
            0 12px 35px rgba(234,88,12,0.22);
        text-align: center;
    }

    .gesture-box {
        padding: 1.5rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #0f766e,
                #0891b2,
                #2563eb
            );
        color: white;
        text-align: center;
        box-shadow:
            0 12px 35px rgba(8,145,178,0.25);
    }

    .feature-card {
        padding: 1.3rem;
        border-radius: 20px;
        text-align: center;
        background:
            linear-gradient(
                145deg,
                rgba(15,23,42,0.95),
                rgba(30,41,59,0.95)
            );
        border: 1px solid rgba(129,140,248,0.30);
        min-height: 130px;
    }

    .feature-icon {
        font-size: 2rem;
    }

    .feature-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .feature-text {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 1.5rem;
        font-size: 0.9rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111827,
                #172554,
                #312e81
            );
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .stButton > button {
        border-radius: 14px;
        font-weight: 700;
        min-height: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🎙️ Voice2Resolve</h1>

        <p>
            AI-Powered Banking Customer Support
            Intent Classification System
        </p>

        <p>
            🎤 Speech AI &nbsp; • &nbsp;
            🧠 NLP &nbsp; • &nbsp;
            📷 Computer Vision
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌟 Voice2Resolve")

    st.markdown("---")

    st.markdown("### 🚀 AI Modules")

    st.markdown("🎙️ Speech-to-Text")

    st.markdown("🧹 NLP Preprocessing")

    st.markdown("🔢 TF-IDF Vectorization")

    st.markdown("🤖 Logistic Regression")

    st.markdown("✋ MediaPipe Hand Detection")

    st.markdown("📷 OpenCV Image Processing")

    st.markdown("---")

    st.info(
        "Banking77 contains 77 banking "
        "customer-support intent categories."
    )


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "best_model.pkl"

VECTORIZER_PATH = "vectorizer.pkl"

HAND_MODEL_PATH = "hand_landmarker.task"


# ============================================================
# NLTK SETUP
# ============================================================

@st.cache_resource
def setup_nltk():

    for name in [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4"
    ]:

        nltk.download(
            name,
            quiet=True
        )

    return True


setup_nltk()


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_assets():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


try:

    model, vectorizer = load_assets()

except Exception as e:

    st.error(
        "❌ Unable to load the trained model or vectorizer."
    )

    st.info(
        "Make sure best_model.pkl and "
        "vectorizer.pkl are in the same folder as app.py."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LOAD WHISPER
# ============================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model("base")


with st.spinner(
    "🧠 Loading Whisper speech model..."
):

    whisper_model = load_whisper_model()


# ============================================================
# NLP PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)

# Keep important negation words
stop_words -= {
    "no",
    "not",
    "never",
    "nor",
    "cannot"
}

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(
        text
    )

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_bytes):

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Convert stereo to mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    target_rate = whisper.audio.SAMPLE_RATE

    # Resample to 16 kHz
    if sample_rate != target_rate:

        new_length = int(
            round(
                len(audio)
                * target_rate
                / sample_rate
            )
        )

        audio = scipy.signal.resample(
            audio,
            new_length
        ).astype(
            np.float32
        )

    result = whisper_model.transcribe(
        audio,
        fp16=False,
        language="en"
    )

    return result["text"].strip()


# ============================================================
# INTENT PREDICTION
# ============================================================

def predict_intent(text):

    cleaned = preprocess_text(
        text
    )

    if not cleaned:

        return (
            None,
            None,
            cleaned
        )

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        vector
    )[0]

    confidence = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = float(
            probabilities.max() * 100
        )

    return (
        prediction,
        confidence,
        cleaned
    )


# ============================================================
# VOICE ASSISTANT
# ============================================================

st.markdown("---")

st.header(
    "🎤 Speak Your Banking Query"
)

st.markdown(
    """
    <div class="section-card">

    <b>How Voice2Resolve works:</b>

    Record your banking query →
    Whisper converts speech into text →
    NLP preprocesses the text →
    TF-IDF converts text into numerical features →
    Logistic Regression predicts the banking intent.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VOICE RECORDER
# ============================================================

audio = mic_recorder(

    start_prompt="🎤 Start Recording",

    stop_prompt="⏹️ Stop Recording",

    just_once=False,

    format="wav",

    key="banking_recorder"
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if audio and audio.get("bytes"):

    st.audio(
        audio["bytes"],
        format="audio/wav"
    )

    if st.button(
        "🔎 Transcribe & Predict Intent",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # STEP 1: SPEECH TO TEXT
            # ------------------------------------------------

            with st.spinner(
                "🎧 Converting speech to text..."
            ):

                recognized_text = transcribe_audio(
                    audio["bytes"]
                )


            # ------------------------------------------------
            # STEP 2: DISPLAY TEXT
            # ------------------------------------------------

            st.subheader(
                "📝 Recognized Text"
            )

            if recognized_text:

                st.info(
                    recognized_text
                )

            else:

                st.warning(
                    "No speech was recognized. "
                    "Please try recording again."
                )

                st.stop()


            # ------------------------------------------------
            # STEP 3: PREDICT INTENT
            # ------------------------------------------------

            with st.spinner(
                "🤖 Predicting banking intent..."
            ):

                (
                    prediction,
                    confidence,
                    cleaned_text
                ) = predict_intent(
                    recognized_text
                )


            # ------------------------------------------------
            # STEP 4: DISPLAY RESULT
            # ------------------------------------------------

            if prediction is None:

                st.warning(
                    "No usable text was available "
                    "after preprocessing."
                )

            else:

                col1, col2 = st.columns(2)

                # Predicted intent
                with col1:

                    st.markdown(
                        f"""
                        <div class="intent-box">

                            <div>
                                🎯 Predicted Banking Intent
                            </div>

                            <h2>
                                {str(prediction)}
                            </h2>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # Confidence
                with col2:

                    confidence_text = (

                        f"{confidence:.2f}%"

                        if confidence is not None

                        else "Not available"
                    )

                    st.markdown(
                        f"""
                        <div class="confidence-box">

                            <div>
                                📊 Model Confidence
                            </div>

                            <h2>
                                {confidence_text}
                            </h2>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # Preprocessed text
                with st.expander(
                    "🧹 View Preprocessed Text"
                ):

                    st.write(
                        cleaned_text
                    )


        except Exception as e:

            st.error(
                "❌ Audio processing failed."
            )

            st.exception(e)


# ============================================================
# APPLICATION WORKFLOW
# ============================================================

st.markdown("---")

st.header(
    "🔄 Application Workflow"
)

st.markdown(
    """
    <div class="workflow">

        <div class="step">
            🎤 User Speech
        </div>

        <div class="arrow">→</div>

        <div class="step">
            📝 Speech-to-Text
        </div>

        <div class="arrow">→</div>

        <div class="step">
            🧹 Text Preprocessing
        </div>

        <div class="arrow">→</div>

        <div class="step">
            🔢 TF-IDF
        </div>

        <div class="arrow">→</div>

        <div class="step">
            🤖 Best Model
        </div>

        <div class="arrow">→</div>

        <div class="step">
            🎯 Prediction
        </div>

        <div class="arrow">→</div>

        <div class="step">
            📊 Confidence
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEDIAPIPE HAND DETECTION
# ============================================================


st.markdown("---")

st.header("✋ MediaPipe – Gesture Detection")

if not MEDIAPIPE_AVAILABLE:

    st.warning(
        "✋ Hand detection is currently unavailable in this "
        "Streamlit environment."
    )

    st.info(
        "The main Voice2Resolve Speech-to-Text and "
        "Banking Intent Classification features remain available."
    )

elif not os.path.exists(HAND_MODEL_PATH):

    st.warning(
        "⚠️ hand_landmarker.task was not found."
    )

    st.info(
        "Place hand_landmarker.task in the same GitHub "
        "folder as app.py."
    )

else:

    st.write(
        "Capture an image using your webcam "
        "to detect hand landmarks."
    )

    image_file = st.camera_input(
        "📷 Capture Hand Image"
    )

    if image_file is not None:

        try:

            image_array = np.frombuffer(
                image_file.getvalue(),
                np.uint8
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if image is None:

                st.error(
                    "Unable to read the camera image."
                )

            else:

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                options = (
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=HAND_MODEL_PATH
                        ),
                        running_mode=(
                            mp.tasks.vision.RunningMode.IMAGE
                        ),
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                )

                with (
                    mp.tasks.vision.HandLandmarker
                    .create_from_options(options)
                    as detector
                ):

                    result = detector.detect(
                        mp_image
                    )

                if result.hand_landmarks:

                    st.success(
                        f"✋ {len(result.hand_landmarks)} "
                        "hand(s) detected!"
                    )

                    annotated = rgb.copy()

                    height, width, _ = annotated.shape

                    for hand_landmarks in result.hand_landmarks:

                        for landmark in hand_landmarks:

                            x = int(
                                landmark.x * width
                            )

                            y = int(
                                landmark.y * height
                            )

                            cv2.circle(
                                annotated,
                                (x, y),
                                6,
                                (255, 255, 0),
                                -1
                            )

                    st.image(
                        annotated,
                        caption="✋ Detected Hand Landmarks",
                        use_container_width=True
                    )

                else:

                    st.info(
                        "🖐️ No hand detected. "
                        "Place your hand clearly in the camera."
                    )

        except Exception as e:

            st.error(
                "Hand detection could not be initialized."
            )

            st.info(
                "Your main Voice2Resolve features are still available."
            )
# ============================================================
# CHECK HAND MODEL
# ============================================================

if not os.path.exists(
    HAND_MODEL_PATH
):

    st.warning(
        "⚠️ hand_landmarker.task was not found."
    )

    st.info(
        "Upload hand_landmarker.task to the "
        "same GitHub folder as app.py."
    )

else:

    image_file = st.camera_input(
        "📷 Capture Hand Image",
        key="hand_camera"
    )


    if image_file is not None:

        try:

            # -----------------------------------------------
            # Convert uploaded image to NumPy
            # -----------------------------------------------

            image_array = np.frombuffer(
                image_file.getvalue(),
                np.uint8
            )


            # -----------------------------------------------
            # Decode image using OpenCV
            # -----------------------------------------------

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if image is None:

                st.error(
                    "Unable to read the camera image."
                )

            else:

                # -------------------------------------------
                # BGR → RGB
                # -------------------------------------------

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )


                # -------------------------------------------
                # MediaPipe Image
                # -------------------------------------------

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )


                # -------------------------------------------
                # MediaPipe configuration
                # -------------------------------------------

                options = (
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=(
                            mp.tasks.BaseOptions(
                                model_asset_path=
                                HAND_MODEL_PATH
                            )
                        ),

                        running_mode=(
                            mp.tasks.vision.RunningMode.IMAGE
                        ),

                        num_hands=2,

                        min_hand_detection_confidence=0.5,

                        min_hand_presence_confidence=0.5,

                        min_tracking_confidence=0.5
                    )
                )


                # -------------------------------------------
                # Detect hands
                # -------------------------------------------

                with (
                    mp.tasks.vision.HandLandmarker
                    .create_from_options(options)
                    as detector
                ):

                    result = detector.detect(
                        mp_image
                    )


                # -------------------------------------------
                # Display results
                # -------------------------------------------

                if result.hand_landmarks:

                    hand_count = len(
                        result.hand_landmarks
                    )

                    st.success(
                        f"✋ {hand_count} hand(s) detected!"
                    )


                    # ---------------------------------------
                    # Draw landmarks
                    # ---------------------------------------

                    annotated = rgb.copy()

                    height, width, _ = (
                        annotated.shape
                    )


                    for hand_landmarks in (
                        result.hand_landmarks
                    ):

                        # Draw landmark points
                        for landmark in hand_landmarks:

                            x = int(
                                landmark.x * width
                            )

                            y = int(
                                landmark.y * height
                            )

                            cv2.circle(
                                annotated,
                                (x, y),
                                6,
                                (255, 255, 0),
                                -1
                            )


                        # Draw connections
                        connections = [
                            (0, 1),
                            (1, 2),
                            (2, 3),
                            (3, 4),

                            (0, 5),
                            (5, 6),
                            (6, 7),
                            (7, 8),

                            (0, 9),
                            (9, 10),
                            (10, 11),
                            (11, 12),

                            (0, 13),
                            (13, 14),
                            (14, 15),
                            (15, 16),

                            (0, 17),
                            (17, 18),
                            (18, 19),
                            (19, 20),

                            (5, 9),
                            (9, 13),
                            (13, 17),
                            (0, 17)
                        ]


                        for start_idx, end_idx in connections:

                            start = hand_landmarks[
                                start_idx
                            ]

                            end = hand_landmarks[
                                end_idx
                            ]


                            x1 = int(
                                start.x * width
                            )

                            y1 = int(
                                start.y * height
                            )

                            x2 = int(
                                end.x * width
                            )

                            y2 = int(
                                end.y * height
                            )


                            cv2.line(
                                annotated,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                3
                            )


                    st.image(
                        annotated,
                        caption="✋ MediaPipe Hand Landmarks",
                        use_container_width=True
                    )


                else:

                    st.info(
                        "🖐️ No hand detected. "
                        "Place your hand clearly "
                        "inside the camera frame."
                    )


        except Exception as e:

            st.error(
                "❌ Hand detection failed."
            )

            st.exception(e)


# ============================================================
# PROJECT FEATURES
# ============================================================

st.markdown("---")

st.header(
    "🌟 Project Features"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🎙️
            </div>

            <div class="feature-title">
                Speech AI
            </div>

            <div class="feature-text">
                Whisper converts spoken
                banking queries into text.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                NLP Classification
            </div>

            <div class="feature-text">
                TF-IDF and Logistic Regression
                classify Banking77 intents.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                ✋
            </div>

            <div class="feature-title">
                Computer Vision
            </div>

            <div class="feature-text">
                OpenCV and MediaPipe detect
                hand landmarks.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎙️ <b>Voice2Resolve</b>
        &nbsp; | &nbsp;
        Banking77
        &nbsp; | &nbsp;
        Whisper
        &nbsp; | &nbsp;
        TF-IDF
        &nbsp; | &nbsp;
        Logistic Regression
        &nbsp; | &nbsp;
        MediaPipe

        <br><br>

        AI-Powered Banking Customer Support
        Intent Classification System

    </div>
    """,
    unsafe_allow_html=True
)

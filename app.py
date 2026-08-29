```python
import io
import re
import joblib
import numpy as np
import streamlit as st
import whisper
import nltk
import soundfile as sf
import scipy.signal
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from streamlit_mic_recorder import mic_recorder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voice2Resolve",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Voice2Resolve")
st.caption("AI-Powered Banking Customer Support Intent Classification")


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


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
        nltk.download(name, quiet=True)

    return True


setup_nltk()


# ============================================================
# LOAD TRAINED MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:
    model, vectorizer = load_assets()

except Exception as e:
    st.error(
        "Unable to load the trained model or vectorizer. "
        "Make sure best_model.pkl and vectorizer.pkl are in the same folder as app.py."
    )
    st.exception(e)
    st.stop()


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


with st.spinner("Loading Whisper speech model..."):
    whisper_model = load_whisper_model()


# ============================================================
# NLP PREPROCESSING
# ============================================================

stop_words = set(stopwords.words("english"))

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

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_bytes):

    # Read recorded WAV audio
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Convert stereo audio to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Whisper uses 16 kHz audio
    target_rate = whisper.audio.SAMPLE_RATE

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
        ).astype(np.float32)

    # Transcribe audio
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

    # NLP preprocessing
    cleaned = preprocess_text(text)

    if not cleaned:
        return None, None, cleaned

    # TF-IDF vectorization
    vector = vectorizer.transform([cleaned])

    # Model prediction
    prediction = model.predict(vector)[0]

    # Confidence score
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(vector)[0]

        confidence = float(
            probabilities.max() * 100
        )

    return prediction, confidence, cleaned


# ============================================================
# APPLICATION WORKFLOW
# ============================================================

st.markdown("---")

st.header("🎤 Speak Your Banking Query")

st.write(
    "Record your voice. Whisper converts speech into text, "
    "then the NLP model predicts one of the 77 Banking77 intents."
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
# PROCESS RECORDED AUDIO
# ============================================================

if audio and audio.get("bytes"):

    # Display recorded audio
    st.audio(
        audio["bytes"],
        format="audio/wav"
    )

    if st.button(
        "🔎 Transcribe & Predict",
        type="primary"
    ):

        try:

            # ------------------------------------------------
            # STEP 1: SPEECH-TO-TEXT
            # ------------------------------------------------

            with st.spinner(
                "Converting speech to text..."
            ):

                recognized_text = transcribe_audio(
                    audio["bytes"]
                )


            # ------------------------------------------------
            # STEP 2: DISPLAY GENERATED TEXT
            # ------------------------------------------------

            st.subheader("📝 Recognized Text")

            st.info(recognized_text)


            # ------------------------------------------------
            # STEP 3: NLP + PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "Predicting banking intent..."
            ):

                prediction, confidence, cleaned_text = (
                    predict_intent(recognized_text)
                )


            # ------------------------------------------------
            # STEP 4: DISPLAY RESULT
            # ------------------------------------------------

            if prediction is None:

                st.warning(
                    "No usable text was available after preprocessing."
                )

            else:

                col1, col2 = st.columns(2)


                # Predicted intent
                with col1:

                    st.subheader(
                        "🎯 Predicted Intent"
                    )

                    st.success(
                        str(prediction)
                    )


                # Confidence score
                with col2:

                    st.subheader(
                        "📊 Confidence Score"
                    )

                    if confidence is not None:

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                    else:

                        st.metric(
                            "Confidence",
                            "Not available"
                        )


                # ------------------------------------------------
                # PREPROCESSED TEXT
                # ------------------------------------------------

                with st.expander(
                    "View Preprocessed Text"
                ):

                    st.write(cleaned_text)


        except Exception as e:

            st.error(
                "Audio processing failed."
            )

            st.exception(e)


# ============================================================
# APPLICATION WORKFLOW DISPLAY
# ============================================================

st.markdown("---")

st.header("🔄 Application Workflow")

st.markdown(
    """
**User Speech**  
↓  
**Speech-to-Text**  
↓  
**Generated Text**  
↓  
**Text Preprocessing**  
↓  
**TF-IDF Vectorization**  
↓  
**Best Model**  
↓  
**Intent Prediction**  
↓  
**Confidence Score**
"""
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📌 About Voice2Resolve")

st.write(
    "Voice2Resolve is an AI-powered banking customer support "
    "system that converts spoken customer queries into text "
    "using Whisper and automatically classifies the query into "
    "one of 77 banking customer-support intents using NLP, "
    "TF-IDF vectorization, and a tuned Logistic Regression model."
)


st.caption(
    "Voice2Resolve | Banking77 | Whisper + TF-IDF + "
    "Tuned Logistic Regression + Streamlit"
)
```

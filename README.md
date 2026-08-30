# 🎙️ AI-Powered Banking Customer Support Intent Classification System

## About

**Voice2Resolve** is an AI-powered banking customer support application that converts spoken customer queries into text and automatically predicts the appropriate banking intent. The system combines **Speech-to-Text, Natural Language Processing, Machine Learning, and Computer Vision** to create an interactive AI application.

The application uses **OpenAI Whisper** to convert customer speech into text. The generated text is then preprocessed, transformed into numerical features using **TF-IDF Vectorization**, and classified using a trained **Logistic Regression model**.

The system is trained on the **Banking77 dataset**, which contains **77 different banking customer support intent categories**. As an additional interactive feature, the project also integrates **MediaPipe and OpenCV** for hand and gesture detection.

---

# 📌 Project Overview

Banking customer support receives a large number of customer queries related to cards, payments, transfers, cash withdrawals, accounts, transactions, and other banking services.

**Voice2Resolve** provides an AI-based solution that allows users to speak their banking queries instead of typing them. The system converts the speech into text and automatically identifies the customer's intent.

The project follows this complete pipeline:

**User Speech → Speech-to-Text → Generated Text → Text Preprocessing → TF-IDF Vectorization → Machine Learning Model → Intent Prediction → Confidence Score**

An additional computer vision module provides:

**Webcam → OpenCV → MediaPipe → Hand/Gesture Detection**

---

# 🎯 Problem Statement

Traditional customer support systems often require customers to manually select categories or type their problems. This can increase response time and make the customer support process less convenient.

The objective of this project is to develop an AI-powered system that can:

* Accept spoken banking customer queries.
* Convert speech into text.
* Clean and preprocess the generated text.
* Convert text into numerical features.
* Predict the correct banking customer support intent.
* Display the predicted category and model confidence score.
* Provide an additional interactive hand and gesture detection feature using MediaPipe and OpenCV.

The system aims to improve the automation and efficiency of banking customer support classification.

---

# 📊 Dataset

This project uses the **Banking77 dataset**.

The Banking77 dataset contains customer support queries related to banking services and is divided into **77 different intent categories**.

### Dataset Columns

The dataset used for this project contains:

| Column  | Description                           |
| ------- | ------------------------------------- |
| `text`  | Customer banking query                |
| `label` | Corresponding banking intent/category |

### Number of Classes

**77 Banking Intents**

Examples of intent categories include:

* Card payment
* Cash withdrawal
* Money transfer
* Card delivery
* Exchange rate
* Balance inquiry
* Transaction issues
* Account-related queries
* Transfer pending
* Cash withdrawal issues
* And many more banking support categories

The dataset was prepared in CSV format and used to train the NLP classification model.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Web Application

* Streamlit

### Speech-to-Text

* OpenAI Whisper
* streamlit-mic-recorder
* SoundFile
* SciPy

### Natural Language Processing

* NLTK
* Regular Expressions
* Stopwords Removal
* Tokenization
* Lemmatization

### Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Joblib

### Computer Vision

* MediaPipe
* OpenCV

### Data Processing

* NumPy
* Pandas

---

# ⚙️ Features Used

## 1. 🎤 Speech-to-Text

Users can record their banking queries directly through the Streamlit application.

The audio is processed using **OpenAI Whisper**, which converts spoken English into text.

Example:

**User Speech:**

> "My card payment is pending."

**Generated Text:**

> my card payment is pending

---

## 2. 🧹 Text Preprocessing

The generated text is cleaned before being sent to the machine learning model.

The preprocessing steps include:

* Converting text to lowercase
* Removing special characters and unnecessary symbols
* Tokenization
* Removing stopwords
* Preserving important negation words such as:

  * no
  * not
  * never
  * nor
  * cannot
* Lemmatization

---

## 3. 🔢 TF-IDF Vectorization

Machine learning models cannot directly understand raw text.

Therefore, the cleaned customer query is converted into numerical features using **TF-IDF Vectorization**.

The TF-IDF representation captures the importance of words and phrases within the customer query.

---

## 4. 🤖 Banking Intent Prediction

The trained machine learning model receives the TF-IDF features and predicts one of the **77 banking intent categories**.

The application displays:

* Predicted Banking Intent
* Model Confidence Score
* Recognized Text
* Preprocessed Text

---

## 5. ✋ MediaPipe Hand and Gesture Detection

As an additional feature, the project integrates **MediaPipe and OpenCV** for hand detection.

The workflow is:

**Webcam → OpenCV Image Processing → MediaPipe → Hand Landmark Detection**

The system can:

* Capture an image from the webcam.
* Detect one or more hands.
* Identify hand landmarks.
* Display detected hand landmark points.
* Draw connections between hand landmarks.

This feature demonstrates the integration of **Computer Vision** with the main Speech AI and NLP application.

---

# 🧠 Machine Learning Model

Multiple machine learning models were evaluated during the project development.

The final selected model was:

## Logistic Regression

The model was trained using:

* **TF-IDF Vectorized Text Features**
* **Tuned Logistic Regression**
* **C = 10**
* **max_iter = 1000**
* **random_state = 42**

### Final Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.06% |
| Precision | 86.75% |
| Recall    | 86.06% |
| F1-Score  | 86.12% |

The tuned Logistic Regression model was selected as the final model for deployment.

The trained model and TF-IDF vectorizer were saved using Joblib.

* `best_model.pkl`
* `vectorizer.pkl`

---

# 🔄 Project Workflow

## Main Voice Classification Workflow

```text
User Speech
     ↓
Speech-to-Text using Whisper
     ↓
Generated Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Tuned Logistic Regression Model
     ↓
Banking Intent Prediction
     ↓
Confidence Score
```

## Gesture Detection Workflow

```text
Webcam
     ↓
OpenCV
     ↓
MediaPipe
     ↓
Hand Landmark Detection
     ↓
Gesture / Interactive Features
```

---

# 📁 Project Structure

```text
AI-Powered-Banking-Customer-Support-Intent-Classification-System/
│
├── app.py
├── requirements.txt
├── packages.txt
├── best_model.pkl
├── vectorizer.pkl
├── hand_landmarker.task
├── dataset.csv
│
└── model_building.ipynb
```


---

# 🌟 Key Features

* 🎙️ Voice recording through the Streamlit application
* 🗣️ Speech-to-Text using OpenAI Whisper
* 🧹 NLP text preprocessing
* 🔢 TF-IDF text vectorization
* 🤖 Banking intent classification
* 📊 Confidence score display
* 📝 Recognized text display
* 🔍 Preprocessed text display
* 🏦 Classification into 77 banking customer support intents
* ✋ MediaPipe hand landmark detection
* 📷 Webcam image capture
* 👁️ OpenCV image processing
* 🎨 Colorful and interactive Streamlit interface

---

# 🚀 Future Enhancements

The following improvements can be added to the project in the future:

* Add support for multiple languages.
* Use real-time streaming speech recognition.
* Integrate a Large Language Model for conversational customer support.
* Add a chatbot interface for follow-up questions.
* Provide recommended solutions for each predicted banking intent.
* Integrate the system with a real banking knowledge base.
* Use deep learning models such as LSTM, BERT, or Transformer-based classifiers.
* Add a database to store customer queries and prediction history.
* Create an admin dashboard for monitoring common customer issues.
* Deploy the application using cloud infrastructure for scalable real-world usage.

---

# 🎯 Conclusion

**Voice2Resolve** demonstrates how multiple AI technologies can be combined into a single real-world application. The project integrates **Speech Recognition, Natural Language Processing, Machine Learning, and Computer Vision** to automate banking customer support intent classification.

Users can speak their banking queries, and the application automatically converts the speech into text, preprocesses the text, applies TF-IDF vectorization, and predicts the most relevant banking intent using a trained Logistic Regression model.

The additional **MediaPipe and OpenCV gesture detection feature** further demonstrates the use of computer vision and interactive AI technologies.

This project shows a complete end-to-end machine learning workflow, from dataset preparation and model development to real-time Streamlit deployment.

---

# 🙏 Acknowledgements

Innomatics Research Labs

Upender Muthyala

Sonam Pawar

Kaggle : banking77

---


# 👨‍💻 Author

# Pindi Bhavana

If you found this project useful, consider giving it a ⭐ on GitHub!

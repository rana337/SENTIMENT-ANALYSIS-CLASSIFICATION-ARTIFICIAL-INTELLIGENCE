import numpy as np
import streamlit as st
import nltk
import joblib
import string
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Title of the GUI
st.title('Sentiment Analysis GUI')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the TF-IDF vectorizer and models
vectorizer = joblib.load('tfidf_vectorizer.pkl')
models = {
    "Naive Bayes": joblib.load('nb_model.pkl'),
    "SVM": joblib.load('svm_model.pkl'),
    "Logistic Regression": joblib.load('lr_model.pkl')
}

# Dropdown for selecting the model
selected_model = st.selectbox("Select Model", list(models.keys()))

# Button to submit the text for sentiment analysis
if st.button('Analyze'):
    # If SVM model is selected, display accuracy and plots
    if selected_model == "SVM":
        st.subheader("SVM Model Evaluation")
        # Load SVM accuracies
        svm_accuracies = joblib.load("svm_accuracies.pkl")
        st.write("Accuracy:", svm_accuracies)
        # Load and display SVM plot
        svm_plot = plt.imread("svm_plot.png")
        plt.imshow(svm_plot)
        plt.axis('off')  # Hide axis
        st.pyplot()

    # If Naive Bayes model is selected, display accuracy and plots
    elif selected_model == "Naive Bayes":
        st.subheader("Naive Bayes Model Evaluation")
        # Load Naive Bayes accuracies
        nb_accuracies = joblib.load("nb_accuracies.pkl")
        st.write("Accuracy:", nb_accuracies)
        # Load and display Naive Bayes plot
        nb_plot = plt.imread("nb_plot.png")
        plt.imshow(nb_plot)
        plt.axis('off')  # Hide axis
        st.pyplot()

    # If Logistic Regression model is selected, display accuracy and plots
    elif selected_model == "Logistic Regression":
        st.subheader("Logistic Regression Model Evaluation")
        # Load Logistic Regression accuracies
        lr_accuracies = joblib.load("lr_accuracies.pkl")
        st.write("Accuracy:", lr_accuracies)
        # Load and display Logistic Regression plot
        lr_plot = plt.imread("lr_plot.png")
        plt.imshow(lr_plot)
        plt.axis('off')  # Hide axis
        st.pyplot()

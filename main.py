"""
# Import Librarys
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn import svm
import joblib


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""# Reading the Dataset into a DataFrama"""

data = pd.read_csv ("sentimentdataset.csv")

"""# Display data"""

data.head()

"""# Drop coulmns"""

x = ['ID','Timestamp','User','Retweets','Likes','Day','Country','Month','Year','Hour']
data.drop(x, axis = 1 , inplace = True)

"""# Display Data"""

data.head()

"""# Replacing the DataSet"""

data.replace(to_replace =" Positive  ", value = "Positive",  inplace = True)
data.replace(to_replace ="Positive ", value = "Positive",  inplace = True)
data.replace(to_replace =" Positive ", value = "Positive",  inplace = True)

data.replace(to_replace =" Negative  ", value = "Negative",  inplace = True)
data.replace(to_replace ="Negative     ", value = "Negative",  inplace = True)
data.replace(to_replace ="'Negative", value = "Negative",  inplace = True)

data.replace(to_replace =" Neutral   ", value = "Neutral",  inplace = True)
data.replace(to_replace ="Neutral", value = "Neutral",  inplace = True)
data.replace(to_replace =" Neutral ", value = "Neutral",  inplace = True)

data.replace(to_replace =" Surprise ", value = "Positive",  inplace = True)
data["Sentiment (Label)"].unique()

"""# Pre_Processing Text column Function"""

def pre_process_text (text):
    # convert text to lower case
    text = text.lower()
    # Remove Punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stop words
    stop_words = set (stopwords.words('english'))
    text = ' '.join ([word for word in text.split() if word not in stop_words])
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

"""# Call the Function"""

data ['clean_text'] = data['Text'].apply(pre_process_text)

"""# Apply label encoding"""

label_encoder = LabelEncoder()
data['Topic_encoded'] = label_encoder.fit_transform(data['Topic'])
data['source_encoded'] = label_encoder.fit_transform(data['Source'])
data['Sentiment_encoded'] = label_encoder.fit_transform(data['Sentiment (Label)'])


"""# Feature Engineering: TF-IDF with bigrams"""

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
x_label=vectorizer.fit_transform(data['clean_text']).toarray()
least_feature=data[['source_encoded','Topic_encoded']].values
x_final=pd.concat([pd.DataFrame(x_label),pd.DataFrame(least_feature)],axis=1)
y=data['Sentiment_encoded']

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(x_final,y)

# Save the trained TF-IDF vectorizer (GUI)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

"""# Split data into training,testing and validation"""

x_train_temp,x_temp,y_train_temp,y_temp=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=30)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.1,random_state=30)

"""# Naive Bayes Model"""

# Model Training: Multinomial Naive Bayes on training data
NB_model = MultinomialNB()
NB_model.fit(x_train_temp, y_train_temp)  # Training on training data

# Save the trained NB model (GUI)
joblib.dump(NB_model, 'nb_model.pkl')

# Predict the response for training dataset
NB_train_pred = NB_model.predict(x_train_temp)
train_accuracy = accuracy_score(y_train_temp,NB_train_pred)
print("Multinomial Naive Bayes Training Performance:")
print(f'Accuracy: {train_accuracy}')
print(classification_report(y_train_temp, NB_train_pred, zero_division=1))

# Model Validation and Evaluation on validation data
NB_val_pred = NB_model.predict(x_val)
val_accuracy = accuracy_score(y_val, NB_val_pred)
print("Multinomial Naive Bayes Validation Performance:")
print(f'Accuracy: {val_accuracy}')
print(classification_report(y_val, NB_val_pred, zero_division=1))

# Model Evaluation on test data
NB_test_pred = NB_model.predict(x_test)
test_accuracy = accuracy_score(y_test, NB_test_pred)
print("\nMultinomial Naive Bayes Test Performance:")
print(f'Accuracy: {test_accuracy}')
print(classification_report(y_test, NB_test_pred, zero_division=1))

# Save the  NB model accuracy (GUI)
joblib.dump(test_accuracy, 'nb_accuracies.pkl')

# Plotting Multinomial Naive Bayes accuracies
labels = ['Training', 'Validation', 'Test']
accuracies = [train_accuracy, val_accuracy, test_accuracy]

# Bar graph
plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.title('Multinomial Naive Bayes Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization
plt.show()

# Scatter plot
plt.scatter(labels, accuracies, color='blue', label='Accuracy')

# Plot
plt.plot(labels, accuracies, color='red', linestyle='-', marker='o')

plt.title('Multinomial Naive Bayes Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization
plt.legend()
# Save the plot as an image (GUI)
plt.savefig("svm_plot.png")
plt.show()


"""# SVM model"""

# Create an SVM Classifier with one-vs-rest strategy for multi-class classification
clf = SVC(kernel='linear', decision_function_shape='ovr')

# Train the model using the training sets
clf.fit(x_train_temp, y_train_temp)

# Save the trained SVM model (GUI)
joblib.dump(clf, 'svm_model.pkl')

# Predict the response for training dataset
svm_train_pred = clf.predict(x_train_temp)
print("Training Accuracy:", train_accuracy)
print("Training Classification Report:\n", classification_report(y_train_temp, svm_train_pred))

# Predict the response for validation dataset
svm_val_pred = clf.predict(x_val)
print("Validation Accuracy:", accuracy_score(y_val, svm_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, svm_val_pred))

# Predict the response for test dataset
svm_test_pred = clf.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, svm_test_pred))
print("Test Classification Report:\n", classification_report(y_test, svm_test_pred))

# Save the SVM model accuracy (GUI)
joblib.dump(test_accuracy, 'svm_accuracies.pkl')

# Plotting SVM accuracies
labels = ['Training', 'Validation', 'Test']
accuracies = [train_accuracy, val_accuracy, test_accuracy]

# Bar graph
plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.title('SVM Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization
plt.show()


# Scatter plot
plt.scatter(labels, accuracies, color='blue', label='Accuracy')

# Plot
plt.plot(labels, accuracies, color='red', linestyle='-', marker='o')

plt.title('SVM Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization
plt.legend()
# Save the plot as an image (GUI)
plt.savefig("nb_plot.png")

plt.show()


plt.close()

"""# Logistic Regression Model"""

# Model Training: Logistic Regression with L2 Regularization on training data
lr_model = LogisticRegression(max_iter=1000, penalty='l2', C=15)
lr_model.fit(x_train_temp, y_train_temp)  # Training on training data

# Save the trained LR model (GUI)
joblib.dump(NB_model, 'lr_model.pkl')

# Predict the response for training dataset
lr_train_pred = lr_model.predict(x_train_temp)
train_accuracy = accuracy_score(y_train_temp,lr_train_pred)
print("Logistic Regression Training Performance:")
print(f'Accuracy: {train_accuracy}')
print(classification_report(y_train_temp, lr_train_pred, zero_division=1))

# Model Validation and Evaluation on validation data
lr_val_pred = lr_model.predict(x_val)
val_accuracy = accuracy_score(y_val, lr_val_pred)
print("Logistic Regression Validation Performance:")
print(f'Accuracy: {val_accuracy}')
print(classification_report(y_val, lr_val_pred, zero_division=1))

# Model Evaluation on test data
lr_test_pred = lr_model.predict(x_test)
test_accuracy = accuracy_score(y_test, lr_test_pred)
print("\nLogistic Regression Test Performance:")
print(f'Accuracy: {test_accuracy}')
print(classification_report(y_test, lr_test_pred, zero_division=1))

# Save the LR model accuracy (GUI)
joblib.dump(test_accuracy, 'lr_accuracies.pkl')

# Plotting Logistic regression accuracies
labels = ['Training', 'Validation', 'Test']
accuracies = [train_accuracy, val_accuracy, test_accuracy]

# Bar graph
plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.title('Logistic regression Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization

# Display the bar plot
plt.show()

# Scatter plot
plt.scatter(labels, accuracies, color='blue', label='Accuracy')

# Plot
plt.plot(labels, accuracies, color='red', linestyle='-', marker='o')

plt.title('Logistic regression Accuracy Comparison')
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for better visualization
plt.legend()

# Save the plot as an image (GUI)
plt.savefig("lr_plot.png")

# Display the scatter plot
plt.show()



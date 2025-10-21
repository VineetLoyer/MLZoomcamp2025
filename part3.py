import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.optim as optim

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# i. Read the data as a Pandas DataFrame and keep only the review body and star rating columns. Rename them to review and rating respectively.
df = pd.read_csv('reviews.tsv', sep='\t',dtype={"star_rating": "Int64", "review_body": "string"},on_bad_lines='skip',quoting=3, low_memory=False)

df = df[['review_body', 'star_rating']]
df = df.rename(columns={'review_body': 'review', 'star_rating': 'rating'})

# ii. Create binary labels from the ratings. Ratings greater than 3 are positive (1), and ratings less than or equal to 2 are negative (0). Discard all reviews with a rating of 3(neutral).
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

positive_count = (df['rating'] > 3).sum()
negative_count = (df['rating'] <= 2).sum()
neutral_count = (df['rating'] == 3).sum()


# iii. Print the number of positive, negative, and neutral reviews from the original dataset before any downsizing. The output must be three separate lines.
print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Neutral reviews: {neutral_count}")

df = df[df['rating'] != 3]
df['label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)


# iv. To create a balanced dataset, randomly sample 100,000 positive reviews and 100,000 negative reviews. Use random state=42 for this sampling step to ensure reproducibility.

df_pos = df[df['label'] == 1].sample(n=100000, random_state=42)
df_neg = df[df['label'] == 0].sample(n=100000, random_state=42)
df_balanced = pd.concat([df_pos, df_neg], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# v. Split your downsized dataset into a training set (80%) and a testing set (20%). Use random state=42.
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)


# Data Cleaning 
#############################################################################################################################################
## Implement a function to preprocess the review text
def clean_text(text):
    if not isinstance(text, str):
        return ""  # or: str(text)
    # i. Convert to lowercase
    text = text.lower()
    # ii. Remove HTML & URLs
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # iii.  Remove non [a-z] characters and extra whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    # iv. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# v. Apply this cleaning function to the training data. Print the average character length of reviews in the training set before and after cleaning, formatted to 4 decimal places. The output must be two separate lines

avg_before = train_df['review'].str.len().mean()
train_df['cleaned_review'] = train_df['review'].apply(clean_text)
avg_after = train_df['cleaned_review'].str.len().mean()

print(f"Average length before cleaning: {avg_before:.4f}")
print(f"Average length after cleaning: {avg_after:.4f}")

#################################################################################################################################################

# Preprocessing
##################################################################################################################################################

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def full_preprocess(text):
    if not isinstance(text, str):
        return ""
    # Basic Cleaning
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)           # HTML tags
    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text) # URLs
    text = re.sub(r'[^a-z\\s]', ' ', text)       # Keep only a-z and whitespace
    text = re.sub(r'\\s+', ' ', text).strip()    # Extra whitespace
    # Tokenize
    tokens = text.split()
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words] # i. Remove stop words using NLTK’s English stop word list
    # Lemmatize (default settings, no POS)
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # ii. Perform lemmatization on the words using NLTK’s WordNetLemmatizer. For reproducibility, do not use Part-of-Speech (POS) tags; lemmatize each word with the default settings.
    # Join back to string
    return ' '.join(tokens)

train_df['preprocessed_review'] = train_df['review'].apply(full_preprocess) 
avg_final = train_df['preprocessed_review'].str.len().mean()
print(f"Average length after preprocessing: {avg_final:.4f}") # iii. Print the average character length of reviews after this final preprocessing stage, formatted to 4 decimal places. 

####################################################################################################################################################


# Feature Extraction
####################################################################################################################################################
# After creating the TF-IDF vectorizer, fit it on the training data and transform both the training and testing data. Print the dimensions (shape) of the resulting training TF-IDF matrix
test_df['preprocessed_review'] = test_df['review'].apply(full_preprocess)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['preprocessed_review'])
X_test = vectorizer.transform(test_df['preprocessed_review'])
print(f"TF-IDF matrix shape: {X_train.shape}")

######################################################################################################################################################



# Model Training and Evaluation
######################################################################################################################################################

# PERCEPTRON

perceptron = Perceptron(max_iter=20,eta0=0.01,tol=None,shuffle=True,random_state=42,fit_intercept=True,penalty=None)
perceptron.fit(X_train, train_df['label'])
train_preds = perceptron.predict(X_train)
test_preds = perceptron.predict(X_test)

train_acc = accuracy_score(train_df['label'], train_preds)
train_prec = precision_score(train_df['label'], train_preds)
train_rec = recall_score(train_df['label'], train_preds)
train_f1 = f1_score(train_df['label'], train_preds)

test_acc = accuracy_score(test_df['label'], test_preds)
test_prec = precision_score(test_df['label'], test_preds)
test_rec = recall_score(test_df['label'], test_preds)
test_f1 = f1_score(test_df['label'], test_preds)

print(f"Perceptron Training Accuracy: {train_acc:.4f}")
print(f"Perceptron Training Precision: {train_prec:.4f}")
print(f"Perceptron Training Recall: {train_rec:.4f}")
print(f"Perceptron Training F1-score: {train_f1:.4f}")
print(f"Perceptron Testing Accuracy: {test_acc:.4f}")
print(f"Perceptron Testing Precision: {test_prec:.4f}")
print(f"Perceptron Testing Recall: {test_rec:.4f}")
print(f"Perceptron Testing F1-score: {test_f1:.4f}")


# SVM
svm = LinearSVC(random_state=42)
svm.fit(X_train, train_df['label'])
svm_train_preds = svm.predict(X_train)
svm_test_preds = svm.predict(X_test)

svm_train_acc = accuracy_score(train_df['label'], svm_train_preds)
svm_train_prec = precision_score(train_df['label'], svm_train_preds)
svm_train_rec = recall_score(train_df['label'], svm_train_preds)
svm_train_f1 = f1_score(train_df['label'], svm_train_preds)

svm_test_acc = accuracy_score(test_df['label'], svm_test_preds)
svm_test_prec = precision_score(test_df['label'], svm_test_preds)
svm_test_rec = recall_score(test_df['label'], svm_test_preds)
svm_test_f1 = f1_score(test_df['label'], svm_test_preds)

print(f"SVM Training Accuracy: {svm_train_acc:.4f}")
print(f"SVM Training Precision: {svm_train_prec:.4f}")
print(f"SVM Training Recall: {svm_train_rec:.4f}")
print(f"SVM Training F1-score: {svm_train_f1:.4f}")
print(f"SVM Testing Accuracy: {svm_test_acc:.4f}")
print(f"SVM Testing Precision: {svm_test_prec:.4f}")
print(f"SVM Testing Recall: {svm_test_rec:.4f}")
print(f"SVM Testing F1-score: {svm_test_f1:.4f}")


# LOGISTIC REGRESSION

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, train_df['label'])

logreg_train_preds = logreg.predict(X_train)
logreg_test_preds = logreg.predict(X_test)

logreg_train_acc = accuracy_score(train_df['label'], logreg_train_preds)
logreg_train_prec = precision_score(train_df['label'], logreg_train_preds)
logreg_train_rec = recall_score(train_df['label'], logreg_train_preds)
logreg_train_f1 = f1_score(train_df['label'], logreg_train_preds)

logreg_test_acc = accuracy_score(test_df['label'], logreg_test_preds)
logreg_test_prec = precision_score(test_df['label'], logreg_test_preds)
logreg_test_rec = recall_score(test_df['label'], logreg_test_preds)
logreg_test_f1 = f1_score(test_df['label'], logreg_test_preds)

print(f"Logistic Regression Training Accuracy: {logreg_train_acc:.4f}")
print(f"Logistic Regression Training Precision: {logreg_train_prec:.4f}")
print(f"Logistic Regression Training Recall: {logreg_train_rec:.4f}")
print(f"Logistic Regression Training F1-score: {logreg_train_f1:.4f}")
print(f"Logistic Regression Testing Accuracy: {logreg_test_acc:.4f}")
print(f"Logistic Regression Testing Precision: {logreg_test_prec:.4f}")
print(f"Logistic Regression Testing Recall: {logreg_test_rec:.4f}")
print(f"Logistic Regression Testing F1-score: {logreg_test_f1:.4f}")
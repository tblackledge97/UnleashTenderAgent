# Training and testing + evaluating the tenders model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import json
import numpy as np
import pickle
import joblib

FILE_PATH = "/Users/tristanblackledge/TenderAusAgent/ai-agent-project/tender_scraper/RSS_tenders_data.json"


def loads_tenders_data(file_path):
    """Load tender data"""
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_data(tenders):
    """Combines features and splits into training/validation/test sets."""

    # feature extraction
    full_text = [tenders[i].get('Title', '') + ' ' +
                 tenders[i].get('description', '') for i in range(len(tenders))]
    keyword_scores = np.array([tender.get('keyword_score', 0) for tender in tenders]).reshape(-1, 1)


    # target feature (labels)
    labels = np.array([tender.get('is_relevant') for tender in tenders])

    # split indicies to keep all features aligned
    indices = np.arange(len(tenders))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels, random_state=42, test_size=0.4, stratify=labels)

    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, random_state=42, test_size=0.5, stratify=y_temp)

    # Fit vectorizer on training data
    full_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    # use the combined text strings corresponding to the training indicies
    train_text = [full_text[i] for i in train_idx]
    full_vectorizer.fit(train_text)

    # Transform all splits using the fitted vectorizer
    X_train = full_vectorizer.transform(train_text)
    X_val = full_vectorizer.transform([full_text[i] for i in val_idx])
    X_test = full_vectorizer.transform([full_text[i] for i in test_idx])

    # add the keyword scores
    X_train_kw = keyword_scores[train_idx]
    X_val_kw = keyword_scores[val_idx]
    X_test_kw = keyword_scores[test_idx]

    # combine the matrices
    X_train = hstack([X_train, csr_matrix(X_train_kw)])
    X_val = hstack([X_val, csr_matrix(X_val_kw)])
    X_test = hstack([X_test, csr_matrix(X_test_kw)])

    return X_train, X_val, X_test, y_train, y_val, y_test, full_vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains the logisitc regression models
    Evaudate the model against the test data."""

    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, C=0.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=True)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=True)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("CM\n", confusion_matrix(y_test, y_pred))

    return model


if __name__ == "__main__":
    tenders_data = loads_tenders_data(FILE_PATH)

    # prep the data
    X_train, x_val, X_test, y_train, y_val, y_test, vectorizer = prepare_data(tenders_data)

    # train model
    final_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # save the model and vectorizer
    joblib.dump(final_model, 'RSS_tender_relevance_model.pkl')
    joblib.dump(vectorizer, 'RSS_tfidf_vectorizer.pkl')
